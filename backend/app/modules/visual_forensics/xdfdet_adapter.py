"""Process-isolated adapter for the pretrained xdfdet face detector."""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any
from uuid import uuid4

from app.modules.video.frame_extraction import FrameExtractionResult
from app.schemas.visual_detection import (
    XdfdetAnalysisResponse,
    XdfdetFrameResult,
)

LOGGER = logging.getLogger(__name__)
CHECKPOINT = "aug-cutout-black"
MINIMUM_EXTRACTED_FRAMES = 12
DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_WORKER_TIMEOUT_SECONDS = 600.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180.0
_STOP = object()

_LIMITATIONS = [
    "xdfdet targets facial manipulation and is not a universal detector for fully AI-generated video.",
    "A high real probability does not prove authenticity; a low real probability does not prove manipulation.",
    "The decision threshold is provisional configuration, not a scientifically validated RMF threshold.",
    "The Hugging Face checkpoint is licensed CC BY-NC 4.0 and is restricted to non-commercial use.",
    "Inference uses the extracted and sampled frames supplied by the existing RMF video pipeline.",
]


def _validate_positive_timeout(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite number greater than zero.")
    return float(value)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_python() -> Path:
    configured_python = os.environ.get("RMF_XDFDET_PYTHON")
    if configured_python is not None:
        return Path(configured_python).expanduser()
    if os.name == "nt":
        return _backend_root() / ".venv-xdfdet" / "Scripts" / "python.exe"
    return _backend_root() / ".venv-xdfdet" / "bin" / "python"


class XdfdetAdapter:
    """Manage a cached xdfdet worker and exchange extracted-frame references."""

    def __init__(
        self,
        python_executable: str | Path | None = None,
        worker_script: str | Path | None = None,
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        if (
            isinstance(decision_threshold, bool)
            or not isinstance(decision_threshold, (int, float))
            or not math.isfinite(decision_threshold)
            or not 0 <= decision_threshold <= 1
        ):
            raise ValueError(
                "Decision threshold must be a finite probability between 0 and 1."
            )

        self.python_executable = (
            Path(python_executable).expanduser()
            if python_executable is not None
            else _default_python()
        )
        self.worker_script = Path(
            worker_script or Path(__file__).with_name("xdfdet_worker.py")
        )
        self.decision_threshold = float(decision_threshold)
        self.worker_timeout_seconds = _validate_positive_timeout(
            worker_timeout_seconds, "Worker timeout"
        )
        self.startup_timeout_seconds = _validate_positive_timeout(
            startup_timeout_seconds, "Worker startup timeout"
        )
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[object] | None = None
        self._model_loading_seconds: float | None = None

    def analyze(
        self, extraction: FrameExtractionResult
    ) -> XdfdetAnalysisResponse:
        """Analyze saved pipeline frames; never reopen or decode the source video."""
        if not isinstance(extraction, FrameExtractionResult):
            return self._error(0, "invalid_input", "A video frame extraction result is required.")
        frames = extraction.frames
        if not frames:
            return self._inconclusive(0, "No frames were extracted from the video.")
        if len(frames) < MINIMUM_EXTRACTED_FRAMES:
            return self._inconclusive(
                len(frames),
                f"At least {MINIMUM_EXTRACTED_FRAMES} extracted frames are required; "
                f"received {len(frames)}.",
            )
        try:
            output_directory = extraction.output_directory.resolve(strict=True)
            for frame in frames:
                frame_path = frame.path.resolve(strict=True)
                if not frame_path.is_relative_to(output_directory):
                    return self._error(
                        len(frames),
                        "invalid_input",
                        "An extracted frame path is outside its output directory.",
                    )
                if not frame_path.is_file():
                    return self._error(
                        len(frames),
                        "frame_not_found",
                        f"Extracted frame does not exist: {frame.path}",
                    )
                if (
                    isinstance(frame.frame_index, bool)
                    or not isinstance(frame.frame_index, int)
                    or frame.frame_index < 0
                    or isinstance(frame.timestamp_seconds, bool)
                    or not isinstance(frame.timestamp_seconds, (int, float))
                    or not math.isfinite(frame.timestamp_seconds)
                    or frame.timestamp_seconds < 0
                ):
                    return self._error(
                        len(frames),
                        "invalid_input",
                        "An extracted frame has invalid index or timestamp metadata.",
                    )
        except OSError as error:
            return self._error(
                len(frames), "frame_not_found", f"Could not access extracted frame: {error}"
            )

        payload = {
            "request_id": uuid4().hex,
            "frames": [
                {
                    "path": str(frame.path.resolve()),
                    "frame_index": frame.frame_index,
                    "timestamp_seconds": frame.timestamp_seconds,
                }
                for frame in frames
            ],
        }
        with self._lock:
            try:
                self._ensure_worker()
                response = self._send_request(payload)
            except TimeoutError as error:
                self._stop_worker()
                return self._error(len(frames), "worker_timeout", str(error))
            except (OSError, RuntimeError) as error:
                self._stop_worker()
                return self._error(len(frames), "worker_unavailable", str(error))

            if response.get("request_id") != payload["request_id"]:
                self._stop_worker()
                return self._error(
                    len(frames),
                    "malformed_worker_response",
                    "Worker response request ID did not match the request.",
                )
            if response.get("status") == "error":
                return self._error(
                    len(frames),
                    str(response.get("error_code", "inference_failed")),
                    str(response.get("error_message", "xdfdet worker reported an error.")),
                )
            if response.get("status") == "inconclusive":
                reason_code = response.get("reason_code")
                if reason_code == "no_faces":
                    result = self._inconclusive(
                        len(frames),
                        "No detectable faces were found in the sampled frames.",
                    )
                    return result.model_copy(
                        update={
                            "frames_with_faces": 0,
                            "faces_detected": 0,
                            "preprocessing_seconds": self._optional_duration(
                                response.get("preprocessing_seconds")
                            ),
                        }
                    )
                self._stop_worker()
                return self._error(
                    len(frames),
                    "malformed_worker_response",
                    f"Worker returned an unsupported inconclusive reason: {reason_code!r}.",
                )
            try:
                return self._success(response, len(frames))
            except (KeyError, TypeError, ValueError) as error:
                self._stop_worker()
                return self._error(
                    len(frames),
                    "malformed_worker_response",
                    f"Worker response failed validation: {error}",
                )

    def close(self) -> None:
        """Stop the persistent worker and release its process resources."""
        with self._lock:
            self._stop_worker()

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if not self.python_executable.is_file():
            raise RuntimeError(
                f"xdfdet Python 3.13 interpreter was not found: {self.python_executable}"
            )
        if not self.worker_script.is_file():
            raise RuntimeError(f"xdfdet worker script was not found: {self.worker_script}")

        responses: queue.Queue[object] = queue.Queue()
        environment = os.environ.copy()
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        environment["HF_HUB_OFFLINE"] = "1"
        process = subprocess.Popen(
            [str(self.python_executable), str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        self._process = process
        self._responses = responses
        threading.Thread(
            target=self._read_stdout,
            args=(process, responses),
            name="xdfdet-worker-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            args=(process,),
            name="xdfdet-worker-stderr",
            daemon=True,
        ).start()
        try:
            ready = responses.get(timeout=self.startup_timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError("xdfdet worker startup timed out.") from error
        if ready is _STOP:
            raise RuntimeError(
                f"xdfdet worker exited during startup (exit code {process.poll()})."
            )
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            raise RuntimeError("xdfdet worker returned a malformed startup response.")
        if ready.get("status") != "ready":
            code = ready.get("error_code", "model_load_failed")
            message = ready.get("error_message", "xdfdet failed to load its model.")
            raise RuntimeError(f"{code}: {message}")
        if ready.get("checkpoint") != CHECKPOINT:
            raise RuntimeError("xdfdet worker loaded an unexpected checkpoint.")
        self._model_loading_seconds = self._required_duration(
            ready.get("model_loading_seconds")
        )

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        responses = self._responses
        if process is None or process.stdin is None or responses is None:
            raise RuntimeError("xdfdet worker is not running.")
        if process.poll() is not None:
            raise RuntimeError(
                f"xdfdet worker exited before inference (exit code {process.returncode})."
            )
        try:
            process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(
                f"Could not send request to xdfdet worker: {error}"
            ) from error
        try:
            response = responses.get(timeout=self.worker_timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError(
                f"xdfdet inference exceeded {self.worker_timeout_seconds:g} seconds."
            ) from error
        if response is _STOP:
            raise RuntimeError(
                f"xdfdet worker exited during inference (exit code {process.poll()})."
            )
        if not isinstance(response, dict):
            raise RuntimeError("xdfdet worker returned malformed JSON.")
        return response

    def _success(
        self, response: dict[str, Any], frames_extracted: int
    ) -> XdfdetAnalysisResponse:
        frame_results_data = response["frame_results"]
        if not isinstance(frame_results_data, list) or not frame_results_data:
            raise ValueError("frame_results must be a non-empty list.")
        frame_results = [
            XdfdetFrameResult.model_validate(item) for item in frame_results_data
        ]
        frames_analyzed = response["frames_analyzed"]
        frames_with_faces = response["frames_with_faces"]
        faces_detected = response["faces_detected"]
        if (
            isinstance(frames_analyzed, bool)
            or not isinstance(frames_analyzed, int)
            or frames_analyzed != len(frame_results)
            or isinstance(frames_with_faces, bool)
            or not isinstance(frames_with_faces, int)
            or not 0 < frames_with_faces <= frames_extracted
            or isinstance(faces_detected, bool)
            or not isinstance(faces_detected, int)
            or faces_detected < frames_with_faces
        ):
            raise ValueError("worker frame and face counts are inconsistent.")
        average = response["average_real_probability"]
        if (
            isinstance(average, bool)
            or not isinstance(average, (int, float))
            or not math.isfinite(average)
            or not 0 <= average <= 1
        ):
            raise ValueError("average_real_probability must be a finite probability.")
        observed_average = sum(item.real_probability for item in frame_results) / len(
            frame_results
        )
        if not math.isclose(average, observed_average, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("aggregate score does not match the frame-level scores.")

        frame_sides = {
            item.real_probability >= self.decision_threshold
            for item in frame_results
        }
        if len(frame_sides) > 1:
            verdict = "inconclusive"
            limitations = [
                *_LIMITATIONS,
                "Frame-level predictions fall on both sides of the provisional decision threshold.",
            ]
        elif average < self.decision_threshold:
            verdict = "likely_manipulated"
            limitations = list(_LIMITATIONS)
        else:
            verdict = "no_manipulation_detected"
            limitations = list(_LIMITATIONS)

        return XdfdetAnalysisResponse(
            status="success",
            verdict=verdict,
            average_real_probability=float(average),
            frames_extracted=frames_extracted,
            frames_analyzed=frames_analyzed,
            frames_with_faces=frames_with_faces,
            faces_detected=faces_detected,
            frame_results=frame_results,
            decision_threshold=self.decision_threshold,
            threshold_source="provisional_configuration",
            limitations=limitations,
            model_loading_seconds=self._model_loading_seconds,
            preprocessing_seconds=self._optional_duration(
                response.get("preprocessing_seconds")
            ),
            inference_seconds=self._optional_duration(
                response.get("inference_seconds")
            ),
        )

    def _inconclusive(
        self, frames_extracted: int, reason: str
    ) -> XdfdetAnalysisResponse:
        return XdfdetAnalysisResponse(
            status="inconclusive",
            verdict="inconclusive",
            average_real_probability=None,
            frames_extracted=frames_extracted,
            frames_analyzed=0,
            frames_with_faces=0,
            decision_threshold=self.decision_threshold,
            threshold_source="provisional_configuration",
            limitations=[*_LIMITATIONS, reason],
        )

    def _error(
        self, frames_extracted: int, code: str, message: str
    ) -> XdfdetAnalysisResponse:
        return XdfdetAnalysisResponse(
            status="error",
            verdict="inconclusive",
            average_real_probability=None,
            frames_extracted=frames_extracted,
            frames_analyzed=0,
            frames_with_faces=0,
            decision_threshold=self.decision_threshold,
            threshold_source="provisional_configuration",
            limitations=list(_LIMITATIONS),
            error_code=code,
            error_message=message,
        )

    @staticmethod
    def _required_duration(value: Any) -> float:
        duration = XdfdetAdapter._optional_duration(value)
        if duration is None:
            raise RuntimeError("xdfdet worker returned an invalid model-load duration.")
        return duration

    @staticmethod
    def _optional_duration(value: Any) -> float | None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return None
        return float(value)

    @staticmethod
    def _read_stdout(
        process: subprocess.Popen[str], responses: queue.Queue[object]
    ) -> None:
        if process.stdout is None:
            responses.put(_STOP)
            return
        try:
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    message = line.rstrip("\r\n")
                responses.put(message)
        finally:
            responses.put(_STOP)

    @staticmethod
    def _read_stderr(process: subprocess.Popen[str]) -> None:
        if process.stderr is None:
            return
        for line in process.stderr:
            LOGGER.warning("xdfdet worker: %s", line.rstrip())

    def _stop_worker(self) -> None:
        process = self._process
        self._process = None
        self._responses = None
        self._model_loading_seconds = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                LOGGER.debug("Failed to close xdfdet worker stdin.", exc_info=True)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


_DEFAULT_ADAPTER: XdfdetAdapter | None = None
_DEFAULT_ADAPTER_LOCK = threading.Lock()


def analyze_extracted_frames(
    extraction: FrameExtractionResult,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> XdfdetAnalysisResponse:
    """Analyze one pipeline extraction with the process-wide cached worker."""
    if (
        isinstance(decision_threshold, bool)
        or not isinstance(decision_threshold, (int, float))
        or not math.isfinite(decision_threshold)
        or not 0 <= decision_threshold <= 1
    ):
        raise ValueError("Decision threshold must be a finite probability between 0 and 1.")
    global _DEFAULT_ADAPTER
    with _DEFAULT_ADAPTER_LOCK:
        if (
            _DEFAULT_ADAPTER is None
            or _DEFAULT_ADAPTER.decision_threshold != decision_threshold
        ):
            if _DEFAULT_ADAPTER is not None:
                _DEFAULT_ADAPTER.close()
            _DEFAULT_ADAPTER = XdfdetAdapter(decision_threshold=decision_threshold)
        return _DEFAULT_ADAPTER.analyze(extraction)


def shutdown_xdfdet_worker() -> None:
    """Explicitly stop the process-wide worker when the host shuts down."""
    global _DEFAULT_ADAPTER
    with _DEFAULT_ADAPTER_LOCK:
        if _DEFAULT_ADAPTER is not None:
            _DEFAULT_ADAPTER.close()
            _DEFAULT_ADAPTER = None


atexit.register(shutdown_xdfdet_worker)
