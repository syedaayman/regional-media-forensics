"""Tests for xdfdet adapter protocol and cautious result handling."""

import json
import os
from pathlib import Path
import sys
import threading

import pytest

from app.modules.video.frame_extraction import (
    ExtractedFrameInfo,
    FrameExtractionResult,
)
from app.modules.visual_forensics.xdfdet_adapter import XdfdetAdapter


def _extraction(directory: Path, count: int = 12) -> FrameExtractionResult:
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(count):
        path = directory / f"frame-{index}.jpg"
        path.write_bytes(b"frame")
        frames.append(
            ExtractedFrameInfo(
                frame_index=index * 30,
                timestamp_seconds=float(index),
                width=64,
                height=48,
                path=path,
            )
        )
    return FrameExtractionResult(
        video_path=directory / "source.mp4",
        output_directory=directory,
        interval_seconds=1.0,
        frames=frames,
    )


def _worker_script(
    tmp_path: Path,
    response_mode: str = "success",
    delay: float = 0,
) -> Path:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        f"""
import json, os, sys, time
print(json.dumps({{"type":"ready","status":"ready","checkpoint":"aug-cutout-black","model_loading_seconds":0.25}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    mode = os.environ.get("RMF_FAKE_WORKER_MODE", {response_mode!r})
    delay = float(os.environ.get("RMF_FAKE_WORKER_DELAY", {delay!r}))
    if mode == "crash":
        os._exit(17)
    if mode == "timeout":
        time.sleep(delay or 2)
    if mode == "malformed":
        print("not-json", flush=True)
        continue
    if mode == "wrong-request":
        request["request_id"] = "wrong-id"
    if mode == "worker-error":
        print(json.dumps({{"request_id":request["request_id"],"status":"error","error_code":"invalid_image","error_message":"bad image"}}), flush=True)
        continue
    if mode == "no-faces":
        print(json.dumps({{"request_id":request["request_id"],"status":"inconclusive","reason_code":"no_faces","frames_extracted":len(request["frames"]),"frames_with_faces":0,"faces_detected":0,"frame_results":[]}}), flush=True)
        continue
    scores = [0.8] * 12
    if mode == "mixed":
        scores = [0.2] * 6 + [0.8] * 6
    average = sum(scores) / len(scores)
    result = {{
        "request_id":request["request_id"],"status":"success",
        "frames_extracted":len(request["frames"]),"frames_analyzed":12,
        "frames_with_faces":12,"faces_detected":12,
        "average_real_probability":average,
        "frame_results":[{{"frame_index":i*30,"timestamp_seconds":float(i),"real_probability":score,"face_detected":True}} for i,score in enumerate(scores)],
        "preprocessing_seconds":0.1,"inference_seconds":0.2
    }}
    print(json.dumps(result), flush=True)
""",
        encoding="utf-8",
    )
    return script


def _adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str = "success",
    *,
    timeout: float = 2,
    startup_timeout: float = 2,
    delay: float = 0,
) -> XdfdetAdapter:
    monkeypatch.setenv("RMF_FAKE_WORKER_MODE", mode)
    monkeypatch.setenv("RMF_FAKE_WORKER_DELAY", str(delay))
    return XdfdetAdapter(
        python_executable=sys.executable,
        worker_script=_worker_script(tmp_path, mode, delay),
        worker_timeout_seconds=timeout,
        startup_timeout_seconds=startup_timeout,
    )


def test_adapter_validates_threshold_and_timeout(tmp_path: Path) -> None:
    script = _worker_script(tmp_path)
    with pytest.raises(ValueError, match="Decision threshold"):
        XdfdetAdapter(sys.executable, script, decision_threshold=1.5)
    with pytest.raises(ValueError, match="Worker timeout"):
        XdfdetAdapter(sys.executable, script, worker_timeout_seconds=0)


def test_adapter_uses_configured_worker_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "isolated-python.exe"
    monkeypatch.setenv("RMF_XDFDET_PYTHON", str(interpreter))

    adapter = XdfdetAdapter()
    result = adapter.analyze(_extraction(tmp_path / "frames"))

    assert adapter.python_executable == interpreter
    assert result.status == "error"
    assert result.error_code == "worker_unavailable"
    assert str(interpreter) in (result.error_message or "")


def test_adapter_forces_worker_to_use_local_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "offline-mode.txt"
    script = tmp_path / "offline-worker.py"
    script.write_text(
        "import json, os, sys\n"
        f"open({str(marker)!r}, 'w').write(os.environ.get('HF_HUB_OFFLINE', ''))\n"
        "print(json.dumps({'type':'ready','status':'ready',"
        "'checkpoint':'aug-cutout-black','model_loading_seconds':0.1}), flush=True)\n"
        "for _line in sys.stdin:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    adapter = XdfdetAdapter(
        python_executable=sys.executable, worker_script=script
    )
    try:
        adapter._ensure_worker()
    finally:
        adapter.close()

    assert marker.read_text(encoding="utf-8") == "1"


def test_successful_request_reuses_worker_and_reports_cautious_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path, monkeypatch)
    extraction = _extraction(tmp_path / "frames")
    try:
        first = adapter.analyze(extraction)
        first_process = adapter._process
        second = adapter.analyze(extraction)
        assert adapter._process is first_process
    finally:
        adapter.close()

    assert first.status == "success"
    assert first.verdict == "no_manipulation_detected"
    assert first.average_real_probability == pytest.approx(0.8)
    assert first.frames_extracted == 12
    assert first.frames_analyzed == 12
    assert first.model_loading_seconds == pytest.approx(0.25)
    assert first.threshold_source == "provisional_configuration"
    assert first is not second


def test_adapter_uses_configured_provisional_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = XdfdetAdapter(
        python_executable=sys.executable,
        worker_script=_worker_script(tmp_path),
        decision_threshold=0.9,
    )
    monkeypatch.setenv("RMF_FAKE_WORKER_MODE", "success")
    try:
        result = adapter.analyze(_extraction(tmp_path / "frames"))
    finally:
        adapter.close()

    assert result.decision_threshold == 0.9
    assert result.threshold_source == "provisional_configuration"
    assert result.verdict == "likely_manipulated"


def test_adapter_returns_inconclusive_for_insufficient_frames(
    tmp_path: Path,
) -> None:
    adapter = XdfdetAdapter(python_executable=tmp_path / "missing-python.exe")
    result = adapter.analyze(_extraction(tmp_path / "frames", count=11))

    assert result.status == "inconclusive"
    assert result.verdict == "inconclusive"
    assert result.average_real_probability is None
    assert result.frames_analyzed == 0
    assert adapter._process is None


def test_adapter_rejects_frames_outside_extraction_directory(
    tmp_path: Path,
) -> None:
    extraction = _extraction(tmp_path / "frames")
    external_path = tmp_path / "outside.jpg"
    external_path.write_bytes(b"frame")
    frame = extraction.frames[0]
    extraction.frames[0] = ExtractedFrameInfo(
        frame_index=frame.frame_index,
        timestamp_seconds=frame.timestamp_seconds,
        width=frame.width,
        height=frame.height,
        path=external_path,
    )
    adapter = XdfdetAdapter(python_executable=tmp_path / "missing-python.exe")

    result = adapter.analyze(extraction)

    assert result.status == "error"
    assert result.error_code == "invalid_input"
    assert adapter._process is None


def test_adapter_rejects_invalid_extraction_input(tmp_path: Path) -> None:
    adapter = XdfdetAdapter(python_executable=tmp_path / "missing-python.exe")

    result = adapter.analyze(None)  # type: ignore[arg-type]

    assert result.status == "error"
    assert result.error_code == "invalid_input"


def test_adapter_returns_inconclusive_for_no_faces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path, monkeypatch, mode="no-faces")
    try:
        result = adapter.analyze(_extraction(tmp_path / "frames"))
    finally:
        adapter.close()

    assert result.status == "inconclusive"
    assert result.verdict == "inconclusive"
    assert result.average_real_probability is None
    assert result.frames_with_faces == 0


def test_adapter_reports_invalid_extracted_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path, monkeypatch, mode="worker-error")
    try:
        result = adapter.analyze(_extraction(tmp_path / "frames"))
    finally:
        adapter.close()

    assert result.status == "error"
    assert result.error_code == "invalid_image"
    assert "bad image" in (result.error_message or "")


@pytest.mark.parametrize("mode", ["crash", "malformed", "wrong-request"])
def test_adapter_handles_worker_crash_and_malformed_responses(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path, monkeypatch, mode=mode)
    try:
        result = adapter.analyze(_extraction(tmp_path / "frames"))
    finally:
        adapter.close()

    assert result.status == "error"
    assert result.verdict == "inconclusive"
    assert result.error_code in {
        "worker_unavailable",
        "malformed_worker_response",
    }


def test_adapter_times_out_and_stops_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(
        tmp_path, monkeypatch, mode="timeout", timeout=0.1, delay=1
    )
    result = adapter.analyze(_extraction(tmp_path / "frames"))

    assert result.status == "error"
    assert result.error_code == "worker_timeout"
    assert adapter._process is None


def test_adapter_returns_structured_model_initialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "load-failure.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'type':'ready','status':'error',"
        "'error_code':'model_load_failed','error_message':'checkpoint missing'}), "
        "flush=True)\n",
        encoding="utf-8",
    )
    adapter = XdfdetAdapter(
        python_executable=sys.executable, worker_script=script
    )
    monkeypatch.delenv("RMF_FAKE_WORKER_MODE", raising=False)

    result = adapter.analyze(_extraction(tmp_path / "frames"))

    assert result.status == "error"
    assert result.error_code == "worker_unavailable"
    assert "model_load_failed" in (result.error_message or "")


def test_conflicting_frame_predictions_are_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path, monkeypatch, mode="mixed")
    try:
        result = adapter.analyze(_extraction(tmp_path / "frames"))
    finally:
        adapter.close()

    assert result.average_real_probability == pytest.approx(0.5)
    assert result.verdict == "inconclusive"
    assert any("both sides" in item for item in result.limitations)


def test_concurrent_requests_are_serialized_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path, monkeypatch)
    extractions = [
        _extraction(tmp_path / "frames-a"),
        _extraction(tmp_path / "frames-b"),
    ]
    results = []

    def run(extraction: FrameExtractionResult) -> None:
        results.append(adapter.analyze(extraction))

    threads = [threading.Thread(target=run, args=(item,)) for item in extractions]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        adapter.close()

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert all(result.status == "success" for result in results)
    assert all(result.frames_extracted == 12 for result in results)
    assert len({frame.path for extraction in extractions for frame in extraction.frames}) == 24


def test_adapter_does_not_create_or_delete_pipeline_frame_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _adapter(tmp_path, monkeypatch)
    extraction = _extraction(tmp_path / "frames")
    expected_paths = {frame.path for frame in extraction.frames}
    try:
        adapter.analyze(extraction)
    finally:
        adapter.close()

    assert expected_paths == {path for path in (tmp_path / "frames").iterdir()}
    assert all(path.is_file() for path in expected_paths)
