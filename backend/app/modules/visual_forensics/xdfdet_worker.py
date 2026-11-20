"""Persistent Python 3.13 JSON-lines worker for the xdfdet model."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger("rmf.xdfdet_worker")
CHECKPOINT = "aug-cutout-black"
MODEL_FRAME_COUNT = 12


@dataclass
class WorkerRuntime:
    """Process-local xdfdet dependencies and cached pretrained model."""

    cv2: Any
    np: Any
    detector: Any
    model: Any
    frame_indices: Any
    normalize: Any
    align_and_crop: Any
    image_size: int
    crop_frame_count: int
    crop_margin: float
    model_loading_seconds: float


def load_runtime() -> WorkerRuntime:
    """Import xdfdet and initialize the pretrained model exactly once."""
    import cv2
    import numpy as np
    from mtcnn import MTCNN
    from xdfdet.config import CROP_FRAMES, IMG_SIZE
    from xdfdet.model import load_model
    from xdfdet.video import MARGIN, _align_and_crop, frame_indices, normalize

    started = time.perf_counter()
    model = load_model(CHECKPOINT)
    model_loading_seconds = time.perf_counter() - started
    return WorkerRuntime(
        cv2=cv2,
        np=np,
        detector=MTCNN(),
        model=model,
        frame_indices=frame_indices,
        normalize=normalize,
        align_and_crop=_align_and_crop,
        image_size=IMG_SIZE,
        crop_frame_count=CROP_FRAMES,
        crop_margin=MARGIN,
        model_loading_seconds=model_loading_seconds,
    )


def _error(
    request_id: str | None, code: str, message: str, frames_extracted: int = 0
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "status": "error",
        "error_code": code,
        "error_message": message,
        "frames_extracted": frames_extracted,
    }


def _analyze_request(
    payload: dict[str, Any], runtime: WorkerRuntime
) -> dict[str, Any]:
    """Analyze the extracted JPEGs without opening or decoding the source video."""
    request_id = payload.get("request_id")
    frame_records = payload.get("frames")
    if not isinstance(request_id, str) or not request_id:
        return _error(None, "invalid_request", "A request ID is required.")
    if not isinstance(frame_records, list) or not frame_records:
        return _error(request_id, "invalid_frames", "At least one extracted frame is required.")

    frames_extracted = len(frame_records)
    records: list[dict[str, Any]] = []
    sampling_indices = runtime.np.linspace(
        0, frames_extracted - 1, runtime.crop_frame_count, dtype=int
    ).tolist()
    selected_records = set(sampling_indices)
    decoded: dict[int, Any] = {}
    for record in frame_records:
        record_index = len(records)
        if not isinstance(record, dict):
            return _error(
                request_id,
                "invalid_frames",
                "Each frame record must be an object.",
                frames_extracted,
            )
        path = record.get("path")
        frame_index = record.get("frame_index")
        timestamp = record.get("timestamp_seconds")
        if (
            not isinstance(path, str)
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
        ):
            return _error(
                request_id,
                "invalid_frames",
                "A frame record has invalid metadata.",
                frames_extracted,
            )

        image = runtime.cv2.imread(path, runtime.cv2.IMREAD_COLOR)
        if image is None or len(image.shape) != 3 or image.shape[2] != 3:
            return _error(
                request_id,
                "invalid_image",
                f"Extracted frame {frame_index} is not a readable color image.",
                frames_extracted,
            )
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            return _error(
                request_id,
                "invalid_image",
                f"Extracted frame {frame_index} has invalid dimensions.",
                frames_extracted,
            )
        if record_index in selected_records:
            decoded[record_index] = runtime.cv2.cvtColor(
                image, runtime.cv2.COLOR_BGR2RGB
            )
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": float(timestamp),
            }
        )

    crops: list[Any] = []
    crop_record_indices: list[int] = []
    faces_by_record: dict[int, int] = {}
    last_face: tuple[Any, Any, Any] | None = None
    detections_by_record: dict[int, list[dict[str, Any]]] = {}
    preprocessing_started = time.perf_counter()

    for record_index in sampling_indices:
        image = decoded[record_index]
        if record_index not in detections_by_record:
            detections_by_record[record_index] = runtime.detector.detect_faces(image)
        faces = detections_by_record[record_index]
        faces_by_record[record_index] = len(faces)
        if faces:
            face = max(faces, key=lambda item: item["confidence"])
            keypoints = face["keypoints"]
            last_face = (
                face["box"],
                keypoints["left_eye"],
                keypoints["right_eye"],
            )
        if last_face is not None:
            crops.append(
                runtime.align_and_crop(
                    image,
                    *last_face,
                    runtime.image_size,
                    runtime.crop_margin,
                )
            )
            crop_record_indices.append(record_index)

    if not crops:
        return {
            "request_id": request_id,
            "status": "inconclusive",
            "reason_code": "no_faces",
            "frames_extracted": frames_extracted,
            "frames_with_faces": 0,
            "faces_detected": 0,
            "frame_results": [],
            "preprocessing_seconds": time.perf_counter() - preprocessing_started,
        }

    while len(crops) < runtime.crop_frame_count:
        crops.insert(0, crops[0])
        crop_record_indices.insert(0, crop_record_indices[0])

    selected_crop_indices = runtime.frame_indices(len(crops))
    sequence = runtime.np.stack(
        [runtime.normalize(crops[index]) for index in selected_crop_indices]
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started

    inference_started = time.perf_counter()
    predictions = runtime.np.asarray(
        runtime.model.predict(sequence[None], verbose=0)
    ).reshape(-1)
    inference_seconds = time.perf_counter() - inference_started
    if len(predictions) != MODEL_FRAME_COUNT or not runtime.np.isfinite(predictions).all():
        return _error(
            request_id,
            "invalid_model_output",
            "xdfdet returned an invalid number of frame scores or non-finite scores.",
            frames_extracted,
        )
    if ((predictions < 0) | (predictions > 1)).any():
        return _error(
            request_id,
            "invalid_model_output",
            "xdfdet returned frame scores outside the probability range.",
            frames_extracted,
        )

    frame_results = []
    for crop_index, score in zip(selected_crop_indices, predictions, strict=True):
        record_index = crop_record_indices[crop_index]
        record = records[record_index]
        frame_results.append(
            {
                **record,
                "real_probability": float(score),
                "face_detected": faces_by_record.get(record_index, 0) > 0,
            }
        )

    return {
        "request_id": request_id,
        "status": "success",
        "frames_extracted": frames_extracted,
        "frames_analyzed": len(frame_results),
        "frames_with_faces": sum(count > 0 for count in faces_by_record.values()),
        "faces_detected": sum(faces_by_record.values()),
        "frame_results": frame_results,
        "average_real_probability": float(predictions.mean()),
        "preprocessing_seconds": preprocessing_seconds,
        "inference_seconds": inference_seconds,
    }


def _write_response(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    """Load one model and serve requests until the parent closes stdin."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(message)s",
    )
    try:
        runtime = load_runtime()
    except Exception as error:
        _write_response(
            {
                "type": "ready",
                "status": "error",
                "error_code": "model_load_failed",
                "error_message": f"{type(error).__name__}: {error}",
            }
        )
        return 1

    _write_response(
        {
            "type": "ready",
            "status": "ready",
            "model_loading_seconds": runtime.model_loading_seconds,
            "checkpoint": CHECKPOINT,
        }
    )
    for line in sys.stdin:
        payload: Any = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Request JSON must be an object.")
            response = _analyze_request(payload, runtime)
        except (json.JSONDecodeError, ValueError) as error:
            response = _error(None, "invalid_request", str(error))
        except Exception as error:
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            LOGGER.exception("xdfdet request failed")
            response = _error(
                request_id if isinstance(request_id, str) else None,
                "inference_failed",
                f"{type(error).__name__}: {error}",
            )
        _write_response(response)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    raise SystemExit(main())
