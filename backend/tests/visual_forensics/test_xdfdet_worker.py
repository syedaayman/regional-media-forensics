"""Unit tests for the Python 3.13 xdfdet worker using mocked inference."""

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.modules.visual_forensics import xdfdet_worker


class _Capture:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def detect_faces(self, image: np.ndarray) -> list[dict[str, object]]:
        self.images.append(image.copy())
        return [
            {
                "confidence": 0.99,
                "box": [1, 1, 4, 4],
                "keypoints": {"left_eye": [2, 2], "right_eye": [4, 2]},
            }
        ]


class _Model:
    def __init__(self, scores: np.ndarray | None = None) -> None:
        self.calls = 0
        self.last_input: np.ndarray | None = None
        self.scores = scores if scores is not None else np.full((1, 12, 1), 0.8)

    def predict(self, images: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert verbose == 0
        self.calls += 1
        self.last_input = images
        return self.scores


def _runtime(
    *,
    detector: _Capture | None = None,
    model: _Model | None = None,
    read_image: np.ndarray | None = None,
) -> tuple[xdfdet_worker.WorkerRuntime, _Capture, _Model]:
    capture = detector or _Capture()
    prediction_model = model or _Model()
    image = (
        read_image
        if read_image is not None
        else np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    )
    cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        COLOR_BGR2RGB=2,
        imread=lambda _path, _mode: image,
        cvtColor=lambda source, _code: source[..., ::-1].copy(),
    )
    runtime = xdfdet_worker.WorkerRuntime(
        cv2=cv2,
        np=np,
        detector=capture,
        model=prediction_model,
        frame_indices=lambda total: list(range(0, 24, 2)),
        normalize=lambda frame: frame.astype(np.float32),
        align_and_crop=lambda image, *_args: image[:4, :4].copy(),
        image_size=4,
        crop_frame_count=32,
        crop_margin=0.3,
        model_loading_seconds=1.25,
    )
    return runtime, capture, prediction_model


def _payload(tmp_path: Path, count: int = 12) -> dict[str, object]:
    paths = []
    for index in range(count):
        image_path = tmp_path / f"frame-{index}.jpg"
        image_path.write_bytes(b"mock jpeg")
        paths.append(
            {
                "path": str(image_path),
                "frame_index": index * 30,
                "timestamp_seconds": float(index),
            }
        )
    return {"request_id": "test-request", "frames": paths}


def test_worker_converts_bgr_to_rgb_and_returns_frame_probabilities(
    tmp_path: Path,
) -> None:
    runtime, detector, model = _runtime()

    result = xdfdet_worker._analyze_request(_payload(tmp_path), runtime)

    assert result["status"] == "success"
    assert result["frames_extracted"] == 12
    assert result["frames_analyzed"] == 12
    assert result["frames_with_faces"] == 12
    assert result["faces_detected"] == 12
    assert result["average_real_probability"] == pytest.approx(0.8)
    assert len(result["frame_results"]) == 12
    assert len(detector.images) == 12
    assert detector.images[0][0, 0].tolist() == [30, 20, 10]
    assert model.last_input is not None
    assert model.last_input.shape == (1, 12, 4, 4, 3)


def test_worker_returns_inconclusive_when_no_faces_are_detected(
    tmp_path: Path,
) -> None:
    detector = _Capture()
    detector.detect_faces = lambda _image: []  # type: ignore[method-assign]
    runtime, _, model = _runtime(detector=detector)

    result = xdfdet_worker._analyze_request(_payload(tmp_path), runtime)

    assert result["status"] == "inconclusive"
    assert result["reason_code"] == "no_faces"
    assert result["frames_with_faces"] == 0
    assert model.calls == 0


def test_worker_rejects_an_invalid_image(tmp_path: Path) -> None:
    runtime, detector, model = _runtime(read_image=None)
    runtime.cv2.imread = lambda _path, _mode: None

    result = xdfdet_worker._analyze_request(_payload(tmp_path), runtime)

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_image"
    assert detector.images == []
    assert model.calls == 0


def test_worker_rejects_non_probability_model_outputs(tmp_path: Path) -> None:
    model = _Model(scores=np.full((1, 12, 1), 1.5))
    runtime, _, _ = _runtime(model=model)

    result = xdfdet_worker._analyze_request(_payload(tmp_path), runtime)

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_model_output"


def test_load_runtime_initializes_model_once_with_expected_checkpoint(
    monkeypatch,
) -> None:
    model = _Model()
    called: list[str] = []
    config = types.ModuleType("xdfdet.config")
    config.CROP_FRAMES = 32
    config.IMG_SIZE = 224
    model_module = types.ModuleType("xdfdet.model")

    def load_model(checkpoint: str) -> _Model:
        called.append(checkpoint)
        return model

    model_module.load_model = load_model
    video = types.ModuleType("xdfdet.video")
    video.MARGIN = 0.3
    video._align_and_crop = lambda image, *_args: image
    video.frame_indices = lambda _total: list(range(0, 24, 2))
    video.normalize = lambda image: image
    package = types.ModuleType("xdfdet")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "xdfdet", package)
    monkeypatch.setitem(sys.modules, "xdfdet.config", config)
    monkeypatch.setitem(sys.modules, "xdfdet.model", model_module)
    monkeypatch.setitem(sys.modules, "xdfdet.video", video)
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    mtcnn = types.ModuleType("mtcnn")
    mtcnn.MTCNN = lambda: object()
    monkeypatch.setitem(sys.modules, "mtcnn", mtcnn)

    first = xdfdet_worker.load_runtime()
    second = xdfdet_worker.load_runtime()

    assert called == ["aug-cutout-black", "aug-cutout-black"]
    assert first.model is model
    assert second.model is model


def test_worker_main_reuses_one_loaded_model_for_multiple_requests(
    tmp_path: Path, monkeypatch
) -> None:
    runtime, _, model = _runtime()
    payload = json.dumps(_payload(tmp_path)) + "\n"
    monkeypatch.setattr(xdfdet_worker, "load_runtime", lambda: runtime)
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload + payload))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    assert xdfdet_worker.main() == 0

    messages = [json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[0]["status"] == "ready"
    assert [message["status"] for message in messages[1:]] == ["success", "success"]
    assert model.calls == 2
