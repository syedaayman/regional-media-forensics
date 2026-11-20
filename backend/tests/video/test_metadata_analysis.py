"""Tests for OpenCV metadata inspection using a mocked capture."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.modules.video import metadata_analysis
from app.modules.video import dependencies
from app.modules.video.dependencies import VideoProcessingDependencyError


class _Frame:
    """Small stand-in for an OpenCV image array."""

    shape = (360, 640, 3)


class _Capture:
    """Fake VideoCapture that exposes a fixed metadata record."""

    def __init__(
        self,
        *,
        opened: bool = True,
        decodable: bool = True,
        fps: float = 25.0,
        frame_count: float = 250.0,
    ) -> None:
        self.opened = opened
        self.decodable = decodable
        self.fps = fps
        self.frame_count = frame_count
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        return {
            1: self.fps,
            2: self.frame_count,
            3: 640.0,
            4: 360.0,
        }[property_id]

    def read(self) -> tuple[bool, _Frame]:
        return self.opened and self.decodable, _Frame()

    def release(self) -> None:
        self.released = True


def _write_mp4(path: Path) -> None:
    path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8)


def _install_fake_opencv(
    monkeypatch: pytest.MonkeyPatch, capture: _Capture
) -> None:
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        VideoCapture=lambda _path: capture,
    )
    monkeypatch.setattr(metadata_analysis, "load_opencv", lambda: fake_cv2)


def test_inspect_video_metadata_reports_stream_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)
    capture = _Capture()
    _install_fake_opencv(monkeypatch, capture)

    metadata = metadata_analysis.inspect_video_metadata(video_path)

    assert metadata.duration_seconds == 10.0
    assert metadata.fps == 25.0
    assert metadata.frame_count == 250
    assert metadata.width == 640
    assert metadata.height == 360
    assert metadata.resolution == "640x360"
    assert metadata.aspect_ratio == pytest.approx(16 / 9)
    assert capture.released


def test_inspect_video_metadata_rejects_video_with_no_decodable_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "corrupt.mp4"
    _write_mp4(video_path)
    capture = _Capture(decodable=False)
    _install_fake_opencv(monkeypatch, capture)

    with pytest.raises(ValueError, match="no decodable frames"):
        metadata_analysis.inspect_video_metadata(video_path)

    assert capture.released


def test_inspect_video_metadata_reports_missing_opencv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)

    def missing_opencv() -> None:
        raise VideoProcessingDependencyError("OpenCV is not installed.")

    monkeypatch.setattr(metadata_analysis, "load_opencv", missing_opencv)

    with pytest.raises(VideoProcessingDependencyError, match="OpenCV"):
        metadata_analysis.inspect_video_metadata(video_path)


def test_load_opencv_explains_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(_name: str) -> None:
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")

    monkeypatch.setattr(dependencies, "import_module", missing_module)

    with pytest.raises(VideoProcessingDependencyError, match="opencv-python-headless"):
        dependencies.load_opencv()


def test_inspect_video_metadata_marks_invalid_fps_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)
    capture = _Capture(fps=0.0)
    _install_fake_opencv(monkeypatch, capture)

    metadata = metadata_analysis.inspect_video_metadata(video_path)

    assert metadata.fps is None
    assert metadata.duration_seconds is None
    assert capture.released
