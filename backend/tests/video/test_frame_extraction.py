"""Tests for interval-based frame extraction with mocked OpenCV."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.modules.video import frame_extraction, metadata_analysis
from app.modules.video.dependencies import VideoProcessingDependencyError


class _Frame:
    """Small stand-in for an OpenCV image array."""

    shape = (240, 320, 3)


class _Capture:
    """Fake capture that produces a fixed number of frames."""

    def __init__(
        self,
        *,
        fps: float = 2.0,
        frame_count: int = 6,
        opened: bool = True,
        timestamp_step: float = 0.5,
    ) -> None:
        self.fps = fps
        self.frame_count = frame_count
        self.opened = opened
        self.timestamp_step = timestamp_step
        self.current_index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        return {
            1: self.fps,
            2: float(self.frame_count),
            3: 320.0,
            4: 240.0,
            5: max(0, self.current_index - 1) * self.timestamp_step * 1000,
        }[property_id]

    def read(self) -> tuple[bool, _Frame]:
        if not self.opened or self.current_index >= self.frame_count:
            return False, _Frame()
        self.current_index += 1
        return True, _Frame()

    def release(self) -> None:
        self.released = True


def _write_mp4(path: Path) -> None:
    path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8)


def _install_fake_opencv(
    monkeypatch: pytest.MonkeyPatch,
    captures: list[_Capture],
    *,
    fps: float = 2.0,
    frame_count: int = 6,
    timestamp_step: float = 0.5,
) -> None:
    def create_capture(_path: str) -> _Capture:
        capture = _Capture(
            fps=fps,
            frame_count=frame_count,
            timestamp_step=timestamp_step,
        )
        captures.append(capture)
        return capture

    def write_frame(filename: str, _image: _Frame) -> bool:
        Path(filename).write_bytes(b"jpeg")
        return True

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_POS_MSEC=5,
        VideoCapture=create_capture,
        imwrite=write_frame,
    )
    monkeypatch.setattr(metadata_analysis, "load_opencv", lambda: fake_cv2)
    monkeypatch.setattr(frame_extraction, "load_opencv", lambda: fake_cv2)


def test_extract_frames_samples_timestamps_respects_limit_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    output_directory = tmp_path / "frames"
    _write_mp4(video_path)
    captures: list[_Capture] = []
    _install_fake_opencv(monkeypatch, captures)

    result = frame_extraction.extract_frames(
        video_path,
        output_directory,
        interval_seconds=1.0,
        max_frames=2,
    )

    assert [frame.frame_index for frame in result.frames] == [0, 2]
    assert [frame.timestamp_seconds for frame in result.frames] == [0.0, 1.0]
    assert [(frame.width, frame.height) for frame in result.frames] == [
        (320, 240),
        (320, 240),
    ]
    assert len(result.frames) == 2
    assert all(frame.path.is_file() for frame in result.frames)
    assert all(frame.path.parent == output_directory.resolve() for frame in result.frames)
    assert len({frame.path.name for frame in result.frames}) == 2
    assert len(captures) == 2
    assert all(capture.released for capture in captures)


def test_extract_frames_uses_source_timestamps_for_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)
    captures: list[_Capture] = []
    _install_fake_opencv(monkeypatch, captures, timestamp_step=0.6)

    result = frame_extraction.extract_frames(
        video_path,
        tmp_path / "frames",
        interval_seconds=1.0,
        max_frames=2,
    )

    assert [frame.frame_index for frame in result.frames] == [0, 2]
    assert [frame.timestamp_seconds for frame in result.frames] == [0.0, 1.2]
    assert all(capture.released for capture in captures)


def test_extract_frames_uses_unique_names_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    output_directory = tmp_path / "frames"
    _write_mp4(video_path)
    captures: list[_Capture] = []
    _install_fake_opencv(monkeypatch, captures)

    first_run = frame_extraction.extract_frames(
        video_path, output_directory, interval_seconds=1.0, max_frames=1
    )
    second_run = frame_extraction.extract_frames(
        video_path, output_directory, interval_seconds=1.0, max_frames=1
    )

    first_path = first_run.frames[0].path
    second_path = second_run.frames[0].path
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()
    assert first_path.read_bytes() == b"jpeg"
    assert second_path.read_bytes() == b"jpeg"


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan")])
def test_extract_frames_rejects_invalid_interval(
    tmp_path: Path, interval: float
) -> None:
    with pytest.raises(ValueError, match="interval"):
        frame_extraction.extract_frames(
            tmp_path / "missing.mp4",
            tmp_path / "frames",
            interval_seconds=interval,
        )


def test_extract_frames_rejects_invalid_video_before_opening_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "invalid.mp4"
    video_path.write_bytes(b"not an mp4")

    def unexpected_opencv() -> None:
        pytest.fail("OpenCV should not be loaded for an invalid video.")

    monkeypatch.setattr(metadata_analysis, "load_opencv", unexpected_opencv)

    with pytest.raises(ValueError, match="container signature"):
        frame_extraction.extract_frames(video_path, tmp_path / "frames")


def test_extract_frames_reports_missing_opencv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)

    def missing_opencv() -> None:
        raise VideoProcessingDependencyError("OpenCV is not installed.")

    monkeypatch.setattr(metadata_analysis, "load_opencv", missing_opencv)

    with pytest.raises(VideoProcessingDependencyError, match="OpenCV"):
        frame_extraction.extract_frames(video_path, tmp_path / "frames")


def test_extract_frames_rejects_invalid_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)
    captures: list[_Capture] = []
    _install_fake_opencv(monkeypatch, captures, fps=0.0)

    with pytest.raises(ValueError, match="FPS"):
        frame_extraction.extract_frames(video_path, tmp_path / "frames")

    assert len(captures) == 1
    assert captures[0].released


def test_extract_frames_releases_capture_when_writing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)
    captures: list[_Capture] = []
    _install_fake_opencv(monkeypatch, captures)

    def fail_write(_filename: str, _image: _Frame) -> bool:
        return False

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_POS_MSEC=5,
        VideoCapture=lambda _path: captures.append(
            _Capture()
        ) or captures[-1],
        imwrite=fail_write,
    )
    monkeypatch.setattr(metadata_analysis, "load_opencv", lambda: fake_cv2)
    monkeypatch.setattr(frame_extraction, "load_opencv", lambda: fake_cv2)

    with pytest.raises(OSError, match="failed to save"):
        frame_extraction.extract_frames(video_path, tmp_path / "frames")

    assert len(captures) == 2
    assert all(capture.released for capture in captures)
