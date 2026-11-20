"""Tests for end-to-end video preprocessing orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.modules.video import frame_extraction, metadata_analysis, video_analysis


class _Frame:
    """Small stand-in for an OpenCV image array."""

    shape = (120, 160, 3)


class _Capture:
    """Fake capture that produces two frames at one frame per second."""

    def __init__(self) -> None:
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            1: 1.0,
            2: 2.0,
            3: 160.0,
            4: 120.0,
            5: max(0, self.index - 1) * 1000.0,
        }[property_id]

    def read(self) -> tuple[bool, _Frame]:
        if self.index >= 2:
            return False, _Frame()
        self.index += 1
        return True, _Frame()

    def release(self) -> None:
        self.released = True


def test_process_video_returns_metadata_and_saved_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8)
    captures: list[_Capture] = []

    def create_capture(_path: str) -> _Capture:
        capture = _Capture()
        captures.append(capture)
        return capture

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_POS_MSEC=5,
        VideoCapture=create_capture,
        imwrite=lambda filename, _frame: Path(filename).write_bytes(b"jpeg") > 0,
    )
    monkeypatch.setattr(metadata_analysis, "load_opencv", lambda: fake_cv2)
    monkeypatch.setattr(frame_extraction, "load_opencv", lambda: fake_cv2)

    result = video_analysis.process_video(
        video_path,
        tmp_path / "frames",
        interval_seconds=1.0,
        max_frames=2,
    )

    assert result.metadata.duration_seconds == 2.0
    assert len(result.extraction.frames) == 2
    assert all(frame.path.is_file() for frame in result.extraction.frames)
    assert len(captures) == 3
    assert all(capture.released for capture in captures)
