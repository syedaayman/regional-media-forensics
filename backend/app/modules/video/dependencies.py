"""Optional third-party dependency loading for video processing."""

from importlib import import_module
from typing import Protocol, cast


class VideoFrame(Protocol):
    """The small part of an OpenCV image array used by frame extraction."""

    @property
    def shape(self) -> tuple[int, ...]: ...


class VideoCapture(Protocol):
    """The OpenCV capture methods used by the video modules."""

    def isOpened(self) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def read(self) -> tuple[bool, VideoFrame]: ...

    def release(self) -> None: ...


class OpenCVModule(Protocol):
    """The OpenCV constants and functions used by the video modules."""

    CAP_PROP_FPS: int
    CAP_PROP_FRAME_COUNT: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_POS_MSEC: int

    def VideoCapture(self, filename: str) -> VideoCapture: ...

    def imwrite(self, filename: str, image: VideoFrame) -> bool: ...


class VideoProcessingDependencyError(RuntimeError):
    """Raised when a required optional video-processing dependency is missing."""


def load_opencv() -> OpenCVModule:
    """Load OpenCV lazily and provide an actionable installation message."""
    try:
        module = import_module("cv2")
    except (ImportError, OSError) as error:
        raise VideoProcessingDependencyError(
            "Video metadata and frame extraction require OpenCV. Install it with "
            '`python -m pip install "opencv-python-headless>=4.10,<5"`.'
        ) from error
    return cast(OpenCVModule, module)
