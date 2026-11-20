"""OpenCV-based inspection of basic video stream metadata."""

from dataclasses import dataclass
import math
from pathlib import Path

from app.modules.video.dependencies import load_opencv
from app.modules.video.preprocessing import validate_video_file


@dataclass(frozen=True)
class VideoMetadata:
    """Basic container and video-stream metadata.

    Stream values that OpenCV cannot determine are represented as ``None``.
    """

    filename: str
    extension: str
    size_bytes: int
    duration_seconds: float | None
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    resolution: str | None
    aspect_ratio: float | None


def _positive_number(value: float) -> float | None:
    """Return finite positive values only."""
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def inspect_video_metadata(
    path: str | Path, max_size_bytes: int | None = None
) -> VideoMetadata:
    """Read duration, FPS, frame count, dimensions, and aspect ratio with OpenCV.

    Raises ``ValueError`` for invalid or undecodable input and a dependency
    error when OpenCV is unavailable. Resources are released on every path.
    """
    video_path = Path(path)
    validation = validate_video_file(video_path, max_size_bytes=max_size_bytes)
    if not validation.valid or validation.size_bytes is None:
        raise ValueError("Invalid video file: " + "; ".join(validation.errors))

    cv2 = load_opencv()
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video file.")

        frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_value = capture.get(cv2.CAP_PROP_FPS)
        width_value = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_value = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        success, _ = capture.read()
        if not success:
            raise ValueError("Video file opened but contains no decodable frames.")

        fps = _positive_number(fps_value)
        raw_frame_count = _positive_number(frame_count_value)
        frame_count = round(raw_frame_count) if raw_frame_count is not None else None
        width_number = _positive_number(width_value)
        height_number = _positive_number(height_value)
        width = round(width_number) if width_number is not None else None
        height = round(height_number) if height_number is not None else None

        duration_seconds = (
            frame_count / fps
            if frame_count is not None and fps is not None
            else None
        )
        resolution = f"{width}x{height}" if width is not None and height is not None else None
        aspect_ratio = (
            width / height if width is not None and height is not None else None
        )

        return VideoMetadata(
            filename=video_path.name,
            extension=video_path.suffix.lower(),
            size_bytes=validation.size_bytes,
            duration_seconds=duration_seconds,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )
    finally:
        capture.release()
