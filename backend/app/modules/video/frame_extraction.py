"""Interval-based frame extraction to a caller-selected output directory."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from uuid import uuid4

from app.modules.video.dependencies import load_opencv
from app.modules.video.metadata_analysis import inspect_video_metadata


@dataclass(frozen=True)
class ExtractedFrameInfo:
    """Details and saved path for one representative frame."""

    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    path: Path


@dataclass(frozen=True)
class FrameExtractionResult:
    """Frame extraction settings and all successfully written frame records."""

    video_path: Path
    output_directory: Path
    interval_seconds: float
    frames: list[ExtractedFrameInfo]


def _safe_filename_stem(path: Path) -> str:
    """Keep only simple filename characters from the video stem."""
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    return safe_stem[:80] or "video"


def extract_frames(
    path: str | Path,
    output_directory: str | Path,
    interval_seconds: float = 1.0,
    max_frames: int = 100,
    max_size_bytes: int | None = None,
) -> FrameExtractionResult:
    """Save representative JPEG frames and return timestamps and dimensions.

    The source is decoded one frame at a time. Frames are sampled using the
    source FPS and the requested interval, and output names contain a sanitized
    source stem, a unique run identifier, and a zero-padded frame index.
    """
    if (
        isinstance(interval_seconds, bool)
        or not isinstance(interval_seconds, (int, float))
        or not math.isfinite(interval_seconds)
        or interval_seconds <= 0
    ):
        raise ValueError("Frame interval must be a finite number greater than zero.")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int):
        raise ValueError("Maximum frame count must be an integer.")
    if max_frames <= 0:
        raise ValueError("Maximum frame count must be greater than zero.")

    video_path = Path(path)
    metadata = inspect_video_metadata(video_path, max_size_bytes=max_size_bytes)
    if metadata.fps is None:
        raise ValueError("Cannot extract frames because the video FPS is invalid or unavailable.")
    if metadata.frame_count is None:
        raise ValueError("Cannot extract frames because the video contains no frames.")

    output_path = Path(output_directory)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("Frame output path exists and is not a directory.")
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_output = output_path.resolve()

    cv2 = load_opencv()
    capture = cv2.VideoCapture(str(video_path))
    extracted_frames: list[ExtractedFrameInfo] = []
    next_sample_timestamp = 0.0
    last_timestamp: float | None = None
    run_id = uuid4().hex

    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video file for frame extraction.")

        for frame_index in range(metadata.frame_count):
            success, image = capture.read()
            if not success:
                if frame_index == 0:
                    raise ValueError("Video file contains no decodable frames.")
                break

            timestamp_seconds = (
                capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            )
            if (
                not math.isfinite(timestamp_seconds)
                or timestamp_seconds < 0
                or (last_timestamp is not None and timestamp_seconds <= last_timestamp)
            ):
                timestamp_seconds = frame_index / metadata.fps
            last_timestamp = timestamp_seconds

            if timestamp_seconds >= next_sample_timestamp:
                if len(image.shape) < 2:
                    raise ValueError("OpenCV returned a frame with invalid dimensions.")
                height, width = image.shape[:2]
                if width <= 0 or height <= 0:
                    raise ValueError("OpenCV returned a frame with zero dimensions.")

                filename = (
                    f"{_safe_filename_stem(video_path)}_{run_id}_"
                    f"frame_{frame_index:06d}.jpg"
                )
                frame_path = (resolved_output / filename).resolve()
                if frame_path.parent != resolved_output:
                    raise ValueError("Generated frame path escapes the output directory.")
                if frame_path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite an existing frame: {frame_path}"
                    )
                if not cv2.imwrite(str(frame_path), image):
                    raise OSError(f"OpenCV failed to save extracted frame: {frame_path}")

                extracted_frames.append(
                    ExtractedFrameInfo(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                        width=width,
                        height=height,
                        path=frame_path,
                    )
                )
                if len(extracted_frames) == max_frames:
                    break
                next_sample_timestamp = timestamp_seconds + interval_seconds
    finally:
        capture.release()

    if not extracted_frames:
        raise ValueError("No video frames could be extracted.")

    return FrameExtractionResult(
        video_path=video_path.resolve(),
        output_directory=resolved_output,
        interval_seconds=interval_seconds,
        frames=extracted_frames,
    )
