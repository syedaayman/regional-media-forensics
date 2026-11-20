"""Orchestration for video validation, metadata, and frame extraction."""

from dataclasses import dataclass
from pathlib import Path

from app.modules.video.frame_extraction import (
    FrameExtractionResult,
    extract_frames,
)
from app.modules.video.metadata_analysis import VideoMetadata, inspect_video_metadata
from app.modules.video.preprocessing import validate_video_file


@dataclass(frozen=True)
class VideoProcessingResult:
    """Infrastructure results for a video; contains no AI classification."""

    metadata: VideoMetadata
    extraction: FrameExtractionResult


def process_video(
    path: str | Path,
    output_directory: str | Path,
    interval_seconds: float = 1.0,
    max_frames: int = 100,
    max_size_bytes: int | None = None,
) -> VideoProcessingResult:
    """Validate a video, inspect its metadata, and save representative frames."""
    validation = validate_video_file(path, max_size_bytes=max_size_bytes)
    if not validation.valid:
        raise ValueError("Invalid video file: " + "; ".join(validation.errors))

    metadata = inspect_video_metadata(path, max_size_bytes=max_size_bytes)
    extraction = extract_frames(
        path,
        output_directory=output_directory,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
        max_size_bytes=max_size_bytes,
    )
    return VideoProcessingResult(metadata=metadata, extraction=extraction)
