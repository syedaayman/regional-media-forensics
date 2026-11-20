"""Schemas for video processing routes."""

from pydantic import BaseModel


class VideoStatusResponse(BaseModel):
    """Describe the current video-processing capabilities."""

    status: str
    detail: str


class VideoValidationResponse(BaseModel):
    """Report basic video container validation results."""

    valid: bool
    filename: str
    size_bytes: int | None
    errors: list[str]


class VideoMetadataResponse(BaseModel):
    """Expose the metadata collected by the existing video pipeline."""

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
