"""Schemas for visual-forensics results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.video import VideoMetadataResponse


class VisualDetectionStatusResponse(BaseModel):
    """Describe the current visual-forensics capabilities."""

    status: str
    detail: str


class XdfdetFrameResult(BaseModel):
    """Prediction for one face crop selected for model inference."""

    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    real_probability: float = Field(ge=0, le=1)
    face_detected: bool


class XdfdetAnalysisResponse(BaseModel):
    """Cautious video-level result from the isolated xdfdet worker."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "inconclusive", "error"]
    detector: Literal["xdfdet"] = "xdfdet"
    model_checkpoint: Literal["aug-cutout-black"] = "aug-cutout-black"
    detection_type: Literal["facial_manipulation"] = "facial_manipulation"
    verdict: Literal[
        "likely_manipulated", "no_manipulation_detected", "inconclusive"
    ]
    video_metadata: VideoMetadataResponse | None = None
    average_real_probability: float | None = Field(default=None, ge=0, le=1)
    average_manipulated_probability: float | None = Field(default=None, ge=0, le=1)
    frames_extracted: int = Field(ge=0)
    frames_analyzed: int = Field(ge=0)
    frames_with_faces: int = Field(ge=0)
    faces_detected: int = Field(default=0, ge=0)
    frame_results: list[XdfdetFrameResult] = Field(default_factory=list)
    decision_threshold: float = Field(ge=0, le=1)
    threshold_source: Literal["provisional_configuration"]
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    model_loading_seconds: float | None = Field(default=None, ge=0)
    preprocessing_seconds: float | None = Field(default=None, ge=0)
    inference_seconds: float | None = Field(default=None, ge=0)
