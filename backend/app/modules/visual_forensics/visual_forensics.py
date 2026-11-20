"""High-level visual-forensics entry points."""

from dataclasses import asdict
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NoReturn

from app.modules.video.video_analysis import process_video
from app.modules.visual_forensics.prediction import predict_video
from app.schemas.visual_detection import XdfdetAnalysisResponse
from app.schemas.video import VideoMetadataResponse


def analyze_visual_content(image_bytes: bytes) -> NoReturn:
    """Report that image-only visual-forensics analysis is not implemented."""
    raise NotImplementedError("End-to-end visual-forensics analysis is not implemented.")


def analyze_video_visual_content(
    video_path: str | Path,
    interval_seconds: float = 1.0,
    max_frames: int = 100,
    decision_threshold: float = 0.5,
) -> XdfdetAnalysisResponse:
    """Process a video once, then analyze its saved frames with isolated xdfdet."""
    if (
        isinstance(decision_threshold, bool)
        or not isinstance(decision_threshold, (int, float))
        or not math.isfinite(decision_threshold)
        or not 0 <= decision_threshold <= 1
    ):
        raise ValueError("Decision threshold must be a finite probability between 0 and 1.")
    try:
        with TemporaryDirectory(prefix="rmf-xdfdet-") as output_directory:
            processing = process_video(
                video_path,
                output_directory=output_directory,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
            )
            result = predict_video(
                processing.extraction,
                decision_threshold=decision_threshold,
            )
            if result.status == "error":
                warnings = [
                    "The visual analysis failed; no authenticity conclusion can be drawn."
                ]
            elif (
                result.status == "inconclusive"
                or result.verdict == "inconclusive"
            ):
                warnings = [
                    "The visual analysis is inconclusive; it is not an authenticity verdict."
                ]
            else:
                warnings = []
            return result.model_copy(
                update={
                    "video_metadata": VideoMetadataResponse.model_validate(
                        asdict(processing.metadata)
                    ),
                    "average_manipulated_probability": (
                        1.0 - result.average_real_probability
                        if result.average_real_probability is not None
                        else None
                    ),
                    "warnings": warnings,
                }
            )
    except (OSError, RuntimeError, ValueError) as error:
        return XdfdetAnalysisResponse(
            status="error",
            verdict="inconclusive",
            average_real_probability=None,
            frames_extracted=0,
            frames_analyzed=0,
            frames_with_faces=0,
            decision_threshold=decision_threshold,
            threshold_source="provisional_configuration",
            warnings=[
                "The visual analysis failed; no authenticity conclusion can be drawn."
            ],
            limitations=[
                "xdfdet targets facial manipulation and is not a universal detector for fully AI-generated video.",
                "A high real probability does not prove authenticity; a low real probability does not prove manipulation.",
                "The Hugging Face checkpoint is licensed CC BY-NC 4.0 and is restricted to non-commercial use.",
            ],
            error_code="video_processing_failed",
            error_message=f"{type(error).__name__}: {error}",
        )
