"""Image and video prediction interfaces."""

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from app.modules.video.frame_extraction import FrameExtractionResult
    from app.schemas.visual_detection import XdfdetAnalysisResponse


def predict_image(image_bytes: bytes) -> NoReturn:
    """Report that visual-forensics prediction is not implemented."""
    raise NotImplementedError("Visual-forensics prediction is not implemented.")


def predict_video(
    extraction: "FrameExtractionResult",
    decision_threshold: float = 0.5,
) -> "XdfdetAnalysisResponse":
    """Predict facial manipulation from pipeline-extracted video frames."""
    from app.modules.visual_forensics.manipulation_detection import (
        detect_video_manipulation,
    )

    return detect_video_manipulation(
        extraction, decision_threshold=decision_threshold
    )
