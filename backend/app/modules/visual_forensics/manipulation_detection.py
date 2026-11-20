"""Image and video manipulation detection interfaces."""

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from app.modules.video.frame_extraction import FrameExtractionResult
    from app.schemas.visual_detection import XdfdetAnalysisResponse


def detect_manipulation(image_bytes: bytes) -> NoReturn:
    """Report that manipulation detection is not implemented."""
    raise NotImplementedError("Image manipulation detection is not implemented.")


def detect_video_manipulation(
    extraction: "FrameExtractionResult",
    decision_threshold: float = 0.5,
) -> "XdfdetAnalysisResponse":
    """Run xdfdet on frames already extracted by the RMF video pipeline."""
    from app.modules.visual_forensics.xdfdet_adapter import (
        analyze_extracted_frames,
    )

    return analyze_extracted_frames(
        extraction, decision_threshold=decision_threshold
    )
