"""Routes exposing current visual-forensics capabilities."""

from fastapi import APIRouter

from app.schemas.visual_detection import VisualDetectionStatusResponse

router = APIRouter(prefix="/visual-detection", tags=["visual-detection"])


@router.get("/status", response_model=VisualDetectionStatusResponse)
def get_visual_detection_status() -> VisualDetectionStatusResponse:
    """Report that visual-forensics model inference is not yet available."""
    return VisualDetectionStatusResponse(
        status="not_implemented",
        detail=(
            "Image signature validation is available. Feature extraction, face "
            "detection, manipulation detection, training, and prediction are not implemented."
        ),
    )
