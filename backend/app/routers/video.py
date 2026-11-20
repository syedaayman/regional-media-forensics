"""Routes exposing current video-processing capabilities."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.modules.video.preprocessing import validate_video_file
from app.modules.visual_forensics.visual_forensics import (
    analyze_video_visual_content,
)
from app.schemas.visual_detection import XdfdetAnalysisResponse
from app.schemas.video import VideoStatusResponse

router = APIRouter(prefix="/video", tags=["video"])
MAX_VIDEO_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.get("/status", response_model=VideoStatusResponse)
def get_video_status() -> VideoStatusResponse:
    """Report the implemented video and visual-forensics capabilities."""
    return VideoStatusResponse(
        status="partial",
        detail=(
            "Video upload validation, metadata inspection, frame extraction, and "
            "isolated xdfdet visual analysis are available."
        ),
    )


@router.post("/analyze", response_model=XdfdetAnalysisResponse)
async def analyze_uploaded_video(
    file: UploadFile = File(...),
) -> XdfdetAnalysisResponse:
    """Validate an uploaded video and analyze it without blocking the event loop."""
    temporary_path: Path | None = None
    original_filename = Path(file.filename or "upload").name or "upload"
    suffix = Path(original_filename).suffix.lower()

    try:
        with NamedTemporaryFile(
            prefix="rmf-video-", suffix=suffix, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            size_bytes = 0
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size_bytes += len(chunk)
                if size_bytes > MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "video_too_large",
                            "message": (
                                "Video exceeds the maximum upload size of "
                                f"{MAX_VIDEO_UPLOAD_BYTES} bytes."
                            ),
                        },
                    )
                temporary_file.write(chunk)

        validation = validate_video_file(
            temporary_path, max_size_bytes=MAX_VIDEO_UPLOAD_BYTES
        )
        if not validation.valid:
            if validation.size_bytes == 0:
                status_code = 400
                code = "empty_video"
            elif any(
                "Unsupported video extension" in error
                or "signature does not match" in error
                for error in validation.errors
            ):
                status_code = 415
                code = "unsupported_video"
            else:
                status_code = 400
                code = "invalid_video"
            raise HTTPException(
                status_code=status_code,
                detail={"code": code, "errors": validation.errors},
            )

        result = await run_in_threadpool(
            analyze_video_visual_content, temporary_path
        )
        if result.video_metadata is not None:
            result = result.model_copy(
                update={
                    "video_metadata": result.video_metadata.model_copy(
                        update={"filename": original_filename}
                    )
                }
            )
        return result
    finally:
        try:
            await file.close()
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
