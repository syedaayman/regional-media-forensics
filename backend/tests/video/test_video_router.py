"""API tests for multipart video upload and mocked visual analysis."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.visual_forensics import visual_forensics
from app.routers import video as video_router
from app.schemas.video import VideoMetadataResponse
from app.schemas.visual_detection import XdfdetAnalysisResponse

client = TestClient(app)
_MP4_HEADER = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8


def _result(
    *,
    status: str = "success",
    verdict: str = "no_manipulation_detected",
    error_code: str | None = None,
) -> XdfdetAnalysisResponse:
    has_prediction = status == "success"
    warnings = []
    if status == "error":
        warnings.append(
            "The visual analysis failed; no authenticity conclusion can be drawn."
        )
    elif status == "inconclusive" or verdict == "inconclusive":
        warnings.append(
            "The visual analysis is inconclusive; it is not an authenticity verdict."
        )
    return XdfdetAnalysisResponse(
        status=status,
        verdict=verdict,
        video_metadata=VideoMetadataResponse(
            filename="temporary.mp4",
            extension=".mp4",
            size_bytes=len(_MP4_HEADER),
            duration_seconds=12.0,
            fps=24.0,
            frame_count=288,
            width=640,
            height=360,
            resolution="640x360",
            aspect_ratio=640 / 360,
        ),
        average_real_probability=0.8 if has_prediction else None,
        average_manipulated_probability=0.2 if has_prediction else None,
        frames_extracted=12,
        frames_analyzed=12 if has_prediction else 0,
        frames_with_faces=12 if has_prediction else 0,
        faces_detected=12 if has_prediction else 0,
        frame_results=[],
        decision_threshold=0.5,
        threshold_source="provisional_configuration",
        warnings=warnings,
        error_code=error_code,
        error_message="worker is unavailable" if error_code else None,
    )


def _track_uploads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    paths: list[Path] = []
    actual_named_temporary_file = video_router.NamedTemporaryFile

    def track_temporary_file(*args, **kwargs):
        temporary_file = actual_named_temporary_file(*args, **kwargs)
        paths.append(Path(temporary_file.name))
        return temporary_file

    monkeypatch.setattr(video_router, "NamedTemporaryFile", track_temporary_file)
    return paths


def test_video_upload_returns_mocked_analysis_and_cleans_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _track_uploads(monkeypatch)
    observed: dict[str, object] = {}

    def mocked_analysis(video_path: Path) -> XdfdetAnalysisResponse:
        observed["bytes"] = video_path.read_bytes()
        observed["exists_during_analysis"] = video_path.is_file()
        return _result()

    monkeypatch.setattr(
        video_router, "analyze_video_visual_content", mocked_analysis
    )
    response = client.post(
        "/video/analyze",
        files={"file": ("sample.mp4", _MP4_HEADER, "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["verdict"] == "no_manipulation_detected"
    assert body["average_real_probability"] == 0.8
    assert body["average_manipulated_probability"] == 0.2
    assert body["model_checkpoint"] == "aug-cutout-black"
    assert body["warnings"] == []
    assert body["video_metadata"]["filename"] == "sample.mp4"
    assert observed == {
        "bytes": _MP4_HEADER,
        "exists_during_analysis": True,
    }
    assert len(paths) == 1
    assert not paths[0].exists()


@pytest.mark.parametrize(
    ("filename", "content", "expected_status", "expected_code"),
    [
        ("not-a-video.txt", b"not a video", 415, "unsupported_video"),
        ("empty.mp4", b"", 400, "empty_video"),
    ],
)
def test_video_upload_rejects_invalid_files_and_cleans_upload(
    filename: str,
    content: bytes,
    expected_status: int,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _track_uploads(monkeypatch)
    response = client.post(
        "/video/analyze",
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert paths and not paths[0].exists()


def test_video_upload_rejects_oversized_file_and_cleans_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_router, "MAX_VIDEO_UPLOAD_BYTES", 16)
    paths = _track_uploads(monkeypatch)

    response = client.post(
        "/video/analyze",
        files={"file": ("large.mp4", _MP4_HEADER + b"x", "video/mp4")},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "video_too_large"
    assert paths and not paths[0].exists()


def test_corrupt_video_returns_structured_error_and_cleans_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _track_uploads(monkeypatch)

    def corrupt_video(*_args, **_kwargs):
        raise ValueError("Video file contains no decodable frames.")

    monkeypatch.setattr(visual_forensics, "process_video", corrupt_video)
    response = client.post(
        "/video/analyze",
        files={"file": ("corrupt.mp4", _MP4_HEADER, "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["verdict"] == "inconclusive"
    assert response.json()["error_code"] == "video_processing_failed"
    assert response.json()["video_metadata"] is None
    assert response.json()["warnings"]
    assert paths and not paths[0].exists()


@pytest.mark.parametrize(
    ("analysis", "expected_status", "expected_code", "expected_frames"),
    [
        (
            _result(status="inconclusive", verdict="inconclusive").model_copy(
                update={"frames_extracted": 4}
            ),
            "inconclusive",
            None,
            4,
        ),
        (
            _result(status="inconclusive", verdict="inconclusive"),
            "inconclusive",
            None,
            12,
        ),
        (
            _result(
                status="error",
                verdict="inconclusive",
                error_code="worker_unavailable",
            ),
            "error",
            "worker_unavailable",
            12,
        ),
    ],
)
def test_inconclusive_and_worker_errors_remain_structured(
    analysis: XdfdetAnalysisResponse,
    expected_status: str,
    expected_code: str | None,
    expected_frames: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        video_router,
        "analyze_video_visual_content",
        lambda _path: analysis,
    )
    response = client.post(
        "/video/analyze",
        files={"file": ("sample.mp4", _MP4_HEADER, "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == expected_status
    assert response.json()["verdict"] == "inconclusive"
    assert response.json()["error_code"] == expected_code
    assert response.json()["frames_extracted"] == expected_frames
    assert response.json()["frames_with_faces"] == 0
    if expected_status == "inconclusive":
        assert response.json()["warnings"]
