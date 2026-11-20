"""Tests for pipeline-to-xdfdet visual-forensics orchestration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.modules.video.metadata_analysis import VideoMetadata
from app.modules.visual_forensics import visual_forensics
from app.schemas.visual_detection import XdfdetAnalysisResponse


def test_video_analysis_uses_pipeline_and_cleans_temporary_frames(
    tmp_path: Path, monkeypatch
) -> None:
    observed_directory: Path | None = None

    def process_video(_path, output_directory, **_kwargs):
        nonlocal observed_directory
        observed_directory = Path(output_directory)
        observed_directory.joinpath("pipeline-frame.jpg").write_bytes(b"frame")
        return SimpleNamespace(
            metadata=VideoMetadata(
                filename="input.mp4",
                extension=".mp4",
                size_bytes=123,
                duration_seconds=12.0,
                fps=24.0,
                frame_count=288,
                width=640,
                height=360,
                resolution="640x360",
                aspect_ratio=640 / 360,
            ),
            extraction=SimpleNamespace(
                frames=[SimpleNamespace(path=observed_directory / "pipeline-frame.jpg")]
            )
        )

    def predict_video(extraction, decision_threshold):
        assert extraction.frames[0].path.is_file()
        assert decision_threshold == 0.4
        return XdfdetAnalysisResponse(
            status="success",
            verdict="no_manipulation_detected",
            average_real_probability=0.8,
            frames_extracted=12,
            frames_analyzed=12,
            frames_with_faces=12,
            faces_detected=12,
            frame_results=[],
            decision_threshold=decision_threshold,
            threshold_source="provisional_configuration",
        )

    monkeypatch.setattr(visual_forensics, "process_video", process_video)
    monkeypatch.setattr(visual_forensics, "predict_video", predict_video)

    result = visual_forensics.analyze_video_visual_content(
        tmp_path / "input.mp4", decision_threshold=0.4
    )

    assert result.verdict == "no_manipulation_detected"
    assert result.average_manipulated_probability == pytest.approx(0.2)
    assert result.video_metadata is not None
    assert result.video_metadata.filename == "input.mp4"
    assert observed_directory is not None
    assert not observed_directory.exists()


def test_video_processing_failures_are_reported_without_predictions(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_processing(*_args, **_kwargs):
        raise ValueError("invalid input video")

    monkeypatch.setattr(visual_forensics, "process_video", fail_processing)
    monkeypatch.setattr(
        visual_forensics,
        "predict_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    result = visual_forensics.analyze_video_visual_content(tmp_path / "bad.mp4")

    assert result.status == "error"
    assert result.verdict == "inconclusive"
    assert result.error_code == "video_processing_failed"
    assert "invalid input video" in (result.error_message or "")


def test_temporary_frames_are_cleaned_when_pipeline_fails(
    tmp_path: Path, monkeypatch
) -> None:
    observed_directory: Path | None = None

    def fail_processing(_path, output_directory, **_kwargs):
        nonlocal observed_directory
        observed_directory = Path(output_directory)
        observed_directory.joinpath("partial-frame.jpg").write_bytes(b"frame")
        raise ValueError("frame extraction failed")

    monkeypatch.setattr(visual_forensics, "process_video", fail_processing)

    result = visual_forensics.analyze_video_visual_content(tmp_path / "bad.mp4")

    assert result.status == "error"
    assert observed_directory is not None
    assert not observed_directory.exists()
