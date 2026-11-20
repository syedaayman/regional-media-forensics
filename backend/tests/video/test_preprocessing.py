"""Tests for video path, size, extension, and container validation."""

from pathlib import Path

from app.modules.video.preprocessing import validate_video_file


def _write_mp4(path: Path) -> None:
    """Write a small MP4-like file header for validation tests."""
    path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8)


def test_validate_video_file_accepts_mp4_container(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)

    result = validate_video_file(video_path)

    assert result.valid
    assert result.filename == "sample.mp4"
    assert result.size_bytes == 20
    assert result.errors == []


def test_validate_video_file_enforces_configured_max_size(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_mp4(video_path)

    result = validate_video_file(video_path, max_size_bytes=19)

    assert not result.valid
    assert "Video file exceeds the maximum allowed size." in result.errors


def test_validate_video_file_rejects_mismatched_signature(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not an mp4")

    result = validate_video_file(video_path)

    assert not result.valid
    assert "Video container signature does not match its extension." in result.errors


def test_validate_video_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.txt"
    video_path.write_text("not a video")

    result = validate_video_file(video_path)

    assert not result.valid
    assert "Unsupported video extension: .txt." in result.errors


def test_validate_video_file_rejects_empty_file(tmp_path: Path) -> None:
    video_path = tmp_path / "empty.mp4"
    video_path.touch()

    result = validate_video_file(video_path)

    assert not result.valid
    assert "Video file is empty." in result.errors


def test_validate_video_file_reports_missing_file(tmp_path: Path) -> None:
    result = validate_video_file(tmp_path / "missing.mp4")

    assert not result.valid
    assert result.size_bytes is None
    assert "Video file does not exist." in result.errors


def test_validate_video_file_rejects_directory(tmp_path: Path) -> None:
    result = validate_video_file(tmp_path)

    assert not result.valid
    assert "Video path is not a file." in result.errors
