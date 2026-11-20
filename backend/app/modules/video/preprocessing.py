"""Video input validation before metadata inspection or frame decoding."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoValidationResult:
    """Result of validating a video file's extension and container signature."""

    valid: bool
    filename: str
    size_bytes: int | None
    errors: list[str]


_SUPPORTED_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_EBML_SIGNATURE = bytes.fromhex("1A45DFA3")


def _has_supported_signature(suffix: str, header: bytes) -> bool:
    """Check common container signatures without decoding the video."""
    if suffix in {".mp4", ".m4v", ".mov"}:
        return len(header) >= 8 and header[4:8] == b"ftyp"
    if suffix == ".avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    if suffix in {".mkv", ".webm"}:
        return header.startswith(_EBML_SIGNATURE)
    return False


def validate_video_file(
    path: str | Path, max_size_bytes: int | None = None
) -> VideoValidationResult:
    """Validate a video's path, extension, size, and recognizable container header.

    The optional size limit is configurable by the caller. Only 16 bytes are
    read for signature validation; this does not prove that the full video is
    decodable or safe to process.
    """
    video_path = Path(path)
    errors: list[str] = []
    size_bytes: int | None = None

    if max_size_bytes is not None and (
        isinstance(max_size_bytes, bool)
        or not isinstance(max_size_bytes, int)
        or max_size_bytes <= 0
    ):
        errors.append("Maximum file size must be greater than zero.")

    try:
        file_exists = video_path.exists()
    except OSError as error:
        file_exists = False
        errors.append(f"Unable to access video path: {error}")

    if not file_exists and not errors:
        errors.append("Video file does not exist.")
    elif file_exists:
        try:
            if not video_path.is_file():
                errors.append("Video path is not a file.")
                return VideoValidationResult(
                    valid=False,
                    filename=video_path.name,
                    size_bytes=None,
                    errors=errors,
                )

            size_bytes = video_path.stat().st_size
            if size_bytes == 0:
                errors.append("Video file is empty.")
            if (
                max_size_bytes is not None
                and isinstance(max_size_bytes, int)
                and not isinstance(max_size_bytes, bool)
                and max_size_bytes > 0
                and size_bytes > max_size_bytes
            ):
                errors.append("Video file exceeds the maximum allowed size.")

            suffix = video_path.suffix.lower()
            if suffix not in _SUPPORTED_SUFFIXES:
                errors.append(f"Unsupported video extension: {suffix or '(none)'}.")
            elif size_bytes > 0:
                with video_path.open("rb") as video_file:
                    header = video_file.read(16)
                if not _has_supported_signature(suffix, header):
                    errors.append("Video container signature does not match its extension.")
        except OSError as error:
            errors.append(f"Unable to read video file: {error}")

    return VideoValidationResult(
        valid=not errors,
        filename=video_path.name,
        size_bytes=size_bytes,
        errors=errors,
    )
