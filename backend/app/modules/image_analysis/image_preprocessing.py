"""Basic image signature validation without external image libraries."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageValidationResult:
    """Result of checking a supported image format's basic header."""

    valid: bool
    image_format: str | None
    width: int | None
    height: int | None
    errors: list[str]


def validate_image_bytes(image_bytes: bytes) -> ImageValidationResult:
    """Recognize common image signatures and read dimensions where available.

    This is a header check, not full decoding or a guarantee that the image is
    well-formed.
    """
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    errors: list[str] = []

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        image_format = "png"
        if len(image_bytes) < 24 or image_bytes[12:16] != b"IHDR":
            errors.append("PNG header is incomplete or missing its IHDR chunk.")
        else:
            width = int.from_bytes(image_bytes[16:20], "big")
            height = int.from_bytes(image_bytes[20:24], "big")
    elif image_bytes.startswith(b"\xff\xd8") and image_bytes.endswith(b"\xff\xd9"):
        image_format = "jpeg"
    elif image_bytes.startswith((b"GIF87a", b"GIF89a")):
        image_format = "gif"
        if len(image_bytes) < 10:
            errors.append("GIF header is incomplete.")
        else:
            width = int.from_bytes(image_bytes[6:8], "little")
            height = int.from_bytes(image_bytes[8:10], "little")
    elif (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        image_format = "webp"
    else:
        errors.append("Unsupported or unrecognized image signature.")

    if image_format is not None and (width == 0 or height == 0):
        errors.append("Image dimensions must be greater than zero.")

    return ImageValidationResult(
        valid=image_format is not None and not errors,
        image_format=image_format,
        width=width,
        height=height,
        errors=errors,
    )


def validate_image_file(path: str | Path) -> ImageValidationResult:
    """Read and validate an image file; report file access failures explicitly."""
    image_path = Path(path)
    if not image_path.exists():
        return ImageValidationResult(
            valid=False,
            image_format=None,
            width=None,
            height=None,
            errors=["Image file does not exist."],
        )
    if not image_path.is_file():
        return ImageValidationResult(
            valid=False,
            image_format=None,
            width=None,
            height=None,
            errors=["Image path is not a file."],
        )

    try:
        image_bytes = image_path.read_bytes()
    except OSError as error:
        return ImageValidationResult(
            valid=False,
            image_format=None,
            width=None,
            height=None,
            errors=[f"Unable to read image file: {error}"],
        )
    return validate_image_bytes(image_bytes)
