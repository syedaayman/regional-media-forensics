"""Image artifact detection interface pending a defined detection method."""

from typing import NoReturn


def detect_artifacts(image_bytes: bytes) -> NoReturn:
    """Report that image artifact detection is not implemented."""
    raise NotImplementedError("Image artifact detection is not implemented.")
