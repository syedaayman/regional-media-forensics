"""Image analysis interface awaiting implemented visual models."""

from typing import NoReturn


def analyze_image(image_bytes: bytes) -> NoReturn:
    """Report that AI image analysis is not implemented."""
    raise NotImplementedError("AI image analysis is not implemented.")
