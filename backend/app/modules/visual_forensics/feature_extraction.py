"""Feature extraction interface pending an image decoding implementation."""

from typing import NoReturn


def extract_visual_features(image_bytes: bytes) -> NoReturn:
    """Report that visual feature extraction is not implemented."""
    raise NotImplementedError("Visual feature extraction is not implemented.")
