"""Face detection interface pending a selected detector and model."""

from typing import NoReturn


def detect_faces(image_bytes: bytes) -> NoReturn:
    """Report that face detection is not implemented."""
    raise NotImplementedError("Face detection is not implemented.")
