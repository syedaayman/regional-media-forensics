"""Tests for basic image signature validation."""

from app.modules.image_analysis.image_preprocessing import validate_image_bytes


def test_validate_image_bytes_reads_png_dimensions() -> None:
    png_header = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\x0dIHDR"
        b"\x00\x00\x00\x20\x00\x00\x00\x10"
    )

    result = validate_image_bytes(png_header)

    assert result.valid
    assert result.image_format == "png"
    assert result.width == 32
    assert result.height == 16


def test_validate_image_bytes_rejects_unknown_signature() -> None:
    result = validate_image_bytes(b"not an image")

    assert not result.valid
    assert result.errors == ["Unsupported or unrecognized image signature."]


def test_validate_image_bytes_rejects_zero_dimension_png() -> None:
    png_header = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\x0dIHDR"
        b"\x00\x00\x00\x00\x00\x00\x00\x10"
    )

    result = validate_image_bytes(png_header)

    assert not result.valid
    assert "Image dimensions must be greater than zero." in result.errors
