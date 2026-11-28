from io import BytesIO
from pathlib import PurePath

import librosa
import numpy as np
import struct


MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 1800.0

SUPPORTED_EXTENSIONS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".aac": "audio/aac",
    ".opus": "audio/ogg",
}


def validate_audio_upload(
    audio_bytes: bytes,
    filename: str | None,
    content_type: str | None,
) -> float:
    if not audio_bytes:
        raise ValueError("Audio file is empty.")
    if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError(
            f"Audio file exceeds the {MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    safe_name = PurePath(filename or "").name
    suffix = PurePath(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported audio format. Supported formats: "
            "WAV, MP3, FLAC, OGG, M4A, and WebM."
        )

    declared_type = (content_type or "").split(";", 1)[0].strip().lower()
    if declared_type and declared_type not in {
        "application/octet-stream",
        SUPPORTED_EXTENSIONS[suffix],
        "audio/x-wav",
        "audio/x-m4a",
    }:
        raise ValueError("Audio MIME type does not match the file extension.")

    if suffix == ".wav" and not (
        audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]
    ):
        raise ValueError("The uploaded WAV file is malformed.")
    if suffix == ".wav" and len(audio_bytes) >= 44:
        try:
            sample_rate = struct.unpack("<I", audio_bytes[24:28])[0]
            byte_rate = struct.unpack("<I", audio_bytes[28:32])[0]
            data_size = len(audio_bytes) - 44
            if sample_rate <= 0 or sample_rate > 192000:
                raise ValueError("The uploaded WAV sample rate is unsupported.")
            if byte_rate <= 0:
                raise ValueError("The uploaded WAV file is malformed.")
            header_duration = data_size / float(byte_rate)
            if header_duration > MAX_AUDIO_DURATION_SECONDS:
                raise ValueError(
                    f"Audio duration exceeds the {int(MAX_AUDIO_DURATION_SECONDS)} second limit."
                )
        except ValueError:
            raise
        except struct.error:
            raise ValueError("The uploaded WAV file is malformed.")
    if suffix == ".mp3" and not (
        audio_bytes.startswith(b"ID3")
        or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    ):
        raise ValueError("The uploaded MP3 file is malformed.")
    if suffix == ".flac" and not audio_bytes.startswith(b"fLaC"):
        raise ValueError("The uploaded FLAC file is malformed.")
    if suffix == ".ogg" and not audio_bytes.startswith(b"OggS"):
        raise ValueError("The uploaded OGG file is malformed.")
    if suffix == ".m4a" and len(audio_bytes) < 12:
        raise ValueError("The uploaded M4A file is malformed.")
    if suffix == ".m4a" and b"ftyp" not in audio_bytes[:32]:
        raise ValueError("The uploaded M4A file is malformed.")
    if suffix == ".webm" and not audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        raise ValueError("The uploaded WebM file is malformed.")
    if suffix == ".aac" and not (
        audio_bytes.startswith(b"\xff\xf1")
        or audio_bytes.startswith(b"\xff\xf9")
        or b"adts" in audio_bytes[:32].lower()
    ):
        raise ValueError("The uploaded AAC file is malformed.")
    if suffix == ".opus" and not audio_bytes.startswith(b"OggS"):
        raise ValueError("The uploaded Opus file is malformed.")

    try:
        audio, sample_rate = librosa.load(
            BytesIO(audio_bytes),
            sr=16000,
            mono=True,
        )
    except Exception as error:
        raise ValueError("The uploaded audio cannot be decoded.") from error

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("The uploaded audio contains no valid samples.")

    duration = float(len(audio) / sample_rate)
    if duration <= 0:
        raise ValueError("The uploaded audio has no duration.")
    if duration > MAX_AUDIO_DURATION_SECONDS:
        raise ValueError(
            f"Audio duration exceeds the {int(MAX_AUDIO_DURATION_SECONDS)} second limit."
        )
    return duration
