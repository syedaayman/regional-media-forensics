from pathlib import Path

from pypdf import PdfReader

from app.modules.forensic_report import generate_pdf


def make_result(analysis_id: str, transcript: str) -> dict:
    return {
        "analysis_id": analysis_id,
        "completed_at": "2026-11-27T00:00:00+00:00",
        "evidence": {
            "evidence_id": "unicode-evidence-" + analysis_id,
            "filename": analysis_id + ".wav",
            "sha256": "0" * 64,
            "upload_timestamp": "2026-11-27T00:00:00+00:00",
            "format": "WAV",
            "duration_seconds": 1.0,
            "sample_rate": 16000,
            "channels": 1,
            "file_size_bytes": 32000,
        },
        "language_analysis": {"detected_language": analysis_id, "confidence": 99.0},
        "transcription": {"model": "Unicode test", "status": "success", "text": transcript},
        "voice_authenticity": {
            "status": "completed",
            "assessment": "inconclusive",
            "model": "test",
            "score": None,
            "threshold": None,
        },
        "audio_quality": {
            "snr_db": 30.0,
            "clipping_percentage": 0.0,
            "silence_percentage": 0.0,
            "rms_energy": 0.1,
            "zero_crossing_rate": 0.1,
            "spectral_centroid_hz": 1000.0,
        },
        "tampering_analysis": {"findings": []},
        "visualizations": {
            "waveform": {"download_endpoint": "/audio/visualizations/unicode/waveform"},
            "spectrogram": {"download_endpoint": "/audio/visualizations/unicode/spectrogram"},
        },
        "overall_assessment": {
            "classification": "inconclusive",
            "summary": "Unicode rendering test.",
            "limitations": ["Controlled rendering test only."],
        },
    }


def test_pdf_generation_for_supported_unicode_scripts() -> None:
    scripts = {
        "kn": ("ಕನ್ನಡದಲ್ಲಿ ನಮಸ್ಕಾರ", "ಕನ್ನಡ"),
        "hi": ("नमस्ते नमस्कार", "नमस्ते"),
        "te": ("నమస్కారం", "నమస్కారం"),
        "ta": ("வணக்கம்", "வணக்கம்"),
        "ml": ("നമസ്കാരം", "നമസ്കാരം"),
        "mr": ("नमस्कार", "नमस्कार"),
        "bn": ("নমস্কার", "নমস্কার"),
        "gu": ("નમસ્તે", "નમસ્તે"),
        "en": ("Hello PDF", "Hello PDF"),
    }

    for code, (transcript, needle) in scripts.items():
        path = generate_pdf(make_result(code, transcript))
        assert path.is_file()
        assert path.stat().st_size > 1000

        reader = PdfReader(str(path))
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        assert extracted
        assert needle in extracted
