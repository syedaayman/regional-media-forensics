from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.audio_forensics import _find_sudden_energy_changes
from app.modules.voice_detection.audio_validation import validate_audio_upload


client = TestClient(app)


def test_audio_routes_are_registered() -> None:
    paths = set(app.openapi()["paths"])
    assert "/audio/transcribe/kannada" in paths
    assert "/audio/transcribe/multilingual" in paths
    assert "/audio/detect" in paths


def test_empty_kannada_upload(monkeypatch) -> None:
    class UnsupportedService:
        def transcribe_audio(self, audio_bytes, language):
            raise AssertionError("model should not run")

    monkeypatch.setattr(
        "app.routers.audio.get_speech_to_text_service",
        lambda: UnsupportedService(),
    )
    response = client.post(
        "/audio/transcribe/kannada",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_audio"


def test_empty_detect_upload(monkeypatch) -> None:
    response = client.post(
        "/audio/detect",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_audio"


def test_invalid_mime_extension(monkeypatch) -> None:
    response = client.post(
        "/audio/transcribe/multilingual",
        params={"language": "kn"},
        files={"file": ("audio.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_audio"


def test_mocked_kannada_success(monkeypatch) -> None:
    class KannadaService:
        def transcribe_audio(self, audio_bytes, language):
            assert audio_bytes.startswith(b"RIFF")
            assert language == "kn"
            return {
                "transcript": "ನಮಸ್ಕಾರ",
                "requested_language": "kn",
                "detected_language": "kn",
                "language_name": "Kannada",
                "language_probability": None,
                "language_detection": "Kannada forced",
                "status": "success",
                "warning": None,
            }

    monkeypatch.setattr(
        "app.routers.audio.validate_audio_upload",
        lambda *_args: 1.0,
    )
    monkeypatch.setattr(
        "app.routers.audio.get_speech_to_text_service",
        lambda: KannadaService(),
    )

    wav = b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * 20
    response = client.post(
        "/audio/transcribe/kannada",
        files={"file": ("sample.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    assert set(response.json()) == {"filename", "transcription"}
    assert set(response.json()["transcription"]) == {
        "transcript", "detected_language", "language_name",
        "language_probability", "status",
    }
    assert response.json()["transcription"]["transcript"] == "ನಮಸ್ಕಾರ"
    assert response.json()["transcription"]["detected_language"] == "kn"
    assert response.json()["transcription"]["status"] == "success"


def test_mocked_automatic_language_detection(monkeypatch) -> None:
    class MultilingualService:
        def transcribe_audio(self, audio_bytes, language):
            assert language is None
            return {
                "transcript": "hello",
                "requested_language": None,
                "detected_language": "en",
                "language_name": "English",
                "language_probability": 99.8,
                "language_detection": "Model auto-detected",
                "status": "success",
                "warning": None,
            }

    monkeypatch.setattr(
        "app.routers.audio.validate_audio_upload",
        lambda *_args: 1.0,
    )
    monkeypatch.setattr(
        "app.routers.audio.get_multilingual_speech_to_text_service",
        lambda: MultilingualService(),
    )

    wav = b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * 20
    response = client.post(
        "/audio/transcribe/multilingual",
        files={"file": ("sample.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    assert set(response.json()) == {"filename", "transcription"}
    assert set(response.json()["transcription"]) == {
        "transcript", "detected_language", "language_name",
        "language_probability", "status",
    }
    assert response.json()["transcription"]["detected_language"] == "en"


def test_multilingual_route_collapses_repeated_sentence_punctuation(
    monkeypatch,
) -> None:
    from types import SimpleNamespace
    import wave
    from app.modules.voice_detection.multilingual_speech_to_text import (
        MultilingualSpeechToText,
    )

    class Model:
        def transcribe(self, _path, **_kwargs):
            segment = SimpleNamespace(
                start=0.0,
                end=1.0,
                text="\u0928\u092e\u0938\u094d\u0915\u093e\u0930" + "\u0964" * 4,
            )
            info = SimpleNamespace(language="hi", language_probability=0.99)
            return iter([segment]), info

    service = MultilingualSpeechToText.__new__(MultilingualSpeechToText)
    service.model = Model()
    monkeypatch.setattr(
        "app.routers.audio.get_multilingual_speech_to_text_service",
        lambda: service,
    )
    audio_buffer = BytesIO()
    with wave.open(audio_buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(b"\x01\x00" * 16000)

    response = client.post(
        "/audio/transcribe/multilingual?language=hi",
        files={
            "file": (
                "hindi.wav",
                audio_buffer.getvalue(),
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200
    transcription = response.json()["transcription"]
    assert transcription["transcript"] == "\u0928\u092e\u0938\u094d\u0915\u093e\u0930\u0964"
    assert transcription["detected_language"] == "hi"
    assert transcription["status"] == "success"


def test_automatic_detection_of_unsupported_language_returns_422(monkeypatch) -> None:
    class MultilingualService:
        def transcribe_audio(self, audio_bytes, language):
            assert language is None
            return {
                "transcript": "bonjour",
                "requested_language": None,
                "detected_language": "fr",
                "language_name": "Unknown / Unsupported",
                "language_probability": 99.0,
                "status": "unsupported_language",
                "warning": "The detected language is not supported.",
            }

    monkeypatch.setattr(
        "app.routers.audio.validate_audio_upload",
        lambda *_args: 1.0,
    )
    monkeypatch.setattr(
        "app.routers.audio.get_multilingual_speech_to_text_service",
        lambda: MultilingualService(),
    )

    wav = b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * 20
    response = client.post(
        "/audio/transcribe/multilingual",
        files={"file": ("sample.wav", wav, "audio/wav")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_language"
    assert "fr" in response.json()["detail"]["message"]


def test_mocked_voice_detection_success(monkeypatch) -> None:
    class VoiceService:
        def analyze_audio(self, audio_bytes):
            return {
                "prediction": "Genuine Human Speech",
                "label": 0,
                "confidence": 66.04,
            }

    monkeypatch.setattr(
        "app.routers.audio.validate_audio_upload",
        lambda *_args: 1.0,
    )
    monkeypatch.setattr(
        "app.routers.audio.get_voice_detection_service",
        lambda: VoiceService(),
    )

    wav = b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * 20
    response = client.post(
        "/audio/detect",
        files={"file": ("sample.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    assert set(response.json()) == {"filename", "analysis"}
    assert set(response.json()["analysis"]) == {
        "prediction", "label", "confidence", "status",
    }
    assert response.json()["analysis"]["prediction"] in {
        "Genuine Human Speech", "AI-Generated Speech",
    }


def test_mocked_model_failure_is_safe(monkeypatch) -> None:
    class FailingKannadaService:
        def transcribe_audio(self, audio_bytes, language):
            raise RuntimeError("sensitive internal detail")

    monkeypatch.setattr(
        "app.routers.audio.validate_audio_upload",
        lambda *_args: 1.0,
    )
    monkeypatch.setattr(
        "app.routers.audio.get_speech_to_text_service",
        lambda: FailingKannadaService(),
    )

    wav = b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * 20
    response = client.post(
        "/audio/transcribe/kannada",
        files={"file": ("sample.wav", wav, "audio/wav")},
    )
    assert response.status_code == 500
    assert "sensitive internal detail" not in response.text


@pytest.mark.parametrize("language", ["xx", "not-a-language"])
def test_unsupported_language(language: str, monkeypatch) -> None:
    class UnsupportedService:
        def transcribe_audio(self, audio_bytes, requested_language):
            from app.modules.voice_detection.multilingual_speech_to_text import (
                UnsupportedLanguageError,
            )
            raise UnsupportedLanguageError(
                f"Unsupported language code: {requested_language}"
            )

    monkeypatch.setattr(
        "app.routers.audio.validate_audio_upload",
        lambda *_args: 1.0,
    )
    monkeypatch.setattr(
        "app.routers.audio.get_multilingual_speech_to_text_service",
        lambda: UnsupportedService(),
    )
    response = client.post(
        "/audio/transcribe/multilingual",
        params={"language": language},
        files={"file": ("audio.wav", b"not-a-valid-wav", "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_language"


def test_corrupted_wav_rejected() -> None:
    with pytest.raises(ValueError, match="malformed"):
        validate_audio_upload(
            b"RIFF-not-a-real-wave",
            "corrupt.wav",
            "audio/wav",
        )


def test_upload_filename_is_reduced_to_safe_basename(monkeypatch) -> None:
    class Service:
        def analyze_audio(self, audio_bytes):
            return {
                "prediction": "Genuine Human Speech",
                "label": 0,
                "confidence": 50.0,
            }

    monkeypatch.setattr(
        "app.routers.audio.validate_audio_upload",
        lambda *_args: 1.0,
    )
    monkeypatch.setattr(
        "app.routers.audio.get_voice_detection_service",
        lambda: Service(),
    )
    response = client.post(
        "/audio/detect",
        files={
            "file": (
                "..\\..\\unsafe.wav",
                b"RIFF" + b"\x00" * 8 + b"WAVE" + b"\x00" * 20,
                "audio/wav",
            )
        },
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "unsafe.wav"


def test_complete_analysis_workflow_and_report(monkeypatch, tmp_path) -> None:
    import wave
    from app.modules import audio_forensics, forensic_report
    from app.modules.audio_forensics import (
        _initialize_database,
        get_analysis,
    )

    test_evidence_root = tmp_path / "audio_evidence"
    test_database_path = tmp_path / "audio_forensics.db"
    test_report_root = tmp_path / "reports"
    monkeypatch.setattr(audio_forensics, "EVIDENCE_ROOT", test_evidence_root)
    monkeypatch.setattr(audio_forensics, "DATABASE_PATH", test_database_path)
    monkeypatch.setattr("app.routers.audio.EVIDENCE_ROOT", test_evidence_root)
    monkeypatch.setattr(forensic_report, "REPORT_ROOT", test_report_root)
    monkeypatch.setattr("app.routers.audio.REPORT_ROOT", test_report_root)
    _initialize_database()

    wav_path = tmp_path / "complete.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    class MultilingualService:
        SUPPORTED_LANGUAGES = {"en": "English"}

        def transcribe_audio(self, audio_bytes, language):
            return {
                "transcript": "hello",
                "detected_language": "en",
                "language_probability": 97.0,
                "language_name": "English",
                "status": "success",
                "segments": [{"start": 0.0, "end": 0.1, "text": "hello"}],
                "model": "test multilingual model",
            }

    class VoiceService:
        def analyze_audio(self, audio_bytes):
            return {
                "prediction": "Genuine Human Speech",
                "label": 0,
                "confidence": 60.0,
            }

    monkeypatch.setattr(
        "app.routers.audio.get_multilingual_speech_to_text_service",
        lambda: MultilingualService(),
    )
    monkeypatch.setattr(
        "app.routers.audio.get_voice_detection_service",
        lambda: VoiceService(),
    )

    response = client.post(
        "/audio/analyze",
        files={"file": ("complete.wav", wav_path.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["evidence"]["sha256"]
    assert body["transcription"]["text"] == "hello"
    assert body["audio_quality"]["status"] == "completed"
    assert body["tampering_analysis"]["status"] == "completed"
    assert body["visualizations"]["waveform"]["status"] == "completed"

    analysis_id = body["analysis_id"]
    status_response = client.get(f"/audio/analyses/{analysis_id}/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    integrity_response = client.get(
        f"/audio/evidence/{body['evidence']['evidence_id']}/integrity"
    )
    assert integrity_response.status_code == 200
    assert integrity_response.json()["verified"] is True

    report_response = client.get(f"/audio/reports/{analysis_id}")
    assert report_response.status_code == 200
    assert report_response.headers["content-type"].startswith("application/pdf")
    assert report_response.content.startswith(b"%PDF-")

    _initialize_database()
    persisted = get_analysis(analysis_id)
    assert persisted is not None
    assert persisted["analysis_id"] == analysis_id
    assert persisted["evidence"]["evidence_id"] == body["evidence"]["evidence_id"]


def test_sudden_energy_filter_ignores_normal_frame_changes() -> None:
    rms = np.array([0.10, 0.11, 0.13, 0.16, 0.14, 0.18, 0.20])

    _, indices, _ = _find_sudden_energy_changes(rms)

    assert len(indices) == 0


def test_sudden_energy_filter_keeps_large_discontinuity() -> None:
    rms = np.array([0.10, 0.11, 0.12, 0.40, 0.41])

    _, indices, _ = _find_sudden_energy_changes(rms)

    assert indices.tolist() == [2]


def test_tampering_analysis_detects_spectral_boundary(tmp_path) -> None:
    import soundfile as sf
    from app.modules.audio_forensics import analyze_quality_and_tampering

    sample_rate = 16000
    time = np.arange(sample_rate * 2) / sample_rate
    audio = np.where(
        time < 1.0,
        0.20 * np.sin(2 * np.pi * 200 * time),
        0.20 * np.sin(2 * np.pi * 3000 * time),
    ).astype(np.float32)
    working_path = tmp_path / "spectral_boundary.wav"
    sf.write(str(working_path), audio, sample_rate)

    _, tampering, _ = analyze_quality_and_tampering(working_path)

    assert any(
        finding["finding_type"] in {
            "spectral_discontinuity",
            "sudden_energy_change",
        }
        for finding in tampering["findings"]
    )


def test_tampering_analysis_detects_dc_offset_boundary(tmp_path) -> None:
    import soundfile as sf
    from app.modules.audio_forensics import analyze_quality_and_tampering

    sample_rate = 16000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)
    audio[:sample_rate] = 0.01
    audio[sample_rate:] = 0.40
    working_path = tmp_path / "boundary.wav"
    sf.write(str(working_path), audio, sample_rate)

    _, tampering, _ = analyze_quality_and_tampering(working_path)

    assert any(
        finding["finding_type"] in {
            "dc_offset_or_copy_boundary",
            "sudden_energy_change",
        }
        for finding in tampering["findings"]
    )
