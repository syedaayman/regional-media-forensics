import io
import wave

import numpy as np
import pytest

from app.modules.voice_detection.multilingual_speech_to_text import (
    SUPPORTED_LANGUAGES,
)
from app.modules.voice_detection.multilingual_speech_to_text import (
    segment_is_repetitive,
)
from app.modules.voice_detection.multilingual_speech_to_text import (
    clean_hallucinated_text as clean_multilingual_text,
)
from app.modules.voice_detection.speech_to_text import (
    clean_hallucinated_text as clean_kannada_text,
)
from app.modules.voice_detection.prediction import predict_voice_authenticity


def make_wav(duration_seconds: float, rms: float = 0.1) -> bytes:
    sample_rate = 16000
    count = int(duration_seconds * sample_rate)
    audio = np.zeros(count, dtype=np.float32)
    if rms:
        audio[:] = rms
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(
            (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        )
    return buffer.getvalue()


def test_exactly_nine_supported_language_codes() -> None:
    assert set(SUPPORTED_LANGUAGES) == {
        "kn", "en", "hi", "te", "ta", "ml", "mr", "bn", "gu"
    }


def test_repetition_is_detected_as_hallucination() -> None:
    assert segment_is_repetitive("ನನನನನನನನನನನನ")
    assert segment_is_repetitive("hello hello hello hello hello hello")


@pytest.mark.parametrize(
    "cleaner",
    [clean_multilingual_text, clean_kannada_text],
)
def test_repeated_sentence_punctuation_is_collapsed(cleaner) -> None:
    salutation = "\u0928\u092e\u0938\u094d\u0915\u093e\u0930"
    danda = "\u0964"

    cleaned, repeated_ratio = cleaner(salutation + danda * 4)

    assert cleaned == salutation + danda
    assert repeated_ratio > 0


def test_short_voice_is_insufficient_not_human() -> None:
    result = predict_voice_authenticity(object(), make_wav(0.25))
    assert result["assessment"] == "Insufficient evidence"
    assert result["status"] == "insufficient_evidence"
    assert result["label"] is None


def test_silent_voice_is_insufficient() -> None:
    result = predict_voice_authenticity(object(), make_wav(2.0, rms=0.0))
    assert result["status"] == "insufficient_evidence"


@pytest.mark.parametrize(
    ("prediction", "assessment"),
    [
        (0, "inconclusive"),
        (1, "potential_ai_generated"),
    ],
)
def test_binary_voice_assessment_is_categorised_safely(
    monkeypatch,
    prediction: int,
    assessment: str,
) -> None:
    class Model:
        classes_ = np.array([0, 1])

        def predict(self, features):
            return np.array([prediction])

        def predict_proba(self, features):
            if prediction == 0:
                return np.array([[0.94, 0.06]])
            return np.array([[0.06, 0.94]])

    monkeypatch.setattr(
        "app.modules.voice_detection.prediction.extract_audio_features",
        lambda _audio_bytes: {
            "rms": 0.1,
            "zero_crossing_rate": 0.1,
            "spectral_centroid": 1000.0,
            "spectral_bandwidth": 500.0,
            **{f"mfcc_{i}": 0.0 for i in range(1, 14)},
        },
    )

    result = predict_voice_authenticity(Model(), make_wav(2.0))
    assert result["assessment"] == assessment


def test_multilingual_asr_chunks_long_audio() -> None:
    from types import SimpleNamespace
    from app.modules.voice_detection.multilingual_speech_to_text import (
        MultilingualSpeechToText,
    )

    class Model:
        calls = 0

        def transcribe(self, _path, **_kwargs):
            self.calls += 1
            segment = SimpleNamespace(
                start=0.0,
                end=1.0,
                text="text",
            )
            info = SimpleNamespace(language="en", language_probability=0.99)
            return iter([segment]), info

    service = MultilingualSpeechToText.__new__(MultilingualSpeechToText)
    service.model = Model()
    result = service.transcribe_audio(make_wav(121.0), "en")

    assert service.model.calls == 2
    assert result["status"] == "success"
