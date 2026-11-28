from pathlib import Path
import joblib

from app.modules.voice_detection.prediction import (
    predict_voice_authenticity
)


EXPECTED_FEATURE_COUNT = 17


class VoiceDetectionService:
    """
    Service for loading the trained voice detection model
    and predicting whether uploaded audio is genuine
    or AI-generated.
    """

    def __init__(self):

        backend_path = Path(__file__).resolve().parents[3]

        self.model_path = (
            backend_path
            / "models"
            / "voice_detection"
            / "checkpoints"
            / "voice_detection_model.joblib"
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Voice detection model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        if not hasattr(self.model, "n_features_in_"):
            raise ValueError(
                "Loaded model does not expose its expected feature count."
            )

        if self.model.n_features_in_ != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Model expects {self.model.n_features_in_} features, "
                f"but the pipeline provides {EXPECTED_FEATURE_COUNT}."
            )

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "Loaded model does not support prediction probabilities."
            )

        if set(self.model.classes_) != {0, 1}:
            raise ValueError(
                f"Unexpected model classes: {self.model.classes_}"
            )

    def analyze_audio(self, audio_bytes: bytes) -> dict:
        """
        Analyze uploaded audio and return its authenticity prediction.
        """

        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise ValueError("A valid, non-empty audio byte sequence is required.")

        result = predict_voice_authenticity(
            self.model,
            audio_bytes
        )

        return result


voice_detection_service: VoiceDetectionService | None = None


def get_voice_detection_service() -> VoiceDetectionService:
    global voice_detection_service
    if voice_detection_service is None:
        voice_detection_service = VoiceDetectionService()
    return voice_detection_service