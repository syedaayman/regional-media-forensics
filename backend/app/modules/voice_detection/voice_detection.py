from pathlib import Path
import joblib

from app.modules.voice_detection.prediction import (
    predict_voice_authenticity
)


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

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Voice detection model not found: {self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    def analyze_audio(self, audio_bytes: bytes) -> dict:
        """
        Analyze uploaded audio and return its authenticity prediction.
        """

        if not audio_bytes:
            raise ValueError("Audio file is empty.")

        result = predict_voice_authenticity(
            self.model,
            audio_bytes
        )

        return result


voice_detection_service = VoiceDetectionService()