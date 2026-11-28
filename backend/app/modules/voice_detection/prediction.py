import numpy as np
import librosa
import io

from app.modules.voice_detection.feature_extraction import (
    extract_audio_features
)


FEATURE_NAMES = [
    "rms",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
] + [f"mfcc_{i}" for i in range(1, 14)]


def predict_voice_authenticity(model, audio_bytes: bytes) -> dict:
    """
    Predict whether audio is genuine or AI-generated.
    """

    # Extract audio features
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
    duration = len(audio) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    if duration < 1.0:
        return {
            "prediction": "Insufficient Audio Evidence",
            "label": None,
            "confidence": 0.0,
            "assessment": "Insufficient evidence",
            "status": "insufficient_evidence",
            "model_score": None,
            "threshold": None,
            "indicators": ["Decoded duration is below 1.0 second."],
            "limitations": [
                "The recording is too short for a reliable voice authenticity assessment.",
                "Automated spoof detection is not definitive proof.",
            ],
        }
    if not np.isfinite(audio).all():
        return {
            "prediction": "Insufficient Audio Evidence",
            "label": None,
            "confidence": 0.0,
            "assessment": "Insufficient evidence",
            "status": "insufficient_evidence",
            "model_score": None,
            "threshold": None,
            "indicators": ["Decoded samples contain non-finite values."],
            "limitations": ["The recording could not be safely evaluated."],
        }
    if rms < 1e-4:
        return {
            "prediction": "Insufficient Audio Evidence",
            "label": None,
            "confidence": 0.0,
            "assessment": "Insufficient evidence",
            "status": "insufficient_evidence",
            "model_score": None,
            "threshold": None,
            "indicators": ["Decoded speech energy is effectively silent."],
            "limitations": ["No reliable voice evidence was available."],
        }

    features = extract_audio_features(audio_bytes)

    # Verify feature names and order
    if list(features.keys()) != FEATURE_NAMES:
        raise ValueError(
            "Audio feature names or order do not match the trained model."
        )

    # Convert features into a 2D NumPy array
    feature_values = np.array(
        [[features[name] for name in FEATURE_NAMES]],
        dtype=float
    )

    # Predict audio class
    prediction = int(model.predict(feature_values)[0])

    # Get prediction probabilities
    probabilities = model.predict_proba(feature_values)[0]

    # Find the probability for the predicted class
    classes = list(model.classes_)

    if prediction not in classes:
        raise ValueError(
            f"Unexpected prediction label: {prediction}"
        )

    class_index = classes.index(prediction)
    confidence = float(probabilities[class_index])

    if prediction not in (0, 1):
        raise ValueError(
            f"Unexpected prediction label: {prediction}"
        )

    return {
        "prediction": "Genuine Human Speech" if prediction == 0 else "AI-Generated Speech",
        "label": prediction,
        "confidence": round(confidence * 100, 2),
        "assessment": (
            "potential_ai_generated"
            if prediction == 1
            else "inconclusive"
        ),
        "status": "completed",
        "model_score": round(confidence, 6),
        "threshold": None,
        "indicators": [
            "Existing checkpoint produced an internal class score.",
        ],
        "limitations": [
            "The existing voice classifier has not been independently validated.",
            "The score is not a calibrated probability of truth.",
            "Automated detection is not definitive proof of human or AI generation.",
        ],
    }