import numpy as np

from app.modules.voice_detection.feature_extraction import (
    extract_audio_features
)


def predict_voice_authenticity(model, audio_bytes: bytes) -> dict:
    """
    Predict whether audio is genuine or AI-generated.
    """

    features = extract_audio_features(audio_bytes)

    feature_values = np.array(
        [list(features.values())],
        dtype=float
    )

    if feature_values.shape[1] != 17:
        raise ValueError(
            f"Expected 17 audio features, "
            f"but received {feature_values.shape[1]}."
        )

    prediction = int(model.predict(feature_values)[0])

    probabilities = model.predict_proba(feature_values)[0]

    confidence = float(probabilities[prediction])

    if prediction == 0:
        result = "Genuine Human Speech"
    elif prediction == 1:
        result = "AI-Generated Speech"
    else:
        raise ValueError(f"Unexpected prediction label: {prediction}")

    return {
        "prediction": result,
        "label": prediction,
        "confidence": round(confidence * 100, 2)
    }