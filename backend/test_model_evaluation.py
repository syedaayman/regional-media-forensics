from pathlib import Path
import joblib

from app.modules.voice_detection.model_training import prepare_features

from app.modules.voice_detection.evaluation import (
    evaluate_voice_model,
    evaluate_language_wise
)


# Project paths
backend_path = Path(__file__).resolve().parent

dataset_path = (
    backend_path.parent
    / "IndicTTS-Download"
    / "data"
)

manifest_path = (
    backend_path
    / "datasets"
    / "voice_detection"
    / "test.csv"
)

model_path = (
    backend_path
    / "models"
    / "voice_detection"
    / "checkpoints"
    / "voice_detection_model.joblib"
)


# Load trained model
print("Loading trained model...")

model = joblib.load(model_path)


# Extract test features
print("\nExtracting test audio features...")

X_test, y_test, languages = prepare_features(
    manifest_path,
    dataset_path
)


# Overall evaluation
print("\nEvaluating overall test performance...")

results = evaluate_voice_model(
    model,
    X_test,
    y_test
)

print("\nFinal test results:")
print(results)


# Detailed language-wise evaluation
evaluate_language_wise(
    model,
    X_test,
    y_test,
    languages
)