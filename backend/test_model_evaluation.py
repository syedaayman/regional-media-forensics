from pathlib import Path
import joblib

from app.modules.voice_detection.model_training import (
    prepare_features,
    FEATURE_NAMES
)

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


# Check required files
if not manifest_path.exists():
    raise FileNotFoundError(
        f"Test manifest not found: {manifest_path}"
    )

if not model_path.exists():
    raise FileNotFoundError(
        f"Trained model not found: {model_path}"
    )

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {dataset_path}"
    )


# Load trained model
print("Loading trained model...")

model = joblib.load(model_path)

if model.n_features_in_ != len(FEATURE_NAMES):
    raise ValueError(
        f"Model expects {model.n_features_in_} features, "
        f"but the current pipeline provides {len(FEATURE_NAMES)}."
    )


# Extract test features
print("\nExtracting test audio features...")

X_test, y_test, languages = prepare_features(
    manifest_path,
    dataset_path
)


# Validate test data
if len(X_test) == 0:
    raise ValueError("No test audio samples were processed.")

if not (
    len(X_test) == len(y_test) == len(languages)
):
    raise ValueError(
        "Test features, labels, and languages have different lengths."
    )

if X_test.shape[1] != len(FEATURE_NAMES):
    raise ValueError(
        f"Expected {len(FEATURE_NAMES)} features, "
        f"but received {X_test.shape[1]}."
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


# Language-wise evaluation
print("\nEvaluating performance by language...")

language_results = evaluate_language_wise(
    model,
    X_test,
    y_test,
    languages
)

print("\nTest evaluation completed successfully.")