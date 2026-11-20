from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from app.modules.voice_detection.dataset_loader import iter_audio_samples
from app.modules.voice_detection.feature_extraction import extract_audio_features


def prepare_features(manifest_path, dataset_path):
    """
    Extract audio features, labels, and language names
    from audio samples.
    """

    X = []
    y = []
    languages = []

    processed = 0
    skipped = 0

    for sample in iter_audio_samples(
        manifest_path,
        dataset_path
    ):
        try:
            features = extract_audio_features(
                sample["audio_bytes"]
            )

            X.append(list(features.values()))
            y.append(sample["label"])
            languages.append(sample["language"])

            processed += 1

            if processed % 500 == 0:
                print(f"Processed {processed} audio samples...")

        except Exception as error:
            skipped += 1
            print(f"Skipping audio sample: {error}")

    print("\nFeature extraction completed.")
    print(f"Successfully processed: {processed}")
    print(f"Skipped recordings: {skipped}")

    return (
        np.asarray(X),
        np.asarray(y),
        np.asarray(languages)
    )


def train_voice_model():

    # Project paths
    backend_path = Path(__file__).resolve().parents[3]

    dataset_path = (
        backend_path.parent
        / "IndicTTS-Download"
        / "data"
    )

    manifest_folder = (
        backend_path
        / "datasets"
        / "voice_detection"
    )

    train_manifest = manifest_folder / "train.csv"
    validation_manifest = manifest_folder / "validation.csv"

    # Prepare training data
    print("\nLoading training dataset...")

    X_train, y_train, _ = prepare_features(
        train_manifest,
        dataset_path
    )

    # Prepare validation data
    print("\nLoading validation dataset...")

    X_val, y_val, _ = prepare_features(
        validation_manifest,
        dataset_path
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "Training or validation dataset is empty."
        )

    # Train model
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Validation
    print("\nEvaluating validation dataset...")

    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    print("\nVOICE DETECTION VALIDATION RESULTS")
    print("=" * 45)

    print(f"Validation Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=[
                "Genuine Speech",
                "AI-Generated Speech"
            ],
            zero_division=0
        )
    )

    # Save model
    model_folder = (
        backend_path
        / "models"
        / "voice_detection"
        / "checkpoints"
    )

    model_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_folder
        / "voice_detection_model.joblib"
    )

    joblib.dump(model, model_path)

    print("\nModel saved successfully!")
    print(f"Model location: {model_path}")


if __name__ == "__main__":
    train_voice_model()