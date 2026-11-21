from pathlib import Path

from app.modules.voice_detection.dataset_loader import (
    iter_audio_samples
)

from app.modules.voice_detection.voice_detection import (
    voice_detection_service
)


# Project paths
backend_path = Path(__file__).resolve().parent

dataset_path = (
    backend_path.parent
    / "IndicTTS-Download"
    / "data"
)

test_manifest = (
    backend_path
    / "datasets"
    / "voice_detection"
    / "test.csv"
)


print("\nLoading a genuine speech sample...")

# Label 0 = Genuine Human Speech
sample = next(
    (
        item
        for item in iter_audio_samples(
            test_manifest,
            dataset_path
        )
        if item["label"] == 0
    ),
    None
)

if sample is None:
    raise ValueError(
        "No genuine speech sample found in the test dataset."
    )

print("Audio sample loaded successfully!")

print("\nRunning voice authenticity prediction...")

result = voice_detection_service.analyze_audio(
    sample["audio_bytes"]
)


print("\nVOICE DETECTION RESULT")
print("=" * 40)

print("Actual label:", sample["label"])
print("Actual class: Genuine Human Speech")
print("Language:", sample["language"])
print("Predicted result:", result["prediction"])
print("Predicted label:", result["label"])
print("Confidence:", result["confidence"], "%")


# Check whether the prediction is correct
if result["label"] == sample["label"]:
    print("\nRESULT: CORRECT PREDICTION")
else:
    print("\nRESULT: INCORRECT PREDICTION")