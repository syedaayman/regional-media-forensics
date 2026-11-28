from pathlib import Path

from app.modules.voice_detection.dataset_loader import (
    iter_audio_samples
)

from app.modules.voice_detection.voice_detection import (
    get_voice_detection_service,
)


def run_voice_prediction_smoke_test() -> None:
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


    # Sample one existing recording from each label for an operational smoke check.
    test_classes = {
        0: "Genuine Human Speech",
        1: "AI-Generated Speech"
    }

    samples_found = {}

    print("\nLoading genuine and AI-generated speech samples...")

    for item in iter_audio_samples(test_manifest, dataset_path):

        label = item["label"]

        if label in test_classes and label not in samples_found:
            samples_found[label] = item

        if len(samples_found) == 2:
            break


    if len(samples_found) != 2:
        raise ValueError(
            "Could not find samples for both genuine and AI-generated speech."
        )


    correct = 0
    voice_service = get_voice_detection_service()

    for actual_label, sample in samples_found.items():

        print("\n" + "=" * 45)
        print("ACTUAL CLASS:", test_classes[actual_label])
        print("Language:", sample["language"])

        print("\nRunning voice authenticity prediction...")

        result = voice_service.analyze_audio(sample["audio_bytes"])

        print("\nPredicted result:", result["prediction"])
        print("Predicted label:", result["label"])
        print("Confidence:", result["confidence"], "%")

        if result["label"] == actual_label:
            print("\nRESULT: CORRECT PREDICTION")
            correct += 1
        else:
            print("\nRESULT: INCORRECT PREDICTION")


    print("\n" + "=" * 45)
    print("FINAL TEST SUMMARY")
    print("Samples tested:", len(samples_found))
    print("Correct predictions:", correct)
    print("Incorrect predictions:", len(samples_found) - correct)
    print("Test accuracy:", f"{correct / len(samples_found) * 100:.2f}%")


if __name__ == "__main__":
    run_voice_prediction_smoke_test()
