from pathlib import Path
from collections import defaultdict
import csv
import os
import tempfile
import unicodedata

from faster_whisper import WhisperModel

from app.modules.voice_detection.dataset_loader import (
    iter_audio_samples
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    BACKEND_DIR
    / "datasets"
    / "voice_detection"
    / "test.csv"
)

DATASET_PATH = (
    BACKEND_DIR.parent
    / "IndicTTS-Download"
    / "data"
)

OUTPUT_DIR = (
    BACKEND_DIR
    / "reports"
    / "voice_detection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. EVALUATION SETTINGS
# --------------------------------------------------

SAMPLES_PER_LANGUAGE = 100

MODEL_NAME = "small"

EXPECTED_CODES = {
    "Assamese": "as",
    "Bengali": "bn",
    "English": "en",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Sanskrit": "sa",
    "Tamil": "ta",
    "Telugu": "te",
}


# --------------------------------------------------
# 3. TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):

    text = unicodedata.normalize("NFC", text.lower())

    text = "".join(
        char
        if (
            unicodedata.category(char)[0] in ("L", "N")
            or char.isspace()
        )
        else " "
        for char in text
    )

    return " ".join(text.split())


# --------------------------------------------------
# 4. WORD ERROR RATE CALCULATION
# --------------------------------------------------

def calculate_wer(reference, hypothesis):

    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()

    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous = list(range(len(hypothesis_words) + 1))

    for i, reference_word in enumerate(
        reference_words, start=1
    ):

        current = [i]

        for j, hypothesis_word in enumerate(
            hypothesis_words, start=1
        ):

            substitution_cost = (
                0 if reference_word == hypothesis_word else 1
            )

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + substitution_cost
                )
            )

        previous = current

    errors = previous[-1]

    return errors / len(reference_words)


# --------------------------------------------------
# 5. COLLECT BALANCED LANGUAGE SAMPLES
# --------------------------------------------------

def collect_samples():

    print("\nCollecting dataset samples...")

    samples_by_language = defaultdict(list)

    for sample in iter_audio_samples(
        MANIFEST_PATH,
        DATASET_PATH,
        batch_size=32
    ):

        language = sample["language"]

        if len(samples_by_language[language]) < SAMPLES_PER_LANGUAGE:

            samples_by_language[language].append(sample)

    print("\nSamples collected:")

    for language in sorted(samples_by_language):

        print(
            language,
            ":",
            len(samples_by_language[language])
        )

    return samples_by_language


# --------------------------------------------------
# 6. TRANSCRIBE AND EVALUATE
# --------------------------------------------------

def evaluate_samples(samples_by_language):

    print("\nLoading Whisper Small model...")

    model = WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8"
    )

    print("Model loaded successfully!")

    results = []

    total_samples = sum(
        len(items)
        for items in samples_by_language.values()
    )

    current = 0

    for language in sorted(samples_by_language):

        for sample in samples_by_language[language]:

            current += 1

            print(
                f"\nProcessing {current}/{total_samples}: "
                f"{language}"
            )

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False
                ) as temp_file:

                    temp_file.write(sample["audio_bytes"])
                    temp_path = temp_file.name

                segments, info = model.transcribe(
                    temp_path,
                    language=None,
                    beam_size=1,
                    vad_filter=True
                )

                predicted_text = " ".join(
                    segment.text.strip()
                    for segment in segments
                    if segment.text.strip()
                )

                reference_text = sample["reference_text"]

                expected_code = EXPECTED_CODES.get(language)

                if expected_code is None:

                    detection_correct = "Not scored"

                else:

                    detection_correct = (
                        info.language == expected_code
                    )

                wer = calculate_wer(
                    reference_text,
                    predicted_text
                )

                results.append({

                    "language": language,

                    "label": sample["label"],

                    "expected_language_code": (
                        expected_code or "Not mapped"
                    ),

                    "detected_language_code": info.language,

                    "language_probability_percent": round(
                        info.language_probability * 100,
                        2
                    ),

                    "language_detection_correct": detection_correct,

                    "reference_text": reference_text,

                    "predicted_text": predicted_text,

                    "wer": round(wer, 4)

                })

            except Exception as error:

                print("Sample processing failed:", error)

            finally:

                if temp_path and os.path.exists(temp_path):

                    os.remove(temp_path)

    return results


# --------------------------------------------------
# 7. SAVE INDIVIDUAL RESULTS
# --------------------------------------------------

def save_detailed_results(results):

    if not results:

        print("\nNo results available.")
        return

    output_file = (
        OUTPUT_DIR
        / "multilingual_test_details.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nDetailed results saved:")
    print(output_file)


# --------------------------------------------------
# 8. GENERATE LANGUAGE-WISE SUMMARY
# --------------------------------------------------

def generate_summary(results):

    summary = []

    languages = sorted(
        set(result["language"] for result in results)
    )

    for language in languages:

        language_results = [
            result
            for result in results
            if result["language"] == language
        ]

        scored = [
            result
            for result in language_results
            if isinstance(
                result["language_detection_correct"], bool
            )
        ]

        if scored:

            correct = sum(
                result["language_detection_correct"]
                for result in scored
            )

            detection_accuracy = (
                correct / len(scored)
            ) * 100

        else:

            detection_accuracy = None

        average_wer = sum(
            result["wer"]
            for result in language_results
        ) / len(language_results)

        summary.append({

            "language": language,

            "samples_evaluated": len(language_results),

            "language_detection_accuracy_percent": (
                round(detection_accuracy, 2)
                if detection_accuracy is not None
                else "Not scored"
            ),

            "average_wer": round(average_wer, 4)

        })

    summary_file = (
        OUTPUT_DIR
        / "multilingual_test_summary.csv"
    )

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\n========== FINAL EVALUATION ==========")

    for row in summary:

        print(
            f"\nLanguage: {row['language']}"
            f"\nSamples: {row['samples_evaluated']}"
            f"\nDetection accuracy: "
            f"{row['language_detection_accuracy_percent']}"
            f"\nAverage WER: {row['average_wer']}"
        )

    print("\nSummary saved:")
    print(summary_file)


# --------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------

def main():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_PATH}"
        )

    samples_by_language = collect_samples()

    if not samples_by_language:

        print("No samples were found.")
        return

    results = evaluate_samples(samples_by_language)

    save_detailed_results(results)

    generate_summary(results)

    print("\nMultilingual evaluation completed!")


if __name__ == "__main__":

    main()