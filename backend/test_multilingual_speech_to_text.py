from pathlib import Path

from app.modules.voice_detection.multilingual_speech_to_text import (
    get_multilingual_speech_to_text_service,
)


def main():

    print("Multilingual Kannada transcription test started.")

    audio_path = Path(
        r"C:\Users\Mypc\Downloads\kannada2.wav"
    )

    if not audio_path.exists():
        print("Audio file not found:", audio_path)
        return

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    print("Transcribing Kannada audio...")

    result = get_multilingual_speech_to_text_service().transcribe_audio(
        audio_bytes,
        language="kn"
    )

    print("\n---------- RESULTS ----------")
    print("Detected language:", result["detected_language"])
    print("Language probability:", result["language_probability"])
    print("Transcript:", result["transcript"])
    print("Status:", result["status"])
    print("-----------------------------")


if __name__ == "__main__":
    main()
