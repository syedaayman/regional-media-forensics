from app.modules.voice_detection.multilingual_speech_to_text import (
    multilingual_speech_to_text
)

print("Multilingual speech-to-text test started.")

audio_path = input("Enter the full audio file path: ")

with open(audio_path, "rb") as audio_file:
    audio_bytes = audio_file.read()

print("\nTranscribing audio...")

result = multilingual_speech_to_text.transcribe_audio(audio_bytes)

print("\n---------- RESULTS ----------")
print("Detected language:", result["detected_language"])
print("Language probability:", result["language_probability"])
print("Transcript:", result["transcript"])
print("Status:", result["status"])
print("-----------------------------")