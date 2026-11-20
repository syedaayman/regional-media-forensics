from app.modules.voice_detection.speech_to_text import (
    speech_to_text_service
)

print("Kannada transcription test started.")

audio_path = input("Enter the full audio file path: ")

with open(audio_path, "rb") as audio_file:
    audio_bytes = audio_file.read()

print("\nTranscribing Kannada audio...")

result = speech_to_text_service.transcribe_audio(
    audio_bytes,
    language="kn"
)

print("\n---------- RESULTS ----------")

print("Detected language:", result["detected_language"])
print("Language name:", result["language_name"])
print("Transcript:", result["transcript"])
print("Status:", result["status"])

print("-----------------------------")