import os
import tempfile

from faster_whisper import WhisperModel


class SpeechToTextService:

    def __init__(self):
        print("Loading Kannada-specialized Whisper model...")

        self.model = WhisperModel(
            "mohankumars26/whisper-kannada-medium-ct2",
            device="cpu",
            compute_type="int8"
        )

        print("Kannada speech-to-text model loaded successfully!")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        language: str = "kn"
    ) -> dict:

        if not audio_bytes:
            raise ValueError("Audio file is empty.")

        if language != "kn":
            raise ValueError(
                "This model currently supports Kannada transcription only."
            )

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:

                temp_file.write(audio_bytes)
                temp_path = temp_file.name

            segments, info = self.model.transcribe(
                temp_path,
                language="kn",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )

            transcript_parts = []

            for segment in segments:
                text = segment.text.strip()

                if text:
                    transcript_parts.append(text)

            transcript = " ".join(transcript_parts)

            return {
                "transcript": transcript,
                "detected_language": "kn",
                "language_probability": None,
                "language_name": "Kannada",
                "requested_language": "kn",
                "language_detection": "Kannada forced",
                "status": (
                    "success" if transcript
                    else "empty_transcript"
                )
            }

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


speech_to_text_service = SpeechToTextService()