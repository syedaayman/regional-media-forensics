import os
import tempfile
from faster_whisper import WhisperModel


class MultilingualSpeechToText:

    def __init__(self):
        print("Loading multilingual Whisper model...")

        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )

        print("Multilingual model loaded successfully!")

    def transcribe_audio(self, audio_bytes: bytes) -> dict:

        if not audio_bytes:
            raise ValueError("Audio file is empty.")

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
                language=None,
                beam_size=1,
                vad_filter=True
            )

            transcript = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )

            return {
                "transcript": transcript,
                "detected_language": info.language,
                "language_probability": round(
                    info.language_probability * 100, 2
                ),
                "status": "success" if transcript else "empty_transcript"
            }

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


multilingual_speech_to_text = MultilingualSpeechToText()