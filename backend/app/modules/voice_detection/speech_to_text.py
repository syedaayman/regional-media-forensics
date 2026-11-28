import os
import tempfile
from io import BytesIO

from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import unicodedata

from app.modules.voice_detection.transcription_evaluation import (
    merge_overlap_segments,
)


def clean_hallucinated_text(text: str) -> tuple[str, float]:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufffd", "")

    words = text.split()
    cleaned_words = []
    consecutive = 0
    previous = None
    for word in words:
        if word == previous:
            consecutive += 1
            if consecutive < 3:
                cleaned_words.append(word)
        else:
            consecutive = 0
            cleaned_words.append(word)
        previous = word

    cleaned = " ".join(cleaned_words)
    result_chars = []
    previous_char = None
    run = 0
    repeated_chars = 0
    for char in cleaned:
        if char == previous_char:
            run += 1
            repeated_chars += 1
            # Avoid forwarding repeated punctuation artifacts while retaining
            # the current bounded-run behavior for letters and numbers.
            if not unicodedata.category(char).startswith("P") and run < 4:
                result_chars.append(char)
        else:
            run = 0
            result_chars.append(char)
        previous_char = char
    cleaned = "".join(result_chars)

    ratio = repeated_chars / max(1, len(cleaned) + repeated_chars)
    return " ".join(cleaned.split()), round(ratio, 4)


def segment_is_repetitive(text: str) -> bool:
    normalized = unicodedata.normalize("NFC", text).strip()
    if not normalized:
        return False
    if len(normalized) >= 8:
        unique = len(set(normalized.replace(" ", "")))
        if unique <= 3:
            return True
    words = normalized.split()
    if len(words) >= 6:
        longest_run = 1
        run = 1
        for index in range(1, len(words)):
            if words[index] == words[index - 1]:
                run += 1
                longest_run = max(longest_run, run)
            else:
                run = 1
        if longest_run >= 4:
            return True
    return False


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
                "This model currently supports Kannada only."
            )

        temp_path = None

        try:
            audio, sample_rate = librosa.load(
                BytesIO(audio_bytes),
                sr=16000,
                mono=True,
            )
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:
                temp_path = temp_file.name
            sf.write(temp_path, audio, sample_rate, format="WAV")
            kwargs = {
                "language": "kn",
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0,
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500),
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.6,
                "log_prob_threshold": -1.0,
                "repetition_penalty": 1.1,
            }

            duration_seconds = len(audio) / sample_rate
            segment_list = []
            info = None
            if duration_seconds <= 120.0:
                segments, info = self.model.transcribe(temp_path, **kwargs)
                segment_list.extend(list(segments))
            else:
                chunk_length = int(120.0 * sample_rate)
                overlap = int(5.0 * sample_rate)
                chunk_paths = []
                try:
                    for chunk_start in range(0, len(audio), chunk_length - overlap):
                        chunk = audio[chunk_start:chunk_start + chunk_length]
                        if len(chunk) < int(0.5 * sample_rate):
                            break
                        with tempfile.NamedTemporaryFile(
                            suffix=".wav",
                            delete=False
                        ) as chunk_file:
                            chunk_path = chunk_file.name
                        chunk_paths.append(chunk_path)
                        sf.write(chunk_path, chunk, sample_rate, format="WAV")
                        chunks, _ = self.model.transcribe(chunk_path, **kwargs)
                        offset = chunk_start / sample_rate
                        for segment in chunks:
                            segment.start = float(segment.start) + offset
                            segment.end = float(segment.end) + offset
                            segment_list.append(segment)
                        if chunk_start + chunk_length >= len(audio):
                            break
                finally:
                    for chunk_path in chunk_paths:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
            raw_segments = [
                segment for segment in segment_list
                if segment.text.strip()
                and not segment_is_repetitive(segment.text)
            ]
            cleaned_segments = []
            hallucination_ratio = 0.0
            for segment in raw_segments:
                cleaned, ratio = clean_hallucinated_text(segment.text)
                hallucination_ratio = max(hallucination_ratio, ratio)
                if cleaned:
                    cleaned_segments.append({
                        "start": round(float(segment.start), 3),
                        "end": round(float(segment.end), 3),
                        "text": cleaned,
                    })
            cleaned_segments = merge_overlap_segments(cleaned_segments)
            transcript = " ".join(item["text"] for item in cleaned_segments)
            segments_output = cleaned_segments

            return {
                "transcript": transcript,
                "detected_language": "kn",
                "language_probability": None,
                "language_name": "Kannada",
                "requested_language": "kn",
                "language_detection": "Kannada forced",
                "warning": None,
                "status": (
                    "success" if transcript
                    else "empty_transcript"
                ),
                "segments": segments_output,
                "model": "mohankumars26/whisper-kannada-medium-ct2",
            }

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


speech_to_text_service: SpeechToTextService | None = None


def get_speech_to_text_service() -> SpeechToTextService:
    global speech_to_text_service
    if speech_to_text_service is None:
        speech_to_text_service = SpeechToTextService()
    return speech_to_text_service
