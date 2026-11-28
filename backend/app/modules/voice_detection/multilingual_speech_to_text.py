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
    text = unicodedata.normalize("NFC", text).replace("\ufffd", "")
    words = text.split()
    result_words = []
    previous = None
    run = 0
    for word in words:
        if word == previous:
            run += 1
            if run < 3:
                result_words.append(word)
        else:
            run = 0
            result_words.append(word)
        previous = word

    cleaned = " ".join(result_words)
    chars = []
    previous_char = None
    run = 0
    repeated = 0
    for char in cleaned:
        if char == previous_char:
            run += 1
            repeated += 1
            # Repeated sentence punctuation in short ASR output is a common
            # hallucination artifact. Keep the first mark; preserve the
            # existing bounded-run behavior for letters and numbers.
            if not unicodedata.category(char).startswith("P") and run < 4:
                chars.append(char)
        else:
            run = 0
            chars.append(char)
        previous_char = char
    cleaned = "".join(chars)
    ratio = repeated / max(1, len(cleaned) + repeated)
    return " ".join(cleaned.split()), round(ratio, 4)


def segment_is_repetitive(text: str) -> bool:
    normalized = unicodedata.normalize("NFC", text).strip()
    if not normalized:
        return False
    chars = normalized.replace(" ", "")
    if len(chars) >= 8 and len(set(chars)) <= 3:
        return True
    words = normalized.split()
    if len(words) >= 6:
        run = 1
        longest = 1
        for index in range(1, len(words)):
            if words[index] == words[index - 1]:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        if longest >= 4:
            return True
    return False


SUPPORTED_LANGUAGES = {
    "bn": "Bengali",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
}


class UnsupportedLanguageError(ValueError):
    pass


class MultilingualSpeechToText:

    def __init__(self):

        print("Loading multilingual Whisper Medium model...")

        self.model = WhisperModel(
            "medium",
            device="cpu",
            compute_type="int8"
        )

        print("Multilingual Whisper Medium model loaded successfully!")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        language: str | None = None
    ) -> dict:

        if not audio_bytes:
            raise ValueError("Audio file is empty.")

        selected_language = None
        if language:
            selected_language = language.strip().lower()
            if selected_language not in SUPPORTED_LANGUAGES:
                raise UnsupportedLanguageError(
                    f"Unsupported language code: {selected_language}"
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
                "language": selected_language,
                "task": "transcribe",
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0,
                "condition_on_previous_text": True,
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500),
                "no_speech_threshold": 0.6,
                "log_prob_threshold": -1.0,
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
                        chunks, chunk_info = self.model.transcribe(chunk_path, **kwargs)
                        if info is None:
                            info = chunk_info
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
            cleaned_segments = []
            for segment in segment_list:
                if not segment.text.strip() or segment_is_repetitive(segment.text):
                    continue
                cleaned, _ = clean_hallucinated_text(segment.text)
                if cleaned:
                    cleaned_segments.append({
                        "start": round(float(segment.start), 3),
                        "end": round(float(segment.end), 3),
                        "text": cleaned,
                    })
            cleaned_segments = merge_overlap_segments(cleaned_segments)

            transcript = " ".join(item["text"] for item in cleaned_segments)
            segments_output = cleaned_segments

            detected_language = info.language

            language_name = SUPPORTED_LANGUAGES.get(
                detected_language,
                "Unknown / Unsupported"
            )

            probability = round(
                info.language_probability * 100,
                2
            )

            if not transcript:

                status = "empty_transcript"

                warning = (
                    "No speech could be transcribed. "
                    "Check the audio quality."
                )

            elif detected_language not in SUPPORTED_LANGUAGES:

                status = "unsupported_language"

                warning = (
                    "The detected language is not supported."
                )

            else:

                status = "success"
                warning = None

            return {
                "transcript": transcript,
                "requested_language": selected_language,
                "detected_language": detected_language,
                "language_name": language_name,
                "language_probability": probability,
                "language_detection": (
                    "Model auto-detected" if selected_language is None
                    else "User-selected language"
                ),
                "status": status,
                "warning": warning
                ,
                "segments": segments_output,
                "model": "medium Whisper model"
            }

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


multilingual_speech_to_text: MultilingualSpeechToText | None = None


def get_multilingual_speech_to_text_service() -> MultilingualSpeechToText:
    global multilingual_speech_to_text
    if multilingual_speech_to_text is None:
        multilingual_speech_to_text = MultilingualSpeechToText()
    return multilingual_speech_to_text
