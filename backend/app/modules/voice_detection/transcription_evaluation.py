import unicodedata
from typing import Iterable


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.lower())
    characters = []
    for char in normalized:
        category = unicodedata.category(char)[0]
        if category in ("L", "N") or char.isspace():
            characters.append(char)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_char in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_char in enumerate(hypothesis, start=1):
            substitution = 0 if reference_char == hypothesis_char else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution,
                )
            )
        previous = current
    return previous[-1]


def calculate_wer(reference: str, hypothesis: str) -> float:
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = 0 if reference_word == hypothesis_word else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution,
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    reference_text = normalize_text(reference)
    hypothesis_text = normalize_text(hypothesis)
    if not reference_text:
        return 0.0 if not hypothesis_text else 1.0
    return edit_distance(reference_text, hypothesis_text) / len(reference_text)


def merge_overlap_segments(segments: Iterable[dict]) -> list[dict]:
    """Merge adjacent ASR chunks and remove duplicated boundary words."""
    ordered = sorted(
        (
            dict(segment)
            for segment in segments
            if isinstance(segment, dict)
            and segment.get("text", "").strip()
        ),
        key=lambda segment: (
            float(segment.get("start", 0.0)),
            float(segment.get("end", 0.0)),
        ),
    )
    merged: list[dict] = []
    for segment in ordered:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        overlaps_in_time = (
            float(segment.get("start", 0.0))
            <= float(previous.get("end", 0.0)) + 0.25
        )
        if not overlaps_in_time:
            merged.append(segment)
            continue

        previous_words = previous["text"].split()
        next_words = segment["text"].split()
        duplicate_count = 0
        for count in range(
            1,
            min(8, len(previous_words), len(next_words)) + 1,
        ):
            if previous_words[-count:] == next_words[:count]:
                duplicate_count = count

        if duplicate_count:
            remaining_words = next_words[duplicate_count:]
            previous["end"] = max(
                float(previous["end"]),
                float(segment["end"]),
            )
            if remaining_words:
                previous["text"] = " ".join(
                    previous_words + remaining_words
                )
        else:
            merged.append(segment)
    return merged
