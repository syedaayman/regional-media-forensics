from app.modules.voice_detection.transcription_evaluation import (
    calculate_cer,
    calculate_wer,
    merge_overlap_segments,
)


def test_identical_transcripts() -> None:
    assert calculate_wer("hello world", "hello world") == 0.0
    assert calculate_cer("ನಮಸ್ಕಾರ", "ನಮಸ್ಕಾರ") == 0.0


def test_one_word_substitution() -> None:
    assert calculate_wer("hello world", "hello there") == 0.5


def test_empty_reference_behavior() -> None:
    assert calculate_wer("", "") == 0.0
    assert calculate_wer("", "unexpected") == 1.0
    assert calculate_cer("", "") == 0.0
    assert calculate_cer("", "unexpected") == 1.0


def test_overlap_segments_remove_duplicate_boundary_words() -> None:
    segments = [
        {"start": 115.0, "end": 120.0, "text": "please repeat this line"},
        {"start": 119.8, "end": 125.0, "text": "repeat this line and continue"},
    ]

    merged = merge_overlap_segments(segments)

    assert len(merged) == 1
    assert merged[0]["text"] == "please repeat this line and continue"
