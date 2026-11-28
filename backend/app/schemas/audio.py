"""Public response schemas for audio routes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptionDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str
    detected_language: Literal["kn"]
    language_name: Literal["Kannada"]
    language_probability: float | None = Field(default=None, ge=0, le=100)
    status: Literal["success", "empty_transcript", "unsupported_language"]


class AudioTranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    transcription: TranscriptionDetails


class MultilingualTranscriptionDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str
    detected_language: Literal["bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"]
    language_name: Literal["Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Tamil", "Telugu"]
    language_probability: float | None = Field(default=None, ge=0, le=100)
    status: Literal["success", "empty_transcript", "unsupported_language"]


class MultilingualTranscriptionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    transcription: MultilingualTranscriptionDetails


class VoiceDetectionAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["Genuine Human Speech", "AI-Generated Speech"]
    label: Literal[0, 1]
    confidence: float = Field(ge=0, le=100)
    status: Literal["completed"]


class VoiceDetectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    analysis: VoiceDetectionAnalysis
