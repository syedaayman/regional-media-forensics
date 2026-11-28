from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from starlette.concurrency import run_in_threadpool
import logging
from pathlib import PurePath
from typing import Any
from fastapi.responses import FileResponse, Response
import librosa
import numpy as np
from pathlib import Path

from app.modules.audio_forensics import (
    EVIDENCE_ROOT,
    register_evidence,
    create_working_copy,
    analyze_quality_and_tampering,
    verify_integrity,
    store_analysis,
    get_analysis,
    get_evidence_for_analysis,
    get_evidence_annotation,
)
from app.modules.forensic_report import add_evidence_annotation, generate_pdf
from app.modules.forensic_report import REPORT_ROOT

from app.modules.voice_detection.speech_to_text import (
    get_speech_to_text_service,
)
from app.modules.voice_detection.multilingual_speech_to_text import (
    get_multilingual_speech_to_text_service,
    UnsupportedLanguageError,
    SUPPORTED_LANGUAGES,
)
from app.modules.voice_detection.voice_detection import (
    get_voice_detection_service,
)
from app.modules.voice_detection.audio_validation import validate_audio_upload
from app.schemas.audio import (
    AudioTranscriptionResponse,
    MultilingualTranscriptionResponse,
    VoiceDetectionResponse,
)
from app.modules.voice_detection.audio_validation import MAX_AUDIO_UPLOAD_BYTES


router = APIRouter(
    prefix="/audio",
    tags=["Audio Forensics"]
)

logger = logging.getLogger("audio_backend")


def safe_upload_filename(filename: str | None) -> str:
    name = (filename or "audio-upload").replace("\\", "/")
    name = PurePath(name).name
    return name[:255] or "audio-upload"


def _transcription_warning(language: str, transcript: str, warning: str | None) -> str | None:
    if warning:
        return warning
    if not transcript.strip():
        return "No speech could be transcribed. Check the audio quality."

    unicode_ranges = {
        "bn": (0x0980, 0x09FF),
        "gu": (0x0A80, 0x0AFF),
        "hi": (0x0900, 0x097F),
        "kn": (0x0C80, 0x0CFF),
        "ml": (0x0D00, 0x0D7F),
        "mr": (0x0900, 0x097F),
        "ta": (0x0B80, 0x0BFF),
        "te": (0x0C00, 0x0C7F),
    }
    if language in unicode_ranges:
        letters = [char for char in transcript if char.isalpha()]
        if letters:
            start, end = unicode_ranges[language]
            native_ratio = sum(start <= ord(char) <= end for char in letters) / len(letters)
            if native_ratio < 0.75:
                return "Transcription may contain wrong-script or incorrect language output."

    words = transcript.split()
    if len(words) >= 6:
        longest_run = 1
        current_run = 1
        for previous, current in zip(words, words[1:]):
            current_run = current_run + 1 if current == previous else 1
            longest_run = max(longest_run, current_run)
        if longest_run >= 4:
            return "Transcription contains repeated text and may be unreliable."
    return None


async def read_bounded_upload(file: UploadFile) -> bytes:
    chunks = []
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "audio_too_large",
                    "message": "Audio file exceeds the upload limit.",
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/detect", response_model=VoiceDetectionResponse)
async def detect_audio(
    file: UploadFile = File(...)
):
    """Detect whether uploaded audio is genuine or AI-generated."""

    try:
        audio_bytes = await read_bounded_upload(file)
        validate_audio_upload(
            audio_bytes,
            file.filename,
            file.content_type,
        )

        service = get_voice_detection_service()
        result = await run_in_threadpool(
            service.analyze_audio,
            audio_bytes,
        )
        if result.get("status") == "insufficient_evidence":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "insufficient_voice_evidence",
                    "message": "The recording is too short, silent, or unusable for voice detection.",
                },
            )
        return {
            "filename": safe_upload_filename(file.filename),
            "analysis": {
                "prediction": result["prediction"],
                "label": result["label"],
                "confidence": result["confidence"],
                "status": "completed",
            },
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_audio", "message": str(error)}
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Voice detection failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "audio_detection_failed",
                "message": "Audio detection could not be completed.",
            }
        )

    finally:
        await file.close()


@router.post("/transcribe/kannada", response_model=AudioTranscriptionResponse)
async def transcribe_kannada(
    file: UploadFile = File(...)
):
    """Transcribe Kannada audio."""

    try:
        audio_bytes = await read_bounded_upload(file)
        validate_audio_upload(
            audio_bytes,
            file.filename,
            file.content_type,
        )

        service = get_speech_to_text_service()
        result = await run_in_threadpool(
            service.transcribe_audio,
            audio_bytes,
            "kn",
        )

        return {
            "filename": safe_upload_filename(file.filename),
            "transcription": {
                "transcript": result["transcript"],
                "detected_language": "kn",
                "language_name": "Kannada",
                "language_probability": result.get("language_probability"),
                "status": result["status"],
            },
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_audio", "message": str(error)}
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Kannada transcription failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "kannada_transcription_failed",
                "message": "Kannada transcription could not be completed.",
            }
        )

    finally:
        await file.close()


@router.post(
    "/transcribe/multilingual",
    response_model=MultilingualTranscriptionResponse,
)
async def transcribe_multilingual(
    file: UploadFile = File(...),
    language: str | None = Query(
        default=None,
        description=(
            "Optional language code; omit for automatic detection: "
            "bn, en, gu, hi, kn, ml, mr, ta, te"
        )
    )
):
    """Transcribe audio using the selected language."""

    try:
        audio_bytes = await read_bounded_upload(file)
        validate_audio_upload(
            audio_bytes,
            file.filename,
            file.content_type,
        )

        service = get_multilingual_speech_to_text_service()
        result = await run_in_threadpool(
            service.transcribe_audio,
            audio_bytes,
            language,
        )

        selected_language = language.strip().lower() if language else None
        if (
            selected_language is None
            and result.get("detected_language") not in SUPPORTED_LANGUAGES
        ):
            detected_language = result.get("detected_language") or "unknown"
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unsupported_language",
                    "message": (
                        "Detected language is not supported: "
                        f"{detected_language}"
                    ),
                },
            )
        public_language = selected_language or result["detected_language"]
        return {
            "filename": safe_upload_filename(file.filename),
            "transcription": {
                "transcript": result["transcript"],
                "detected_language": public_language,
                "language_name": (
                    "Kannada" if public_language == "kn" else
                    "English" if public_language == "en" else
                    "Hindi" if public_language == "hi" else
                    "Telugu" if public_language == "te" else
                    "Tamil" if public_language == "ta" else
                    "Malayalam" if public_language == "ml" else
                    "Marathi" if public_language == "mr" else
                    "Bengali" if public_language == "bn" else
                    "Gujarati"
                ),
                "language_probability": (
                    None if selected_language is not None
                    else result.get("language_probability")
                ),
                "status": result["status"],
            },
        }

    except UnsupportedLanguageError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "unsupported_language", "message": str(error)}
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_audio", "message": str(error)}
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Multilingual transcription failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "multilingual_transcription_failed",
                "message": "Multilingual transcription could not be completed.",
            }
        )

    finally:
        await file.close()


def _aggregate_assessment(
    voice: dict[str, Any],
    tampering: dict[str, Any],
    quality: dict[str, Any],
    transcription: dict[str, Any],
) -> dict[str, Any]:
    indicators = []
    if voice["assessment"] == "potential_ai_generated":
        indicators.append("voice_classifier_indicator")
    if tampering["findings"]:
        indicators.append("tampering_indicators")
    if quality["clipping_percentage"] > 1.0:
        indicators.append("clipping")

    if len(indicators) >= 2:
        classification = "multiple_suspicious_indicators_detected"
    elif voice["assessment"] == "potential_ai_generated":
        classification = "potential_ai_generated_voice_indicators"
    elif tampering["findings"]:
        classification = "potential_audio_manipulation_indicators"
    elif transcription["status"] in ("failed", "empty_transcript"):
        classification = "insufficient_evidence"
    else:
        classification = "inconclusive"

    summary = {
        "inconclusive": "No definitive conclusion was established; measurements and model output are insufficient or unvalidated.",
        "insufficient_evidence": "Available analysis is incomplete or insufficient for a reliable forensic conclusion.",
        "potential_ai_generated_voice_indicators": "The classifier produced a synthetic-speech-associated indicator, but its real-world reliability is not verified.",
        "potential_audio_manipulation_indicators": "One or more measured signal characteristics are possible manipulation indicators.",
        "multiple_suspicious_indicators_detected": "Multiple independent signal indicators were detected; this is not proof of manipulation.",
    }[classification]

    return {
        "classification": classification,
        "summary": summary,
        "limitations": [
            "No suspicious-indicator finding proves that audio is fake.",
            "The voice classifier is not independently validated for this recording.",
            "Transcription accuracy depends on model and recording quality.",
        ],
    }


def _write_visualizations(analysis_id: str, working_path: Path) -> dict[str, Any]:
    evidence_dir = working_path.parent
    audio, sr = librosa.load(str(working_path), sr=16000, mono=True)

    waveform_path = evidence_dir / "waveform.svg"
    width, height = 800, 180
    points = 800
    sampled = np.interp(
        np.linspace(0, len(audio) - 1, points),
        np.arange(len(audio)),
        audio,
    )
    coords = []
    for i, value in enumerate(sampled):
        x = i
        y = height / 2 - float(value) * height * 0.42
        coords.append(f"{x},{y:.2f}")
    waveform_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" stroke="#999"/>'
        f'<polyline fill="none" stroke="#1f2937" stroke-width="1" points="{" ".join(coords)}"/>'
        f'</svg>'
    )
    waveform_path.write_text(waveform_svg, encoding="utf-8")

    spectrogram_path = evidence_dir / "spectrogram.svg"
    stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    display = db[:, :: max(1, db.shape[1] // 200)]
    display = np.nan_to_num(display)
    display = np.clip((display + 80) / 80, 0, 1)
    bins = display.shape[0]
    rects = []
    box_width = 800 / display.shape[1]
    box_height = 300 / bins
    for x in range(display.shape[1]):
        for y in range(0, bins, 8):
            value = float(np.mean(display[y:min(y + 8, bins), x]))
            color = int(value * 255)
            rects.append(
                f'<rect x="{x*box_width:.2f}" y="{300-(y+8)*box_height:.2f}" '
                f'width="{box_width:.2f}" height="{box_height*8:.2f}" fill="rgb({color},{color},{color})"/>'
            )
    spectrogram_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="300">'
        '<rect width="100%" height="100%" fill="white"/>'
        + "".join(rects)
        + "</svg>"
    )
    spectrogram_path.write_text(spectrogram_svg, encoding="utf-8")

    return {
        "waveform": {
            "status": "completed",
            "download_endpoint": f"/audio/visualizations/{analysis_id}/waveform",
        },
        "spectrogram": {
            "status": "completed",
            "download_endpoint": f"/audio/visualizations/{analysis_id}/spectrogram",
        },
    }


@router.post("/analyze", status_code=202)
async def analyze_audio(file: UploadFile = File(...)):
    """Register and run a complete real audio forensic analysis."""
    try:
        audio_bytes = await read_bounded_upload(file)
        validate_audio_upload(audio_bytes, file.filename, file.content_type)

        evidence = register_evidence(
            safe_upload_filename(file.filename),
            audio_bytes,
            file.content_type,
        )
        with_lock = evidence["evidence_id"]
        evidence_dir = EVIDENCE_ROOT / with_lock
        original_path = evidence_dir / f"original{Path(safe_upload_filename(file.filename)).suffix.lower()}"
        working_path = create_working_copy(with_lock, original_path)
        quality, tampering, _ = analyze_quality_and_tampering(working_path)
        visualizations = _write_visualizations(evidence["analysis_id"], working_path)

        transcription = {
            "status": "failed",
            "text": "",
            "segments": [],
            "model": "Unavailable",
        }
        language_analysis = {
            "detected_language": None,
            "confidence": None,
            "supported": False,
        }
        try:
            service = get_multilingual_speech_to_text_service()
            transcription_result = await run_in_threadpool(
                service.transcribe_audio,
                audio_bytes,
                None,
            )
            transcription = {
                "status": "completed" if transcription_result["status"] == "success" else transcription_result["status"],
                "text": transcription_result["transcript"],
                "segments": transcription_result["segments"],
                "model": transcription_result["model"],
            }
            detected = transcription_result["detected_language"]
            language_analysis = {
                "detected_language": detected,
                "confidence": transcription_result["language_probability"],
                "supported": detected in service.SUPPORTED_LANGUAGES if hasattr(service, "SUPPORTED_LANGUAGES") else detected is not None,
            }
        except Exception as error:
            logger.exception("Complete-analysis transcription failed")
            transcription["status"] = "failed"
            transcription["text"] = ""

        voice_authenticity = {
            "status": "completed",
            "assessment": "inconclusive",
            "model": "RandomForest voice_detection_model.joblib",
            "score": None,
            "threshold": None,
            "indicators": [],
            "limitations": [
                "The existing classifier has not been independently validated.",
                "Model confidence is not a calibrated probability of voice authenticity.",
            ],
        }
        try:
            voice_service = get_voice_detection_service()
            voice_result = await run_in_threadpool(voice_service.analyze_audio, audio_bytes)
            voice_authenticity["status"] = voice_result.get("status", "completed")
            voice_authenticity["assessment"] = voice_result.get("assessment", "Inconclusive")
            voice_authenticity["score"] = voice_result.get("model_score")
            voice_authenticity["threshold"] = voice_result.get("threshold")
            voice_authenticity["indicators"] = voice_result.get("indicators", [])
            voice_authenticity["limitations"] = voice_result.get("limitations", voice_authenticity["limitations"])
        except Exception as error:
            logger.exception("Complete-analysis voice detection failed")
            voice_authenticity["status"] = "failed"
            voice_authenticity["limitations"].append("The voice detector could not complete.")

        overall = _aggregate_assessment(
            voice_authenticity,
            tampering,
            quality,
            transcription,
        )
        result = {
            "analysis_id": evidence["analysis_id"],
            "status": "completed",
            "completed_at": evidence["upload_timestamp"],
            "evidence": evidence,
            "language_analysis": language_analysis,
            "transcription": transcription,
            "voice_authenticity": voice_authenticity,
            "tampering_analysis": tampering,
            "audio_quality": quality,
            "visualizations": visualizations,
            "overall_assessment": overall,
            "report": {
                "status": "completed",
                "download_endpoint": f"/audio/reports/{evidence['analysis_id']}",
            },
        }
        store_analysis(result)
        generate_pdf(result)
        return result
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail={"code": "invalid_audio", "message": str(error)})
    except Exception as error:
        logger.exception("Complete audio analysis failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "complete_analysis_failed",
                "message": "Complete audio analysis could not be completed.",
            },
        )
    finally:
        await file.close()


@router.get("/analyses/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    result = get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": "analysis_not_found", "message": "Analysis was not found."})
    return {"analysis_id": analysis_id, "status": result["status"]}


@router.get("/analyses/{analysis_id}")
async def retrieve_analysis(analysis_id: str):
    result = get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": "analysis_not_found", "message": "Analysis was not found."})
    return result


@router.get("/evidence/{evidence_id}/integrity")
async def evidence_integrity(evidence_id: str):
    try:
        return verify_integrity(evidence_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"code": "evidence_not_found", "message": "Evidence was not found."})


@router.get("/visualizations/{analysis_id}/{kind}")
async def get_visualization(analysis_id: str, kind: str):
    result = get_analysis(analysis_id)
    if result is not None:
        evidence_id = result["evidence"]["evidence_id"]
    else:
        evidence_link = get_evidence_for_analysis(analysis_id)
        if evidence_link is None:
            raise HTTPException(status_code=404, detail={"code": "analysis_not_found", "message": "Analysis was not found."})
        evidence_id = evidence_link["evidence_id"]
    if kind not in {"waveform", "spectrogram"}:
        raise HTTPException(status_code=404, detail={"code": "visualization_not_found", "message": "Visualization was not found."})
    evidence_dir = EVIDENCE_ROOT / evidence_id
    path = evidence_dir / f"{kind}.svg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "visualization_not_found", "message": "Visualization was not found."})
    return FileResponse(path, media_type="image/svg+xml", filename=f"{kind}.svg")


@router.get("/reports/{analysis_id}")
async def get_report(analysis_id: str):
    result = get_analysis(analysis_id)
    if result is not None:
        evidence_id = result.get("evidence", {}).get("evidence_id")
    else:
        evidence_link = get_evidence_for_analysis(analysis_id)
        if evidence_link is None:
            raise HTTPException(status_code=404, detail={"code": "analysis_not_found", "message": "Analysis was not found."})
        evidence_id = evidence_link["evidence_id"]
    report_path = REPORT_ROOT / f"{analysis_id}.pdf"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail={"code": "report_not_found", "message": "Report was not found."})
    annotation = get_evidence_annotation(evidence_id) if evidence_id else None
    headers = {}
    if annotation is not None:
        headers["X-Evidence-Classification"] = annotation["classification"]
        headers["X-Evidence-Notice"] = annotation["note"]
        annotated_pdf = add_evidence_annotation(
            report_path.read_bytes(),
            annotation["classification"],
            annotation["note"],
        )
        headers["Content-Disposition"] = (
            f'attachment; filename="audio-forensic-report-{analysis_id}.pdf"'
        )
        return Response(
            content=annotated_pdf,
            media_type="application/pdf",
            headers=headers,
        )
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"audio-forensic-report-{analysis_id}.pdf",
        headers=headers,
    )
