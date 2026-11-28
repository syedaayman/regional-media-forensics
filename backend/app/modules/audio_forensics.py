import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


BACKEND_PATH = Path(__file__).resolve().parents[3]
STORAGE_ROOT = BACKEND_PATH / "storage"
EVIDENCE_ROOT = STORAGE_ROOT / "audio_evidence"
EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = STORAGE_ROOT / "audio_forensics.db"
KNOWN_TEST_EVIDENCE_ANNOTATIONS = {
    "689b96d4-ca14-4627-a03d-92703672f57c": {
        "analysis_id": "dc05c4de-9de4-4434-adbe-bcd112a7a875",
        "sha256": "2976da01e205a110c9fa41d47659e238a5c6d3c3f3137582f2949853faa201dd",
        "classification": "known_test_fixture",
        "note": (
            "Synthetic 0.1-second silent API regression fixture. Preserve the "
            "record, but do not treat it as a genuine forensic submission."
        ),
    }
}
_migration_status: dict[str, Any] = {
    "database_path": str(DATABASE_PATH),
    "existing_evidence_seeded": 0,
    "existing_evidence_preserved": 0,
    "existing_evidence_skipped": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _database_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    return connection


def _initialize_database() -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    with _database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL UNIQUE,
                original_filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                upload_timestamp TEXT NOT NULL,
                original_path TEXT NOT NULL,
                format TEXT NOT NULL,
                subtype TEXT,
                duration_seconds REAL,
                sample_rate INTEGER,
                channels INTEGER,
                frames INTEGER,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL,
                status TEXT NOT NULL,
                completed_at TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_evidence_id
                ON analyses(evidence_id);

            CREATE TABLE IF NOT EXISTS evidence_annotations (
                annotation_id TEXT PRIMARY KEY,
                evidence_id TEXT NOT NULL UNIQUE,
                classification TEXT NOT NULL,
                note TEXT NOT NULL,
                annotated_at TEXT NOT NULL,
                FOREIGN KEY (evidence_id) REFERENCES evidence(evidence_id)
            );

            """
        )

        for evidence_dir in sorted(
            (path for path in EVIDENCE_ROOT.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            integrity_path = evidence_dir / "integrity.json"
            if not integrity_path.is_file():
                _migration_status["existing_evidence_skipped"] += 1
                continue
            try:
                info = json.loads(integrity_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                _migration_status["existing_evidence_skipped"] += 1
                continue

            evidence_id = info.get("evidence_id")
            analysis_id = info.get("analysis_id")
            original_path = Path(info.get("original_path", ""))
            sha256 = info.get("sha256")
            if not evidence_id or not analysis_id or not sha256 or not original_path.is_file():
                _migration_status["existing_evidence_skipped"] += 1
                continue

            existing = connection.execute(
                "SELECT evidence_id, analysis_id, original_filename, sha256, "
                "original_path FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            if existing is not None:
                comparable = (
                    existing["analysis_id"],
                    existing["original_filename"],
                    existing["sha256"],
                    existing["original_path"],
                )
                incoming = (
                    analysis_id,
                    info.get("original_filename"),
                    sha256,
                    str(original_path),
                )
                if comparable != incoming:
                    raise RuntimeError(
                        "Existing SQLite evidence differs from integrity.json."
                    )
                _migration_status["existing_evidence_preserved"] += 1
                continue

            connection.execute(
                """
                INSERT INTO evidence (
                    evidence_id, analysis_id, original_filename, sha256,
                    upload_timestamp, original_path, format, subtype,
                    duration_seconds, sample_rate, channels, frames,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    analysis_id,
                    info.get("original_filename", evidence_dir.name),
                    sha256,
                    info.get("upload_timestamp", utc_now()),
                    str(original_path),
                    info.get("format", ""),
                    info.get("subtype"),
                    info.get("duration_seconds"),
                    info.get("sample_rate"),
                    info.get("channels"),
                    info.get("frames"),
                    json.dumps(info, ensure_ascii=False),
                    utc_now(),
                ),
            )
            _migration_status["existing_evidence_seeded"] += 1

        for evidence_id, annotation in KNOWN_TEST_EVIDENCE_ANNOTATIONS.items():
            evidence_row = connection.execute(
                "SELECT analysis_id, sha256 FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            if (
                evidence_row is None
                or evidence_row["analysis_id"] != annotation["analysis_id"]
                or evidence_row["sha256"] != annotation["sha256"]
            ):
                continue
            # Earlier unisolated tests registered this same fixture more than
            # once with different UUIDs. Match the immutable content hash, then
            # independently re-hash each referenced original before annotating.
            matching_records = connection.execute(
                "SELECT evidence_id, original_path FROM evidence WHERE sha256 = ?",
                (annotation["sha256"],),
            ).fetchall()
            for record in matching_records:
                original_path = Path(record["original_path"])
                if not original_path.is_file():
                    continue
                digest = hashlib.sha256()
                with original_path.open("rb") as original_file:
                    for chunk in iter(lambda: original_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != annotation["sha256"]:
                    continue
                connection.execute(
                    """
                    INSERT OR IGNORE INTO evidence_annotations (
                        annotation_id, evidence_id, classification, note, annotated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid5(uuid.NAMESPACE_URL, record["evidence_id"])),
                        record["evidence_id"],
                        annotation["classification"],
                        annotation["note"],
                        utc_now(),
                    ),
                )


_initialize_database()


def register_evidence(filename: str, audio_bytes: bytes, content_type: str | None) -> dict[str, Any]:
    evidence_id = str(uuid.uuid4())
    analysis_id = str(uuid.uuid4())
    evidence_dir = EVIDENCE_ROOT / evidence_id
    evidence_dir.mkdir(parents=True, exist_ok=False)

    safe_suffix = Path(filename or "recording").suffix.lower() or ".wav"
    original_path = evidence_dir / f"original{safe_suffix}"
    original_path.write_bytes(audio_bytes)

    sha256 = hashlib.sha256(audio_bytes).hexdigest()
    metadata = extract_metadata(original_path)
    upload_timestamp = utc_now()
    integrity = {
        "evidence_id": evidence_id,
        "analysis_id": analysis_id,
        "original_filename": filename,
        "sha256": sha256,
        "upload_timestamp": upload_timestamp,
        "original_path": str(original_path),
        **metadata,
    }
    evidence_dir.joinpath("integrity.json").write_text(
        json.dumps(integrity, indent=2),
        encoding="utf-8",
    )

    with _database_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO evidence (
                    evidence_id, analysis_id, original_filename, sha256,
                    upload_timestamp, original_path, format, subtype,
                    duration_seconds, sample_rate, channels, frames,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    analysis_id,
                    filename,
                    sha256,
                    upload_timestamp,
                    str(original_path),
                    metadata["format"],
                    metadata["subtype"],
                    metadata["duration_seconds"],
                    metadata["sample_rate"],
                    metadata["channels"],
                    metadata["frames"],
                    json.dumps(integrity, ensure_ascii=False),
                    upload_timestamp,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise RuntimeError("Evidence could not be stored safely.") from error

    evidence = {
        "evidence_id": evidence_id,
        "analysis_id": analysis_id,
        "filename": filename,
        "sha256": sha256,
        "upload_timestamp": upload_timestamp,
        "format": metadata["format"],
        "file_size_bytes": len(audio_bytes),
        "duration_seconds": metadata["duration_seconds"],
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
    }
    return evidence


def extract_metadata(path: Path) -> dict[str, Any]:
    try:
        with sf.SoundFile(str(path)) as sound:
            sample_rate = int(sound.samplerate)
            channels = int(sound.channels)
            frames = int(sound.frames)
            format_name = sound.format
            subtype = sound.subtype
        duration = frames / sample_rate if sample_rate else 0.0
    except Exception as error:
        raise ValueError("The uploaded audio could not be decoded.") from error
    return {
        "format": format_name,
        "subtype": subtype,
        "duration_seconds": round(float(duration), 6),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
    }


def load_original(evidence_id: str) -> tuple[dict[str, Any], Path]:
    with _database_connection() as connection:
        row = connection.execute(
            "SELECT metadata_json FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
    if row is None:
        raise KeyError("Evidence not found")
    try:
        info = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise KeyError("Evidence metadata is malformed") from error
    original_path_value = info.get("original_path")
    if not original_path_value:
        raise KeyError("Evidence original path is missing")
    return info, Path(original_path_value)


def verify_integrity(evidence_id: str) -> dict[str, Any]:
    info, original_path = load_original(evidence_id)
    if not original_path.is_file():
        raise KeyError("Original evidence file is missing")
    calculated = hashlib.sha256(original_path.read_bytes()).hexdigest()
    result = {
        "evidence_id": evidence_id,
        "original_filename": info["original_filename"],
        "expected_sha256": info["sha256"],
        "calculated_sha256": calculated,
        "verified": calculated == info["sha256"],
        "checked_at": utc_now(),
    }
    annotation = get_evidence_annotation(evidence_id)
    if annotation is not None:
        result["record_disposition"] = annotation["classification"]
        result["disposition_note"] = annotation["note"]
    return result


def get_evidence_annotation(evidence_id: str) -> dict[str, str] | None:
    with _database_connection() as connection:
        row = connection.execute(
            "SELECT classification, note, annotated_at "
            "FROM evidence_annotations WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "classification": row["classification"],
        "note": row["note"],
        "annotated_at": row["annotated_at"],
    }


def create_working_copy(evidence_id: str, original_path: Path) -> Path:
    working_dir = EVIDENCE_ROOT / evidence_id
    working_path = working_dir / "working_16k_mono.wav"
    audio, sr = librosa.load(str(original_path), sr=16000, mono=True)
    sf.write(str(working_path), audio, 16000, format="WAV", subtype="PCM_16")
    return working_path


def _find_sudden_energy_changes(
    rms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    rms_diff = np.abs(np.diff(rms))
    if not len(rms_diff):
        return rms_diff, np.array([], dtype=int), 0.15

    median_diff = float(np.median(rms_diff))
    median_absolute_deviation = float(
        np.median(np.abs(rms_diff - median_diff))
    )
    threshold = max(median_diff + 6.0 * median_absolute_deviation, 0.15)
    return rms_diff, np.where(rms_diff > threshold)[0], threshold


def _find_robust_frame_discontinuities(
    values: np.ndarray,
    minimum_threshold: float,
    mad_scale: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    differences = np.abs(np.diff(values))
    if not len(differences):
        return differences, np.array([], dtype=int), minimum_threshold
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    threshold = max(median + mad_scale * mad, minimum_threshold)
    return differences, np.where(differences > threshold)[0], threshold


def analyze_quality_and_tampering(working_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audio, sr = librosa.load(str(working_path), sr=16000, mono=True)
    if len(audio) == 0:
        raise ValueError("Audio has no decoded samples")

    finite = np.isfinite(audio)
    if not finite.all():
        raise ValueError("Audio contains non-finite samples")

    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=1024, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=1024, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    frame_duration = 512 / sr
    silence = float(np.mean(rms < max(1e-5, float(np.percentile(rms, 10)))))
    peak = float(np.max(np.abs(audio)))
    clipping = float(np.mean(np.abs(audio) >= 0.999))
    mean_rms = float(np.mean(rms))
    noise_floor = float(np.percentile(rms, 10))
    signal_level = float(np.percentile(rms, 90))
    snr_db = float(20 * np.log10((signal_level + 1e-12) / (noise_floor + 1e-12)))
    snr_db = max(-100.0, min(100.0, snr_db))

    quality = {
        "status": "completed",
        "duration_seconds": round(float(len(audio) / sr), 6),
        "sample_rate": sr,
        "channels": 1,
        "snr_db": round(snr_db, 3),
        "snr_is_estimated": True,
        "clipping_percentage": round(clipping * 100, 3),
        "silence_percentage": round(silence * 100, 3),
        "rms_energy": round(mean_rms, 6),
        "zero_crossing_rate": round(float(np.mean(zcr)), 6),
        "spectral_centroid_hz": round(float(np.mean(centroid)), 3),
        "peak_amplitude": round(peak, 6),
        "method_limitations": [
            "SNR is a frame-based estimate, not a calibrated microphone measurement.",
            "The absence of a tampering indicator is not proof of authenticity."
        ],
    }

    findings = []
    if clipping > 0.001:
        locations = np.where(np.abs(audio) >= 0.999)[0]
        findings.append(make_finding(
            "clipping",
            "medium",
            float(locations[0] / sr),
            float(locations[-1] / sr),
            f"{clipping * 100:.3f}% of samples reached or exceeded 0.999",
            "Peak threshold analysis",
        ))

    if len(rms) > 2:
        rms_diff, indices, threshold = _find_sudden_energy_changes(rms)
        for index in indices[:20]:
            start = float(times[index])
            end = float(min(times[min(index + 1, len(times) - 1)], len(audio) / sr))
            findings.append(make_finding(
                "sudden_energy_change",
                "medium" if rms_diff[index] > threshold * 1.5 else "low",
                start,
                end,
                f"RMS change {rms_diff[index]:.5f} exceeded threshold {threshold:.5f}",
                "Adjacent short-frame RMS discontinuity",
            ))

    if silence > 0.25:
        findings.append(make_finding(
            "long_silence_region",
            "low",
            0.0,
            float(len(audio) / sr),
            f"{silence * 100:.2f}% of frames were classified as quiet",
            "Frame RMS threshold relative to percentile",
        ))

    # Spectral-flatness spikes can expose isolated noise substitution or
    # splicing that energy and clipping checks miss.
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=1024, hop_length=512
    )[0]
    flatness_threshold = float(np.percentile(flatness, 95))
    if flatness_threshold > 0.22:
        flat_indices = np.where(flatness >= max(flatness_threshold, 0.22))[0]
        if len(flat_indices):
            start = float(times[min(int(flat_indices[0]), len(times) - 1)])
            end = float(times[min(int(flat_indices[-1]), len(times) - 1)])
            findings.append(make_finding(
                "spectral_flatness_spike",
                "medium",
                start,
                end,
                f"Flatness percentile-95 reached {flatness_threshold:.3f}",
                "Short-frame spectral flatness",
            ))

    # Frame-level mean jumps detect DC offset or cut-and-paste boundaries even
    # when the RMS level is otherwise nearly constant.
    frame_means = librosa.util.frame(audio, frame_length=1024, hop_length=512).mean(axis=0)
    mean_diff, mean_indices, mean_threshold = _find_robust_frame_discontinuities(
        frame_means,
        minimum_threshold=0.025,
    )
    if len(mean_diff):
        for index in mean_indices[:10]:
            start = float(times[min(index, len(times) - 1)])
            end = float(times[min(index + 1, len(times) - 1)])
            findings.append(make_finding(
                "dc_offset_or_copy_boundary",
                "medium",
                start,
                end,
                f"Frame mean change {mean_diff[index]:.5f} exceeded threshold {mean_threshold:.5f}",
                "Adjacent-frame mean discontinuity",
            ))

    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=512,
    )[0]
    centroid_diff, centroid_indices, centroid_threshold = (
        _find_robust_frame_discontinuities(
            spectral_centroids,
            minimum_threshold=250.0,
        )
    )
    for index in centroid_indices[:10]:
        start = float(times[min(index, len(times) - 1)])
        end = float(times[min(index + 1, len(times) - 1)])
        findings.append(make_finding(
            "spectral_discontinuity",
            "medium",
            start,
            end,
            f"Spectral centroid change {centroid_diff[index]:.1f} Hz exceeded threshold {centroid_threshold:.1f} Hz",
            "Adjacent-frame spectral-centroid discontinuity",
        ))

    tampering = {
        "status": "completed",
        "findings": findings,
        "methods_used": [
            "Clipping threshold analysis",
            "Adjacent-frame RMS discontinuity",
            "Frame RMS silence percentage",
            "Short-frame spectral flatness",
            "Adjacent-frame mean discontinuity",
            "Adjacent-frame spectral-centroid discontinuity",
        ],
        "limitations": [
            "These are indicator methods, not definitive evidence of editing.",
            "Speech pauses and normal recording effects may produce findings."
        ],
    }

    visualization = {
        "waveform": {"status": "completed"},
        "spectrogram": {"status": "completed"},
    }
    return quality, tampering, visualization


def make_finding(finding_type: str, severity: str, start: float, end: float,
                 evidence: str, method: str) -> dict[str, Any]:
    return {
        "finding_id": str(uuid.uuid4()),
        "finding_type": finding_type,
        "start_timestamp": round(max(0.0, start), 3),
        "end_timestamp": round(max(0.0, end), 3),
        "severity": severity,
        "measured_evidence": evidence,
        "explanation": "A measured signal characteristic was observed by the tested method.",
        "method_used": method,
        "limitations": "This finding is an indicator, not proof of manipulation.",
    }


def store_analysis(result: dict[str, Any]) -> None:
    analysis_id = result.get("analysis_id")
    evidence_id = result.get("evidence", {}).get("evidence_id")
    status = result.get("status")
    if not analysis_id or not evidence_id or not status:
        raise RuntimeError("Analysis payload has missing required identifiers.")

    now = utc_now()
    with _database_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO analyses (
                    analysis_id, evidence_id, status, completed_at,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    evidence_id,
                    status,
                    result.get("completed_at"),
                    json.dumps(result, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise RuntimeError(
                "Analysis was not inserted because it conflicts with existing evidence or an existing analysis."
            ) from error


def get_migration_status() -> dict[str, Any]:
    with _database_connection() as connection:
        evidence_count = connection.execute(
            "SELECT COUNT(*) FROM evidence"
        ).fetchone()[0]
        analysis_count = connection.execute(
            "SELECT COUNT(*) FROM analyses"
        ).fetchone()[0]
    return {
        **_migration_status,
        "evidence_rows": evidence_count,
        "analysis_rows": analysis_count,
    }

def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    with _database_connection() as connection:
        row = connection.execute(
            "SELECT payload_json FROM analyses WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
    if row is not None:
        try:
            result = json.loads(row["payload_json"]) if row["payload_json"] else None
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            result = None
        if isinstance(result, dict):
            evidence_id = result.get("evidence", {}).get("evidence_id")
            annotation = get_evidence_annotation(evidence_id) if evidence_id else None
            if annotation is not None:
                limitations = result.setdefault("overall_assessment", {}).setdefault(
                    "limitations", []
                )
                if annotation["note"] not in limitations:
                    limitations.append(annotation["note"])
            return result

    evidence_link = get_evidence_for_analysis(analysis_id)
    if evidence_link is None:
        return None

    evidence_id = evidence_link["evidence_id"]
    result = {
        "analysis_id": analysis_id,
        "status": "evidence_only",
        "completed_at": None,
        "evidence": {
            "evidence_id": evidence_id,
            "analysis_id": analysis_id,
            "filename": evidence_link["original_filename"],
            "sha256": evidence_link["sha256"],
            "upload_timestamp": evidence_link["metadata"].get("upload_timestamp"),
            "format": evidence_link["metadata"].get("format"),
            "file_size_bytes": (
                Path(evidence_link["original_path"]).stat().st_size
                if Path(evidence_link["original_path"]).is_file()
                else None
            ),
            "duration_seconds": evidence_link["metadata"].get("duration_seconds"),
            "sample_rate": evidence_link["metadata"].get("sample_rate"),
            "channels": evidence_link["metadata"].get("channels"),
        },
        "language_analysis": {
            "detected_language": None,
            "confidence": None,
            "supported": False,
        },
        "transcription": {
            "status": "missing",
            "text": "",
            "segments": [],
            "model": "Legacy record: analysis data unavailable",
        },
        "voice_authenticity": {
            "status": "missing",
            "assessment": "inconclusive",
            "model": None,
            "score": None,
            "threshold": None,
            "indicators": [],
            "limitations": ["This legacy record has no stored voice-analysis payload."],
        },
        "tampering_analysis": {
            "status": "missing",
            "findings": [],
            "methods_used": [],
            "limitations": ["This legacy record has no stored tampering-analysis payload."],
        },
        "audio_quality": {
            "status": "missing",
            "method_limitations": ["This legacy record has no stored audio-quality payload."],
        },
        "visualizations": {},
        "overall_assessment": {
            "classification": "insufficient_evidence",
            "summary": "The evidence is registered, but its original analysis payload is missing.",
            "limitations": [
                "No analysis result can be reconstructed without the original evidence processing.",
                "Legacy evidence should be reprocessed to create a complete analysis.",
            ],
        },
        "report": {
            "status": "unavailable",
            "download_endpoint": None,
        },
    }
    annotation = get_evidence_annotation(evidence_id)
    if annotation is not None:
        result["overall_assessment"]["limitations"].append(annotation["note"])
    return result


def get_evidence_for_analysis(analysis_id: str) -> dict[str, Any] | None:
    """Return registered evidence linkage when a legacy analysis payload is absent."""
    with _database_connection() as connection:
        row = connection.execute(
            "SELECT evidence_id, analysis_id, original_filename, sha256, "
            "original_path, metadata_json FROM evidence WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        metadata = {}
    return {
        "evidence_id": row["evidence_id"],
        "analysis_id": row["analysis_id"],
        "original_filename": row["original_filename"],
        "sha256": row["sha256"],
        "original_path": row["original_path"],
        "metadata": metadata,
    }
