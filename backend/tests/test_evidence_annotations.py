"""Tests for non-destructive evidence provenance annotations."""

import json
import uuid
from io import BytesIO
import hashlib
import wave

from fastapi.testclient import TestClient
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from app.modules import audio_forensics
from app.main import app


def test_annotation_is_visible_on_retrieved_analysis(monkeypatch, tmp_path) -> None:
    evidence_id = str(uuid.uuid4())
    analysis_id = str(uuid.uuid4())
    monkeypatch.setattr(audio_forensics, "DATABASE_PATH", tmp_path / "forensics.db")
    monkeypatch.setattr(audio_forensics, "EVIDENCE_ROOT", tmp_path / "audio")
    audio_forensics._initialize_database()

    metadata = {
        "evidence_id": evidence_id,
        "analysis_id": analysis_id,
        "original_filename": "fixture.wav",
        "sha256": "fixture-hash",
    }
    timestamp = audio_forensics.utc_now()
    with audio_forensics._database_connection() as connection:
        connection.execute(
            """INSERT INTO evidence (
                evidence_id, analysis_id, original_filename, sha256,
                upload_timestamp, original_path, format, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence_id,
                analysis_id,
                "fixture.wav",
                "fixture-hash",
                timestamp,
                str(tmp_path / "fixture.wav"),
                "WAV",
                json.dumps(metadata),
                timestamp,
            ),
        )
        connection.execute(
            """INSERT INTO analyses (
                analysis_id, evidence_id, status, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                analysis_id,
                evidence_id,
                "completed",
                json.dumps({
                    "analysis_id": analysis_id,
                    "evidence": {"evidence_id": evidence_id},
                    "overall_assessment": {"limitations": []},
                }),
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """INSERT INTO evidence_annotations (
                annotation_id, evidence_id, classification, note, annotated_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                evidence_id,
                "known_test_fixture",
                "Synthetic QA record; not genuine evidence.",
                timestamp,
            ),
        )

    result = audio_forensics.get_analysis(analysis_id)
    assert result is not None
    assert result["overall_assessment"]["limitations"] == [
        "Synthetic QA record; not genuine evidence."
    ]

    with audio_forensics._database_connection() as connection:
        stored = json.loads(
            connection.execute(
                "SELECT payload_json FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()[0]
        )
    assert stored["overall_assessment"]["limitations"] == []


def test_annotated_report_download_is_visibly_marked_without_overwriting(
    monkeypatch, tmp_path
) -> None:
    analysis_id = str(uuid.uuid4())
    evidence_id = str(uuid.uuid4())
    report_root = tmp_path / "reports"
    report_root.mkdir()
    report_path = report_root / f"{analysis_id}.pdf"
    source = BytesIO()
    pdf = canvas.Canvas(source)
    pdf.drawString(36, 780, "Original analysis report")
    pdf.save()
    original_bytes = source.getvalue()
    report_path.write_bytes(original_bytes)

    monkeypatch.setattr("app.routers.audio.REPORT_ROOT", report_root)
    monkeypatch.setattr(
        "app.routers.audio.get_analysis",
        lambda _analysis_id: {"evidence": {"evidence_id": evidence_id}},
    )
    monkeypatch.setattr(
        "app.routers.audio.get_evidence_annotation",
        lambda _evidence_id: {
            "classification": "known_test_fixture",
            "note": "Synthetic QA fixture; not genuine evidence.",
        },
    )

    response = TestClient(app).get(f"/audio/reports/{analysis_id}")

    assert response.status_code == 200
    assert response.headers["x-evidence-classification"] == "known_test_fixture"
    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(response.content)).pages
    )
    assert "EVIDENCE CLASSIFICATION: KNOWN_TEST_FIXTURE" in extracted
    assert "not genuine evidence" in extracted
    assert report_path.read_bytes() == original_bytes


def test_migration_marks_every_verified_copy_of_known_fixture(
    monkeypatch, tmp_path
) -> None:
    database_path = tmp_path / "fixture.db"
    evidence_root = tmp_path / "audio"
    monkeypatch.setattr(audio_forensics, "DATABASE_PATH", database_path)
    monkeypatch.setattr(audio_forensics, "EVIDENCE_ROOT", evidence_root)
    audio_forensics._initialize_database()

    wav_bytes = BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    fixture_bytes = wav_bytes.getvalue()
    fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
    expected_hash = audio_forensics.KNOWN_TEST_EVIDENCE_ANNOTATIONS[
        "689b96d4-ca14-4627-a03d-92703672f57c"
    ]["sha256"]
    assert fixture_hash == expected_hash

    timestamp = audio_forensics.utc_now()
    fixture_ids = [
        "689b96d4-ca14-4627-a03d-92703672f57c",
        str(uuid.uuid4()),
    ]
    for evidence_id in fixture_ids:
        evidence_directory = evidence_root / evidence_id
        evidence_directory.mkdir(parents=True)
        original_path = evidence_directory / "original.wav"
        original_path.write_bytes(fixture_bytes)
        analysis_id = (
            "dc05c4de-9de4-4434-adbe-bcd112a7a875"
            if evidence_id == fixture_ids[0]
            else str(uuid.uuid4())
        )
        with audio_forensics._database_connection() as connection:
            connection.execute(
                """INSERT INTO evidence (
                    evidence_id, analysis_id, original_filename, sha256,
                    upload_timestamp, original_path, format, duration_seconds,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence_id,
                    analysis_id,
                    "complete.wav",
                    fixture_hash,
                    timestamp,
                    str(original_path),
                    "WAV",
                    0.1,
                    "{}",
                    timestamp,
                ),
            )

    nonmatch_id = str(uuid.uuid4())
    nonmatch_path = tmp_path / "different.wav"
    nonmatch_path.write_bytes(fixture_bytes + b"different")
    with audio_forensics._database_connection() as connection:
        connection.execute(
            """INSERT INTO evidence (
                evidence_id, analysis_id, original_filename, sha256,
                upload_timestamp, original_path, format, duration_seconds,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                nonmatch_id,
                str(uuid.uuid4()),
                "complete.wav",
                fixture_hash,
                timestamp,
                str(nonmatch_path),
                "WAV",
                0.1,
                "{}",
                timestamp,
            ),
        )

    audio_forensics._initialize_database()

    assert all(
        audio_forensics.get_evidence_annotation(evidence_id) is not None
        for evidence_id in fixture_ids
    )
    assert audio_forensics.get_evidence_annotation(nonmatch_id) is None


def test_malformed_legacy_metadata_is_handled_without_crashing(
    monkeypatch, tmp_path
) -> None:
    evidence_id = str(uuid.uuid4())
    analysis_id = str(uuid.uuid4())
    monkeypatch.setattr(audio_forensics, "DATABASE_PATH", tmp_path / "legacy.db")
    monkeypatch.setattr(audio_forensics, "EVIDENCE_ROOT", tmp_path / "audio")
    audio_forensics._initialize_database()
    with audio_forensics._database_connection() as connection:
        connection.execute(
            """INSERT INTO evidence (
                evidence_id, analysis_id, original_filename, sha256,
                upload_timestamp, original_path, format, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence_id,
                analysis_id,
                "legacy.wav",
                "malformed-hash",
                audio_forensics.utc_now(),
                str(tmp_path / "missing.wav"),
                "WAV",
                "{not-json",
                audio_forensics.utc_now(),
            ),
        )

    legacy = audio_forensics.get_evidence_for_analysis(analysis_id)
    assert legacy is not None
    assert legacy["metadata"] == {}


def test_legacy_registered_evidence_can_retrieve_existing_artifacts_without_analysis(
    monkeypatch, tmp_path
) -> None:
    evidence_id = str(uuid.uuid4())
    analysis_id = str(uuid.uuid4())
    evidence_root = tmp_path / "audio"
    report_root = tmp_path / "reports"
    evidence_directory = evidence_root / evidence_id
    evidence_directory.mkdir(parents=True)
    report_root.mkdir()
    monkeypatch.setattr(audio_forensics, "DATABASE_PATH", tmp_path / "legacy.db")
    monkeypatch.setattr(audio_forensics, "EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr("app.routers.audio.EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr("app.routers.audio.REPORT_ROOT", report_root)
    audio_forensics._initialize_database()

    original_path = evidence_directory / "original.wav"
    original_path.write_bytes(b"legacy-original-audio")
    timestamp = audio_forensics.utc_now()
    metadata = {
        "evidence_id": evidence_id,
        "analysis_id": analysis_id,
        "original_filename": "legacy.wav",
        "sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "original_path": str(original_path),
    }
    with audio_forensics._database_connection() as connection:
        connection.execute(
            """INSERT INTO evidence (
                evidence_id, analysis_id, original_filename, sha256,
                upload_timestamp, original_path, format, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence_id,
                analysis_id,
                "legacy.wav",
                metadata["sha256"],
                timestamp,
                str(original_path),
                "WAV",
                json.dumps(metadata),
                timestamp,
            ),
        )

    (evidence_directory / "waveform.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8"
    )
    (evidence_directory / "spectrogram.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8"
    )
    source_pdf = BytesIO()
    canvas.Canvas(source_pdf).save()
    (report_root / f"{analysis_id}.pdf").write_bytes(source_pdf.getvalue())

    client = TestClient(app)
    waveform = client.get(f"/audio/visualizations/{analysis_id}/waveform")
    spectrogram = client.get(f"/audio/visualizations/{analysis_id}/spectrogram")
    report = client.get(f"/audio/reports/{analysis_id}")
    analysis = client.get(f"/audio/analyses/{analysis_id}")

    assert waveform.status_code == 200
    assert spectrogram.status_code == 200
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF-")
    assert analysis.status_code == 200
    assert analysis.json()["status"] == "evidence_only"
    assert analysis.json()["overall_assessment"]["classification"] == "insufficient_evidence"
    assert audio_forensics.get_analysis(analysis_id)["status"] == "evidence_only"
