from pathlib import Path
from typing import Any
from io import BytesIO

from app.modules.audio_forensics import BACKEND_PATH


REPORT_ROOT = BACKEND_PATH / "storage" / "reports"


def add_evidence_annotation(pdf_bytes: bytes, classification: str, note: str) -> bytes:
    """Return a visibly annotated PDF copy without changing the stored report."""
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen import canvas

    source = PdfReader(BytesIO(pdf_bytes))
    writer = PdfWriter()
    for page in source.pages:
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        overlay_bytes = BytesIO()
        overlay = canvas.Canvas(overlay_bytes, pagesize=(width, height))
        banner_height = min(42.0, height * 0.08)
        overlay.setFillColor(HexColor("#8b1e1e"))
        overlay.rect(0, height - banner_height, width, banner_height, fill=1, stroke=0)
        overlay.setFillColor(white)
        overlay.setFont("Helvetica-Bold", 10)
        overlay.drawString(
            24,
            height - 16,
            f"EVIDENCE CLASSIFICATION: {classification.upper()}",
        )
        overlay.setFont("Helvetica", 8)
        overlay.drawString(24, height - 30, note.replace("\n", " ")[:220])
        overlay.save()
        overlay_bytes.seek(0)
        page.merge_page(PdfReader(overlay_bytes).pages[0])
        writer.add_page(page)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
    )[:1000]


def _generate_reportlab_fallback(report_path: Path, html_document: str) -> None:
    """Best-effort fallback for servers without a Chromium PDF renderer."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = BACKEND_PATH / "backend" / "assets" / "fonts" / "NotoSansEnglish.ttf"
    if font_path.is_file():
        pdfmetrics.registerFont(TTFont("NotoSansFallback", str(font_path)))
        font_name = "NotoSansFallback"
    else:
        font_name = "Helvetica"

    text = " ".join(
        html_fragment
        for html_fragment in html_document.replace("<", " <").split()
    )
    text = html.unescape(text)
    text = text.replace(">", " ")

    pdf = canvas.Canvas(str(report_path), pagesize=A4)
    pdf.setFont(font_name, 9)
    y = 790
    for line in text.split():
        pdf.drawString(50, y, line[:160])
        y -= 12
        if y < 40:
            pdf.showPage()
            pdf.setFont(font_name, 9)
            y = 790
    pdf.save()


def _script_font_name(char: str) -> str:
    code = ord(char)
    if 0x0980 <= code <= 0x09FF:
        return "NotoSansBengali"
    if 0x0900 <= code <= 0x097F:
        return "NotoSansDevanagari"
    if 0x0A80 <= code <= 0x0AFF:
        return "NotoSansGujarati"
    if 0x0C80 <= code <= 0x0CFF:
        return "NotoSansKannada"
    if 0x0D00 <= code <= 0x0D7F:
        return "NotoSansMalayalam"
    if 0x0B80 <= code <= 0x0BFF:
        return "NotoSansTamil"
    if 0x0C00 <= code <= 0x0C7F:
        return "NotoSansTelugu"
    return "NotoSansEnglish"


def _script_runs(text: str) -> list[tuple[str, str]]:
    runs = []
    current_font = None
    current = []
    for char in text:
        font_name = _script_font_name(char)
        if current_font is None:
            current_font = font_name
        if font_name != current_font:
            runs.append((current_font, "".join(current)))
            current = []
            current_font = font_name
        current.append(char)
    if current:
        runs.append((current_font or "NotoSansEnglish", "".join(current)))
    return runs


def _generate_shaped_report(report_path: Path, result: dict[str, Any]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from fontTools.pens.basePen import BasePen
    from fontTools.ttLib import TTFont as FontToolsTTFont
    import uharfbuzz as hb

    class PDFGlyphPen(BasePen):
        def __init__(self, glyph_set, path):
            super().__init__(glyph_set)
            self.path = path

        def _moveTo(self, point):
            self.path.moveTo(point[0], point[1])

        def _lineTo(self, point):
            self.path.lineTo(point[0], point[1])

        def _curveToOne(self, p1, p2, p3):
            self.path.curveTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

        def _closePath(self):
            self.path.close()

    font_dir = BACKEND_PATH / "backend" / "assets" / "fonts"
    font_files = {
        "NotoSansEnglish": "NotoSansEnglish.ttf",
        "NotoSansDevanagari": "NotoSansDevanagari.ttf",
        "NotoSansKannada": "NotoSansKannada.ttf",
        "NotoSansTelugu": "NotoSansTelugu.ttf",
        "NotoSansTamil": "NotoSansTamil.ttf",
        "NotoSansMalayalam": "NotoSansMalayalam.ttf",
        "NotoSansBengali": "NotoSansBengali.ttf",
        "NotoSansGujarati": "NotoSansGujarati.ttf",
    }
    for name, filename in font_files.items():
        path = font_dir / filename
        if path.is_file():
            pdfmetrics.registerFont(TTFont(name, str(path)))

    evidence = result["evidence"]
    quality = result["audio_quality"]
    findings = result["tampering_analysis"]["findings"]

    pdf = canvas.Canvas(str(report_path), pagesize=A4)
    width, height = A4
    y = height - 50

    def line(text: str, size: float = 9, bold: bool = False) -> None:
        nonlocal y
        pdf.setFillColor(colors.black)
        pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        pdf.drawString(50, y, str(text)[:1000])
        y -= size + 5

    def heading(text: str) -> None:
        nonlocal y
        y -= 8
        line(text, 12, True)
        pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
        pdf.line(50, y + 3, width - 50, y + 3)
        y -= 4

    def field(label: str, value: Any) -> None:
        line(f"{label}: {value}", 9)

    line("REGIONAL MEDIA FORENSICS - AUDIO FORENSIC REPORT", 16, True)
    heading("EVIDENCE")
    field("Report ID", result["analysis_id"])
    field("Evidence ID", evidence["evidence_id"])
    field("Original filename", evidence["filename"])
    field("SHA-256", evidence["sha256"])
    field("Uploaded", evidence["upload_timestamp"])

    heading("AUDIO METADATA")
    field("Format", evidence["format"])
    field("Duration", f"{evidence['duration_seconds']} seconds")
    field("Sample rate", f"{evidence['sample_rate']} Hz")
    field("Channels", evidence["channels"])
    field("File size", f"{evidence['file_size_bytes']} bytes")

    heading("LANGUAGE AND TRANSCRIPTION")
    field("Detected language", result["language_analysis"]["detected_language"])
    field("Language confidence", result["language_analysis"]["confidence"])
    field("Model", result["transcription"]["model"])
    field("Status", result["transcription"]["status"])
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(50, y, "Transcript:")
    y -= 14

    def draw_unicode(text: str, x: float, baseline: float, size: float = 9) -> float:
        cursor = x
        for font_name, run in _script_runs(text):
            font_path = font_dir / font_files[font_name]
            if not font_path.is_file():
                continue

            face = hb.Face(font_path.read_bytes())
            hb_font = hb.Font(face)
            tt_font = FontToolsTTFont(str(font_path), lazy=True)
            units_per_em = tt_font["head"].unitsPerEm
            glyph_set = tt_font.getGlyphSet()

            buffer = hb.Buffer()
            buffer.add_str(run)
            buffer.guess_segment_properties()
            hb.shape(hb_font, buffer)
            scale = size / units_per_em
            advance = 0.0
            glyph_x = cursor

            for info, position in zip(buffer.glyph_infos, buffer.glyph_positions):
                glyph_name = tt_font.getGlyphName(info.codepoint)
                glyph = glyph_set[glyph_name]
                pdf_path = pdf.beginPath()
                pen = PDFGlyphPen(glyph_set, pdf_path)
                glyph.draw(pen)
                pdf.saveState()
                pdf.translate(
                    glyph_x + position.x_offset * scale,
                    baseline + position.y_offset * scale,
                )
                pdf.scale(scale, scale)
                pdf.setFillColor(colors.black)
                pdf.drawPath(pdf_path, fill=1, stroke=0)
                pdf.restoreState()
                glyph_x += position.x_advance * scale
                advance += position.x_advance * scale

            cursor += advance
            tt_font.close()

        text_object = pdf.beginText()
        text_object.setTextOrigin(x, baseline)
        text_object.setTextRenderMode(3)
        for font_name, run in _script_runs(text):
            if font_name in pdfmetrics.getRegisteredFontNames():
                text_object.setFont(font_name, size)
            text_object.textOut(run)
        pdf.drawText(text_object)
        return cursor

    transcript = str(result["transcription"]["text"])
    cursor = 50
    for chunk in [transcript[i:i + 110] for i in range(0, len(transcript), 110)]:
        draw_unicode(chunk, cursor, y, 9)
        y -= 14
        if y < 40:
            pdf.showPage()
            y = height - 50

    heading("VOICE AUTHENTICITY")
    voice = result["voice_authenticity"]
    field("Status", voice["status"])
    field("Assessment", voice["assessment"])
    field("Model", voice["model"])
    field("Actual model score", voice["score"])
    field("Decision threshold", voice["threshold"])

    heading("AUDIO QUALITY")
    for key in ("snr_db", "clipping_percentage", "silence_percentage",
                "rms_energy", "zero_crossing_rate", "spectral_centroid_hz"):
        field(key, quality.get(key))

    heading("TAMPERING AND MANIPULATION INDICATORS")
    if findings:
        for finding in findings:
            field(
                str(finding["finding_type"]),
                f"{finding['start_timestamp']}s-{finding['end_timestamp']}s | "
                f"{finding['severity']} | {finding['measured_evidence']}",
            )
    else:
        line("No indicator was detected by the tested methods.", 9)

    heading("VISUALIZATIONS")
    field("Waveform", result["visualizations"]["waveform"].get("download_endpoint"))
    field("Spectrogram", result["visualizations"]["spectrogram"].get("download_endpoint"))

    heading("OVERALL ASSESSMENT")
    field("Classification", result["overall_assessment"]["classification"])
    field("Summary", result["overall_assessment"]["summary"])

    heading("LIMITATIONS")
    for limitation in result["overall_assessment"]["limitations"]:
        line("- " + str(limitation), 9)

    heading("PROCESSING HISTORY")
    field("Completed", result["completed_at"])
    field("Integrity verification", "Hash matches only if it has not changed since upload.")

    pdf.save()


def generate_pdf(result: dict[str, Any]) -> Path:
    report_dir = REPORT_ROOT
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{result['analysis_id']}.pdf"
    _generate_shaped_report(report_path, result)
    return report_path
