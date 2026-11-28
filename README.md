# Regional Media Forensics

Regional Media Forensics is a local-development application with a React +
TypeScript + Vite frontend and a Python + FastAPI backend. The backend includes
a video visual-forensics API and an Audio Backend for transcription and audio
forensic analysis. Audio evidence and analysis records are stored in SQLite;
the application does not provide a general credibility score.

## Requirements

- Windows 11
- Python 3.10 or newer
- Node.js 18 or newer, including npm

## Run the backend

Open PowerShell in the project root, then run:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The backend runs at `http://127.0.0.1:8000`. Check it by opening
`http://127.0.0.1:8000/health` in a browser. It should return:

```json
{"status":"ok"}
```

To run the backend test suite, open a second PowerShell window in the project
root:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

## Audio Backend Overview

The Audio Backend accepts uploaded recordings, validates and decodes the
audio, performs independent forensic analysis stages, preserves evidence, and
returns structured JSON responses. It runs locally and does not require a paid
API or external transcription service.

### Core objectives

- Transcribe supported regional languages and English.
- Process recordings up to 30 minutes.
- Identify AI-generated voice indicators with cautious result reporting.
- Detect measurable clipping, discontinuity, silence, spectral, DC-offset,
  and copy-boundary indicators.
- Preserve original evidence and analysis traceability.
- Return consistent, inspectable API responses.

## Audio Processing Workflow

1. The upload is checked for size, extension, MIME type, signature, duration,
   decoded samples, and supported format.
2. The original upload is preserved in an evidence directory with integrity
   metadata.
3. SHA-256 integrity metadata and SQLite evidence records are written.
4. The audio is decoded to 16 kHz mono for model processing.
5. Transcription, voice detection, quality measurement, and tampering
   indicators run independently.
6. SVG waveform/spectrogram artifacts and a PDF report are generated.
7. The final analysis and evidence linkage are stored for later retrieval.

## Supported Languages

The multilingual ASR service supports these language codes:

| Code | Language |
| --- | --- |
| `bn` | Bengali |
| `en` | English |
| `gu` | Gujarati |
| `hi` | Hindi |
| `kn` | Kannada |
| `ml` | Malayalam |
| `mr` | Marathi |
| `ta` | Tamil |
| `te` | Telugu |

A language can be selected explicitly or automatically detected. A specialized
Kannada Whisper checkpoint is used by the dedicated Kannada endpoint.

## Audio Architecture

```text
regional-media-forensics/
  storage/                       Runtime evidence, database, and reports
  backend/
    requirements.txt
    assets/
      fonts/
        OFL.txt
    app/
      main.py
      routers/
        audio.py                 Audio HTTP endpoints
        video.py                 Existing video endpoint
      modules/
        audio_forensics.py       Evidence, integrity, quality, and tampering
        forensic_report.py       PDF report generation
        voice_detection/
          audio_validation.py    Upload validation
          speech_to_text.py      Kannada transcription
          multilingual_speech_to_text.py
          feature_extraction.py  17 voice features
          prediction.py          Voice classifier prediction
          voice_detection.py     Model lifecycle
          transcription_evaluation.py
      schemas/
        audio.py                 Audio response schemas
    tests/
      test_audio_api.py
      test_evidence_annotations.py
    models/                      Existing local model files
    datasets/                    Existing evaluation data
  frontend/                      React + TypeScript + Vite application
```

`storage/` is created at the repository root at runtime. It contains audio
evidence, the SQLite database, generated visualizations, and PDF reports; it is
local application data, not source code.

## Speech-to-Text

The backend uses Faster-Whisper with these configured models:

- Specialized Kannada model:
  `mohankumars26/whisper-kannada-medium-ct2`
- Multilingual Whisper medium model, CPU `int8`

The model weights must be available to Faster-Whisper. If they are not already
cached locally, model initialization may retrieve them. Installing Python
requirements does not itself install the model weights.

The transcription pipeline:

- Normalizes audio to 16 kHz mono PCM.
- Uses VAD and repetition controls.
- Processes long recordings in 120-second chunks with overlap.
- Orders segments chronologically.
- Removes duplicated words at overlap boundaries.
- Cleans repeated characters and repeated punctuation.
- The dedicated transcription endpoints return transcript text, language
  information, and status. The complete-analysis endpoint also stores
  timestamped transcription segments.

## AI-Generated Voice Detection

The existing scikit-learn Random Forest checkpoint classifies a 17-feature
representation of the decoded audio. Features include RMS energy, zero-crossing
rate, spectral centroid, spectral bandwidth, and 13 mean MFCC values.

The API distinguishes:

- Short, silent, or unusable recordings.
- Completed model predictions.
- Cautious genuine or AI-generated labels.

Classifier confidence is returned as model output, not presented as a calibrated
probability of authenticity.

## Audio Tampering Detection

The tampering analysis uses measurable signal indicators:

- Clipping and peak thresholds.
- RMS discontinuities between adjacent frames.
- Quiet-frame or long-silence proportions.
- Spectral-flatness spikes.
- Frame-mean/DC-offset discontinuities.
- Spectral-centroid discontinuities.

Findings include timestamps, severity, measured evidence, method, and
limitations. They are indicators for forensic review, not definitive proof of
editing.

## Evidence and Database Management

The backend preserves:

- The original uploaded file.
- SHA-256 hash.
- Filename and upload metadata.
- Recording format, duration, sample rate, channels, and frame count.
- SQLite evidence and analysis records.
- Generated waveform and spectrogram artifacts.
- Generated PDF reports.
- Legacy evidence and evidence-only fallback responses.

Integrity can be verified after the original evidence is stored.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process health check |
| `POST` | `/audio/transcribe/kannada` | Kannada transcription |
| `POST` | `/audio/transcribe/multilingual` | Multilingual transcription |
| `POST` | `/audio/detect` | AI-generated voice indicator |
| `POST` | `/audio/analyze` | Complete forensic analysis (HTTP 202) |
| `GET` | `/audio/analyses/{analysis_id}/status` | Analysis status |
| `GET` | `/audio/analyses/{analysis_id}` | Complete stored result |
| `GET` | `/audio/evidence/{evidence_id}/integrity` | SHA-256 integrity check |
| `GET` | `/audio/visualizations/{analysis_id}/{kind}` | Waveform or spectrogram |
| `GET` | `/audio/reports/{analysis_id}` | Download PDF report |

Swagger UI is available at `http://127.0.0.1:8000/docs`.

## Supported Upload Formats

- WAV
- MP3
- FLAC
- OGG
- M4A
- WebM
- AAC
- Opus

Uploads are limited to 25 MiB and 30 minutes. The upload validator checks
supported extensions, file signatures, MIME type compatibility, and decoded
audio duration.

## Example Requests

### Kannada transcription

```powershell
curl.exe -F "file=@C:\path\to\kannada-recording.wav" "http://127.0.0.1:8000/audio/transcribe/kannada"
```

### Multilingual transcription

```powershell
curl.exe -F "file=@C:\path\to\hindi-recording.wav" "http://127.0.0.1:8000/audio/transcribe/multilingual?language=hi"
```

Automatic language detection:

```powershell
curl.exe -F "file=@C:\path\to\english-recording.wav" "http://127.0.0.1:8000/audio/transcribe/multilingual"
```

### Voice detection

```powershell
curl.exe -F "file=@C:\path\to\recording.wav" "http://127.0.0.1:8000/audio/detect"
```

### Complete forensic analysis

```powershell
curl.exe -F "file=@C:\path\to\recording.wav" "http://127.0.0.1:8000/audio/analyze"
```

## Example Response Shape

`POST /audio/transcribe/multilingual` returns a response shaped like:

```json
{
  "filename": "hindi-recording.wav",
  "transcription": {
    "transcript": "Transcribed text",
    "detected_language": "hi",
    "language_name": "Hindi",
    "language_probability": 98.5,
    "status": "success"
  }
}
```

`POST /audio/detect` returns a classifier result:

```json
{
  "filename": "recording.wav",
  "analysis": {
    "prediction": "Genuine Human Speech",
    "label": 0,
    "confidence": 82.4,
    "status": "completed"
  }
}
```

`POST /audio/analyze` returns HTTP 202 and a complete analysis payload. It
includes an `analysis_id`, evidence metadata, transcription segments,
voice-authenticity output, quality metrics, tampering findings, visualization
links, an overall assessment, and a report download endpoint.

## Technologies

- Python
- FastAPI and Pydantic
- Faster-Whisper
- librosa
- soundfile
- NumPy
- pandas and pyarrow
- joblib
- scikit-learn
- SQLite
- ReportLab
- uharfbuzz, fontTools, and pypdf for multilingual PDF text
- pytest

## Audio Backend Environment

The backend setup above installs the Audio Backend dependencies into
`backend\.venv`. The core Audio routes use local ASR and require neither a paid
API key nor a `backend/.env` file. Faster-Whisper model weights are separate
from Python dependencies. If the required weights are not cached locally,
Faster-Whisper may retrieve them when the model is initialized.

The backend is served at `http://127.0.0.1:8000`; its Swagger UI is available
at `http://127.0.0.1:8000/docs`.

## Audio Backend Tests

Run the Audio-focused regression tests from the project root in PowerShell:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q test_dataset_loader.py test_feature_extraction.py tests\test_audio_api.py tests\test_evidence_annotations.py tests\test_pdf_unicode.py tests\test_transcription_evaluation.py tests\test_voice_and_language_hardening.py
```

## Evaluation Scripts

The evaluation script uses the existing local test manifest, dataset, and model
checkpoint. From the project root, run:

```powershell
cd backend
.\.venv\Scripts\python.exe test_model_evaluation.py
```

The evaluation reports accuracy, precision, recall, F1-score, and language-wise
results for the existing test manifest.

## Folder Structure

```text
regional-media-forensics/
  README.md
  frontend/
  storage/                         Created locally at runtime
  backend/
    requirements.txt
    app/
      main.py
      routers/
        audio.py
        video.py
      modules/
        audio_forensics.py
        forensic_report.py
        voice_detection/
          audio_validation.py
          multilingual_speech_to_text.py
          speech_to_text.py
          feature_extraction.py
          prediction.py
      schemas/
        audio.py
    tests/
      test_audio_api.py
      test_evidence_annotations.py
    assets/
      fonts/
        OFL.txt
    models/
    datasets/
```

## Credits

This project uses openly available machine-learning tooling and existing local
model checkpoints. Whisper-related functionality is provided through
Faster-Whisper; audio processing uses librosa and soundfile; API services use
FastAPI. The bundled Noto Sans font files are licensed under SIL Open Font
License 1.1; their copyright notices and license text are in
`backend/assets/fonts/OFL.txt`. The font families are from the Noto Project:
[Bengali](https://github.com/notofonts/bengali),
[Devanagari](https://github.com/notofonts/devanagari),
[Latin, Greek, and Cyrillic](https://github.com/notofonts/latin-greek-cyrillic),
[Gujarati](https://github.com/notofonts/gujarati),
[Kannada](https://github.com/notofonts/kannada),
[Malayalam](https://github.com/notofonts/malayalam),
[Tamil](https://github.com/notofonts/tamil), and
[Telugu](https://github.com/notofonts/telugu).

## Run the frontend

Open another PowerShell window in the project root, then run:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`. The page
checks the backend when it loads and displays either **Connected** or
**Not connected**, including an error message if the health request fails.

The frontend defaults to `http://127.0.0.1:8000` for its backend URL. To use
another URL, copy `frontend/.env.example` to `frontend/.env.local`, edit
`VITE_API_BASE_URL`, and restart the Vite development server.

## Verify frontend-to-backend communication

1. Start the backend and confirm `/health` returns `{"status":"ok"}`.
2. Start the frontend and open its Vite URL.
3. Confirm the page shows **Connected**.
4. Stop the backend and reload the frontend page. It should show
   **Not connected** with a request error.

The backend and frontend are run in separate terminal windows. Dependency
installation is intentionally left to the developer using the commands above.

## Visual-forensics API

The backend exposes `POST /video/analyze` as a multipart upload with a single
`file` field. Uploads are limited to 500 MiB. The response includes video
metadata, frame and face counts, available per-frame probabilities, aggregate
real/manipulated probabilities, the cautious verdict, processing timings, and
limitations. Processing errors and inconclusive results remain explicitly
marked; neither is an authenticity verdict.

xdfdet runs in a separate Python 3.13 environment, not the backend environment.
Set `RMF_XDFDET_PYTHON` to that environment's Python executable when it is not
at `backend/.venv-xdfdet`. The worker requires xdfdet, TensorFlow/Keras, OpenCV,
NumPy, MTCNN, and its other model dependencies in that isolated environment.
The backend environment separately requires OpenCV and the multipart parser
listed in `backend/requirements.txt`. The
`aug-cutout-black` checkpoint must already be present in the local Hugging Face
cache; the worker is started in offline mode and will not fetch a checkpoint.
The backend does not install TensorFlow or Keras.

The API is not deployment-ready until the isolated worker environment and local
checkpoint are installed and verified in the target deployment environment.
