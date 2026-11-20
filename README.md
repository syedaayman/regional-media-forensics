# Regional Media Forensics

Regional Media Forensics is a local-development application with a React +
TypeScript + Vite frontend and a Python + FastAPI backend. The backend includes
a video visual-forensics API; the application does not include a database or a
general credibility score.

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

To run the backend test, open a second PowerShell window in the project root:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

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
