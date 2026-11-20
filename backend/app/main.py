from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.routers.video import router as video_router

app = FastAPI(title="Regional Media Forensics API")

app.include_router(video_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
