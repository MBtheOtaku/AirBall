import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.auth import router as auth_router
from .routes.analysis import router as analysis_router


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _get_cors_origins() -> list[str]:
    raw_origins = os.getenv("AIRBALL_ALLOWED_ORIGINS") or os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app = FastAPI(
    title="AirBall API",
    description="Backend API for AirBall application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(analysis_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}