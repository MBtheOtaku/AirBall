from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.auth import router as auth_router
from .routes.analysis import router as analysis_router
from .config import get_cors_origins


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(
    title="AirBall API",
    description="Backend API for AirBall application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
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