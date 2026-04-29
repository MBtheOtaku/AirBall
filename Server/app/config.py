import os


def get_cors_origins() -> list[str]:
    raw_origins = os.getenv("AIRBALL_ALLOWED_ORIGINS") or os.getenv("CORS_ORIGINS", "http://localhost:3000")
    parsed = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

    # Preserve order while removing duplicates.
    deduped = list(dict.fromkeys(parsed))
    return deduped or ["http://localhost:3000"]
