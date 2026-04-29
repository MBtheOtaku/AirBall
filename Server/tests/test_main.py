from app.config import get_cors_origins


def test_get_cors_origins_prefers_airball_var(monkeypatch):
    monkeypatch.setenv("AIRBALL_ALLOWED_ORIGINS", "https://a.com, https://b.com")
    monkeypatch.setenv("CORS_ORIGINS", "https://ignored.com")

    origins = get_cors_origins()

    assert origins == ["https://a.com", "https://b.com"]


def test_get_cors_origins_dedupes_and_trims(monkeypatch):
    monkeypatch.setenv("AIRBALL_ALLOWED_ORIGINS", " https://a.com ,https://a.com, , https://b.com ")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    origins = get_cors_origins()

    assert origins == ["https://a.com", "https://b.com"]


def test_get_cors_origins_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AIRBALL_ALLOWED_ORIGINS", " ,   ")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    origins = get_cors_origins()

    assert origins == ["http://localhost:3000"]


def test_get_cors_origins_uses_legacy_env_when_primary_missing(monkeypatch):
    monkeypatch.delenv("AIRBALL_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,https://prod.example.com")

    origins = get_cors_origins()

    assert origins == ["http://localhost:3000", "https://prod.example.com"]
