from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import List, Optional


# Sentinel values that should NEVER survive into production. The validator below
# rejects any of these when DEBUG=False.
_INSECURE_DEFAULTS = {
    "MINIO_ACCESS_KEY": {"", "minioadmin"},
    "MINIO_SECRET_KEY": {"", "minioadmin"},
    "ARANGODB_PASSWORD": {""},
    "GROQ_API_KEY": {""},
}


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "MediCortex Orchestrator"
    DEBUG: bool = False  # SEC-3: secure-by-default; explicit DEBUG=true required in dev .env

    # CORS — DEPLOY-3: explicit allowlist, never "*"
    # Comma-separated env var, e.g. ALLOWED_ORIGINS="http://localhost:5173,https://user.github.io"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173"]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/medicortex"
    SQLALCHEMY_POOL_SIZE: int = 20  # OPS-1: 10-user concurrency requires headroom
    SQLALCHEMY_MAX_OVERFLOW: int = 10
    SQLALCHEMY_POOL_TIMEOUT: int = 30

    # Concurrency / worker affinity (OPS-3)
    WEB_CONCURRENCY: int = 1

    # Ollama (Router / Aggregator / Extractor / Agent Planner / Fallback synthesizer)
    OLLAMA_CLOUD_URL: str = "http://homeserver:11434"
    OLLAMA_CLOUD_MODEL: str = "gemma4:e2b"
    OLLAMA_TIMEOUT_SECONDS: int = 60  # OPS-4

    # MinIO
    MINIO_URL: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET_NAME: str = "medicortex-uploads"
    # SEC-2: Presigned URL TTL — 1 hour by default (was 7 days)
    MINIO_PRESIGN_TTL_SECONDS: int = 3600
    # Override for presigned URL host (e.g. tailnet FQDN). When set, presigned
    # URLs use this host so the user's browser can resolve them — the internal
    # MINIO_URL may point at host.docker.internal which browsers cannot reach.
    MINIO_PUBLIC_URL: Optional[str] = None

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_SOCKET_TIMEOUT: int = 2  # OPS-7

    # MedGemma
    MEDGEMMA_API_URL: str = "http://localhost:8000/predict"
    # DEPLOY-2: cold-start handling
    MEDGEMMA_TIMEOUT_SECONDS: int = 30  # was 120; fall back to Gemma 4 fast on RunPod cold start
    MEDGEMMA_KEEPWARM_URL: Optional[str] = None  # if set, periodically pinged in lifespan
    MEDGEMMA_KEEPWARM_INTERVAL_SECONDS: int = 240  # 4 min
    RUNPOD_API_KEY: Optional[str] = None  # Bearer token for RunPod /runsync

    # Upload / download size caps (SEC-1)
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024   # 50 MB
    MAX_PDF_BYTES: int = 100 * 1024 * 1024     # 100 MB

    # Rate limiting (OPS-5)
    RATELIMIT_ENABLED: bool = True
    RATELIMIT_CHAT_STREAM: str = "10/minute"
    RATELIMIT_CHAT: str = "30/minute"
    RATELIMIT_UPLOAD: str = "20/minute"

    # ArangoDB
    ARANGODB_HOST: str = "http://homeserver:8529"
    ARANGODB_USERNAME: str = "root"
    ARANGODB_PASSWORD: str = ""
    ARANGODB_DB_NAME: str = "clinical_ontology"

    # ── Model-as-Judge (A2A §5.2) ─────────────────────────────────────
    GROQ_API_KEY: str = ""
    JUDGE_ENABLED: bool = True
    JUDGE_SAMPLE_RATE: float = 1.0
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"
    JUDGE_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    JUDGE_MAX_INPUT_TOKENS: int = 500

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v):
        # Allow ALLOWED_ORIGINS env var to be a comma-separated string.
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def _validate_prod_secrets(self):
        # SEC-3: in production (DEBUG=False), refuse to start with default/empty secrets.
        if self.DEBUG:
            return self
        violations = []
        for key, bad in _INSECURE_DEFAULTS.items():
            value = getattr(self, key, None)
            if value in bad:
                violations.append(f"  - {key} is unset or uses an insecure default")
        # ALLOWED_ORIGINS must not include "*" in production
        if "*" in self.ALLOWED_ORIGINS:
            violations.append("  - ALLOWED_ORIGINS contains '*' (forbidden in production)")
        if violations:
            raise ValueError(
                "Refusing to start with insecure production configuration:\n"
                + "\n".join(violations)
                + "\nSet DEBUG=true for local dev or supply the missing values via .env."
            )
        return self

    # Model Configuration
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


settings = Settings()
