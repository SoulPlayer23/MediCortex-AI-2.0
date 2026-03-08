from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "MediCortex Orchestrator"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/medicortex"
    
    # OpenAI
    OPENAI_API_KEY: str
    
    # MinIO
    MINIO_URL: str = "http://localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "medicortex-uploads"

    # Redis Cache Backend
    REDIS_URL: str = "redis://localhost:6379/0"

    # MedGemma API (homeserver via Tailscale)
    MEDGEMMA_API_URL: str = "http://100.107.2.102:8000/predict"

    # ── Model-as-Judge (A2A §5.2) ─────────────────────────────────────
    GROQ_API_KEY: str = ""
    JUDGE_ENABLED: bool = True
    JUDGE_SAMPLE_RATE: float = 1.0        # 0.0–1.0; 1.0 = judge every request
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"
    JUDGE_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    JUDGE_MAX_INPUT_TOKENS: int = 500     # truncate aggregated response before sending

    # Model Configuration
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")

settings = Settings()
