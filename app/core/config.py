"""
DocAI Configuration — centralized settings loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # --- API Keys ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # --- LLM ---
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    # --- OCR ---
    ocr_engine: str = Field(default="tesseract", alias="OCR_ENGINE")
    ocr_confidence_threshold: float = Field(default=0.7, alias="OCR_CONFIDENCE_THRESHOLD")

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="docai_documents", alias="CHROMA_COLLECTION_NAME")

    # --- MLflow ---
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="docai-extraction", alias="MLFLOW_EXPERIMENT_NAME")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")

    # --- Monitoring ---
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    accuracy_threshold: float = Field(default=0.85, alias="ACCURACY_THRESHOLD")

    # --- Database ---
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/docai.db", alias="DATABASE_URL"
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
