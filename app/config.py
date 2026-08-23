"""Application configuration (Phase 2).

Every key, URL, and tunable loads from environment variables (and .env) through
pydantic-settings. Nothing is hardcoded in the codebase: the app reads all
secrets from the environment. Field names map to the UPPER_SNAKE_CASE env var
via `validation_alias`, so the env var each setting reads is explicit here.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Vector store ---
    database_url: str = Field(
        default="postgresql://localhost:5432/qna-database",
        validation_alias="DATABASE_URL",
    )

    # --- Azure OpenAI ---
    azure_endpoint: str = Field(default="", validation_alias="AZURE_OPENAI_ENDPOINT")
    azure_api_key: str = Field(default="", validation_alias="AZURE_OPENAI_API_KEY")
    azure_api_version: str = Field(
        default="2024-10-21", validation_alias="AZURE_OPENAI_API_VERSION"
    )
    chat_deployment: str = Field(
        default="", validation_alias="AZURE_OPENAI_CHAT_DEPLOYMENT"
    )
    embedding_deployment: str = Field(
        default="", validation_alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
    embedding_dim: int = Field(default=1536, validation_alias="EMBEDDING_DIM")

    # --- Retrieval / chunking (tuned in Phase 7) ---
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, validation_alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, validation_alias="TOP_K")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
