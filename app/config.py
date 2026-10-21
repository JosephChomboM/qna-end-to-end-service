"""Application configuration (Phase 2).

Every key, URL, and tunable loads from environment variables (and .env) through
pydantic-settings. Nothing is hardcoded in the codebase: the app reads all
secrets from the environment. Field names map to the UPPER_SNAKE_CASE env var
via `validation_alias`, so the env var each setting reads is explicit here.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
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

    # --- Public-demo guards ---
    # Per-IP and global request caps on /query keep the LLM quota from being
    # drained by one visitor (or a bot). In-memory, so they reset on restart.
    rate_limit_per_minute: int = Field(default=10, validation_alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_day: int = Field(default=300, validation_alias="RATE_LIMIT_PER_DAY")
    # If set, POST /ingest requires this value in the X-Admin-Token header
    # (ingest drops and rebuilds the table, so it must not be open on a public URL).
    admin_token: str = Field(default="", validation_alias="ADMIN_TOKEN")

    # --- Retrieval / chunking (tuned in Phase 7) ---
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, validation_alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, validation_alias="TOP_K")

    # --- Langfuse (Phase 8) — live tracing of /query. Optional: if the keys are
    # empty, tracing is disabled and the service behaves exactly as before. ---
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        # Accept either LANGFUSE_HOST (the SDK's own name) or LANGFUSE_BASE_URL.
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
