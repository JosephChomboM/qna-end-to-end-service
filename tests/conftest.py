"""Shared pytest fixtures.

Tests are layered by what infrastructure they need:
  - pure unit tests (chunking): no DB, no network.
  - `db` tests (retrieval): a running PostgreSQL + pgvector, but NO LLM API
    (embeddings are faked deterministically).
  - `eval` tests: real Azure OpenAI (embeddings + chat).
Missing infrastructure -> the relevant tests skip rather than error.
"""

from __future__ import annotations

import psycopg
import pytest

from app.config import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def db_available(settings) -> bool:
    """True if we can connect to Postgres and the vector extension is present."""
    try:
        with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def azure_configured(settings) -> bool:
    return all(
        [
            settings.azure_endpoint,
            settings.azure_api_key,
            settings.chat_deployment,
            settings.embedding_deployment,
            "<" not in settings.azure_endpoint,  # not the placeholder
        ]
    )
