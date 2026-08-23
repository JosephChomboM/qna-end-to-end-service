"""Langfuse wiring (Phase 8) — live tracing of the running service.

The PRODUCTION stage of the three-way distinction:

    RAG eval (Phase 4)  -> pass/fail gate on a fixed question set
    MLflow   (Phase 7)  -> offline comparison of runs
    Langfuse (Phase 8)  -> live request monitoring                   [this file]

Same faithfulness/retrieval concerns as Phases 4 and 7, but here they're observed
on real, live requests instead of a frozen set. Tracing is entirely optional: if
the Langfuse keys are absent, `init_langfuse` returns None and the API runs
exactly as before.
"""

from __future__ import annotations

import os

from app.config import Settings


def langfuse_configured(s: Settings) -> bool:
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def init_langfuse(s: Settings):
    """Return a configured Langfuse client, or None if not configured.

    Also exports the keys to the process environment so the `langfuse.openai`
    drop-in wrapper (used by the instrumented client) shares the same config.
    """
    if not langfuse_configured(s):
        return None

    os.environ["LANGFUSE_PUBLIC_KEY"] = s.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = s.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = s.langfuse_host

    from langfuse import Langfuse

    return Langfuse()
