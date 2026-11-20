"""
Structured logging configuration.

Why structured (JSON) logs instead of plain text:
    In Phase 16 (observability) we'll want to answer questions like "what was
    the p95 retrieval latency for user X in the last hour?" That's only
    possible if logs are machine-parseable from day one. Retrofitting
    structured logging onto a codebase full of `print()` and free-text
    `logger.info("did the thing")` calls is painful — so we set the
    convention now, even though nothing sophisticated consumes these logs yet.

We use Python's standard `logging` module (no extra dependency yet) with a
JSON formatter. When Langfuse/OpenTelemetry are added in Phase 16, this is
the module that gets extended, not replaced.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Allow callers to attach structured context:
        #   logger.info("chunk_retrieved", extra={"context": {"chunk_id": ..., "score": ...}})
        if hasattr(record, "context"):
            payload["context"] = record.context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    # Avoid duplicate handlers on reload (uvicorn --reload re-imports modules)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
