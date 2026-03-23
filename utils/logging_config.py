"""Centralized backend logging for GraphRAG.

Writes JSON-per-line logs to output/logs with rotation.
"""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "output", "logs")
APP_LOG_PATH = os.path.join(LOG_DIR, "app.log")
DEBUG_LOG_PATH = os.path.join(LOG_DIR, "debug.log")
ERROR_LOG_PATH = os.path.join(LOG_DIR, "errors.log")


class JsonLineFormatter(logging.Formatter):
    """Render each log record as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra = getattr(record, "event_payload", None)
        if isinstance(extra, dict):
            payload.update(extra)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def get_logging_config() -> dict[str, Any]:
    return {
        "verbose": _env_bool("LOG_VERBOSE", False),
        "prompt_full": _env_bool("LOG_PROMPT_FULL", True),
        "retrieval_content_full": _env_bool("LOG_RETRIEVAL_CONTENT_FULL", True),
        "max_field_chars": int(os.getenv("LOG_MAX_FIELD_CHARS", "20000")),
    }


def setup_logging() -> None:
    """Initialize rotating JSONL file handlers (idempotent)."""
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = JsonLineFormatter()
    max_bytes = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    specs = [
        ("graphrag.app", logging.INFO, APP_LOG_PATH),
        ("graphrag.debug", logging.DEBUG, DEBUG_LOG_PATH),
        ("graphrag.error", logging.ERROR, ERROR_LOG_PATH),
    ]

    for logger_name, level, file_path in specs:
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            continue
        logger.setLevel(level)
        logger.propagate = False

        handler = RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def _trim(value: Any, max_chars: int) -> Any:
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "... [TRUNCATED]"
    if isinstance(value, list):
        return [_trim(v, max_chars) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, max_chars) for k, v in value.items()}
    return value


def log_event(
    logger_name: str,
    message: str,
    *,
    trace_id: str | None = None,
    route: str | None = None,
    stage: str | None = None,
    event: str | None = None,
    duration_ms: float | None = None,
    payload: dict[str, Any] | None = None,
    level: int = logging.INFO,
    trim_payload: bool = True,
) -> None:
    """Emit one structured event."""
    setup_logging()
    config = get_logging_config()

    data: dict[str, Any] = {
        "trace_id": trace_id,
        "route": route,
        "stage": stage,
        "event": event,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
    }
    if payload:
        data["payload"] = _trim(payload, config["max_field_chars"]) if trim_payload else payload

    # Remove nulls for cleaner JSONL output.
    data = {k: v for k, v in data.items() if v is not None}

    logger = logging.getLogger(logger_name)
    logger.log(level, message, extra={"event_payload": data})


def get_log_paths() -> dict[str, str]:
    return {
        "app": APP_LOG_PATH,
        "debug": DEBUG_LOG_PATH,
        "errors": ERROR_LOG_PATH,
    }
