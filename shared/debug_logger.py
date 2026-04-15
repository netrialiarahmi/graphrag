"""
Structured debug logger for GraphRAG pipeline.
Writes JSON-lines to output/logs/debug.log for verbose tracing.
"""
import json
import os
import uuid
from datetime import datetime, timezone

_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "logs"))
_LOG_PATH = os.path.join(_LOG_DIR, "debug.log")
os.makedirs(_LOG_DIR, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_verbose_debug_enabled() -> bool:
    return _env_bool("GRAPHRAG_VERBOSE_DEBUG", False)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def get_log_path() -> str:
    return _LOG_PATH


def log_event(
    *,
    trace_id: str,
    route: str,
    stage: str,
    event: str,
    message: str,
    payload: dict | None = None,
):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "INFO",
        "logger": "graphrag.debug",
        "message": message,
        "trace_id": trace_id,
        "route": route,
        "stage": stage,
        "event": event,
        "payload": payload or {},
    }
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_verbose_event(
    *,
    route: str,
    stage: str,
    event: str,
    message: str,
    payload: dict | None = None,
    trace_id: str = "",
):
    """Write one structured event only when verbose debug mode is enabled."""
    if not is_verbose_debug_enabled():
        return
    log_event(
        trace_id=trace_id or new_trace_id(),
        route=route,
        stage=stage,
        event=event,
        message=message,
        payload=payload,
    )
