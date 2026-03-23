"""
Structured debug logger for GraphRAG pipeline.
Writes JSON-lines to chatbot/debug.log for every LLM prompt/response.
"""
import json
import os
import uuid
from datetime import datetime, timezone

_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chatbot"))
_LOG_PATH = os.path.join(_LOG_DIR, "debug.log")
os.makedirs(_LOG_DIR, exist_ok=True)


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
