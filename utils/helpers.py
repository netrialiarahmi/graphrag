"""Helper functions extracted from app.py for modular reuse.

Includes environment parsing, logging, and checkpoint initialization.
"""
import os
import sys
import logging
import sqlite3
from typing import Optional


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.
    
    Args:
        name: Environment variable name.
        default: Default value if not set.
    
    Returns:
        True if var is "1", "true", "yes", or "on" (case-insensitive).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def setup_file_logger(log_dir: str = None, log_filename: str = "app.log") -> logging.Logger:
    """Set up a file logger for the application.
    
    Args:
        log_dir: Directory for log files. If None, defaults to output/logs.
        log_filename: Name of the log file.
    
    Returns:
        Configured logger instance.
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "output", "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger("graphrag.backend")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(fh)
        logger.propagate = False
    
    return logger


def write_log(logger: logging.Logger, lines: list[str] = None, query: str = "", latency: float = 0.0):
    """Append agent debug logs to file logger.
    
    Args:
        logger: Logger instance from setup_file_logger().
        lines: List of log lines to record.
        query: User query (for context).
        latency: Response latency in seconds.
    """
    if not lines:
        return
    logger.info("query=%s | latency=%.1fs", query, latency)
    for line in lines:
        logger.info("  %s", line)


def setup_checkpointer(deployed: bool = False):
    """Initialize a LangGraph checkpointer appropriate for the environment.
    
    Args:
        deployed: If True, use InMemorySaver. If False, use SqliteSaver.
    
    Returns:
        Checkpointer instance or None if unavailable.
    """
    try:
        if deployed:
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()
            print("[CHECKPOINTER] ✅ InMemorySaver initialized", file=sys.stderr)
            return checkpointer
        else:
            from langgraph.checkpoint.sqlite import SqliteSaver
            checkpoint_db = os.path.join(os.path.dirname(__file__), "..", "data", "db", "checkpointer.db")
            os.makedirs(os.path.dirname(checkpoint_db), exist_ok=True)
            conn = sqlite3.connect(checkpoint_db, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            checkpointer = SqliteSaver(conn)
            print(f"[CHECKPOINTER] ✅ SqliteSaver initialized at {checkpoint_db}", file=sys.stderr)
            return checkpointer
    except ImportError:
        print("[CHECKPOINTER] ⚠️  Checkpointer library not available", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[CHECKPOINTER] ⚠️  Failed to init checkpointer: {e}", file=sys.stderr)
        return None


def detect_deployment() -> bool:
    """Detect if running in a deployed environment.
    
    Returns:
        True if running on Streamlit Cloud, Docker, or production.
    """
    return any([
        "STREAMLIT_SERVER_RUNDIR" in os.environ,
        os.environ.get("ENVIRONMENT") == "production",
        os.path.exists("/.dockerenv"),
    ])
