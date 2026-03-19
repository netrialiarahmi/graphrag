"""LangSmith tracing & observability configuration.
All features degrade gracefully when LANGSMITH_API_KEY is not set."""

import os


def init_langsmith():
    """Activate LangSmith tracing if API key is present.
    Safe to call multiple times; no-op when key is missing."""
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if not api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "graphrag-legal"))
    return True


def get_traceable():
    """Return the @traceable decorator if langsmith is available, else a no-op."""
    try:
        from langsmith import traceable
        if os.getenv("LANGSMITH_API_KEY"):
            return traceable
    except ImportError:
        pass
    # Return identity decorator
    def _noop(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f
    return _noop
