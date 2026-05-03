#!/usr/bin/env python
"""Detailed app import diagnostics."""
import sys
import os
import traceback
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

steps = [
    ("FastAPI", lambda: __import__("fastapi")),
    ("Pydantic", lambda: __import__("pydantic")),
    ("app.schemas", lambda: __import__("app.schemas")),
    ("app.services.agent", lambda: __import__("app.services.agent")),
    ("app.services.graph", lambda: __import__("app.services.graph")),
    ("app.api.main", lambda: __import__("app.api.main")),
]

for name, loader in steps:
    try:
        print(f"[→] Importing {name}...", file=sys.stderr, flush=True)
        loader()
        print(f"[✅] {name} OK", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[❌] {name} FAILED", file=sys.stderr, flush=True)
        print(f"Error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

print("\n✅ All imports successful!", file=sys.stderr, flush=True)
sys.exit(0)
