#!/usr/bin/env python
"""Quick test to verify FastAPI app imports and basic setup."""
import sys
import os

# Suppress numpy warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    print("[TEST] Importing FastAPI app...", file=sys.stderr)
    from app.api.main import app
    print("✅ FastAPI app imported successfully", file=sys.stderr)
    
    print("[TEST] Checking app routes...", file=sys.stderr)
    routes = [route.path for route in app.routes]
    print(f"✅ Routes available: {routes}", file=sys.stderr)
    
    print("[TEST] All checks passed!", file=sys.stderr)
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
