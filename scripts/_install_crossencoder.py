#!/usr/bin/env python3
"""Install and verify sentence-transformers."""
import subprocess, sys

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

# Install
print("\nInstalling sentence-transformers...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "sentence-transformers"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    # Try with --break-system-packages
    print("\nRetrying with --break-system-packages...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--break-system-packages", "sentence-transformers"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        sys.exit(1)

print("Install output (last 200 chars):", result.stdout[-200:])

# Verify
try:
    from sentence_transformers import CrossEncoder
    print("\n✅ CrossEncoder imported successfully!")
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)
