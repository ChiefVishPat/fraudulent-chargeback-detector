#!/usr/bin/env python3
"""
Simple startup script for the fraud detection system.
Handles environment setup for macOS compatibility.
"""
import os
import sys

# Set up environment for macOS OpenMP compatibility
if sys.platform == "darwin":  # macOS
    # Add OpenMP library path for LightGBM
    omp_paths = [
        "/opt/homebrew/opt/libomp/lib",
        "/usr/local/opt/libomp/lib",
        "/opt/homebrew/lib",
        "/usr/local/lib"
    ]
    
    for path in omp_paths:
        if os.path.exists(path):
            dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            if path not in dyld_path:
                os.environ["DYLD_LIBRARY_PATH"] = f"{path}:{dyld_path}".rstrip(":")
            break
    
    # Also set fallback paths
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:/usr/local/lib:/usr/lib"

if __name__ == "__main__":
    try:
        from app.main import main
        main()
    except ImportError as e:
        if "lightgbm" in str(e).lower():
            print("⚠️  LightGBM import error detected. Trying to fix...")
            print("💡 Run this command first: brew install libomp")
            print("🔄 Then try: uv run run.py")
            sys.exit(1)
        else:
            raise