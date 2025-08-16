#!/usr/bin/env python3
"""
Fraud Detection System - Main Entry Point
Run with: uv run main.py
"""
import sys
from pathlib import Path

# Ensure the project root is in Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the fraud detection system."""
    import uvicorn
    import os
    
    # Import the FastAPI app (this will work with absolute imports now)
    from app.api import app
    
    # Configuration
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print("🚀 Starting Fraud Detection System...")
    print("=" * 50)
    print(f"🌐 Server: http://localhost:{port}")
    print(f"📱 Web Interface: http://localhost:{port}")
    print(f"📚 API Documentation: http://localhost:{port}/docs")
    print(f"💾 Interactive API: http://localhost:{port}/redoc")
    print("=" * 50)
    print()
    
    try:
        uvicorn.run(app, host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()