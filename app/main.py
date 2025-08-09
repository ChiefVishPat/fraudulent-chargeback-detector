"""
Main entry point for the fraud detection application.
Sets up logging and runs the FastAPI server.
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import structlog
import uvicorn

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Configure structured logging for the application."""
    
    # Create logs directory
    logs_dir = project_root / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    # Configure structlog
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    # File handler for structured logs
    log_file = logs_dir / f"fraud_detection_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    
    # Console processor
    console_processor = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FILENAME,
                          structlog.processors.CallsiteParameter.LINENO]
            ),
            console_processor,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set up file handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger = structlog.get_logger()
    logger.info("Logging configured", log_file=str(log_file))
    
    return logger

def create_app():
    """Create and configure the FastAPI application."""
    
    # Import here to ensure logging is configured first
    from app.api import app
    
    logger = structlog.get_logger()
    
    # Ensure results directories exist
    results_dirs = [
        "results/models",
        "results/validation", 
        "results/logs",
        "results/plots"
    ]
    
    for dir_path in results_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info("Created directory", path=dir_path)
    
    # Log application startup
    logger.info("Fraud detection application starting",
               project_root=str(project_root),
               python_version=sys.version,
               working_directory=os.getcwd())
    
    return app

def main():
    """Main entry point."""
    
    # Setup logging first
    logger = setup_logging()
    
    try:
        # Create the app
        app = create_app()
        
        # Configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "false").lower() == "true"
        
        logger.info("Starting fraud detection server",
                   host=host, port=port, reload=reload)
        
        # Run the server
        uvicorn.run(
            "app.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application shutting down (KeyboardInterrupt)")
    except Exception as e:
        logger.error("Application startup failed", error=str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()