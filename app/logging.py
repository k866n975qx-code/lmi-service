import logging
import os
import structlog
import sys
from pathlib import Path
from dotenv import load_dotenv

def setup_logging():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    error_log_path = os.getenv("LOG_ERROR_FILE", "").strip()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if error_log_path:
        Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(error_log_path)
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(log_level)
