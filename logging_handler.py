"""
logging_handler.py
Custom logging handler that captures all logs for Streamlit display.
Streams logs to a list that Streamlit can display in real-time.
"""
import logging
from typing import List, Dict, Any
from datetime import datetime


class StreamlitLoggingHandler(logging.Handler):
    """Custom logging handler that stores logs for Streamlit display."""
    
    def __init__(self):
        super().__init__()
        self.logs: List[Dict[str, Any]] = []
    
    def emit(self, record: logging.LogRecord):
        """Capture log record and store it."""
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": record.module,
            }
            self.logs.append(log_entry)
        except Exception:
            self.handleError(record)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Return all collected logs."""
        return self.logs
    
    def clear_logs(self):
        """Clear all logs."""
        self.logs = []


# Global handler instance
_streamlit_handler = None


def get_streamlit_handler() -> StreamlitLoggingHandler:
    """Get or create the global Streamlit logging handler."""
    global _streamlit_handler
    if _streamlit_handler is None:
        _streamlit_handler = StreamlitLoggingHandler()
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(_streamlit_handler)
    return _streamlit_handler


def setup_logging_capture():
    """Set up logging capture for Streamlit."""
    handler = get_streamlit_handler()
    
    # Set logging level to capture everything
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure formatter
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    
    return handler
