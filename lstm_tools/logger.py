import logging
import sys
from typing import Optional

# Create a logger for the package
logger = logging.getLogger("lstm_tools")

def configure_logging(level: int = logging.INFO, 
                     log_file: Optional[str] = None,
                     console: bool = True,
                     format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    Configure the logging for the lstm_tools package.

    Parameters
    ----------
    level : int, optional
        Logging level, by default logging.INFO
    log_file : Optional[str], optional
        Path to log file, by default None. If None, logs will only be output to console.
    console : bool, optional
        Whether to output logs to console, by default True
    format_string : str, optional
        Format string for log messages, by default "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Reset any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set the level
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Configure default logger
configure_logging()

# Helper functions for common log levels
def debug(msg: str):
    """Log a debug message."""
    logger.debug(msg)

def info(msg: str):
    """Log an info message."""
    logger.info(msg)

def warning(msg: str):
    """Log a warning message."""
    logger.warning(msg)

def error(msg: str):
    """Log an error message."""
    logger.error(msg)

def critical(msg: str):
    """Log a critical message."""
    logger.critical(msg) 