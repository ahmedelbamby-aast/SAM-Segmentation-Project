"""
Utility functions for logging, helpers, and common operations.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    name: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with file and console handlers.
    
    Args:
        log_file: Path to log file (optional)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        name: Logger name (uses root logger if None)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2h 30m 15s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_size(size_bytes: int) -> str:
    """
    Format file size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string like "1.5 GB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def estimate_eta(
    processed: int, 
    total: int, 
    elapsed_seconds: float
) -> str:
    """
    Estimate time remaining.
    
    Args:
        processed: Number of items processed
        total: Total number of items
        elapsed_seconds: Time elapsed so far
        
    Returns:
        Formatted ETA string
    """
    if processed == 0:
        return "Calculating..."
    
    rate = processed / elapsed_seconds
    remaining = total - processed
    eta_seconds = remaining / rate
    
    return format_duration(eta_seconds)


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
