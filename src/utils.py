"""
Utility functions: duration formatting, size formatting, ETA estimation,
timestamps, and directory helpers.

Note: Logging is handled globally by :class:`~src.logging_system.LoggingSystem`.
This module no longer provides a ``setup_logging`` shim — callers should use
``LoggingSystem.get_logger(__name__)`` directly.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
from pathlib import Path
from datetime import datetime

from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


@trace
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


@trace
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


@trace
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


@trace
def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@trace
def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
