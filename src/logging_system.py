"""Central logging system for the SAM 3 Segmentation Pipeline.

Provides a singleton LoggingSystem with structured JSON file output,
Rich console output, per-job correlation IDs, automatic log rotation,
and a @trace decorator for entry/exit/duration tracing on public methods.

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
_F = TypeVar("_F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Correlation-ID filter
# ---------------------------------------------------------------------------

class _CorrelationFilter(logging.Filter):
    """Injects the current thread's correlation ID into every log record."""

    _local = threading.local()

    @classmethod
    def set(cls, correlation_id: str) -> None:
        cls._local.correlation_id = correlation_id

    @classmethod
    def get(cls) -> str:
        return getattr(cls._local, "correlation_id", "global")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.correlation_id = self.get()  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emits one JSON object per log line for structured log analysis."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "global"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)  # type: ignore[arg-type]
        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LoggingSystem singleton
# ---------------------------------------------------------------------------

class LoggingSystem:
    """Singleton logging system for the SAM 3 pipeline.

    Usage::

        LoggingSystem.initialize(config)
        logger = LoggingSystem.get_logger(__name__)
        LoggingSystem.set_correlation_id("job_001")

    Config is the ``logging`` section dataclass (``LoggingConfig``), but the
    class also accepts a plain dict for flexibility in tests.
    """

    _instance: Optional["LoggingSystem"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    # Rich console — shared across all loggers
    _console: Console = Console(stderr=True)

    def __new__(cls) -> "LoggingSystem":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Public class-level API
    # ------------------------------------------------------------------

    @classmethod
    def initialize(
        cls,
        *,
        level: str = "INFO",
        log_file: Optional[str] = None,
        json_output: bool = True,
        max_file_size_mb: int = 50,
        console_rich: bool = True,
    ) -> None:
        """Configure the root logger and all handlers.

        Args:
            level: Logging level string (``"DEBUG"``, ``"INFO"``, …).
            log_file: Path for the rotating JSON log file. ``None`` disables
                file logging.
            json_output: When ``True`` the file handler emits JSON lines.
                When ``False`` plain text is emitted.
            max_file_size_mb: Max size in MB before log rotation.
            console_rich: When ``True`` use Rich console handler; when
                ``False`` use plain ``StreamHandler``.

        Raises:
            RuntimeError: If initialization fails (e.g. bad log path).
        """
        inst = cls()
        with cls._lock:
            if cls._initialized:
                return
            cls._initialized = True

        numeric_level = getattr(logging, level.upper(), logging.INFO)

        root = logging.getLogger()
        root.setLevel(numeric_level)

        # Remove any existing handlers (e.g. from pytest caplog)
        root.handlers.clear()

        corr_filter = _CorrelationFilter()

        # ---- console handler ----
        if console_rich:
            console_handler = RichHandler(
                console=cls._console,
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s [%(correlation_id)s] "
                    "%(name)s: %(message)s"
                )
            )
        console_handler.addFilter(corr_filter)
        console_handler.setLevel(numeric_level)
        root.addHandler(console_handler)

        # ---- file handler (JSON rotating) ----
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            if json_output:
                file_handler.setFormatter(_JsonFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s %(levelname)-8s [%(correlation_id)s] "
                        "%(name)s: %(message)s"
                    )
                )
            file_handler.addFilter(corr_filter)
            file_handler.setLevel(numeric_level)
            root.addHandler(file_handler)

        # ---- unhandled exception hook ----
        def _excepthook(
            exc_type: type,
            exc_value: BaseException,
            exc_tb: Any,
        ) -> None:
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            root.critical(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_tb),
            )

        sys.excepthook = _excepthook

        inst._logger = logging.getLogger("logging_system")
        inst._logger.debug("LoggingSystem initialized (level=%s)", level)

    @classmethod
    def initialize_from_config(cls, config: Any) -> None:
        """Initialize from a ``LoggingConfig`` dataclass or compatible dict.

        Args:
            config: Object with attributes ``level``, ``log_file``,
                ``json_output``, ``max_file_size_mb``, ``console_rich``;
                OR a plain dict with the same keys.
        """
        if isinstance(config, dict):
            cls.initialize(
                level=config.get("level", "INFO"),
                log_file=config.get("log_file"),
                json_output=config.get("json_output", True),
                max_file_size_mb=config.get("max_file_size_mb", 50),
                console_rich=config.get("console_rich", True),
            )
        else:
            cls.initialize(
                level=getattr(config, "level", "INFO"),
                log_file=getattr(config, "log_file", None),
                json_output=getattr(config, "json_output", True),
                max_file_size_mb=getattr(config, "max_file_size_mb", 50),
                console_rich=getattr(config, "console_rich", True),
            )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Return a named logger.

        Args:
            name: Typically ``__name__`` of the calling module.

        Returns:
            A standard :class:`logging.Logger` instance.
        """
        return logging.getLogger(name)

    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Bind a correlation ID to the current thread.

        Args:
            correlation_id: Job name or run ID attached to every log record
                emitted from this thread.
        """
        _CorrelationFilter.set(correlation_id)

    @classmethod
    def get_correlation_id(cls) -> str:
        """Return the current thread's correlation ID.

        Returns:
            The correlation ID string, or ``"global"`` if none is set.
        """
        return _CorrelationFilter.get()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only).

        Removes all handlers and clears the initialized flag so that
        :meth:`initialize` can be called again in the same process.
        """
        with cls._lock:
            root = logging.getLogger()
            for handler in root.handlers[:]:
                handler.close()
                root.removeHandler(handler)
            cls._initialized = False
            cls._instance = None


# ---------------------------------------------------------------------------
# @trace decorator
# ---------------------------------------------------------------------------

def trace(func: _F) -> _F:
    """Decorator that logs entry, exit, and wall-clock duration.

    Should be applied to every public method in pipeline modules.
    Uses the module-level logger (``logging.getLogger(func.__module__)``),
    emitting at DEBUG level to avoid noise in production logs.

    Example::

        class MyProcessor:
            @trace
            def process_image(self, path: Path) -> SegmentationResult:
                ...

    Args:
        func: The callable to wrap.

    Returns:
        Wrapped callable with the same signature.
    """
    logger = logging.getLogger(func.__module__)
    qualname = f"{func.__qualname__}"

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        logger.debug("→ %s enter", qualname)
        try:
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug("← %s exit (%.1f ms)", qualname, elapsed)
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "✗ %s raised %s after %.1f ms: %s",
                qualname,
                type(exc).__name__,
                elapsed,
                exc,
            )
            raise

    return _wrapper  # type: ignore[return-value]
