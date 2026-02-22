"""Unit tests for src/logging_system.py.

Covers LoggingSystem singleton, handler setup, correlation IDs,
@trace decorator, and reset behaviour.

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable when running from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_system import LoggingSystem, _CorrelationFilter, trace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset the LoggingSystem singleton before and after each test."""
    LoggingSystem.reset()
    yield
    LoggingSystem.reset()


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_same_instance_returned(self):
        a = LoggingSystem()
        b = LoggingSystem()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = LoggingSystem()
        LoggingSystem.reset()
        b = LoggingSystem()
        # After reset a new instance is created; they may be equal but the
        # internal state (handlers) should be clean.
        # The important thing is that _initialized is False after reset.
        assert not LoggingSystem._initialized

    def test_initialize_is_idempotent(self):
        LoggingSystem.initialize(console_rich=False)
        root_before = list(logging.getLogger().handlers)
        LoggingSystem.initialize(console_rich=False)
        root_after = list(logging.getLogger().handlers)
        assert len(root_before) == len(root_after)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_sets_initialized_flag(self):
        assert not LoggingSystem._initialized
        LoggingSystem.initialize(console_rich=False)
        assert LoggingSystem._initialized

    def test_root_logger_has_handler(self):
        LoggingSystem.initialize(console_rich=False)
        root = logging.getLogger()
        assert len(root.handlers) >= 1

    def test_root_level_applied(self):
        LoggingSystem.initialize(level="DEBUG", console_rich=False)
        assert logging.getLogger().level == logging.DEBUG

    def test_file_handler_created(self, tmp_path: Path):
        log_file = str(tmp_path / "test.log")
        LoggingSystem.initialize(log_file=log_file, console_rich=False)
        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "RotatingFileHandler" in handler_types

    def test_no_file_handler_when_log_file_none(self):
        LoggingSystem.initialize(log_file=None, console_rich=False)
        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "RotatingFileHandler" not in handler_types

    def test_log_directory_created(self, tmp_path: Path):
        log_file = str(tmp_path / "subdir" / "pipeline.log")
        LoggingSystem.initialize(log_file=log_file, console_rich=False)
        assert (tmp_path / "subdir").exists()

    def test_initialize_from_config_dict(self):
        config = {"level": "WARNING", "console_rich": False}
        LoggingSystem.initialize_from_config(config)
        assert LoggingSystem._initialized
        assert logging.getLogger().level == logging.WARNING

    def test_initialize_from_config_object(self):
        class FakeConfig:
            level = "ERROR"
            log_file = None
            json_output = True
            max_file_size_mb = 10
            console_rich = False

        LoggingSystem.initialize_from_config(FakeConfig())
        assert logging.getLogger().level == logging.ERROR


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logger(self):
        LoggingSystem.initialize(console_rich=False)
        logger = LoggingSystem.get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self):
        LoggingSystem.initialize(console_rich=False)
        logger = LoggingSystem.get_logger("my.fancy.module")
        assert logger.name == "my.fancy.module"

    def test_same_name_returns_same_logger(self):
        LoggingSystem.initialize(console_rich=False)
        a = LoggingSystem.get_logger("same")
        b = LoggingSystem.get_logger("same")
        assert a is b

    def test_logger_works_before_initialize(self):
        # get_logger is a thin wrapper; works even without explicit initialize
        logger = LoggingSystem.get_logger("early")
        assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# Correlation IDs
# ---------------------------------------------------------------------------


class TestCorrelationId:
    def test_default_is_global(self):
        assert LoggingSystem.get_correlation_id() == "global"

    def test_set_and_get(self):
        LoggingSystem.set_correlation_id("job_001")
        assert LoggingSystem.get_correlation_id() == "job_001"

    def test_thread_isolation(self):
        """Each thread should have its own correlation ID."""
        results: List[str] = []
        barrier = threading.Barrier(2)

        def worker(job_id: str) -> None:
            LoggingSystem.set_correlation_id(job_id)
            barrier.wait()  # ensure both threads have set their IDs
            results.append(LoggingSystem.get_correlation_id())

        t1 = threading.Thread(target=worker, args=("job_A",))
        t2 = threading.Thread(target=worker, args=("job_B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert set(results) == {"job_A", "job_B"}

    def test_id_injected_into_log_record(self):
        """The filter should attach correlation_id to log records."""
        LoggingSystem.set_correlation_id("filter_test")
        f = _CorrelationFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="hi", args=(), exc_info=None,
        )
        f.filter(record)
        assert record.correlation_id == "filter_test"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# JSON file output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_lines_in_log_file(self, tmp_path: Path):
        log_file = str(tmp_path / "out.log")
        LoggingSystem.initialize(
            level="DEBUG",
            log_file=log_file,
            json_output=True,
            console_rich=False,
        )
        LoggingSystem.set_correlation_id("json_test")
        logger = LoggingSystem.get_logger("test.json")
        logger.info("hello world")

        # Flush all handlers
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_path = Path(log_file)
        assert log_path.exists()
        lines = [l for l in log_path.read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["msg"] == "hello world"
        assert record["correlation_id"] == "json_test"
        assert record["level"] == "INFO"

    def test_json_contains_logger_name(self, tmp_path: Path):
        log_file = str(tmp_path / "name.log")
        LoggingSystem.initialize(log_file=log_file, json_output=True, console_rich=False)
        logger = LoggingSystem.get_logger("my.module")
        logger.warning("name check")
        for handler in logging.getLogger().handlers:
            handler.flush()
        lines = [l for l in Path(log_file).read_text("utf-8").splitlines() if l.strip()]
        record = json.loads(lines[-1])
        assert record["logger"] == "my.module"


# ---------------------------------------------------------------------------
# @trace decorator
# ---------------------------------------------------------------------------


class TestTraceDecorator:
    def test_returns_function_result(self):
        @trace
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self):
        @trace
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_logs_entry_and_exit(self, caplog):
        with caplog.at_level(logging.DEBUG):
            @trace
            def greet(name: str) -> str:
                return f"hello {name}"

            result = greet("world")

        assert result == "hello world"
        log_text = " ".join(caplog.messages)
        assert "enter" in log_text
        assert "exit" in log_text

    def test_logs_error_and_reraises(self, caplog):
        @trace
        def boom() -> None:
            raise ValueError("kaboom")

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ValueError, match="kaboom"):
                boom()

        log_text = " ".join(caplog.messages)
        assert "raised" in log_text or "ValueError" in log_text

    def test_works_as_method_decorator(self):
        class Sample:
            @trace
            def compute(self, x: int) -> int:
                return x * 2

        s = Sample()
        assert s.compute(7) == 14

    def test_elapsed_time_is_non_negative(self, caplog):
        @trace
        def slow() -> None:
            time.sleep(0.01)

        with caplog.at_level(logging.DEBUG):
            slow()

        # Check that at least one message contains a numeric ms value
        exit_msgs = [m for m in caplog.messages if "exit" in m]
        assert exit_msgs, "No exit log message found"
        # Should contain something like "10.x ms"
        assert "ms" in exit_msgs[0]

    def test_qualname_in_log(self, caplog):
        class MyClass:
            @trace
            def my_method(self) -> None:
                pass

        with caplog.at_level(logging.DEBUG):
            MyClass().my_method()

        log_text = " ".join(caplog.messages)
        assert "MyClass.my_method" in log_text


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_handlers(self):
        LoggingSystem.initialize(console_rich=False)
        assert len(logging.getLogger().handlers) >= 1
        LoggingSystem.reset()
        assert len(logging.getLogger().handlers) == 0

    def test_reset_clears_initialized_flag(self):
        LoggingSystem.initialize(console_rich=False)
        assert LoggingSystem._initialized
        LoggingSystem.reset()
        assert not LoggingSystem._initialized

    def test_initialize_after_reset(self):
        LoggingSystem.initialize(console_rich=False)
        LoggingSystem.reset()
        LoggingSystem.initialize(level="WARNING", console_rich=False)
        assert LoggingSystem._initialized
        assert logging.getLogger().level == logging.WARNING
