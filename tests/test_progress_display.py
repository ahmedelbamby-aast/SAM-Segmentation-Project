"""Unit tests for src/progress_display.py.

Tests cover:
- StageProgress dataclass properties
- ModuleProgressManager lifecycle (start/stop/context-manager)
- Stage registration and management
- ProgressCallback protocol compliance
- Stage-specific callback methods
- Stats reporting (get_stage_stats, get_all_stats, reset_stats)
- Thread safety (concurrent on_item_complete calls)
- Graceful degradation when Rich is unavailable

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""

from __future__ import annotations

import sys
import time
from threading import Thread
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.progress_display import (
    PIPELINE_STAGES,
    ModuleProgressManager,
    StageProgress,
)
from src.interfaces import ProgressCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _running_manager(stage: str = "Segment", total: int = 10) -> ModuleProgressManager:
    """Return a started manager with one active stage."""
    mgr = ModuleProgressManager()
    mgr.start()
    mgr.start_stage(stage, total=total)
    return mgr


# ---------------------------------------------------------------------------
# StageProgress
# ---------------------------------------------------------------------------


class TestStageProgress:
    """Tests for the StageProgress dataclass."""

    def test_elapsed_is_non_negative(self) -> None:
        sp = StageProgress(stage_name="Segment", total=100)
        assert sp.elapsed >= 0.0

    def test_throughput_zero_before_completions(self) -> None:
        sp = StageProgress(stage_name="Segment", total=100)
        assert sp.throughput == 0.0

    def test_throughput_computed_after_completions(self) -> None:
        sp = StageProgress(stage_name="Segment", total=100)
        sp.start_time = time.monotonic() - 10.0  # pretend 10s elapsed
        sp.completed = 20
        thr = sp.throughput
        assert 1.9 < thr < 2.1  # ~2.0 img/s

    def test_eta_str_before_completions(self) -> None:
        sp = StageProgress(stage_name="Segment", total=100)
        # No completions — should return dash sentinel
        assert sp.eta_str == "—"

    def test_eta_str_after_completions(self) -> None:
        sp = StageProgress(stage_name="Segment", total=100)
        sp.start_time = time.monotonic() - 5.0
        sp.completed = 50
        eta = sp.eta_str
        assert isinstance(eta, str)
        assert len(eta) > 0

    def test_defaults(self) -> None:
        sp = StageProgress(stage_name="NMS")
        assert sp.total == 0
        assert sp.completed == 0
        assert sp.errors == 0
        assert sp.task_id is None
        assert sp.active is False


# ---------------------------------------------------------------------------
# PIPELINE_STAGES
# ---------------------------------------------------------------------------


class TestPipelineStages:
    """Tests for the PIPELINE_STAGES constant."""

    def test_contains_all_nine_stages(self) -> None:
        expected = {
            "Scan", "Preprocess", "Segment", "Remap", "NMS",
            "Filter", "Annotate", "Upload", "Validate",
        }
        assert expected == set(PIPELINE_STAGES)

    def test_stages_are_ordered(self) -> None:
        assert PIPELINE_STAGES[0] == "Scan"
        assert PIPELINE_STAGES[-1] == "Validate"


# ---------------------------------------------------------------------------
# ModuleProgressManager — lifecycle
# ---------------------------------------------------------------------------


class TestModuleProgressManagerLifecycle:
    """Tests for the start/stop/context-manager lifecycle."""

    def test_start_stop_does_not_raise(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.stop()

    def test_stop_idempotent(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.stop()
        mgr.stop()  # second stop must not raise

    def test_context_manager_stops_on_exit(self) -> None:
        with ModuleProgressManager() as mgr:
            assert mgr._running is True
        assert mgr._running is False

    def test_context_manager_returns_manager_instance(self) -> None:
        with ModuleProgressManager() as mgr:
            assert isinstance(mgr, ModuleProgressManager)

    def test_start_sets_running_flag(self) -> None:
        mgr = ModuleProgressManager()
        assert not mgr._running
        mgr.start()
        assert mgr._running
        mgr.stop()

    def test_graceful_when_rich_unavailable(self) -> None:
        """Manager must not raise if Rich is not installed."""
        with patch("src.progress_display._RICH_AVAILABLE", False):
            mgr = ModuleProgressManager()
            mgr.start()
            mgr.start_stage("Segment", total=5)
            mgr.on_item_start("img_001")
            mgr.on_item_complete("img_001")
            mgr.on_item_error("img_002", RuntimeError("test"))
            mgr.finish_stage("Segment")
            mgr.stop()


# ---------------------------------------------------------------------------
# ModuleProgressManager — stage management
# ---------------------------------------------------------------------------


class TestModuleProgressManagerStages:
    """Tests for start_stage / finish_stage."""

    def test_start_stage_registers_stage(self) -> None:
        mgr = _running_manager("NMS", total=50)
        assert "NMS" in mgr._stages
        mgr.stop()

    def test_start_stage_sets_total(self) -> None:
        mgr = _running_manager("Segment", total=123)
        assert mgr._stages["Segment"].total == 123
        mgr.stop()

    def test_start_unknown_stage_raises(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        with pytest.raises(ValueError, match="Unknown stage"):
            mgr.start_stage("UnknownStage", total=10)
        mgr.stop()

    def test_finish_stage_marks_inactive(self) -> None:
        mgr = _running_manager("Filter", total=10)
        mgr.finish_stage("Filter")
        assert mgr._stages["Filter"].active is False
        mgr.stop()

    def test_finish_unknown_stage_does_not_raise(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.finish_stage("NonExistentStage")  # must not raise
        mgr.stop()

    def test_multiple_stages_coexist(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        for stage in ("Scan", "Segment", "NMS"):
            mgr.start_stage(stage, total=100)
        assert set(mgr._stages.keys()) == {"Scan", "Segment", "NMS"}
        mgr.stop()

    def test_all_nine_stages_can_be_registered(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        for stage in PIPELINE_STAGES:
            mgr.start_stage(stage, total=10)
        assert len(mgr._stages) == 9
        mgr.stop()


# ---------------------------------------------------------------------------
# ModuleProgressManager — ProgressCallback protocol
# ---------------------------------------------------------------------------


class TestProgressCallbackProtocol:
    """Tests for on_item_start / on_item_complete / on_item_error."""

    def test_implements_progress_callback_protocol(self) -> None:
        """ModuleProgressManager must satisfy the ProgressCallback protocol."""
        mgr = ModuleProgressManager()
        # Runtime check via isinstance (ProgressCallback is @runtime_checkable)
        assert isinstance(mgr, ProgressCallback)

    def test_on_item_complete_increments_count(self) -> None:
        mgr = _running_manager("Segment", total=10)
        mgr.on_item_complete("img_001")
        assert mgr._stages["Segment"].completed == 1
        mgr.stop()

    def test_multiple_on_item_complete_accumulates(self) -> None:
        mgr = _running_manager("Segment", total=10)
        for i in range(5):
            mgr.on_item_complete(f"img_{i:03d}")
        assert mgr._stages["Segment"].completed == 5
        mgr.stop()

    def test_on_item_error_increments_error_count(self) -> None:
        mgr = _running_manager("Segment", total=10)
        mgr.on_item_error("img_001", RuntimeError("oops"))
        assert mgr._stages["Segment"].errors == 1
        mgr.stop()

    def test_on_item_start_does_not_change_counts(self) -> None:
        mgr = _running_manager("Segment", total=10)
        mgr.on_item_start("img_001")
        stage = mgr._stages["Segment"]
        assert stage.completed == 0
        assert stage.errors == 0
        mgr.stop()

    def test_on_item_complete_without_stage_does_not_raise(self) -> None:
        """Callbacks with no active stage must be silently ignored."""
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.on_item_complete("img_001")  # no stage registered
        mgr.stop()

    def test_active_stage_filter(self) -> None:
        """With active_stage set, only that stage receives events."""
        mgr = ModuleProgressManager(active_stage="NMS")
        mgr.start()
        mgr.start_stage("NMS", total=5)
        mgr.start_stage("Filter", total=5)
        mgr.on_item_complete("img_001")
        assert mgr._stages["NMS"].completed == 1
        assert mgr._stages["Filter"].completed == 0
        mgr.stop()


# ---------------------------------------------------------------------------
# Stage-specific callbacks
# ---------------------------------------------------------------------------


class TestStageSpecificCallbacks:
    """Tests for on_stage_item_complete / on_stage_item_error."""

    def test_on_stage_item_complete_targets_correct_stage(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.start_stage("NMS", total=5)
        mgr.start_stage("Filter", total=5)
        mgr.on_stage_item_complete("NMS", "img_001")
        assert mgr._stages["NMS"].completed == 1
        assert mgr._stages["Filter"].completed == 0
        mgr.stop()

    def test_on_stage_item_error_targets_correct_stage(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.start_stage("NMS", total=5)
        mgr.start_stage("Filter", total=5)
        mgr.on_stage_item_error("Filter", "img_001", RuntimeError("err"))
        assert mgr._stages["Filter"].errors == 1
        assert mgr._stages["NMS"].errors == 0
        mgr.stop()

    def test_on_stage_item_complete_unknown_stage_silent(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.on_stage_item_complete("UnknownStage", "img_001")  # must not raise
        mgr.stop()


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------


class TestStatsReporting:
    """Tests for get_stage_stats / get_all_stats / reset_stats."""

    def test_get_stage_stats_returns_none_for_unknown_stage(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        assert mgr.get_stage_stats("NotHere") is None
        mgr.stop()

    def test_get_stage_stats_returns_dict(self) -> None:
        mgr = _running_manager("Segment", total=100)
        stats = mgr.get_stage_stats("Segment")
        assert stats is not None
        assert stats["stage"] == "Segment"
        assert stats["total"] == 100
        mgr.stop()

    def test_get_stage_stats_keys(self) -> None:
        mgr = _running_manager("Segment", total=100)
        stats = mgr.get_stage_stats("Segment")
        for key in ("stage", "total", "completed", "errors", "elapsed", "throughput", "eta"):
            assert key in stats
        mgr.stop()

    def test_get_stage_stats_reflects_completions(self) -> None:
        mgr = _running_manager("Segment", total=100)
        for _ in range(7):
            mgr.on_item_complete("x")
        stats = mgr.get_stage_stats("Segment")
        assert stats["completed"] == 7
        mgr.stop()

    def test_get_all_stats_returns_all_registered_stages(self) -> None:
        mgr = ModuleProgressManager()
        mgr.start()
        mgr.start_stage("Scan", total=10)
        mgr.start_stage("NMS", total=10)
        all_stats = mgr.get_all_stats()
        assert set(all_stats.keys()) == {"Scan", "NMS"}
        mgr.stop()

    def test_reset_stats_clears_all_stages(self) -> None:
        mgr = _running_manager("Segment", total=100)
        mgr.reset_stats()
        assert len(mgr._stages) == 0
        mgr.stop()

    def test_reset_stats_does_not_stop_display(self) -> None:
        mgr = _running_manager("Segment", total=100)
        mgr.reset_stats()
        assert mgr._running is True
        mgr.stop()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for concurrent access to ModuleProgressManager."""

    def test_concurrent_on_item_complete_is_safe(self) -> None:
        """1000 concurrent increments must produce the correct total."""
        mgr = _running_manager("Segment", total=1000)
        total_calls = 1000

        def worker(n: int) -> None:
            for i in range(n):
                mgr.on_item_complete(f"img_{i:04d}")

        threads = [Thread(target=worker, args=(250,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mgr._stages["Segment"].completed == total_calls
        mgr.stop()

    def test_concurrent_start_stage_does_not_raise(self) -> None:
        """Registering stages from multiple threads must not corrupt state."""
        mgr = ModuleProgressManager()
        mgr.start()

        errors: List[Exception] = []

        def register(name: str) -> None:
            try:
                mgr.start_stage(name, total=10)
            except Exception as exc:
                errors.append(exc)

        stages = ["Scan", "Segment", "NMS", "Filter"]
        threads = [Thread(target=register, args=(s,)) for s in stages]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        mgr.stop()
