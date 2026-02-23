"""Module-level Rich progress display for the SAM 3 Segmentation Pipeline.

Provides :class:`ModuleProgressManager`, an ephemeral (in-memory only) progress
display built on :mod:`rich.progress`.  It implements the
:class:`~src.interfaces.ProgressCallback` protocol so it can receive
``on_item_start`` / ``on_item_complete`` / ``on_item_error`` events from any
pipeline stage.

Durable persistence is handled separately by
:class:`~src.progress_tracker.ProgressTracker` (SQLite).  This module is
display-only and carries no I/O side effects.

**Stage names** (canonical, used as keys throughout):
``"Scan"``, ``"Preprocess"``, ``"Segment"``, ``"Remap"``, ``"NMS"``,
``"Filter"``, ``"Annotate"``, ``"Upload"``, ``"Validate"``.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional

from .interfaces import ProgressCallback  # protocol import
from .logging_system import LoggingSystem, trace
from .utils import estimate_eta

try:
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

_logger = LoggingSystem.get_logger(__name__)

# Ordered list of all pipeline stages — used to build the bar group.
PIPELINE_STAGES = [
    "Scan",
    "Preprocess",
    "Segment",
    "Remap",
    "NMS",
    "Filter",
    "Annotate",
    "Upload",
    "Validate",
]


# ---------------------------------------------------------------------------
# Internal state dataclass
# ---------------------------------------------------------------------------


@dataclass
class StageProgress:
    """Tracks live progress statistics for a single pipeline stage.

    Args:
        stage_name: Human-readable stage label (e.g. ``"Segment"``).
        total: Expected total number of items in this stage.
    """

    stage_name: str
    total: int = 0
    completed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.monotonic)
    task_id: Optional["TaskID"] = None  # Rich task ID, set after start_stage()
    active: bool = False

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds since this stage was started."""
        return time.monotonic() - self.start_time

    @property
    def throughput(self) -> float:
        """Items per second (0 if not enough data yet)."""
        elapsed = self.elapsed
        if elapsed < 0.01 or self.completed == 0:
            return 0.0
        return self.completed / elapsed

    @property
    def eta_str(self) -> str:
        """Human-readable estimated time remaining for this stage."""
        elapsed = self.elapsed
        if elapsed < 0.01 or self.completed == 0:
            return "—"
        return estimate_eta(self.completed, self.total, elapsed)


# ---------------------------------------------------------------------------
# ModuleProgressManager
# ---------------------------------------------------------------------------


class ModuleProgressManager:
    """Ephemeral Rich progress display implementing :class:`ProgressCallback`.

    Manages a group of per-stage :class:`rich.progress.Progress` bars, one
    per pipeline stage.  Each bar shows:

    ``[Stage] ████████░░ 156/400 images | 12.3 img/s | ETA 00:19``

    Usage example (single-stage)::

        mgr = ModuleProgressManager()
        mgr.start()
        mgr.start_stage("Segment", total=400)
        for img in images:
            mgr.on_item_start(img.stem)
            process(img)
            mgr.on_item_complete(img.stem)
        mgr.finish_stage("Segment")
        mgr.stop()

    Usage example (context manager)::

        with ModuleProgressManager() as mgr:
            mgr.start_stage("NMS", total=200)
            …

    The manager is thread-safe: a single instance can receive events from
    multiple worker threads simultaneously.

    Args:
        active_stage: If provided, only progress for this stage is tracked
            and all others are hidden (useful when running a standalone CLI
            stage command).  Pass ``None`` to show all registered stages.
        show_gpu_memory: Whether to include GPU memory usage in the display.
            Requires :mod:`torch`.
    """

    def __init__(
        self,
        active_stage: Optional[str] = None,
        show_gpu_memory: bool = True,
    ) -> None:
        self._active_stage = active_stage
        self._show_gpu_memory = show_gpu_memory
        self._lock = Lock()
        self._stages: Dict[str, StageProgress] = {}
        self._rich_progress: Optional["Progress"] = None
        self._live: Optional["Live"] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @trace
    def start(self) -> None:
        """Start the Rich :class:`~rich.live.Live` display."""
        if not _RICH_AVAILABLE:
            _logger.warning("Rich not installed — progress bars disabled")
            self._running = True
            return
        self._rich_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.fields[stage]:10}[/bold cyan]"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[throughput]}[/green]"),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            TextColumn("[red]Err: {task.fields[errors]}[/red]"),
            expand=False,
        )
        self._live = Live(self._rich_progress, refresh_per_second=10, transient=False)
        self._live.start()
        self._running = True
        _logger.info("ModuleProgressManager started")

    @trace
    def stop(self) -> None:
        """Stop the Rich :class:`~rich.live.Live` display."""
        if not self._running:
            return
        self._running = False
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.stop()
            self._live = None
        self._rich_progress = None
        _logger.info("ModuleProgressManager stopped")

    def __enter__(self) -> "ModuleProgressManager":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    @trace
    def start_stage(self, stage_name: str, total: int) -> None:
        """Register and start a new progress bar for *stage_name*.

        Args:
            stage_name: One of :data:`PIPELINE_STAGES` (case-sensitive).
            total: Expected item count for this stage.

        Raises:
            ValueError: If *stage_name* is not in :data:`PIPELINE_STAGES`.
        """
        if stage_name not in PIPELINE_STAGES:
            raise ValueError(
                f"Unknown stage {stage_name!r}.  Valid stages: {PIPELINE_STAGES}"
            )
        with self._lock:
            sp = StageProgress(stage_name=stage_name, total=total, active=True)
            self._stages[stage_name] = sp
            if _RICH_AVAILABLE and self._rich_progress is not None:
                sp.task_id = self._rich_progress.add_task(
                    stage_name,
                    total=total,
                    stage=f"[{stage_name}]",
                    throughput="—",
                    errors=0,
                )
        _logger.info("Stage started: %s (total=%d)", stage_name, total)

    @trace
    def finish_stage(self, stage_name: str) -> None:
        """Mark *stage_name* as finished and log final stats.

        Args:
            stage_name: The stage to finish.
        """
        with self._lock:
            sp = self._stages.get(stage_name)
            if sp is None:
                _logger.warning("finish_stage called for unknown stage: %s", stage_name)
                return
            sp.active = False
            if _RICH_AVAILABLE and self._rich_progress is not None and sp.task_id is not None:
                self._rich_progress.update(
                    sp.task_id,
                    completed=sp.total,
                    throughput=f"{sp.throughput:.1f} img/s",
                    errors=sp.errors,
                )
        _logger.info(
            "Stage finished: %s — %d/%d completed, %d errors, %.1f img/s",
            stage_name, sp.completed, sp.total, sp.errors, sp.throughput,
        )

    # ------------------------------------------------------------------
    # ProgressCallback protocol
    # ------------------------------------------------------------------

    @trace
    def on_item_start(self, item_id: str) -> None:
        """Signal that *item_id* has begun processing in the active stage.

        This method is part of the :class:`~src.interfaces.ProgressCallback` protocol.

        Args:
            item_id: Unique item identifier (e.g. image path stem).
        """
        # Nothing to update on start — stage is already shown
        pass

    @trace
    def on_item_complete(self, item_id: str) -> None:
        """Signal that *item_id* finished processing successfully.

        Args:
            item_id: Unique item identifier.
        """
        stage_name = self._active_stage or self._last_active_stage()
        if stage_name is None:
            return
        with self._lock:
            sp = self._stages.get(stage_name)
            if sp is None:
                return
            sp.completed += 1
            self._refresh_task(sp)

    @trace
    def on_item_error(self, item_id: str, error: Exception) -> None:
        """Signal that *item_id* raised *error* during processing.

        Args:
            item_id: Unique item identifier.
            error: The exception that was raised.
        """
        stage_name = self._active_stage or self._last_active_stage()
        if stage_name is None:
            return
        with self._lock:
            sp = self._stages.get(stage_name)
            if sp is None:
                return
            sp.errors += 1
            self._refresh_task(sp)
        _logger.error("Item error in stage %s: %s — %s", stage_name, item_id, error)

    # ------------------------------------------------------------------
    # Stage-specific callbacks (for multi-stage pipelines where the
    # caller explicitly targets a stage rather than relying on active_stage)
    # ------------------------------------------------------------------

    @trace
    def on_stage_item_complete(self, stage_name: str, item_id: str) -> None:
        """``on_item_complete`` targeted at a specific stage.

        Args:
            stage_name: The stage receiving the completion event.
            item_id: Unique item identifier.
        """
        with self._lock:
            sp = self._stages.get(stage_name)
            if sp is None:
                return
            sp.completed += 1
            self._refresh_task(sp)

    @trace
    def on_stage_item_error(self, stage_name: str, item_id: str, error: Exception) -> None:
        """``on_item_error`` targeted at a specific stage.

        Args:
            stage_name: The stage receiving the error event.
            item_id: Unique item identifier.
            error: The exception that was raised.
        """
        with self._lock:
            sp = self._stages.get(stage_name)
            if sp is None:
                return
            sp.errors += 1
            self._refresh_task(sp)
        _logger.error("Item error in stage %s: %s — %s", stage_name, item_id, error)

    # ------------------------------------------------------------------
    # Stats / reporting
    # ------------------------------------------------------------------

    @trace
    def get_stage_stats(self, stage_name: str) -> Optional[Dict[str, object]]:
        """Return a snapshot of stats for *stage_name*.

        Args:
            stage_name: Target stage.

        Returns:
            Dict with keys ``stage``, ``total``, ``completed``, ``errors``,
            ``elapsed``, ``throughput``, ``eta``; or ``None`` if not found.
        """
        sp = self._stages.get(stage_name)
        if sp is None:
            return None
        return {
            "stage": sp.stage_name,
            "total": sp.total,
            "completed": sp.completed,
            "errors": sp.errors,
            "elapsed": round(sp.elapsed, 2),
            "throughput": round(sp.throughput, 2),
            "eta": sp.eta_str,
        }

    @trace
    def get_all_stats(self) -> Dict[str, Dict[str, object]]:
        """Return stats snapshots for all registered stages.

        Returns:
            Dict mapping stage name → stats dict (same shape as
            :meth:`get_stage_stats`).
        """
        return {
            name: self.get_stage_stats(name)  # type: ignore[misc]
            for name in self._stages
        }

    @trace
    def get_stats(self) -> Dict[str, Any]:
        """Return stats for all stages (alias for :meth:`get_all_stats`).

        Satisfies the ``get_stats()`` / ``reset_stats()`` convention
        required by the project instructions.

        Returns:
            Dict mapping stage name → stats dict.
        """
        return self.get_all_stats()  # type: ignore[return-value]

    @trace
    def reset_stats(self) -> None:
        """Reset all stage statistics (does not stop the display)."""
        with self._lock:
            self._stages.clear()
        _logger.info("ModuleProgressManager stats reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_task(self, sp: StageProgress) -> None:
        """Update the Rich task for *sp* if the display is running.

        Must be called with ``self._lock`` held.
        """
        if _RICH_AVAILABLE and self._rich_progress is not None and sp.task_id is not None:
            thr_str = f"{sp.throughput:.1f} img/s" if sp.throughput > 0 else "—"
            self._rich_progress.update(
                sp.task_id,
                completed=sp.completed,
                throughput=thr_str,
                errors=sp.errors,
            )

    def _last_active_stage(self) -> Optional[str]:
        """Return the name of the most recently started active stage."""
        active = [s for s in self._stages.values() if s.active]
        if not active:
            # fall back to last stage registered
            if self._stages:
                return list(self._stages.keys())[-1]
            return None
        return active[-1].stage_name
