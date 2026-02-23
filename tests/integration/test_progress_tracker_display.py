"""
Integration tests: ProgressTracker + ModuleProgressManager event flow.

Verifies that:
- ``ModuleProgressManager`` correctly tracks stage progress in memory.
- ``ProgressTracker`` correctly persists job/image state in SQLite.
- Both systems can work together through a shared ``ProgressCallback`` pattern.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.progress_tracker import ProgressTracker
from src.progress_display import ModuleProgressManager


# ---------------------------------------------------------------------------
# 1. ModuleProgressManager standalone lifecycle
# ---------------------------------------------------------------------------

class TestModuleProgressManagerLifecycle:
    def test_start_and_stop_without_error(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        mgr.stop()

    def test_context_manager_cleans_up(self):
        with ModuleProgressManager(show_gpu_memory=False) as mgr:
            mgr.start_stage("Segment", total=5)
            mgr.on_item_complete("img_0")

    def test_start_stage_registers_stage(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        mgr.start_stage("NMS", total=10)
        stats = mgr.get_stage_stats("NMS")
        assert stats is not None
        assert stats["total"] == 10
        mgr.stop()

    def test_on_item_complete_increments_counter(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        mgr.start_stage("Segment", total=3)
        mgr.on_item_complete("img_0")
        mgr.on_item_complete("img_1")
        stats = mgr.get_stage_stats("Segment")
        assert stats["completed"] == 2
        mgr.stop()

    def test_on_item_error_increments_error_counter(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        mgr.start_stage("Segment", total=2)
        mgr.on_item_error("img_bad", RuntimeError("simulated"))
        stats = mgr.get_stage_stats("Segment")
        assert stats["errors"] >= 1
        mgr.stop()

    def test_finish_stage_marks_stage_done(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        mgr.start_stage("Remap", total=1)
        mgr.on_item_complete("item_0")
        mgr.finish_stage("Remap")
        stats = mgr.get_stage_stats("Remap")
        assert stats is not None
        mgr.stop()

    def test_reset_stats_clears_stage_data(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        mgr.start_stage("Segment", total=5)
        mgr.on_item_complete("img_0")
        mgr.reset_stats()
        # After reset the cleared stage should not appear
        assert mgr.get_stage_stats("Segment") is None
        mgr.stop()


# ---------------------------------------------------------------------------
# 2. ModuleProgressManager: get_all_stats aggregation
# ---------------------------------------------------------------------------

class TestGetAllStats:
    def test_returns_stats_for_all_started_stages(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        for stage in ("Segment", "NMS", "Filter"):
            mgr.start_stage(stage, total=2)
            mgr.on_item_complete(f"{stage}_item0")
        all_stats = mgr.get_all_stats()
        for stage in ("Segment", "NMS", "Filter"):
            assert stage in all_stats
        mgr.stop()

    def test_all_stages_completed_count_matches_events(self):
        mgr = ModuleProgressManager(show_gpu_memory=False)
        mgr.start()
        mgr.start_stage("Annotate", total=4)
        for i in range(4):
            mgr.on_item_complete(f"img_{i}")
        stats = mgr.get_all_stats()["Annotate"]
        assert stats["completed"] == 4
        mgr.stop()


# ---------------------------------------------------------------------------
# 3. ProgressTracker SQLite persistence
# ---------------------------------------------------------------------------

class TestProgressTrackerPersistence:
    @pytest.fixture()
    def tracker(self, tmp_path):
        db = tmp_path / "progress.db"
        t = ProgressTracker(db_path=db)
        yield t
        t.close()

    def test_create_job_and_retrieve_id(self, tracker, tmp_path):
        imgs = [tmp_path / f"img_{i}.jpg" for i in range(3)]
        splits = ["train"] * 3
        job_id = tracker.create_job("test_job", imgs, splits)
        assert isinstance(job_id, int)
        assert tracker.get_job_id("test_job") == job_id

    def test_pending_images_returned_after_create_job(self, tracker, tmp_path):
        imgs = [tmp_path / f"img_{i}.jpg" for i in range(5)]
        job_id = tracker.create_job("pending_test", imgs, ["train"] * 5)
        pending = tracker.get_pending_images(job_id, limit=10)
        assert len(pending) == 5

    def test_mark_processing_removes_from_pending(self, tracker, tmp_path):
        imgs = [tmp_path / f"img_{i}.jpg" for i in range(3)]
        job_id = tracker.create_job("proc_test", imgs, ["train"] * 3)
        pending = tracker.get_pending_images(job_id)
        ids = [row[0] for row in pending]
        tracker.mark_processing(ids)
        pending_after = tracker.get_pending_images(job_id)
        assert len(pending_after) == 0

    def test_mark_completed_updates_progress(self, tracker, tmp_path):
        imgs = [tmp_path / "img_0.jpg"]
        job_id = tracker.create_job("complete_test", imgs, ["train"])
        pending = tracker.get_pending_images(job_id)
        img_id = pending[0][0]
        tracker.mark_processing([img_id])
        tracker.mark_completed(img_id)
        tracker.checkpoint(job_id)  # flush image status counts into jobs table
        progress = tracker.get_progress(job_id)
        assert progress["processed_count"] >= 1

    def test_mark_error_recorded_in_progress(self, tracker, tmp_path):
        imgs = [tmp_path / "img_e.jpg"]
        job_id = tracker.create_job("error_test", imgs, ["train"])
        pending = tracker.get_pending_images(job_id)
        img_id = pending[0][0]
        tracker.mark_processing([img_id])
        tracker.mark_error(img_id, "simulated error")
        tracker.checkpoint(job_id)  # flush image status counts into jobs table
        progress = tracker.get_progress(job_id)
        assert progress["error_count"] >= 1

    def test_job_id_returns_none_for_unknown_job(self, tracker):
        assert tracker.get_job_id("no_such_job") is None

    def test_progress_by_split_groups_correctly(self, tracker, tmp_path):
        imgs = [tmp_path / f"img_{i}.jpg" for i in range(4)]
        splits = ["train", "train", "valid", "test"]
        job_id = tracker.create_job("split_test", imgs, splits)
        split_progress = tracker.get_progress_by_split(job_id)
        assert "train" in split_progress
        assert "valid" in split_progress
        assert "test" in split_progress


# ---------------------------------------------------------------------------
# 4. ProgressTracker + ModuleProgressManager: combined event flow
# ---------------------------------------------------------------------------

class TestCombinedEventFlow:
    """Simulate a pipeline stage that fires tracker SQLite writes and
    display Manager updates in sequence (both share same logical flow)."""

    def test_events_arrive_in_order(self, tmp_path):
        db = tmp_path / "combined.db"
        tracker = ProgressTracker(db_path=db)
        mgr = ModuleProgressManager(show_gpu_memory=False)

        imgs = [tmp_path / f"img_{i}.jpg" for i in range(3)]
        job_id = tracker.create_job("combined_job", imgs, ["train"] * 3)

        mgr.start()
        mgr.start_stage("Segment", total=3)

        pending = tracker.get_pending_images(job_id, limit=10)
        for img_id, img_path, split in pending:
            tracker.mark_processing([img_id])
            mgr.on_item_start(str(img_path))
            tracker.mark_completed(img_id)
            mgr.on_item_complete(str(img_path))

        tracker.checkpoint(job_id)  # flush counts to jobs table
        mgr.finish_stage("Segment")
        mgr.stop()

        progress = tracker.get_progress(job_id)
        assert progress["processed_count"] == 3
        assert progress["pending_count"] == 0

        stats = mgr.get_stage_stats("Segment")
        assert stats["completed"] == 3

        tracker.close()

    def test_error_events_tracked_in_both_systems(self, tmp_path):
        db = tmp_path / "err_combined.db"
        tracker = ProgressTracker(db_path=db)
        mgr = ModuleProgressManager(show_gpu_memory=False)

        imgs = [tmp_path / "bad_img.jpg"]
        job_id = tracker.create_job("err_job", imgs, ["train"])

        mgr.start()
        mgr.start_stage("Segment", total=1)

        pending = tracker.get_pending_images(job_id)
        img_id, img_path, _ = pending[0]
        tracker.mark_processing([img_id])
        err = RuntimeError("GPU OOM")
        tracker.mark_error(img_id, str(err))
        tracker.checkpoint(job_id)  # flush counts to jobs table
        mgr.on_item_error(str(img_path), err)

        mgr.stop()

        progress = tracker.get_progress(job_id)
        assert progress["error_count"] == 1

        stage_stats = mgr.get_stage_stats("Segment")
        assert stage_stats["errors"] >= 1

        tracker.close()
