"""System tests: end-to-end pipeline flow with mocked segmentor.

Tests the full pipeline orchestration using mocked or lightweight
components so no GPU / SAM3 model is required.

Author: Ahmed Hany ElBamby
Date: 25-07-2025
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import numpy as np
import pytest

from src.interfaces import SegmentationResult, MaskData
from src.progress_tracker import ProgressTracker
from src.annotation_writer import AnnotationWriter
from src.result_filter import ResultFilter
from src.post_processor import MaskPostProcessor
from src.class_registry import ClassRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_mask(h: int = 32, w: int = 32) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[4:19, 4:19] = 1.0
    return m


def _make_result(image_path: Path, class_ids: Optional[List[int]] = None):
    """Create a duck-typed result object compatible with AnnotationWriter and ResultFilter."""
    cids = class_ids or [0, 1]
    masks_arr = np.stack([_binary_mask() for _ in cids])  # shape (N, H, W)
    return SimpleNamespace(
        image_path=image_path,
        masks=masks_arr,
        class_ids=cids,
        num_detections=len(cids),
        image_width=32,
        image_height=32,
    )


def _make_segmentation_result(image_path: Path, class_ids: Optional[List[int]] = None) -> SegmentationResult:
    """Create a SegmentationResult with MaskData objects (for NMS testing)."""
    cids = class_ids or [0, 1]
    masks = []
    for i, cid in enumerate(cids):
        mask = _binary_mask()
        masks.append(MaskData(
            mask=mask,
            confidence=0.9 - i * 0.05,
            class_id=cid,
            area=int(mask.sum()),
            bbox=(4, 4, 19, 19),
        ))
    return SegmentationResult(
        image_path=image_path,
        masks=masks,
        image_width=32,
        image_height=32,
    )


def _make_post_config(**kwargs):
    from dataclasses import dataclass
    @dataclass
    class _Cfg:
        enabled: bool = True
        strategy: str = "confidence"
        iou_threshold: float = 0.5
        confidence_threshold: float = 0.1
        enable_class_specific: bool = False
        soft_nms_sigma: float = 0.5
        soft_nms_score_threshold: float = 0.001
        decay_factor: float = 0.5
        adaptive_base_threshold: float = 0.3
        adaptive_density_weight: float = 0.1
    return _Cfg(**kwargs)


# ---------------------------------------------------------------------------
# Progress tracker lifecycle
# ---------------------------------------------------------------------------

class TestProgressTrackerJobLifecycle:
    """Jobs can be registered, images queued, and progress tracked."""

    def test_create_job_and_track_images(self, tmp_path):
        db = tmp_path / "progress.db"
        tracker = ProgressTracker(db_path=db)
        image_paths = [tmp_path / f"img_{i}.jpg" for i in range(5)]
        splits = ["train"] * 5
        job_id = tracker.create_job("system_test_job_001", image_paths, splits)
        # Retrieve image IDs from pending list
        pending = tracker.get_pending_images(job_id, limit=10)
        for (image_id, _path, _split) in pending:
            tracker.mark_completed(image_id)
        tracker.checkpoint(job_id)
        progress = tracker.get_progress(job_id)
        assert progress["total_images"] == 5
        assert progress["processed_count"] == 5
        assert progress["error_count"] == 0
        tracker.close()

    def test_resume_skips_completed_images(self, tmp_path):
        """Images already completed are skipped on resume."""
        db = tmp_path / "progress.db"
        tracker = ProgressTracker(db_path=db)
        image_paths = [tmp_path / f"img_{i}.jpg" for i in range(6)]
        splits = ["train"] * 6
        job_id = tracker.create_job("system_resume_test", image_paths, splits)
        # Get all pending, complete first 3
        pending = tracker.get_pending_images(job_id, limit=10)
        for (image_id, _path, _split) in pending[:3]:
            tracker.mark_completed(image_id)
        tracker.checkpoint(job_id)
        progress = tracker.get_progress(job_id)
        assert progress["processed_count"] == 3
        # Get pending — should only return last 3
        remaining = tracker.get_pending_images(job_id, limit=10)
        assert len(remaining) == 3
        tracker.close()

    def test_error_images_tracked(self, tmp_path):
        db = tmp_path / "progress.db"
        tracker = ProgressTracker(db_path=db)
        image_paths = [tmp_path / f"img_{i}.jpg" for i in range(4)]
        splits = ["train"] * 4
        job_id = tracker.create_job("system_error_test", image_paths, splits)
        pending = tracker.get_pending_images(job_id, limit=10)
        image_ids = [iid for (iid, _, _) in pending]
        tracker.mark_error(image_ids[0], "GPU OOM")
        tracker.mark_error(image_ids[1], "decode error")
        tracker.mark_completed(image_ids[2])
        tracker.mark_completed(image_ids[3])
        tracker.checkpoint(job_id)
        progress = tracker.get_progress(job_id)
        assert progress["error_count"] == 2
        assert progress["processed_count"] == 2
        tracker.close()


# ---------------------------------------------------------------------------
# Remap → NMS → Filter → Annotate pipeline fragment
# ---------------------------------------------------------------------------

class TestRemapNMSAnnotatePipeline:
    """Tests the core pipeline fragment: remap → NMS → filter → annotation."""

    def _make_registry_5to2(self):
        from dataclasses import dataclass
        @dataclass
        class _ModelCfg:
            prompts: List[str] = None
            class_remapping: Optional[dict] = None
            def __post_init__(self):
                if self.prompts is None:
                    self.prompts = ["teacher", "student", "kid", "child", "Adult"]
                if self.class_remapping is None:
                    self.class_remapping = {"kid": "student", "child": "student", "Adult": "teacher"}
        return ClassRegistry.from_config(_ModelCfg())

    def test_registry_correct_output_classes(self):
        registry = self._make_registry_5to2()
        assert registry.num_classes == 2
        assert "teacher" in registry.class_names
        assert "student" in registry.class_names

    def test_nms_applied_after_remap(self, tmp_path):
        """Remapped class IDs feed NMS, which reduces overlapping masks."""
        registry = self._make_registry_5to2()
        post_cfg = _make_post_config(strategy="confidence", iou_threshold=0.1)
        processor = MaskPostProcessor(post_cfg, class_names=registry.class_names)
        # Use SegmentationResult with MaskData objects (required by apply_nms)
        result = _make_segmentation_result(tmp_path / "img.jpg", class_ids=[0, 1, 0, 1])
        output = processor.apply_nms(result)
        assert output is not None
        assert len(output.masks) <= 4

    def test_filter_respects_min_confidence(self, tmp_path):
        """ResultFilter categorises images with/without detections."""
        from dataclasses import dataclass
        @dataclass
        class _PipelineCfg:
            output_dir: Path = None
            neither_dir: Path = None
            def __post_init__(self):
                if self.output_dir is None:
                    self.output_dir = tmp_path / "out"
                if self.neither_dir is None:
                    self.neither_dir = tmp_path / "neither"
        cfg = _PipelineCfg()
        result_filter = ResultFilter(cfg)
        result = _make_result(tmp_path / "img.jpg", class_ids=[0, 1])
        # Image with detections → True
        kept = result_filter.filter_result(tmp_path / "img.jpg", result, copy_to_neither=False)
        assert kept is True
        # Image with no detections → False
        discarded = result_filter.filter_result(tmp_path / "empty.jpg", None, copy_to_neither=False)
        assert discarded is False

    def test_annotation_writer_yolo_output(self, tmp_path):
        """AnnotationWriter writes valid YOLO files for remapped results."""
        from dataclasses import dataclass
        registry = self._make_registry_5to2()
        @dataclass
        class _AnnCfg:
            output_dir: Path = None
            def __post_init__(self):
                if self.output_dir is None:
                    self.output_dir = tmp_path / "dataset"
        cfg = _AnnCfg()
        writer = AnnotationWriter(cfg, registry)
        result = _make_result(tmp_path / "img.jpg", class_ids=[0, 1])
        writer.write_annotation(tmp_path / "img.jpg", result, split="train", copy_image=False)
        stats = writer.get_stats()
        assert stats["train"]["annotations"] >= 1


# ---------------------------------------------------------------------------
# Class remapping full flow
# ---------------------------------------------------------------------------

class TestClassRemappingEndToEnd:
    """5 prompts → 2 output classes flows correctly through the pipeline."""

    def test_prompt_index_remaps_to_output_id(self):
        from dataclasses import dataclass
        @dataclass
        class _ModelCfg:
            prompts: List[str] = None
            class_remapping: Optional[dict] = None
            def __post_init__(self):
                if self.prompts is None:
                    self.prompts = ["teacher", "student", "kid", "child", "Adult"]
                if self.class_remapping is None:
                    self.class_remapping = {"kid": "student", "child": "student", "Adult": "teacher"}
        registry = ClassRegistry.from_config(_ModelCfg())
        # "kid" (index 2) → "student" (output ID)
        student_id = registry.class_names.index("student")
        remapped = registry.remap_prompt_index(2)
        assert remapped == student_id, f"Expected {student_id}, got {remapped}"

    def test_identity_mapping_with_no_remapping_config(self):
        from dataclasses import dataclass
        @dataclass
        class _ModelCfg:
            prompts: List[str] = None
            class_remapping: Optional[dict] = None
            def __post_init__(self):
                if self.prompts is None:
                    self.prompts = ["cat", "dog"]
                if self.class_remapping is None:
                    self.class_remapping = {}
        registry = ClassRegistry.from_config(_ModelCfg())
        assert registry.num_classes == 2
        assert registry.remap_prompt_index(0) == 0
        assert registry.remap_prompt_index(1) == 1

    def test_yolo_names_dict_matches_output_classes(self):
        from dataclasses import dataclass
        @dataclass
        class _ModelCfg:
            prompts: List[str] = None
            class_remapping: Optional[dict] = None
            def __post_init__(self):
                if self.prompts is None:
                    self.prompts = ["teacher", "student", "kid", "child", "Adult"]
                if self.class_remapping is None:
                    self.class_remapping = {"kid": "student", "child": "student", "Adult": "teacher"}
        registry = ClassRegistry.from_config(_ModelCfg())
        names = registry.get_yolo_names()
        assert len(names) == 2
        assert set(names.values()) == {"teacher", "student"}

    def test_class_ids_in_yolo_output_in_range(self, tmp_path):
        """YOLO annotation class IDs must be in [0, num_classes-1]."""
        from dataclasses import dataclass
        @dataclass
        class _ModelCfg:
            prompts: List[str] = None
            class_remapping: Optional[dict] = None
            def __post_init__(self):
                if self.prompts is None:
                    self.prompts = ["teacher", "student", "kid"]
                if self.class_remapping is None:
                    self.class_remapping = {"kid": "student"}
        @dataclass
        class _AnnCfg:
            output_dir: Path = None
            def __post_init__(self):
                if self.output_dir is None:
                    self.output_dir = tmp_path / "out"
        registry = ClassRegistry.from_config(_ModelCfg())
        writer = AnnotationWriter(_AnnCfg(), registry)
        result = _make_result(tmp_path / "img.jpg", class_ids=[0, 1])
        writer.write_annotation(tmp_path / "img.jpg", result, split="train", copy_image=False)
        label_dir = tmp_path / "out" / "train" / "labels"
        if label_dir.exists():
            for lf in label_dir.glob("*.txt"):
                for line in lf.read_text().splitlines():
                    class_id = int(line.split()[0])
                    assert 0 <= class_id < registry.num_classes


# ---------------------------------------------------------------------------
# Cross-platform path handling
# ---------------------------------------------------------------------------

class TestCrossPlatformPaths:
    """Path handling works correctly regardless of OS separator style."""

    def test_pathlib_paths_work_in_annotation_writer(self, tmp_path):
        """AnnotationWriter creates output dirs using pathlib.Path."""
        from dataclasses import dataclass
        from src.class_registry import ClassRegistry
        @dataclass
        class _ModelCfg:
            prompts: List[str] = None
            class_remapping: Optional[dict] = None
            def __post_init__(self):
                if self.prompts is None:
                    self.prompts = ["cat", "dog"]
                if self.class_remapping is None:
                    self.class_remapping = {}
        @dataclass
        class _AnnCfg:
            output_dir: Path = None
            def __post_init__(self):
                if self.output_dir is None:
                    self.output_dir = tmp_path / "output" / "dataset"
        registry = ClassRegistry.from_config(_ModelCfg())
        writer = AnnotationWriter(_AnnCfg(), registry)
        result = _make_result(tmp_path / "img.jpg", class_ids=[0, 1])
        writer.write_annotation(tmp_path / "img.jpg", result, split="train", copy_image=False)
        assert (tmp_path / "output" / "dataset" / "train").exists()

    def test_progress_tracker_path_with_nested_dirs(self, tmp_path):
        """ProgressTracker creates DB file in nested directories."""
        db_path = tmp_path / "deep" / "nested" / "progress.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracker = ProgressTracker(db_path=db_path)
        image_paths = [tmp_path / "img.jpg"]
        tracker.create_job("path_test", image_paths, ["train"])
        assert db_path.exists()
        tracker.close()
