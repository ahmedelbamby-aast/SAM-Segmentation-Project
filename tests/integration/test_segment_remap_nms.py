"""Integration tests — Segment → Remap → NMS pipeline flow.

Tests the interaction between :class:`~src.class_registry.ClassRegistry`
(remap stage), :class:`~src.post_processor.MaskPostProcessor` (NMS stage),
and the :class:`~src.pipeline.SegmentationPipeline._remap_result()` static
helper, using synthetic :class:`~src.interfaces.SegmentationResult` objects
instead of the real SAM3 model (which requires a GPU and the model file).

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from src.class_registry import ClassRegistry
from src.interfaces import MaskData, SegmentationResult
from src.pipeline import SegmentationPipeline
from src.post_processor import MaskPostProcessor, create_post_processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask(h: int, w: int, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[r1:r2, c1:c2] = True
    return m


@dataclass
class _FakePostConfig:
    enabled: bool = True
    iou_threshold: float = 0.5
    strategy: str = "confidence"
    class_priority: List[str] = field(default_factory=list)
    soft_nms_sigma: float = 0.5
    min_confidence_after_decay: float = 0.1
    weighted_nms_sigma: float = 0.5
    adaptive_nms_density_factor: float = 0.1
    diou_nms_beta: float = 1.0
    mask_merge_threshold: float = 0.7
    enable_class_specific: bool = False


def _make_result(specs: list, w: int = 100, h: int = 100) -> SegmentationResult:
    """Build SegmentationResult from (mask, confidence, raw_class_id) tuples."""
    masks = [
        MaskData(mask=m, confidence=c, class_id=cls,
                 area=int(np.asarray(m).sum()),
                 bbox=(0, 0, w, h))
        for m, c, cls in specs
    ]
    return SegmentationResult(
        image_path=Path("fake.jpg"),
        masks=masks,
        image_width=w,
        image_height=h,
    )


# ---------------------------------------------------------------------------
# TestRemapStage
# ---------------------------------------------------------------------------

class TestRemapStage:
    """Tests for SegmentationPipeline._remap_result() (remap pipeline stage)."""

    def test_identity_remap_1_to_1(self):
        """With no class_remapping, prompt indices pass through unchanged."""
        registry = ClassRegistry(["teacher", "student"])
        m1 = _mask(10, 10, 0, 0, 5, 5)
        m2 = _mask(10, 10, 5, 5, 10, 10)
        result = _make_result([(m1, 0.9, 0), (m2, 0.7, 1)])
        remapped = SegmentationPipeline._remap_result(result, registry)
        # 0→0, 1→1 (identity)
        assert remapped.masks[0].class_id == 0
        assert remapped.masks[1].class_id == 1

    def test_many_to_one_remap(self):
        """Multiple prompt indices collapse into the same output class ID."""
        registry = ClassRegistry(
            ["teacher", "student", "kid", "child"],
            class_remapping={"kid": "student", "child": "student"},
        )
        # prompt idx 0=teacher, 1=student, 2=kid→student, 3=child→student
        m = _mask(10, 10, 0, 0, 5, 5)
        result = _make_result([(m, 0.9, 0), (m, 0.8, 1), (m, 0.7, 2), (m, 0.6, 3)])
        remapped = SegmentationPipeline._remap_result(result, registry)
        assert remapped.masks[0].class_id == 0  # teacher → 0
        assert remapped.masks[1].class_id == 1  # student → 1
        assert remapped.masks[2].class_id == 1  # kid → 1 (student)
        assert remapped.masks[3].class_id == 1  # child → 1 (student)

    def test_remap_preserves_metadata(self):
        """_remap_result preserves all non-class_id fields exactly."""
        registry = ClassRegistry(["teacher", "student"])
        m = _mask(10, 10, 0, 0, 8, 8)
        result = _make_result([(m, 0.95, 0)])
        remapped = SegmentationPipeline._remap_result(result, registry)
        md = remapped.masks[0]
        assert md.confidence == pytest.approx(0.95)
        assert md.area == int(m.sum())
        assert remapped.image_path == result.image_path
        assert remapped.image_width == result.image_width
        assert remapped.image_height == result.image_height

    def test_remap_empty_result(self):
        """An empty SegmentationResult passes through without error."""
        registry = ClassRegistry(["teacher"])
        result = _make_result([])
        remapped = SegmentationPipeline._remap_result(result, registry)
        assert len(remapped.masks) == 0


# ---------------------------------------------------------------------------
# TestRemapThenNMS
# ---------------------------------------------------------------------------

class TestRemapThenNMS:
    """End-to-end remap → NMS interaction tests."""

    def _run(self, specs, remapping=None, strategy="confidence", iou_threshold=0.5):
        prompts = ["teacher", "student", "kid", "child"]
        registry = ClassRegistry(prompts, class_remapping=remapping or {})
        cfg = _FakePostConfig(strategy=strategy, iou_threshold=iou_threshold)
        nms = create_post_processor(cfg, class_names=registry.class_names)

        result = _make_result(specs)
        remapped = SegmentationPipeline._remap_result(result, registry)
        final = nms.apply_nms(remapped)
        return final, registry

    def test_nms_receives_remapped_ids_not_raw_prompt_indices(self):
        """After remap, NMS operates on output class IDs (0..M-1) not on prompt indices."""
        m = _mask(50, 50, 0, 0, 40, 40)
        # raw indices 2 and 3 both remap to "student" (output id 1)
        final, registry = self._run(
            specs=[(m, 0.9, 2), (m, 0.4, 3)],
            remapping={"kid": "student", "child": "student"},
        )
        # Both masks had class_id=1 after remap; NMS suppresses the lower-confidence one
        assert len(final.masks) == 1
        assert final.masks[0].class_id == 1  # remapped "student" id

    def test_non_overlapping_masks_both_survive_pipeline(self):
        m1 = _mask(20, 20, 0, 0, 8, 8)
        m2 = _mask(20, 20, 12, 12, 20, 20)
        final, _ = self._run(specs=[(m1, 0.9, 0), (m2, 0.7, 1)])
        assert len(final.masks) == 2

    def test_remap_then_class_specific_nms(self):
        """Class-specific NMS after remap suppresses within output class only."""
        m = _mask(20, 20, 0, 0, 18, 18)
        # prompt idx 0=teacher, 2=kid→student (id 1), 3=child→student (id 1)
        # After remap: 2 teacher masks (class 0) + 2 student masks (class 1)
        # Class-specific NMS: keep best per class → 2 remain
        specs = [(m, 0.9, 0), (m, 0.3, 0), (m, 0.8, 2), (m, 0.2, 3)]
        remapping = {"kid": "student", "child": "student"}
        prompts = ["teacher", "student", "kid", "child"]
        registry = ClassRegistry(prompts, class_remapping=remapping)
        cfg = _FakePostConfig(strategy="confidence", iou_threshold=0.5, enable_class_specific=True)
        nms = create_post_processor(cfg, class_names=registry.class_names)

        result = _make_result(specs, w=20, h=20)
        remapped = SegmentationPipeline._remap_result(result, registry)
        final = nms.apply_nms(remapped)
        class_ids = [md.class_id for md in final.masks]
        assert class_ids.count(0) == 1   # best teacher
        assert class_ids.count(1) == 1   # best student


# ---------------------------------------------------------------------------
# TestRemapConstraint
# ---------------------------------------------------------------------------

class TestRemapConstraint:
    """Verify the mandatory constraint: remap happens BEFORE NMS sees any class IDs."""

    def test_raw_prompt_ids_are_never_seen_by_nms(self):
        """
        With 5 prompts mapping to 2 output classes, NMS class IDs must be in {0, 1}.
        If remap were skipped, we'd see IDs in {0, 1, 2, 3, 4} — this test fails then.
        """
        prompts = ["teacher", "student", "kid", "child", "Adult"]
        remapping = {"kid": "student", "child": "student", "Adult": "teacher"}
        registry = ClassRegistry(prompts, class_remapping=remapping)
        expected_output_ids = {0, 1}

        m = _mask(10, 10, 0, 0, 9, 9)
        specs = [(m, float(0.5 + i*0.05), i) for i in range(5)]
        result = _make_result(specs)
        remapped = SegmentationPipeline._remap_result(result, registry)

        actual_ids = {md.class_id for md in remapped.masks}
        assert actual_ids == expected_output_ids, (
            f"Expected output class IDs {expected_output_ids}, got {actual_ids}. "
            "This means remap is not applied before NMS."
        )

    def test_remap_before_nms_does_not_change_num_masks(self):
        """Remap itself never adds or removes masks — only NMS does."""
        registry = ClassRegistry(["a", "b", "c"], class_remapping={"c": "b"})
        specs = [
            (_mask(10, 10, 0, 0, 5, 5), 0.9, i) for i in range(3)
        ]
        result = _make_result(specs)
        remapped = SegmentationPipeline._remap_result(result, registry)
        assert len(remapped.masks) == len(result.masks)
