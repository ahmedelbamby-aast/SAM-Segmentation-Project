"""System tests: each NMS strategy produces valid output.

Verifies that all 10 registered NMS strategies can be instantiated,
run on realistic SegmentationResult data, and return structurally-valid
results without crashing.

Author: Ahmed Hany ElBamby
Date: 25-07-2025
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.interfaces import SegmentationResult, MaskData
from src.post_processor import MaskPostProcessor, NMSStrategyFactory


# ---------------------------------------------------------------------------
# Minimal PostProcessingConfig stub
# ---------------------------------------------------------------------------

@dataclass
class _NMSConfig:
    enabled: bool = True
    strategy: str = "confidence"
    iou_threshold: float = 0.4
    confidence_threshold: float = 0.3
    enable_class_specific: bool = False
    soft_nms_sigma: float = 0.5
    soft_nms_score_threshold: float = 0.001
    decay_factor: float = 0.5
    adaptive_base_threshold: float = 0.3
    adaptive_density_weight: float = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(h: int, w: int, fill_rows: tuple = (2, 18), fill_cols: tuple = (2, 18)):
    """Create a float32 binary mask with a rectangular filled region."""
    m = np.zeros((h, w), dtype=np.float32)
    r0, r1 = fill_rows
    c0, c1 = fill_cols
    m[r0:r1, c0:c1] = 1.0
    return m


def _make_overlapping_result(num_masks: int = 4, class_ids: Optional[List[int]] = None) -> SegmentationResult:
    """Create a SegmentationResult with overlapping masks for NMS testing."""
    H, W = 64, 64
    masks: List[MaskData] = []
    for i in range(num_masks):
        offset = i * 3  # slight shift creates overlap
        mask = _make_mask(H, W, fill_rows=(2 + offset, 20 + offset), fill_cols=(2 + offset, 20 + offset))
        cid = class_ids[i] if class_ids else (i % 2)
        area = int(mask.sum())
        masks.append(MaskData(
            mask=mask,
            confidence=0.9 - i * 0.1,
            class_id=cid,
            area=area,
            bbox=(2 + offset, 2 + offset, 20 + offset, 20 + offset),
        ))
    return SegmentationResult(
        image_path=Path("test_image.jpg"),
        masks=masks,
        image_width=W,
        image_height=H,
    )


# ---------------------------------------------------------------------------
# Parametrized strategy tests
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    "confidence",
    "area",
    "class_priority",
    "gaussian_soft_nms",
    "linear_soft_nms",
    "weighted_nms",
    "adaptive_nms",
    "diou_nms",
    "matrix_nms",
    "mask_merge_nms",
]


@pytest.mark.parametrize("strategy_name", ALL_STRATEGIES)
class TestNMSStrategyOutput:
    """All 10 strategies must produce valid SegmentationResult without crashing."""

    def test_strategy_is_registered(self, strategy_name: str):
        """Strategy must be found in the factory registry."""
        strategy = NMSStrategyFactory.create(strategy_name)
        assert strategy is not None

    def test_apply_nms_does_not_crash(self, strategy_name: str):
        """apply_nms() must not crash for any registered strategy."""
        config = _NMSConfig(strategy=strategy_name)
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=4)
        output = processor.apply_nms(result)
        assert isinstance(output, SegmentationResult)

    def test_output_masks_are_subset_or_modified(self, strategy_name: str):
        """Output mask count must be <= input (suppression or merge, never addition)."""
        config = _NMSConfig(strategy=strategy_name, iou_threshold=0.1)
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=6)
        output = processor.apply_nms(result)
        assert len(output.masks) <= len(result.masks)

    def test_output_masks_have_valid_class_ids(self, strategy_name: str):
        """All output masks must have non-negative class_ids."""
        config = _NMSConfig(strategy=strategy_name)
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=4, class_ids=[0, 1, 0, 1])
        output = processor.apply_nms(result)
        for mask in output.masks:
            assert mask.class_id >= 0

    def test_output_masks_have_valid_confidence(self, strategy_name: str):
        """All surviving masks must have confidence in [0, 1]."""
        config = _NMSConfig(strategy=strategy_name)
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=4)
        output = processor.apply_nms(result)
        for mask in output.masks:
            assert 0.0 <= mask.confidence <= 1.0

    def test_output_image_path_preserved(self, strategy_name: str):
        """image_path must be preserved in output."""
        config = _NMSConfig(strategy=strategy_name)
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=3)
        output = processor.apply_nms(result)
        assert output.image_path == result.image_path

    def test_get_stats_returns_dict(self, strategy_name: str):
        """get_stats() must return a dict after processing."""
        config = _NMSConfig(strategy=strategy_name)
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=4)
        processor.apply_nms(result)
        stats = processor.get_stats()
        assert isinstance(stats, dict)
        assert "total_processed" in stats
        assert stats["total_processed"] == 4


class TestNMSDisabledPassthrough:
    """When NMS is disabled, results must pass through unchanged."""

    def test_disabled_nms_returns_original_masks(self):
        config = _NMSConfig(enabled=False)
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=5)
        output = processor.apply_nms(result)
        assert len(output.masks) == 5

    def test_empty_masks_passthrough(self):
        """Empty mask list is handled gracefully by all strategies."""
        for strategy_name in ALL_STRATEGIES:
            config = _NMSConfig(strategy=strategy_name)
            processor = MaskPostProcessor(config)
            empty_result = SegmentationResult(
                image_path=Path("empty.jpg"),
                masks=[],
                image_width=64,
                image_height=64,
            )
            output = processor.apply_nms(empty_result)
            assert output.masks == [], f"Strategy {strategy_name!r} failed on empty masks"


class TestNMSResetStats:
    """reset_stats() brings all counters back to zero."""

    def test_reset_after_processing(self):
        config = _NMSConfig(strategy="confidence")
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=4)
        processor.apply_nms(result)
        processor.reset_stats()
        stats = processor.get_stats()
        assert all(v == 0 for v in stats.values())

    def test_stats_accumulate_across_calls(self):
        config = _NMSConfig(strategy="confidence")
        processor = MaskPostProcessor(config)
        result = _make_overlapping_result(num_masks=3)
        processor.apply_nms(result)
        processor.apply_nms(result)
        stats = processor.get_stats()
        assert stats["total_processed"] == 6
