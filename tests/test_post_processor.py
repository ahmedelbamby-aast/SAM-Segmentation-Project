"""Unit tests for src/post_processor.py (NMS Strategy Pattern).

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.post_processor import (
    OverlapStrategy,
    NMSStrategy,
    NMSStrategyFactory,
    ConfidenceNMS,
    AreaNMS,
    ClassPriorityNMS,
    GaussianSoftNMS,
    LinearSoftNMS,
    WeightedNMS,
    AdaptiveNMS,
    DIoUNMS,
    MatrixNMS,
    MaskMergeNMS,
    MaskPostProcessor,
    create_post_processor,
    _calculate_mask_iou,
    _calculate_mask_overlap,
)
from src.interfaces import SegmentationResult, MaskData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_mask(h: int, w: int, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[r1:r2, c1:c2] = True
    return m


@dataclass
class _FakeConfig:
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


def _make_result(mask_specs: list, base_path: str = "img.jpg") -> SegmentationResult:
    """Build SegmentationResult from list of (mask_array, confidence, class_id)."""
    masks = []
    for mask_arr, conf, cls in mask_specs:
        masks.append(MaskData(
            mask=mask_arr,
            confidence=conf,
            class_id=cls,
            area=int(np.asarray(mask_arr).sum()),
            bbox=(0, 0, mask_arr.shape[1], mask_arr.shape[0]),
        ))
    return SegmentationResult(
        image_path=Path(base_path),
        masks=masks,
        image_width=100,
        image_height=100,
    )


# ---------------------------------------------------------------------------
# TestModuleLevelHelpers
# ---------------------------------------------------------------------------

class TestModuleLevelHelpers:
    def test_iou_full_overlap(self):
        m = _square_mask(10, 10, 0, 0, 5, 5)
        assert _calculate_mask_iou(m, m) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        m1 = _square_mask(10, 10, 0, 0, 4, 4)
        m2 = _square_mask(10, 10, 5, 5, 9, 9)
        assert _calculate_mask_iou(m1, m2) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        m1 = _square_mask(10, 10, 0, 0, 6, 6)   # 36 pixels
        m2 = _square_mask(10, 10, 4, 4, 10, 10)  # 36 pixels  inter=4 union=68
        iou = _calculate_mask_iou(m1, m2)
        assert 0.0 < iou < 1.0

    def test_iou_empty_mask_returns_zero(self):
        empty = np.zeros((10, 10), dtype=bool)
        full  = _square_mask(10, 10, 0, 0, 10, 10)
        assert _calculate_mask_iou(empty, full) == pytest.approx(0.0)

    def test_overlap_asymmetric(self):
        big   = _square_mask(10, 10, 0, 0, 10, 10)  # 100 px
        small = _square_mask(10, 10, 0, 0, 5, 5)    # 25 px  (subset of big)
        o_big, o_small = _calculate_mask_overlap(big, small)
        assert o_big   == pytest.approx(0.25)
        assert o_small == pytest.approx(1.0)

    def test_overlap_empty_returns_zeros(self):
        empty = np.zeros((10, 10), dtype=bool)
        full  = _square_mask(10, 10, 0, 0, 10, 10)
        o1, o2 = _calculate_mask_overlap(empty, full)
        assert o1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestNMSStrategyFactory
# ---------------------------------------------------------------------------

class TestNMSStrategyFactory:
    def test_available_contains_all_10(self):
        expected = {
            "confidence", "area", "class_priority",
            "gaussian_soft_nms", "linear_soft_nms", "weighted_nms",
            "adaptive_nms", "diou_nms", "matrix_nms", "mask_merge_nms",
        }
        assert expected.issubset(set(NMSStrategyFactory.available()))

    def test_create_returns_correct_type(self):
        assert isinstance(NMSStrategyFactory.create("confidence"), ConfidenceNMS)
        assert isinstance(NMSStrategyFactory.create("area"), AreaNMS)
        assert isinstance(NMSStrategyFactory.create("gaussian_soft_nms"), GaussianSoftNMS)
        assert isinstance(NMSStrategyFactory.create("mask_merge_nms"), MaskMergeNMS)

    def test_create_legacy_soft_nms_alias(self):
        s = NMSStrategyFactory.create("soft_nms")
        assert isinstance(s, GaussianSoftNMS)

    def test_create_from_enum_value(self):
        s = NMSStrategyFactory.create(OverlapStrategy.CONFIDENCE)
        assert isinstance(s, ConfidenceNMS)

    def test_create_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            NMSStrategyFactory.create("nonexistent_strategy")

    def test_create_all_10_smoke(self):
        names = [
            "confidence", "area", "class_priority", "gaussian_soft_nms",
            "linear_soft_nms", "weighted_nms", "adaptive_nms",
            "diou_nms", "matrix_nms", "mask_merge_nms",
        ]
        for name in names:
            s = NMSStrategyFactory.create(name)
            assert isinstance(s, NMSStrategy)


# ---------------------------------------------------------------------------
# TestConfidenceNMS
# ---------------------------------------------------------------------------

class TestConfidenceNMS:
    _s = ConfidenceNMS()
    _cfg = _FakeConfig()

    def test_suppresses_lower_confidence_mask(self):
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=0.4,
            class_i=0, class_j=0, iou=0.8, config=self._cfg
        )
        assert score == pytest.approx(0.0)
        assert self._s.should_suppress(score, self._cfg)

    def test_keeps_higher_confidence_mask(self):
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.3, conf_j=0.9,
            class_i=0, class_j=0, iou=0.8, config=self._cfg
        )
        assert score == pytest.approx(0.9)
        assert not self._s.should_suppress(score, self._cfg)

    def test_equal_conf_suppresses(self):
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.7, conf_j=0.7,
            class_i=0, class_j=0, iou=0.6, config=self._cfg
        )
        assert self._s.should_suppress(score, self._cfg)


# ---------------------------------------------------------------------------
# TestAreaNMS
# ---------------------------------------------------------------------------

class TestAreaNMS:
    _s = AreaNMS()
    _cfg = _FakeConfig()

    def test_suppresses_smaller_area(self):
        big   = _square_mask(10, 10, 0, 0, 8, 8)
        small = _square_mask(10, 10, 0, 0, 4, 4)
        score = self._s.compute_suppression_score(
            mask_i=big, mask_j=small, conf_i=0.5, conf_j=0.9,
            class_i=0, class_j=0, iou=0.8, config=self._cfg
        )
        assert self._s.should_suppress(score, self._cfg)

    def test_keeps_larger_area_j(self):
        small = _square_mask(10, 10, 0, 0, 3, 3)
        big   = _square_mask(10, 10, 0, 0, 9, 9)
        score = self._s.compute_suppression_score(
            mask_i=small, mask_j=big, conf_i=0.9, conf_j=0.5,
            class_i=0, class_j=0, iou=0.7, config=self._cfg
        )
        assert not self._s.should_suppress(score, self._cfg)


# ---------------------------------------------------------------------------
# TestClassPriorityNMS
# ---------------------------------------------------------------------------

class TestClassPriorityNMS:
    _s = ClassPriorityNMS()

    def test_priority_order_respected(self):
        cfg = _FakeConfig(class_priority=["teacher", "student"])
        # mask_i = teacher (priority 0), mask_j = student (priority 1) → suppress j
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.5, conf_j=0.9,
            class_i=0, class_j=1, iou=0.9, config=cfg,
            class_names=["teacher", "student"]
        )
        assert self._s.should_suppress(score, cfg)

    def test_no_priority_falls_back_to_confidence(self):
        cfg = _FakeConfig(class_priority=[])
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=0.3,
            class_i=0, class_j=1, iou=0.8, config=cfg, class_names=None
        )
        assert self._s.should_suppress(score, cfg)


# ---------------------------------------------------------------------------
# TestGaussianSoftNMS
# ---------------------------------------------------------------------------

class TestGaussianSoftNMS:
    _s = GaussianSoftNMS()
    _cfg = _FakeConfig(soft_nms_sigma=0.5, min_confidence_after_decay=0.1)

    def test_decays_confidence(self):
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=1.0,
            class_i=0, class_j=0, iou=0.7, config=self._cfg
        )
        assert 0.0 < score < 1.0

    def test_high_iou_decays_more(self):
        s_low  = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=1.0,
            class_i=0, class_j=0, iou=0.2, config=self._cfg
        )
        s_high = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=1.0,
            class_i=0, class_j=0, iou=0.9, config=self._cfg
        )
        assert s_high < s_low

    def test_should_suppress_below_threshold(self):
        assert self._s.should_suppress(0.05, self._cfg)
        assert not self._s.should_suppress(0.5, self._cfg)


# ---------------------------------------------------------------------------
# TestLinearSoftNMS
# ---------------------------------------------------------------------------

class TestLinearSoftNMS:
    _s = LinearSoftNMS()
    _cfg = _FakeConfig(min_confidence_after_decay=0.1)

    def test_decays_linearly(self):
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=1.0,
            class_i=0, class_j=0, iou=0.5, config=self._cfg
        )
        assert score == pytest.approx(0.5)

    def test_full_iou_zeroes_confidence(self):
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=1.0,
            class_i=0, class_j=0, iou=1.0, config=self._cfg
        )
        assert score == pytest.approx(0.0)
        assert self._s.should_suppress(score, self._cfg)


# ---------------------------------------------------------------------------
# TestWeightedNMS
# ---------------------------------------------------------------------------

class TestWeightedNMS:
    _s = WeightedNMS()
    _cfg = _FakeConfig()

    def test_never_hard_suppresses(self):
        for v in [0.0, 0.001, 0.5, 1.0]:
            assert not self._s.should_suppress(v, self._cfg)

    def test_blended_score_positive(self):
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.6, conf_j=0.4,
            class_i=0, class_j=0, iou=0.9, config=self._cfg
        )
        assert score > 0.0


# ---------------------------------------------------------------------------
# TestAdaptiveNMS
# ---------------------------------------------------------------------------

class TestAdaptiveNMS:
    _s = AdaptiveNMS()

    def test_suppresses_when_iou_above_adaptive_threshold(self):
        cfg = _FakeConfig(iou_threshold=0.5, adaptive_nms_density_factor=0.1)
        # adaptive threshold = 0.5 + 0.1 = 0.6; iou=0.8 > 0.6 → suppress
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=0.7,
            class_i=0, class_j=0, iou=0.8, config=cfg
        )
        assert self._s.should_suppress(score, cfg)

    def test_keeps_when_iou_below_adaptive_threshold(self):
        cfg = _FakeConfig(iou_threshold=0.5, adaptive_nms_density_factor=0.1)
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=0.5,
            class_i=0, class_j=0, iou=0.4, config=cfg
        )
        assert not self._s.should_suppress(score, cfg)


# ---------------------------------------------------------------------------
# TestDIoUNMS
# ---------------------------------------------------------------------------

class TestDIoUNMS:
    _s = DIoUNMS()
    _cfg = _FakeConfig(iou_threshold=0.5)

    def test_suppresses_overlapping(self):
        m1 = _square_mask(10, 10, 0, 0, 8, 8)
        m2 = _square_mask(10, 10, 0, 0, 8, 8)  # identical → high iou + zero dist
        score = self._s.compute_suppression_score(
            mask_i=m1, mask_j=m2, conf_i=0.9, conf_j=0.5,
            class_i=0, class_j=0, iou=1.0, config=self._cfg
        )
        assert self._s.should_suppress(score, self._cfg)

    def test_keeps_distant_masks(self):
        m1 = _square_mask(20, 20, 0, 0, 5, 5)
        m2 = _square_mask(20, 20, 14, 14, 20, 20)
        iou = _calculate_mask_iou(m1, m2)
        score = self._s.compute_suppression_score(
            mask_i=m1, mask_j=m2, conf_i=0.9, conf_j=0.5,
            class_i=0, class_j=0, iou=iou, config=self._cfg
        )
        assert not self._s.should_suppress(score, self._cfg)


# ---------------------------------------------------------------------------
# TestMatrixNMS
# ---------------------------------------------------------------------------

class TestMatrixNMS:
    _s = MatrixNMS()
    _cfg = _FakeConfig(min_confidence_after_decay=0.1)

    def test_high_confidence_i_decays_j_more(self):
        s_high = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=1.0,
            class_i=0, class_j=0, iou=0.8, config=self._cfg
        )
        s_low = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.1, conf_j=1.0,
            class_i=0, class_j=0, iou=0.8, config=self._cfg
        )
        assert s_high < s_low

    def test_should_suppress_below_threshold(self):
        assert self._s.should_suppress(0.05, self._cfg)
        assert not self._s.should_suppress(0.5, self._cfg)


# ---------------------------------------------------------------------------
# TestMaskMergeNMS
# ---------------------------------------------------------------------------

class TestMaskMergeNMS:
    _s = MaskMergeNMS()

    def test_merge_sentinel_same_class_high_iou(self):
        cfg = _FakeConfig(mask_merge_threshold=0.7, iou_threshold=0.5)
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=0.8,
            class_i=0, class_j=0, iou=0.85, config=cfg
        )
        assert score == pytest.approx(-1.0)

    def test_suppress_different_class(self):
        cfg = _FakeConfig(mask_merge_threshold=0.7)
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.9, conf_j=0.3,
            class_i=0, class_j=1, iou=0.9, config=cfg
        )
        assert self._s.should_suppress(score, cfg)

    def test_keep_different_class_higher_conf_j(self):
        cfg = _FakeConfig(mask_merge_threshold=0.7)
        score = self._s.compute_suppression_score(
            mask_i=None, mask_j=None, conf_i=0.3, conf_j=0.9,
            class_i=0, class_j=1, iou=0.9, config=cfg
        )
        assert not self._s.should_suppress(score, cfg)


# ---------------------------------------------------------------------------
# TestMaskPostProcessor
# ---------------------------------------------------------------------------

class TestMaskPostProcessor:

    def _processor(self, **kw) -> MaskPostProcessor:
        cfg = _FakeConfig(**kw)
        return MaskPostProcessor(cfg, class_names=["teacher", "student"])

    def test_init_creates_correct_strategy(self):
        p = self._processor(strategy="gaussian_soft_nms")
        assert isinstance(p._strategy, GaussianSoftNMS)

    def test_apply_nms_empty_result_returns_unchanged(self):
        p = self._processor()
        result = _make_result([])
        out = p.apply_nms(result)
        assert len(out.masks) == 0

    def test_apply_nms_disabled_returns_unchanged(self):
        p = self._processor(enabled=False)
        m1 = _square_mask(10, 10, 0, 0, 9, 9)
        m2 = _square_mask(10, 10, 0, 0, 9, 9)
        result = _make_result([(m1, 0.9, 0), (m2, 0.5, 0)])
        out = p.apply_nms(result)
        assert len(out.masks) == 2

    def test_apply_nms_removes_overlapping_lower_confidence(self):
        p = self._processor(strategy="confidence", iou_threshold=0.5)
        m1 = _square_mask(50, 50, 0, 0, 40, 40)  # large, conf=0.9
        m2 = _square_mask(50, 50, 0, 0, 40, 40)  # identical, conf=0.3
        result = _make_result([(m1, 0.9, 0), (m2, 0.3, 0)])
        out = p.apply_nms(result)
        assert len(out.masks) == 1
        assert out.masks[0].confidence == pytest.approx(0.9)

    def test_apply_nms_non_overlapping_keeps_both(self):
        p = self._processor(strategy="confidence", iou_threshold=0.5)
        m1 = _square_mask(20, 20, 0, 0, 8, 8)
        m2 = _square_mask(20, 20, 12, 12, 20, 20)
        result = _make_result([(m1, 0.9, 0), (m2, 0.3, 1)])
        out = p.apply_nms(result)
        assert len(out.masks) == 2

    def test_stats_incremented_on_suppression(self):
        p = self._processor(strategy="confidence", iou_threshold=0.5)
        p.reset_stats()
        m = _square_mask(10, 10, 0, 0, 9, 9)
        result = _make_result([(m, 0.9, 0), (m, 0.3, 0)])
        p.apply_nms(result)
        stats = p.get_stats()
        assert stats["masks_suppressed"] >= 1

    def test_reset_stats_zeroes_all(self):
        p = self._processor()
        m = _square_mask(10, 10, 0, 0, 9, 9)
        p.apply_nms(_make_result([(m, 0.9, 0), (m, 0.3, 0)]))
        p.reset_stats()
        for v in p.get_stats().values():
            assert v == 0

    def test_class_specific_nms_groups_by_class(self):
        p = self._processor(strategy="confidence", iou_threshold=0.5, enable_class_specific=True)
        m = _square_mask(10, 10, 0, 0, 9, 9)
        # 2 overlapping class-0, 2 overlapping class-1 → should keep 1 of each = 2 total
        result = _make_result([(m, 0.9, 0), (m, 0.3, 0), (m, 0.8, 1), (m, 0.2, 1)])
        out = p.apply_nms(result)
        classes = [md.class_id for md in out.masks]
        assert classes.count(0) == 1
        assert classes.count(1) == 1

    def test_legacy_calculate_mask_iou_delegate(self):
        p = self._processor()
        m = _square_mask(10, 10, 0, 0, 8, 8)
        assert p.calculate_mask_iou(m, m) == pytest.approx(1.0)

    def test_legacy_calculate_mask_overlap_delegate(self):
        p = self._processor()
        big   = _square_mask(10, 10, 0, 0, 10, 10)
        small = _square_mask(10, 10, 0, 0, 5, 5)
        _, o_small = p.calculate_mask_overlap(big, small)
        assert o_small == pytest.approx(1.0)

    def test_protocol_compliance(self):
        from src.interfaces import PostProcessor
        p = self._processor()
        assert isinstance(p, PostProcessor)

    def test_result_metadata_preserved(self):
        p = self._processor(strategy="confidence", iou_threshold=0.5)
        m1 = _square_mask(50, 50, 0, 0, 40, 40)
        m2 = _square_mask(50, 50, 0, 0, 40, 40)
        result = _make_result([(m1, 0.9, 0), (m2, 0.3, 0)])
        out = p.apply_nms(result)
        assert out.image_path == result.image_path
        assert out.image_width == result.image_width
        assert out.image_height == result.image_height
        assert out.device == result.device


# ---------------------------------------------------------------------------
# TestCreatePostProcessorFactory
# ---------------------------------------------------------------------------

class TestCreatePostProcessorFactory:
    def test_returns_mask_post_processor(self):
        cfg = _FakeConfig()
        p = create_post_processor(cfg)
        assert isinstance(p, MaskPostProcessor)

    def test_with_class_names(self):
        cfg = _FakeConfig(strategy="class_priority")
        p = create_post_processor(cfg, class_names=["teacher", "student"])
        assert p.class_names == ["teacher", "student"]

    def test_without_class_names_defaults_to_none(self):
        cfg = _FakeConfig()
        p = create_post_processor(cfg)
        assert p.class_names is None


# ---------------------------------------------------------------------------
# TestOverlapStrategyEnum
# ---------------------------------------------------------------------------

class TestOverlapStrategyEnum:
    def test_all_strategies_have_string_values(self):
        for member in OverlapStrategy:
            assert isinstance(member.value, str)

    def test_soft_nms_is_alias_for_gaussian(self):
        assert OverlapStrategy.SOFT_NMS.value == OverlapStrategy.GAUSSIAN_SOFT_NMS.value
