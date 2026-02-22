"""Post-processing module: NMS for overlapping segmentation masks.

Implements mask-aware Non-Maximum Suppression (NMS) via the **Strategy Pattern**
(OCP-compliant).  Ten strategies are provided out of the box.  New strategies
can be added by:

  1. Subclassing :class:`NMSStrategy`.
  2. Decorating the subclass with ``@NMSStrategyFactory.register("name")``.
  3. Adding any needed config fields to :class:`PostProcessingConfig` in
     ``src/config_manager.py``.

**Critical pipeline constraint**: ``apply_nms()`` receives *remapped* output
class IDs (0...M-1), never raw SAM3 prompt indices.  Remapping is applied by the
pipeline remap stage before NMS is called.

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from .logging_system import LoggingSystem, trace
from .interfaces import PostProcessor, SegmentationResult, MaskData, ProgressCallback

_logger = LoggingSystem.get_logger(__name__)


class OverlapStrategy(Enum):
    """All supported NMS strategy names."""
    CONFIDENCE = "confidence"
    AREA = "area"
    CLASS_PRIORITY = "class_priority"
    GAUSSIAN_SOFT_NMS = "gaussian_soft_nms"
    LINEAR_SOFT_NMS = "linear_soft_nms"
    WEIGHTED_NMS = "weighted_nms"
    ADAPTIVE_NMS = "adaptive_nms"
    DIOU_NMS = "diou_nms"
    MATRIX_NMS = "matrix_nms"
    MASK_MERGE_NMS = "mask_merge_nms"
    SOFT_NMS = "gaussian_soft_nms"  # legacy alias


class NMSStrategy(ABC):
    """Abstract base for all NMS suppression strategies."""

    @abstractmethod
    def compute_suppression_score(
        self, *, mask_i, mask_j, conf_i, conf_j, class_i, class_j,
        iou, config, class_names=None,
    ) -> float:
        """Return updated confidence for mask_j (0.0 = hard suppress, -1.0 = merge)."""

    @abstractmethod
    def should_suppress(self, new_conf: float, config) -> bool:
        """Return True if mask_j should be removed."""


class NMSStrategyFactory:
    """Registry for NMS strategies  self-register via @NMSStrategyFactory.register(name)."""

    _registry: Dict[str, Type[NMSStrategy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[NMSStrategy]], Type[NMSStrategy]]:
        def decorator(strategy_cls):
            cls._registry[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def create(cls, name: str) -> NMSStrategy:
        if isinstance(name, OverlapStrategy):
            name = name.value
        if name == "soft_nms":
            name = "gaussian_soft_nms"
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"NMSStrategyFactory: unknown strategy {name!r}. Available: {available}")
        return cls._registry[name]()

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry.keys())


@NMSStrategyFactory.register("confidence")
class ConfidenceNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_i, conf_j, **kw) -> float:
        return 0.0 if conf_i >= conf_j else conf_j

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf <= 0.0


@NMSStrategyFactory.register("area")
class AreaNMS(NMSStrategy):
    def compute_suppression_score(self, *, mask_i, mask_j, conf_j, **kw) -> float:
        return 0.0 if mask_i.sum() >= mask_j.sum() else conf_j

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf <= 0.0


@NMSStrategyFactory.register("class_priority")
class ClassPriorityNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_i, conf_j, class_i, class_j, config, class_names=None, **kw) -> float:
        priority = getattr(config, "class_priority", []) or []
        if class_names and priority:
            ni = class_names[class_i] if class_i < len(class_names) else str(class_i)
            nj = class_names[class_j] if class_j < len(class_names) else str(class_j)
            if ni in priority and nj in priority:
                return 0.0 if priority.index(ni) <= priority.index(nj) else conf_j
        return 0.0 if conf_i >= conf_j else conf_j

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf <= 0.0


@NMSStrategyFactory.register("gaussian_soft_nms")
class GaussianSoftNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_j, iou, config, **kw) -> float:
        sigma = getattr(config, "soft_nms_sigma", 0.5)
        return conf_j * float(np.exp(-(iou ** 2) / sigma))

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf < getattr(config, "min_confidence_after_decay", 0.1)


@NMSStrategyFactory.register("linear_soft_nms")
class LinearSoftNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_j, iou, **kw) -> float:
        return conf_j * (1.0 - iou)

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf < getattr(config, "min_confidence_after_decay", 0.1)


@NMSStrategyFactory.register("weighted_nms")
class WeightedNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_i, conf_j, **kw) -> float:
        total = conf_i + conf_j
        return (conf_i * conf_i + conf_j * conf_j) / total if total > 0 else 0.0

    def should_suppress(self, new_conf, config) -> bool:
        return False  # masks are blended, never hard-suppressed


@NMSStrategyFactory.register("adaptive_nms")
class AdaptiveNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_j, iou, config, **kw) -> float:
        density_factor = getattr(config, "adaptive_nms_density_factor", 0.1)
        threshold = getattr(config, "iou_threshold", 0.5) + density_factor
        return 0.0 if iou > threshold else conf_j

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf <= 0.0


@NMSStrategyFactory.register("diou_nms")
class DIoUNMS(NMSStrategy):
    @staticmethod
    def _bbox(mask):
        rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
        if not rows.any():
            return 0.0, 0.0, 0.0, 0.0
        r1, r2 = np.where(rows)[0][[0, -1]]
        c1, c2 = np.where(cols)[0][[0, -1]]
        return float(c1), float(r1), float(c2), float(r2)

    def compute_suppression_score(self, *, mask_i, mask_j, conf_j, iou, config, **kw) -> float:
        x1i, y1i, x2i, y2i = self._bbox(mask_i)
        x1j, y1j, x2j, y2j = self._bbox(mask_j)
        cx_i, cy_i = (x1i+x2i)/2, (y1i+y2i)/2
        cx_j, cy_j = (x1j+x2j)/2, (y1j+y2j)/2
        d2 = (cx_i-cx_j)**2 + (cy_i-cy_j)**2
        diag2 = (max(x2i,x2j)-min(x1i,x1j))**2 + (max(y2i,y2j)-min(y1i,y1j))**2
        diou = iou - (d2/diag2 if diag2 > 0 else 0)
        return 0.0 if diou > getattr(config, "iou_threshold", 0.5) else conf_j

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf <= 0.0


@NMSStrategyFactory.register("matrix_nms")
class MatrixNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_i, conf_j, iou, **kw) -> float:
        return conf_j * max(0.0, 1.0 - iou * conf_i)

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf < getattr(config, "min_confidence_after_decay", 0.1)


@NMSStrategyFactory.register("mask_merge_nms")
class MaskMergeNMS(NMSStrategy):
    def compute_suppression_score(self, *, conf_i, conf_j, class_i, class_j, iou, config, **kw) -> float:
        merge_thresh = getattr(config, "mask_merge_threshold", 0.7)
        if class_i == class_j and iou > merge_thresh:
            return -1.0  # merge sentinel
        return 0.0 if conf_i >= conf_j else conf_j

    def should_suppress(self, new_conf, config) -> bool:
        return new_conf <= 0.0


def _calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1, m2 = mask1.astype(bool), mask2.astype(bool)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def _calculate_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float]:
    m1, m2 = mask1.astype(bool), mask2.astype(bool)
    inter = np.logical_and(m1, m2).sum()
    a1, a2 = m1.sum(), m2.sum()
    return (float(inter/a1) if a1 > 0 else 0.0, float(inter/a2) if a2 > 0 else 0.0)


class MaskPostProcessor:
    """Decoupled NMS post-processor.  Implements PostProcessor protocol.

    Receives only a PostProcessingConfig slice (ISP) and optional class_names
    from ClassRegistry.  Never receives a raw Config object.
    """

    def __init__(self, config: Any, class_names: Optional[List[str]] = None) -> None:
        self.config = config
        self.class_names = class_names
        strategy_name = getattr(config, "strategy", "confidence")
        self._strategy: NMSStrategy = NMSStrategyFactory.create(strategy_name)
        self._stats: Dict[str, int] = {
            "total_processed": 0, "overlaps_detected": 0,
            "masks_suppressed": 0, "confidence_decayed": 0, "masks_merged": 0,
        }
        _logger.info("MaskPostProcessor: strategy=%s, iou_threshold=%.2f",
                     strategy_name, getattr(config, "iou_threshold", 0.5))

    @trace
    def apply_nms(self, result: SegmentationResult, *, callback: Optional[ProgressCallback] = None) -> SegmentationResult:
        if not getattr(self.config, "enabled", True) or len(result.masks) == 0:
            return result
        if getattr(self.config, "enable_class_specific", False):
            return self._apply_class_specific(result)
        filtered = self._run_nms(
            list(result.masks),
            [m.class_id for m in result.masks],
            [m.confidence for m in result.masks],
        )
        return self._rebuild(result, filtered)

    @trace
    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    @trace
    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0

    def _run_nms(self, mask_data: List[MaskData], class_ids: List[int], confidences: List[float]) -> List[MaskData]:
        n = len(mask_data)
        self._stats["total_processed"] += n
        confs = np.array(confidences, dtype=float)
        sorted_idx = np.argsort(-confs)
        keep = np.ones(n, dtype=bool)
        merge_targets: Dict[int, int] = {}

        for i_pos, idx_i in enumerate(sorted_idx):
            if not keep[idx_i]:
                continue
            md_i = mask_data[idx_i]
            mask_i = np.asarray(md_i.mask)
            conf_i = float(confs[idx_i])
            cls_i = class_ids[idx_i]

            for j_pos in range(i_pos + 1, n):
                idx_j = sorted_idx[j_pos]
                if not keep[idx_j]:
                    continue
                md_j = mask_data[idx_j]
                mask_j = np.asarray(md_j.mask)
                conf_j = float(confs[idx_j])
                cls_j = class_ids[idx_j]

                iou = _calculate_mask_iou(mask_i, mask_j)
                if iou <= getattr(self.config, "iou_threshold", 0.5):
                    continue

                self._stats["overlaps_detected"] += 1
                new_conf = self._strategy.compute_suppression_score(
                    mask_i=mask_i, mask_j=mask_j,
                    conf_i=conf_i, conf_j=conf_j,
                    class_i=cls_i, class_j=cls_j,
                    iou=iou, config=self.config, class_names=self.class_names,
                )

                if new_conf == -1.0:
                    merge_targets[int(idx_j)] = int(idx_i)
                    keep[idx_j] = False
                    self._stats["masks_merged"] += 1
                elif self._strategy.should_suppress(new_conf, self.config):
                    keep[idx_j] = False
                    self._stats["masks_suppressed"] += 1
                else:
                    confs[idx_j] = new_conf
                    self._stats["confidence_decayed"] += 1
                    if self._strategy.should_suppress(new_conf, self.config):
                        keep[idx_j] = False
                        self._stats["masks_suppressed"] += 1

        for j_idx, i_idx in merge_targets.items():
            merged = np.logical_or(np.asarray(mask_data[i_idx].mask), np.asarray(mask_data[j_idx].mask))
            md = mask_data[i_idx]
            mask_data[i_idx] = MaskData(mask=merged, confidence=md.confidence,
                                         class_id=md.class_id, area=int(merged.sum()),
                                         bbox=md.bbox, polygon=None)

        return [mask_data[i] for i in np.where(keep)[0]]

    def _apply_class_specific(self, result: SegmentationResult) -> SegmentationResult:
        groups: Dict[int, List[int]] = {}
        for idx, md in enumerate(result.masks):
            groups.setdefault(md.class_id, []).append(idx)
        final: List[MaskData] = []
        for cls_id, indices in groups.items():
            grp = [result.masks[i] for i in indices]
            final.extend(self._run_nms(grp, [m.class_id for m in grp], [m.confidence for m in grp]))
        return self._rebuild(result, final)

    @staticmethod
    def _rebuild(original: SegmentationResult, masks: List[MaskData]) -> SegmentationResult:
        return SegmentationResult(
            image_path=original.image_path, masks=masks,
            image_width=original.image_width, image_height=original.image_height,
            inference_time_ms=original.inference_time_ms, device=original.device,
        )

    def calculate_mask_iou(self, m1: np.ndarray, m2: np.ndarray) -> float:
        return _calculate_mask_iou(m1, m2)

    def calculate_mask_overlap(self, m1: np.ndarray, m2: np.ndarray) -> Tuple[float, float]:
        return _calculate_mask_overlap(m1, m2)


def create_post_processor(post_processing_config: Any, class_names: Optional[List[str]] = None) -> MaskPostProcessor:
    """Factory: build MaskPostProcessor from a PostProcessingConfig slice."""
    return MaskPostProcessor(post_processing_config, class_names=class_names)
