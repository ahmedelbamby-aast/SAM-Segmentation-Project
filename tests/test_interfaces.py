"""Unit tests for src/interfaces.py.

Covers data structures (MaskData, SegmentationResult, ProcessingStats)
and Protocol runtime_checkable isinstance checks for all Protocols.

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interfaces import (
    Filter,
    MaskData,
    PostProcessor,
    ProgressCallback,
    Processor,
    ProcessingStats,
    SegmentationResult,
    Segmentor,
    Tracker,
    Uploader,
    Writer,
)


# ---------------------------------------------------------------------------
# MaskData
# ---------------------------------------------------------------------------


class TestMaskData:
    def test_basic_construction(self):
        mask = MaskData(
            mask=None,
            confidence=0.9,
            class_id=0,
            area=1024,
            bbox=(10, 20, 50, 60),
        )
        assert mask.confidence == 0.9
        assert mask.class_id == 0
        assert mask.area == 1024
        assert mask.bbox == (10, 20, 50, 60)
        assert mask.polygon is None

    def test_polygon_can_be_set(self):
        mask = MaskData(
            mask=None,
            confidence=0.5,
            class_id=1,
            area=200,
            bbox=(0, 0, 10, 10),
            polygon=[0.1, 0.1, 0.9, 0.1, 0.9, 0.9],
        )
        assert mask.polygon == [0.1, 0.1, 0.9, 0.1, 0.9, 0.9]

    def test_mask_accepts_any_type(self):
        import array
        arr = array.array("b", [0, 1, 0, 1])
        mask = MaskData(mask=arr, confidence=0.8, class_id=0, area=2, bbox=(0, 0, 2, 1))
        assert mask.mask is arr

    def test_fields_are_mutable(self):
        mask = MaskData(mask=None, confidence=0.3, class_id=0, area=10, bbox=(0, 0, 1, 1))
        mask.confidence = 0.99
        assert mask.confidence == 0.99


# ---------------------------------------------------------------------------
# SegmentationResult
# ---------------------------------------------------------------------------


class TestSegmentationResult:
    def _make_mask(self) -> MaskData:
        return MaskData(mask=None, confidence=0.7, class_id=0, area=100, bbox=(0, 0, 10, 10))

    def test_basic_construction(self):
        result = SegmentationResult(
            image_path=Path("/tmp/img.jpg"),
            masks=[self._make_mask()],
            image_width=640,
            image_height=480,
        )
        assert result.image_path == Path("/tmp/img.jpg")
        assert len(result.masks) == 1
        assert result.image_width == 640
        assert result.image_height == 480
        assert result.inference_time_ms == 0.0
        assert result.device == "cpu"

    def test_custom_defaults(self):
        result = SegmentationResult(
            image_path=Path("/a/b.png"),
            masks=[],
            image_width=100,
            image_height=100,
            inference_time_ms=42.5,
            device="cuda:0",
        )
        assert result.inference_time_ms == 42.5
        assert result.device == "cuda:0"

    def test_empty_masks_list(self):
        result = SegmentationResult(
            image_path=Path("/no/masks.jpg"),
            masks=[],
            image_width=320,
            image_height=240,
        )
        assert result.masks == []

    def test_multiple_masks(self):
        masks = [self._make_mask() for _ in range(5)]
        result = SegmentationResult(
            image_path=Path("/x.jpg"), masks=masks, image_width=640, image_height=480
        )
        assert len(result.masks) == 5


# ---------------------------------------------------------------------------
# ProcessingStats
# ---------------------------------------------------------------------------


class TestProcessingStats:
    def test_default_values(self):
        stats = ProcessingStats()
        assert stats.processed == 0
        assert stats.skipped == 0
        assert stats.errors == 0
        assert stats.total_time_ms == 0.0
        assert stats.extra == {}

    def test_custom_values(self):
        stats = ProcessingStats(processed=10, skipped=2, errors=1, total_time_ms=500.0)
        assert stats.processed == 10
        assert stats.skipped == 2
        assert stats.errors == 1
        assert stats.total_time_ms == 500.0

    def test_extra_dict_is_mutable(self):
        stats = ProcessingStats()
        stats.extra["masks_merged"] = 3
        assert stats.extra["masks_merged"] == 3

    def test_extra_not_shared_between_instances(self):
        a = ProcessingStats()
        b = ProcessingStats()
        a.extra["x"] = 1
        assert "x" not in b.extra


# ---------------------------------------------------------------------------
# ProgressCallback Protocol
# ---------------------------------------------------------------------------


class TestProgressCallbackProtocol:
    def _make_impl(self) -> ProgressCallback:
        class ConcreteCallback:
            def on_item_start(self, item_id: str) -> None:
                pass
            def on_item_complete(self, item_id: str) -> None:
                pass
            def on_item_error(self, item_id: str, error: Exception) -> None:
                pass

        return ConcreteCallback()

    def test_isinstance_check_passes(self):
        impl = self._make_impl()
        assert isinstance(impl, ProgressCallback)

    def test_missing_method_fails_isinstance(self):
        class Incomplete:
            def on_item_start(self, item_id: str) -> None:
                pass
            # missing on_item_complete and on_item_error

        assert not isinstance(Incomplete(), ProgressCallback)

    def test_all_three_methods_callable(self):
        impl = self._make_impl()
        impl.on_item_start("img_001")
        impl.on_item_complete("img_001")
        impl.on_item_error("img_002", RuntimeError("oops"))


# ---------------------------------------------------------------------------
# Segmentor Protocol
# ---------------------------------------------------------------------------


class TestSegmentorProtocol:
    def _make_impl(self) -> Any:
        class ConcreteSegmentor:
            def process_image(self, image_path: Path, *, callback=None) -> SegmentationResult:
                return SegmentationResult(
                    image_path=image_path, masks=[], image_width=0, image_height=0
                )

            def process_batch(self, image_paths, *, callback=None):
                return iter([])

            def get_device_info(self) -> Dict[str, Any]:
                return {"device": "cpu"}

            def cleanup(self) -> None:
                pass

        return ConcreteSegmentor()

    def test_isinstance_passes(self):
        assert isinstance(self._make_impl(), Segmentor)

    def test_missing_cleanup_fails(self):
        class Incomplete:
            def process_image(self, p, *, callback=None): ...
            def process_batch(self, paths, *, callback=None): ...
            def get_device_info(self): ...
            # no cleanup

        assert not isinstance(Incomplete(), Segmentor)

    def test_process_image_returns_result(self):
        impl = self._make_impl()
        result = impl.process_image(Path("/x.jpg"))
        assert isinstance(result, SegmentationResult)


# ---------------------------------------------------------------------------
# PostProcessor Protocol
# ---------------------------------------------------------------------------


class TestPostProcessorProtocol:
    def _make_impl(self) -> Any:
        class ConcretePostProcessor:
            def apply_nms(self, result: SegmentationResult, *, callback=None) -> SegmentationResult:
                return result
            def get_stats(self) -> ProcessingStats:
                return ProcessingStats()
            def reset_stats(self) -> None:
                pass

        return ConcretePostProcessor()

    def test_isinstance_passes(self):
        assert isinstance(self._make_impl(), PostProcessor)

    def test_missing_reset_stats_fails(self):
        class Incomplete:
            def apply_nms(self, r, *, callback=None): ...
            def get_stats(self): ...

        assert not isinstance(Incomplete(), PostProcessor)


# ---------------------------------------------------------------------------
# Filter Protocol
# ---------------------------------------------------------------------------


class TestFilterProtocol:
    def _make_impl(self) -> Any:
        class ConcreteFilter:
            def filter_result(self, result, *, callback=None): return result
            def get_stats(self): return ProcessingStats()
            def reset_stats(self): pass

        return ConcreteFilter()

    def test_isinstance_passes(self):
        assert isinstance(self._make_impl(), Filter)


# ---------------------------------------------------------------------------
# Writer Protocol
# ---------------------------------------------------------------------------


class TestWriterProtocol:
    def _make_impl(self) -> Any:
        class ConcreteWriter:
            def write_annotation(self, result, *, split="train", callback=None): return Path("/out.txt")
            def write_data_yaml(self): return Path("/data.yaml")
            def get_stats(self): return ProcessingStats()
            def reset_stats(self): pass

        return ConcreteWriter()

    def test_isinstance_passes(self):
        assert isinstance(self._make_impl(), Writer)


# ---------------------------------------------------------------------------
# Tracker Protocol
# ---------------------------------------------------------------------------


class TestTrackerProtocol:
    def _make_impl(self) -> Any:
        class ConcreteTracker:
            def create_job(self, job_name, total_images): pass
            def mark_completed(self, image_path, stage): pass
            def mark_error(self, image_path, stage, error): pass
            def get_progress(self, job_name): return {}
            def checkpoint(self, job_name): return []

        return ConcreteTracker()

    def test_isinstance_passes(self):
        assert isinstance(self._make_impl(), Tracker)

    def test_missing_checkpoint_fails(self):
        class Incomplete:
            def create_job(self, *a): ...
            def mark_completed(self, *a): ...
            def mark_error(self, *a): ...
            def get_progress(self, *a): ...

        assert not isinstance(Incomplete(), Tracker)


# ---------------------------------------------------------------------------
# Uploader Protocol
# ---------------------------------------------------------------------------


class TestUploaderProtocol:
    def _make_impl(self) -> Any:
        class ConcreteUploader:
            def queue_batch(self, image_paths, annotation_paths): pass
            def wait_for_uploads(self): return ProcessingStats()
            def shutdown(self, wait=True): pass

        return ConcreteUploader()

    def test_isinstance_passes(self):
        assert isinstance(self._make_impl(), Uploader)


# ---------------------------------------------------------------------------
# Processor Protocol
# ---------------------------------------------------------------------------


class TestProcessorProtocol:
    def _make_impl(self) -> Any:
        class ConcreteProcessor:
            def start(self): pass
            def process_batch(self, image_paths, *, callback=None): return iter([])
            def shutdown(self, wait=True): pass

        return ConcreteProcessor()

    def test_isinstance_passes(self):
        assert isinstance(self._make_impl(), Processor)
