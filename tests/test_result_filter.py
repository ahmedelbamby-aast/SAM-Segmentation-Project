"""
Unit tests for the ResultFilter module.

Tests cover all public methods of :class:`~src.result_filter.ResultFilter`
and the :class:`~src.result_filter.FilterStats` dataclass.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch
import pytest

from src.result_filter import ResultFilter, FilterStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_config(tmp_path: Path):
    """Return a minimal pipeline config mock aimed at ``tmp_path``."""
    cfg = MagicMock()
    cfg.output_dir = str(tmp_path / "output")
    cfg.neither_dir = str(tmp_path / "neither")
    return cfg


def _make_result(num_det: Optional[int] = None, masks=None, class_ids=None):
    """Build a lightweight SegmentationResult-like object."""
    obj = SimpleNamespace()
    if num_det is not None:
        obj.num_detections = num_det
    if masks is not None:
        obj.masks = masks
    if class_ids is not None:
        obj.class_ids = class_ids
    return obj


def _make_image(tmp_path: Path, name: str = "frame_001.jpg") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header
    return p


# ---------------------------------------------------------------------------
# FilterStats tests
# ---------------------------------------------------------------------------

class TestFilterStats:
    def test_defaults(self):
        stats = FilterStats()
        assert stats.total_processed == 0
        assert stats.with_detections == 0
        assert stats.no_detections == 0

    def test_detection_rate_zero_when_no_images(self):
        stats = FilterStats()
        assert stats.detection_rate == 0.0

    def test_detection_rate_100_percent(self):
        stats = FilterStats(total_processed=10, with_detections=10, no_detections=0)
        assert stats.detection_rate == pytest.approx(100.0)

    def test_detection_rate_50_percent(self):
        stats = FilterStats(total_processed=4, with_detections=2, no_detections=2)
        assert stats.detection_rate == pytest.approx(50.0)

    def test_detection_rate_partial(self):
        stats = FilterStats(total_processed=3, with_detections=1, no_detections=2)
        assert stats.detection_rate == pytest.approx(100 / 3)


# ---------------------------------------------------------------------------
# ResultFilter.__init__ tests
# ---------------------------------------------------------------------------

class TestResultFilterInit:
    def test_creates_neither_images_directory(self, tmp_path):
        cfg = _make_pipeline_config(tmp_path)
        rf = ResultFilter(cfg)
        assert (tmp_path / "neither" / "images").is_dir()

    def test_stats_start_at_zero(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        stats = rf.get_stats()
        assert stats["total_processed"] == 0
        assert stats["with_detections"] == 0
        assert stats["no_detections"] == 0

    def test_filtered_images_start_empty(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        assert rf.get_filtered_images() == []

    def test_neither_dir_stored_correctly(self, tmp_path):
        cfg = _make_pipeline_config(tmp_path)
        rf = ResultFilter(cfg)
        assert rf.neither_dir == tmp_path / "neither"

    def test_output_dir_stored_correctly(self, tmp_path):
        cfg = _make_pipeline_config(tmp_path)
        rf = ResultFilter(cfg)
        assert rf.output_dir == tmp_path / "output"


# ---------------------------------------------------------------------------
# ResultFilter.filter_result tests
# ---------------------------------------------------------------------------

class TestFilterResult:
    def test_returns_true_for_result_with_detections(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        result = _make_result(num_det=2)
        assert rf.filter_result(img, result) is True

    def test_returns_false_for_none_result(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        assert rf.filter_result(img, None) is False

    def test_returns_false_for_zero_detections(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        result = _make_result(num_det=0)
        assert rf.filter_result(img, result) is False

    def test_increments_total_processed(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, _make_result(num_det=1))
        rf.filter_result(img, None)
        assert rf.stats.total_processed == 2

    def test_increments_with_detections(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, _make_result(num_det=3))
        assert rf.stats.with_detections == 1
        assert rf.stats.no_detections == 0

    def test_increments_no_detections(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, None)
        assert rf.stats.no_detections == 1
        assert rf.stats.with_detections == 0

    def test_copies_to_neither_by_default(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path, "test.jpg")
        rf.filter_result(img, None)
        assert (tmp_path / "neither" / "images" / "test.jpg").exists()

    def test_does_not_copy_when_copy_to_neither_false(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path, "test.jpg")
        rf.filter_result(img, None, copy_to_neither=False)
        assert not (tmp_path / "neither" / "images" / "test.jpg").exists()

    def test_image_added_to_filtered_list_on_no_detection(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, None)
        assert img in rf.get_filtered_images()

    def test_image_not_added_to_filtered_list_on_detection(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, _make_result(num_det=1))
        assert rf.get_filtered_images() == []


# ---------------------------------------------------------------------------
# ResultFilter._has_valid_detections tests
# ---------------------------------------------------------------------------

class TestHasValidDetections:
    def test_none_result_returns_false(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        assert rf._has_valid_detections(None) is False

    def test_num_detections_positive_returns_true(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        assert rf._has_valid_detections(_make_result(num_det=1)) is True

    def test_num_detections_zero_returns_false(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        assert rf._has_valid_detections(_make_result(num_det=0)) is False

    def test_fallback_with_masks_and_class_ids(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        result = _make_result(masks=["mask1"], class_ids=[0])
        assert rf._has_valid_detections(result) is True

    def test_fallback_empty_masks_returns_false(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        result = _make_result(masks=[], class_ids=[])
        assert rf._has_valid_detections(result) is False

    def test_fallback_none_masks_returns_false(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        result = _make_result(masks=None, class_ids=[])
        assert rf._has_valid_detections(result) is False

    def test_result_with_no_known_attrs_returns_false(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        assert rf._has_valid_detections(SimpleNamespace()) is False


# ---------------------------------------------------------------------------
# ResultFilter.get_stats tests
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_dict_with_required_keys(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        stats = rf.get_stats()
        for key in ("total_processed", "with_detections", "no_detections",
                    "detection_rate", "neither_folder"):
            assert key in stats

    def test_detection_rate_formatted_as_string(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, _make_result(num_det=1))
        rate = rf.get_stats()["detection_rate"]
        assert isinstance(rate, str)
        assert "%" in rate

    def test_neither_folder_as_string(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        assert isinstance(rf.get_stats()["neither_folder"], str)


# ---------------------------------------------------------------------------
# ResultFilter.get_filtered_images tests
# ---------------------------------------------------------------------------

class TestGetFilteredImages:
    def test_returns_copy_not_reference(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, None)
        copy = rf.get_filtered_images()
        copy.clear()
        # Internal list must not be affected
        assert len(rf.get_filtered_images()) == 1

    def test_multiple_filtered_images(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        for i in range(3):
            img = _make_image(tmp_path, f"img_{i}.jpg")
            rf.filter_result(img, None, copy_to_neither=False)
        assert len(rf.get_filtered_images()) == 3


# ---------------------------------------------------------------------------
# ResultFilter.get_neither_count tests
# ---------------------------------------------------------------------------

class TestGetNeitherCount:
    def test_zero_initially(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        assert rf.get_neither_count() == 0

    def test_count_increases_after_filter(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        for i in range(2):
            img = _make_image(tmp_path, f"img_{i}.jpg")
            rf.filter_result(img, None)
        assert rf.get_neither_count() == 2

    def test_ignores_images_not_copied(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, None, copy_to_neither=False)
        assert rf.get_neither_count() == 0

    def test_returns_zero_when_images_dir_missing(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        # Manually remove the directory created in __init__
        shutil.rmtree(tmp_path / "neither")
        assert rf.get_neither_count() == 0


# ---------------------------------------------------------------------------
# ResultFilter.write_neither_manifest tests
# ---------------------------------------------------------------------------

class TestWriteNeitherManifest:
    def test_creates_manifest_file(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path, "abc.jpg")
        rf.filter_result(img, None, copy_to_neither=False)
        manifest = rf.write_neither_manifest()
        assert manifest.exists()

    def test_returns_path_to_manifest(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        manifest = rf.write_neither_manifest()
        assert isinstance(manifest, Path)
        assert manifest.name == "manifest.txt"

    def test_manifest_contains_image_names(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path, "img_a.jpg")
        rf.filter_result(img, None, copy_to_neither=False)
        manifest = rf.write_neither_manifest()
        text = manifest.read_text(encoding="utf-8")
        assert "img_a.jpg" in text

    def test_manifest_contains_header_comment(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        manifest = rf.write_neither_manifest()
        text = manifest.read_text(encoding="utf-8")
        assert text.startswith("#")

    def test_manifest_empty_when_no_filtered_images(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        manifest = rf.write_neither_manifest()
        text = manifest.read_text(encoding="utf-8")
        # Should only have comment lines, no image names
        data_lines = [l for l in text.splitlines() if l and not l.startswith("#")]
        assert data_lines == []


# ---------------------------------------------------------------------------
# ResultFilter.reset_stats tests
# ---------------------------------------------------------------------------

class TestResetStats:
    def test_resets_counters_to_zero(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, _make_result(num_det=1))
        rf.filter_result(img, None, copy_to_neither=False)
        rf.reset_stats()
        stats = rf.get_stats()
        assert stats["total_processed"] == 0
        assert stats["with_detections"] == 0
        assert stats["no_detections"] == 0

    def test_clears_filtered_images_list(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        img = _make_image(tmp_path)
        rf.filter_result(img, None, copy_to_neither=False)
        rf.reset_stats()
        assert rf.get_filtered_images() == []


# ---------------------------------------------------------------------------
# ResultFilter._move_to_neither error handling
# ---------------------------------------------------------------------------

class TestMoveToNeitherError:
    def test_logs_error_on_copy_failure(self, tmp_path):
        rf = ResultFilter(_make_pipeline_config(tmp_path))
        nonexistent = tmp_path / "ghost.jpg"  # file does not exist
        # Should not raise — errors are logged, not propagated
        rf._move_to_neither(nonexistent)
