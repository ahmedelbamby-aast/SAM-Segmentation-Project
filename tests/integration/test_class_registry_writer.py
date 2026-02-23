"""
Integration tests: ClassRegistry + AnnotationWriter YOLO output pipeline.

Verifies that:
- ``ClassRegistry`` drives YOLO class IDs and ``data.yaml`` name mapping.
- Many-to-one remapping flows through ``AnnotationWriter`` correctly.
- Written label files contain YOLO polygon format with correct class IDs.
- ``data.yaml`` class names match the registry's output class names.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
import pytest
import yaml

from src.class_registry import ClassRegistry
from src.annotation_writer import AnnotationWriter, MaskConverter, DatasetMetadataWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_config(tmp_path: Path):
    cfg = MagicMock()
    cfg.output_dir = str(tmp_path / "output")
    return cfg


def _make_image(tmp_path: Path, name: str = "frame_01.jpg") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    return p


def _make_binary_mask(h: int = 32, w: int = 32) -> np.ndarray:
    """Return a simple filled-rectangle binary mask (0/1 values as float32).
    
    NOTE: mask_to_polygon() multiplies by 255 internally, so masks must use
    0/1 values here, not 0/255. The filled region must be large enough to
    pass the cv2 contour area > 100 threshold.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    # fill a 15x15 region → area = 225 > 100 threshold in mask_to_polygon
    mask[4:19, 4:19] = 1.0
    return mask


def _make_result(class_ids, masks):
    """Build a minimal segmentation-result-like object."""
    obj = SimpleNamespace()
    obj.masks = masks
    obj.class_ids = class_ids
    obj.num_detections = len(class_ids)
    return obj


# ---------------------------------------------------------------------------
# 1. ClassRegistry → AnnotationWriter: identity (1:1) mapping
# ---------------------------------------------------------------------------

class TestIdentityMapping:
    """ClassRegistry with no remapping → each prompt is its own class."""

    def test_two_classes_produce_two_yolo_ids(self, tmp_path):
        registry = ClassRegistry(prompts=["teacher", "student"])
        cfg = _make_pipeline_config(tmp_path)
        writer = AnnotationWriter(cfg, registry)

        img = _make_image(tmp_path)
        masks = np.stack([_make_binary_mask(), _make_binary_mask()])
        result = _make_result(class_ids=[0, 1], masks=masks)

        label_path = writer.write_annotation(img, result, split="train", copy_image=False)
        assert label_path is not None and label_path.exists()

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        ids = {int(line.split()[0]) for line in lines}
        assert ids == {0, 1}

    def test_data_yaml_names_match_registry(self, tmp_path):
        registry = ClassRegistry(prompts=["teacher", "student"])
        cfg = _make_pipeline_config(tmp_path)
        writer = AnnotationWriter(cfg, registry)

        yaml_path = writer.write_data_yaml()
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

        assert data["nc"] == 2
        assert data["names"][0] == "teacher"
        assert data["names"][1] == "student"

    def test_yolo_polygon_label_format(self, tmp_path):
        """Label file must be: <class_id> <x1> <y1> ... with floats."""
        registry = ClassRegistry(prompts=["teacher"])
        cfg = _make_pipeline_config(tmp_path)
        writer = AnnotationWriter(cfg, registry)

        img = _make_image(tmp_path)
        masks = np.stack([_make_binary_mask()])
        result = _make_result(class_ids=[0], masks=masks)

        label_path = writer.write_annotation(img, result, split="train", copy_image=False)
        line = label_path.read_text(encoding="utf-8").strip()
        parts = line.split()
        assert parts[0] == "0"
        # All coordinate values must be parseable floats
        for coord in parts[1:]:
            float(coord)


# ---------------------------------------------------------------------------
# 2. ClassRegistry → AnnotationWriter: many-to-one remapping
# ---------------------------------------------------------------------------

class TestManyToOneRemapping:
    """5 prompts collapsed to 2 output classes via class_remapping."""

    def _make_registry(self):
        return ClassRegistry(
            prompts=["teacher", "student", "kid", "child", "Adult"],
            class_remapping={"Adult": "teacher", "kid": "student", "child": "student"},
        )

    def test_num_classes_is_two(self):
        reg = self._make_registry()
        assert reg.num_classes == 2

    def test_class_names_are_teacher_and_student(self):
        reg = self._make_registry()
        assert set(reg.class_names) == {"teacher", "student"}

    def test_adult_remapped_to_teacher_id(self):
        reg = self._make_registry()
        # "Adult" is prompt index 4
        adult_idx = reg.prompts.index("Adult")
        remapped_id = reg.remap_prompt_index(adult_idx)
        assert reg.get_class_name(remapped_id) == "teacher"

    def test_kid_remapped_to_student_id(self):
        reg = self._make_registry()
        kid_idx = reg.prompts.index("kid")
        remapped_id = reg.remap_prompt_index(kid_idx)
        assert reg.get_class_name(remapped_id) == "student"

    def test_remapped_class_ids_written_to_yolo_label(self, tmp_path):
        """After remapping 5 prompts → 2 classes, annotations must use 0/1 only."""
        reg = self._make_registry()
        cfg = _make_pipeline_config(tmp_path)
        writer = AnnotationWriter(cfg, reg)

        img = _make_image(tmp_path)
        # Simulate 3 detections, each of a different remapped output class
        teacher_id = reg.remap_prompt_index(reg.prompts.index("teacher"))  # stays teacher
        student_id = reg.remap_prompt_index(reg.prompts.index("student"))  # stays student
        adult_id   = reg.remap_prompt_index(reg.prompts.index("Adult"))    # → teacher

        masks = np.stack([_make_binary_mask()] * 3)
        result = _make_result(
            class_ids=[teacher_id, student_id, adult_id],
            masks=masks,
        )

        label_path = writer.write_annotation(img, result, split="train", copy_image=False)
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        written_ids = {int(l.split()[0]) for l in lines}
        assert written_ids.issubset({0, 1})

    def test_data_yaml_has_two_classes_with_correct_names(self, tmp_path):
        reg = self._make_registry()
        m = DatasetMetadataWriter(tmp_path / "output", reg.class_names)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)
        yaml_path = m.write_data_yaml()
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert data["nc"] == 2
        assert set(data["names"].values()) == {"teacher", "student"}


# ---------------------------------------------------------------------------
# 3. ClassRegistry → AnnotationWriter: IPC round-trip (to_dict / from_dict)
# ---------------------------------------------------------------------------

class TestIPCRoundTrip:
    def test_from_dict_produces_identical_registry(self):
        reg = ClassRegistry(
            prompts=["teacher", "student", "adult"],
            class_remapping={"adult": "teacher"},
        )
        reg2 = ClassRegistry.from_dict(reg.to_dict())

        assert reg2.class_names == reg.class_names
        assert reg2.num_classes == reg.num_classes
        for i in range(reg.num_prompts):
            assert reg2.remap_prompt_index(i) == reg.remap_prompt_index(i)

    def test_from_dict_writer_produces_same_yaml(self, tmp_path):
        reg = ClassRegistry(prompts=["a", "b"])
        reg2 = ClassRegistry.from_dict(reg.to_dict())

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        out1.mkdir()
        out2.mkdir()

        m1 = DatasetMetadataWriter(out1, reg.class_names)
        m2 = DatasetMetadataWriter(out2, reg2.class_names)
        y1 = yaml.safe_load(m1.write_data_yaml().read_text(encoding="utf-8"))
        y2 = yaml.safe_load(m2.write_data_yaml().read_text(encoding="utf-8"))

        assert y1["names"] == y2["names"]


# ---------------------------------------------------------------------------
# 4. MaskConverter standalone: output feeds into AnnotationWriter correctly
# ---------------------------------------------------------------------------

class TestMaskConverterIntegration:
    def test_masks_to_polygons_class_ids_preserved(self):
        converter = MaskConverter()
        masks = np.stack([_make_binary_mask()] * 3)
        polygons = converter.masks_to_polygons(masks, class_ids=[0, 1, 0])
        assert len(polygons) == 3
        ids = [p[0] for p in polygons]
        assert ids == [0, 1, 0]

    def test_polygon_coords_are_normalised(self):
        converter = MaskConverter()
        mask = _make_binary_mask(16, 16)
        polygon = converter.mask_to_polygon(mask)
        assert all(0.0 <= v <= 1.0 for v in polygon)

    def test_empty_mask_gives_no_polygon(self):
        converter = MaskConverter()
        blank = np.zeros((16, 16), dtype=np.uint8)
        polygon = converter.mask_to_polygon(blank)
        assert len(polygon) == 0

    def test_masks_to_polygons_skips_empty_masks(self):
        converter = MaskConverter()
        good_mask = _make_binary_mask()
        blank_mask = np.zeros_like(good_mask)
        masks = np.stack([good_mask, blank_mask])
        polygons = converter.masks_to_polygons(masks, class_ids=[0, 1])
        # Only the good mask should yield a result
        assert len(polygons) == 1
        assert polygons[0][0] == 0


# ---------------------------------------------------------------------------
# 5. Empty / edge-case annotations
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_result_writes_no_label(self, tmp_path):
        registry = ClassRegistry(prompts=["teacher"])
        cfg = _make_pipeline_config(tmp_path)
        writer = AnnotationWriter(cfg, registry)
        img = _make_image(tmp_path)
        result = _make_result(class_ids=[], masks=np.empty((0, 16, 16)))
        label = writer.write_annotation(img, result, split="train", copy_image=False)
        assert label is None

    def test_none_result_writes_empty_annotation(self, tmp_path):
        registry = ClassRegistry(prompts=["teacher"])
        cfg = _make_pipeline_config(tmp_path)
        writer = AnnotationWriter(cfg, registry)
        img = _make_image(tmp_path)
        label = writer.write_empty_annotation(img, split="train")
        assert label.exists()
        assert label.read_text(encoding="utf-8") == ""

    def test_get_stats_increments_on_write(self, tmp_path):
        registry = ClassRegistry(prompts=["teacher"])
        cfg = _make_pipeline_config(tmp_path)
        writer = AnnotationWriter(cfg, registry)
        img = _make_image(tmp_path)
        masks = np.stack([_make_binary_mask()])
        result = _make_result(class_ids=[0], masks=masks)
        writer.write_annotation(img, result, split="train", copy_image=False)
        stats = writer.get_stats()
        # get_stats() returns {split: {"images": N, "labels": N, "annotations": N}}
        assert stats["train"]["labels"] >= 1
