"""
Integration tests: Config loading → module initialisation → execution.

Verifies that each module can be initialised from a real (or minimal mock)
config object and that its basic execution path runs without errors.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.config_manager import load_config, PipelineConfig
from src.class_registry import ClassRegistry
from src.annotation_writer import AnnotationWriter
from src.result_filter import ResultFilter
from src.validator import Validator
from src.progress_tracker import ProgressTracker
from src.progress_display import ModuleProgressManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


def _minimal_pipeline_config(tmp_path: Path):
    """Return a minimal PipelineConfig-like mock pointing at tmp_path."""
    cfg = MagicMock(spec=PipelineConfig)
    cfg.output_dir = str(tmp_path / "output")
    cfg.neither_dir = str(tmp_path / "neither")
    cfg.input_dir = str(tmp_path / "input")
    cfg.resolution = 640
    cfg.supported_formats = [".jpg", ".png"]
    cfg.num_workers = 1
    cfg.input_mode = "flat"
    cfg.train_percent = 0.7
    cfg.valid_percent = 0.15
    cfg.test_percent = 0.15
    return cfg


# ---------------------------------------------------------------------------
# 1. load_config → Config object structure
# ---------------------------------------------------------------------------

class TestLoadConfig:
    @pytest.mark.skipif(
        not CONFIG_PATH.exists(),
        reason="config/config.yaml not present in workspace",
    )
    def test_load_config_returns_config_object(self):
        config = load_config(str(CONFIG_PATH))
        assert hasattr(config, "pipeline")
        assert hasattr(config, "model")
        assert hasattr(config, "progress")
        assert hasattr(config, "roboflow")

    @pytest.mark.skipif(
        not CONFIG_PATH.exists(),
        reason="config/config.yaml not present in workspace",
    )
    def test_pipeline_config_has_required_fields(self):
        config = load_config(str(CONFIG_PATH))
        pipeline = config.pipeline
        assert hasattr(pipeline, "input_dir")
        assert hasattr(pipeline, "output_dir")
        assert hasattr(pipeline, "resolution")

    @pytest.mark.skipif(
        not CONFIG_PATH.exists(),
        reason="config/config.yaml not present in workspace",
    )
    def test_model_config_has_prompts(self):
        config = load_config(str(CONFIG_PATH))
        assert hasattr(config.model, "prompts")
        assert len(config.model.prompts) >= 1


# ---------------------------------------------------------------------------
# 2. Config → ClassRegistry initialisation
# ---------------------------------------------------------------------------

class TestClassRegistryFromConfig:
    @pytest.mark.skipif(
        not CONFIG_PATH.exists(),
        reason="config/config.yaml not present in workspace",
    )
    def test_from_config_builds_registry(self):
        config = load_config(str(CONFIG_PATH))
        # from_config accepts ModelConfig (config.model), not the full Config
        registry = ClassRegistry.from_config(config.model)
        assert registry.num_prompts >= 1
        assert registry.num_classes >= 1
        assert len(registry.class_names) == registry.num_classes

    def test_from_explicit_prompts(self):
        registry = ClassRegistry(prompts=["teacher", "student"])
        assert registry.num_prompts == 2
        assert registry.num_classes == 2
        assert registry.class_names == ["teacher", "student"]


# ---------------------------------------------------------------------------
# 3. Config slice → AnnotationWriter initialisation + directory creation
# ---------------------------------------------------------------------------

class TestAnnotationWriterInit:
    def test_init_creates_output_directories(self, tmp_path):
        cfg = _minimal_pipeline_config(tmp_path)
        registry = ClassRegistry(prompts=["teacher", "student"])
        writer = AnnotationWriter(cfg, registry)

        output = tmp_path / "output"
        for split in ("train", "valid", "test"):
            assert (output / split / "images").is_dir()
            assert (output / split / "labels").is_dir()

    def test_init_accepts_pipeline_config_slice(self, tmp_path):
        """ISP: AnnotationWriter must accept only the pipeline config slice."""
        cfg = _minimal_pipeline_config(tmp_path)
        registry = ClassRegistry(prompts=["teacher"])
        # Should not raise even though no full Config is passed
        writer = AnnotationWriter(cfg, registry)
        assert writer is not None


# ---------------------------------------------------------------------------
# 4. Config slice → ResultFilter initialisation
# ---------------------------------------------------------------------------

class TestResultFilterInit:
    def test_init_creates_neither_directory(self, tmp_path):
        cfg = _minimal_pipeline_config(tmp_path)
        rf = ResultFilter(cfg)
        assert (tmp_path / "neither" / "images").is_dir()

    def test_accepts_pipeline_config_slice(self, tmp_path):
        cfg = _minimal_pipeline_config(tmp_path)
        rf = ResultFilter(cfg)
        assert rf.neither_dir == tmp_path / "neither"


# ---------------------------------------------------------------------------
# 5. Config slice → Validator initialisation
# ---------------------------------------------------------------------------

class TestValidatorInit:
    def test_init_with_pipeline_config_slice(self, tmp_path):
        cfg = _minimal_pipeline_config(tmp_path)
        db_path = tmp_path / "val.db"
        validator = Validator(cfg, db_path=db_path)
        assert validator is not None
        validator.close()

    def test_validator_scan_output_returns_empty_for_new_dir(self, tmp_path):
        cfg = _minimal_pipeline_config(tmp_path)
        db_path = tmp_path / "val.db"
        validator = Validator(cfg, db_path=db_path)
        # No output files yet → scan returns a dict mapping split → set of paths
        output_images = validator.scan_output_directory()
        assert isinstance(output_images, dict)
        # All splits should be empty sets
        for split_set in output_images.values():
            assert isinstance(split_set, set)
            assert len(split_set) == 0
        validator.close()


# ---------------------------------------------------------------------------
# 6. Config slice → ProgressTracker initialisation
# ---------------------------------------------------------------------------

class TestProgressTrackerInit:
    def test_init_creates_sqlite_db(self, tmp_path):
        db = tmp_path / "progress.db"
        tracker = ProgressTracker(db_path=db)
        assert db.exists()
        tracker.close()

    def test_create_job_after_init(self, tmp_path):
        db = tmp_path / "progress.db"
        tracker = ProgressTracker(db_path=db)
        imgs = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
        job_id = tracker.create_job("init_test", imgs, ["train", "valid"])
        assert isinstance(job_id, int)
        tracker.close()


# ---------------------------------------------------------------------------
# 7. Full config → multiple modules chained
# ---------------------------------------------------------------------------

class TestMultiModuleChaining:
    """Simulate the pipeline stage: config → registry → writer → filter."""

    def test_registry_writer_filter_chained(self, tmp_path):
        import numpy as np
        from types import SimpleNamespace

        cfg = _minimal_pipeline_config(tmp_path)
        registry = ClassRegistry(prompts=["teacher", "student"])
        writer = AnnotationWriter(cfg, registry)
        rf = ResultFilter(cfg)

        # Create a dummy image
        img = tmp_path / "img_001.jpg"
        img.write_bytes(b"\xff\xd8" * 10)

        # Build a segmentation result with 1 detection (teacher)
        # mask values must be 0/1 float (mask_to_polygon multiplies by 255 internally)
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[4:19, 4:19] = 1.0  # 15x15 = 225 > 100 contour area threshold
        masks = np.stack([mask])
        result = SimpleNamespace(
            masks=masks,
            class_ids=[0],
            num_detections=1,
        )

        # Writer produces a label file
        label = writer.write_annotation(img, result, split="train", copy_image=False)
        assert label is not None and label.exists()

        # Filter reports as "has detections"
        has_det = rf.filter_result(img, result)
        assert has_det is True

        # Stats are consistent
        stats = writer.get_stats()
        assert stats["train"]["labels"] >= 1
        assert rf.get_stats()["with_detections"] == 1

    def test_no_detection_flow(self, tmp_path):
        cfg = _minimal_pipeline_config(tmp_path)
        registry = ClassRegistry(prompts=["teacher"])
        writer = AnnotationWriter(cfg, registry)
        rf = ResultFilter(cfg)

        img = tmp_path / "empty.jpg"
        img.write_bytes(b"\xff\xd8" * 4)

        # No detections → filter sends to neither
        no_result = None
        has_det = rf.filter_result(img, no_result, copy_to_neither=False)
        assert has_det is False
        assert rf.get_stats()["no_detections"] == 1

        # Empty annotation file
        label = writer.write_empty_annotation(img, split="train")
        assert label.exists()
        assert label.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# 8. Config loading error handling
# ---------------------------------------------------------------------------

class TestConfigLoadingErrors:
    def test_load_nonexistent_config_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_empty_yaml_raises_or_returns_default(self, tmp_path):
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        with pytest.raises(Exception):
            load_config(str(empty_yaml))
