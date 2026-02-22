"""
Unit tests for src/cli/ entry points.

Covers argument parsing, config loading, error handling, and correct
concrete-class wiring for every CLI entry point.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CONFIG = "config/config.yaml"
BAD_CONFIG = "no_such_file.yaml"


def _args(*extra: str) -> List[str]:
    """Build an argv list with a valid config prepended."""
    return ["--config", VALID_CONFIG] + list(extra)


# ---------------------------------------------------------------------------
# Test: sam3-preprocess
# ---------------------------------------------------------------------------

class TestPreprocessCLI:
    """Tests for src.cli.preprocess.main"""

    def test_help_exits_zero(self):
        from src.cli.preprocess import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.preprocess import main
        rc = main(["--config", BAD_CONFIG])
        assert rc == 1

    def test_valid_config_no_images_returns_0(self):
        from src.cli.preprocess import main
        with patch("src.cli.preprocess.load_config") as mock_cfg, \
             patch("src.cli.preprocess.ImagePreprocessor") as MockPrep, \
             patch("src.cli.preprocess.ModuleProgressManager") as MockMgr:
            mock_cfg.return_value = MagicMock()
            inst = MockPrep.return_value
            inst.scan_directory.return_value = []
            rc = main(_args())
        assert rc == 0

    def test_fast_flag_sets_fast_scan(self):
        from src.cli.preprocess import main
        with patch("src.cli.preprocess.load_config") as mock_cfg, \
             patch("src.cli.preprocess.ImagePreprocessor") as MockPrep, \
             patch("src.cli.preprocess.ModuleProgressManager"):
            mock_cfg.return_value = MagicMock()
            inst = MockPrep.return_value
            inst.scan_directory.return_value = []
            main(_args("--fast"))
        inst.set_fast_scan.assert_called_once_with(True)

    def test_input_dir_override(self):
        from src.cli.preprocess import main
        with patch("src.cli.preprocess.load_config") as mock_cfg, \
             patch("src.cli.preprocess.ImagePreprocessor") as MockPrep, \
             patch("src.cli.preprocess.ModuleProgressManager"):
            cfg = MagicMock()
            mock_cfg.return_value = cfg
            MockPrep.return_value.scan_directory.return_value = []
            main(_args("--input-dir", "/tmp/images"))
        assert cfg.pipeline.input_dir == Path("/tmp/images")

    def test_processes_images_with_progress(self):
        from src.cli.preprocess import main
        fake_paths = [Path("/img/a.jpg"), Path("/img/b.jpg")]
        with patch("src.cli.preprocess.load_config") as mock_cfg, \
             patch("src.cli.preprocess.ImagePreprocessor") as MockPrep, \
             patch("src.cli.preprocess.ModuleProgressManager") as MockMgr:
            mock_cfg.return_value = MagicMock()
            inst = MockPrep.return_value
            inst.scan_directory.return_value = fake_paths
            ctx = MockMgr.return_value.__enter__.return_value
            ctx.get_stage_stats.return_value = {"completed": 2}
            rc = main(_args())
        assert rc == 0
        assert ctx.start_stage.called
        assert ctx.finish_stage.called


# ---------------------------------------------------------------------------
# Test: sam3-download
# ---------------------------------------------------------------------------

class TestDownloadCLI:
    """Tests for src.cli.download.main"""

    def test_help_exits_zero(self):
        from src.cli.download import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_status_model_exists(self, tmp_path):
        from src.cli.download import main
        model_file = tmp_path / "sam3.pt"
        model_file.write_text("fake")
        rc = main(["--output-dir", str(tmp_path), "--status"])
        assert rc == 0

    def test_status_model_missing(self, tmp_path):
        from src.cli.download import main
        rc = main(["--output-dir", str(tmp_path), "--status"])
        assert rc == 0  # status check itself succeeds; just prints missing

    def test_no_token_returns_1(self, tmp_path, monkeypatch):
        from src.cli.download import main
        monkeypatch.delenv("HF_TOKEN", raising=False)
        rc = main(["--output-dir", str(tmp_path)])
        assert rc == 1

    def test_download_success(self, tmp_path, monkeypatch):
        from src.cli.download import main
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        with patch("src.cli.download.download_sam3_model") as mock_dl:
            mock_dl.return_value = tmp_path / "sam3.pt"
            rc = main(["--output-dir", str(tmp_path)])
        assert rc == 0
        mock_dl.assert_called_once()

    def test_download_failure_returns_1(self, tmp_path, monkeypatch):
        from src.cli.download import main
        monkeypatch.setenv("HF_TOKEN", "fake-token")
        with patch("src.cli.download.download_sam3_model") as mock_dl:
            mock_dl.side_effect = RuntimeError("network error")
            rc = main(["--output-dir", str(tmp_path)])
        assert rc == 1


# ---------------------------------------------------------------------------
# Test: sam3-progress
# ---------------------------------------------------------------------------

class TestProgressCLI:
    """Tests for src.cli.progress.main"""

    def test_help_exits_zero(self):
        from src.cli.progress import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.progress import main
        rc = main(["--config", BAD_CONFIG, "--job-name", "test"])
        assert rc == 1

    def test_unknown_job_prints_message(self):
        from src.cli.progress import main
        with patch("src.cli.progress.load_config") as mock_cfg, \
             patch("src.cli.progress.ProgressTracker") as MockTracker:
            mock_cfg.return_value = MagicMock()
            inst = MockTracker.return_value
            inst.get_job_id.return_value = None
            inst.get_progress.return_value = {}
            inst.get_progress_by_split.return_value = {}
            rc = main(_args("--job-name", "no-such-job"))
        assert rc == 0  # prints "not found" but does not error out

    def test_reset_stuck_calls_reset_processing(self):
        from src.cli.progress import main
        with patch("src.cli.progress.load_config") as mock_cfg, \
             patch("src.cli.progress.ProgressTracker") as MockTracker:
            mock_cfg.return_value = MagicMock()
            inst = MockTracker.return_value
            inst.get_job_id.return_value = 42
            inst.get_progress.return_value = {}
            inst.get_progress_by_split.return_value = {}
            rc = main(_args("--job-name", "myjob", "--reset-stuck"))
        inst.reset_processing_images.assert_called_once_with(42)
        assert rc == 0

    def test_reset_stuck_unknown_job_returns_1(self):
        from src.cli.progress import main
        with patch("src.cli.progress.load_config") as mock_cfg, \
             patch("src.cli.progress.ProgressTracker") as MockTracker:
            mock_cfg.return_value = MagicMock()
            inst = MockTracker.return_value
            inst.get_job_id.return_value = None
            rc = main(_args("--job-name", "ghost", "--reset-stuck"))
        assert rc == 1


# ---------------------------------------------------------------------------
# Test: sam3-segment
# ---------------------------------------------------------------------------

class TestSegmentCLI:
    """Tests for src.cli.segment.main"""

    def test_help_exits_zero(self):
        from src.cli.segment import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.segment import main
        rc = main(["--config", BAD_CONFIG])
        assert rc == 1

    def test_no_images_returns_0(self):
        from src.cli.segment import main
        with patch("src.cli.segment.load_config") as mock_cfg, \
             patch("src.cli.segment.ClassRegistry"), \
             patch("src.cli.segment.create_processor") as MockProc:
            cfg = MagicMock()
            cfg.pipeline.supported_formats = [".jpg"]
            # input_dir.rglob("*") returns empty iterator → no images
            cfg.pipeline.input_dir.rglob.return_value = []
            mock_cfg.return_value = cfg
            rc = main(_args())
        assert rc == 0

    def test_device_override(self):
        from src.cli.segment import main
        with patch("src.cli.segment.load_config") as mock_cfg, \
             patch("src.cli.segment.ClassRegistry"), \
             patch("src.cli.segment.create_processor") as MockProc:
            cfg = MagicMock()
            cfg.pipeline.supported_formats = [".jpg"]
            cfg.pipeline.input_dir.rglob.return_value = []
            mock_cfg.return_value = cfg
            main(_args("--device", "cpu"))
        assert cfg.model.device == "cpu"

    def test_workers_override(self):
        from src.cli.segment import main
        with patch("src.cli.segment.load_config") as mock_cfg, \
             patch("src.cli.segment.ClassRegistry"), \
             patch("src.cli.segment.create_processor") as MockProc:
            cfg = MagicMock()
            cfg.pipeline.supported_formats = [".jpg"]
            cfg.pipeline.input_dir.rglob.return_value = []
            mock_cfg.return_value = cfg
            main(_args("--workers", "4"))
        assert cfg.model.parallel_workers == 4


# ---------------------------------------------------------------------------
# Test: sam3-postprocess
# ---------------------------------------------------------------------------

class TestPostprocessCLI:
    """Tests for src.cli.postprocess.main"""

    def test_help_exits_zero(self):
        from src.cli.postprocess import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.postprocess import main
        rc = main(["--config", BAD_CONFIG])
        assert rc == 1

    def test_valid_config_returns_0(self):
        from src.cli.postprocess import main
        with patch("src.cli.postprocess.load_config") as mock_cfg, \
             patch("src.cli.postprocess.ClassRegistry") as MockReg, \
             patch("src.cli.postprocess.MaskPostProcessor") as MockNMS:
            cfg = MagicMock()
            cfg.post_processing.strategy = "confidence"
            cfg.post_processing.iou_threshold = 0.5
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock(class_names=["a"])
            rc = main(_args())
        assert rc == 0

    def test_strategy_override(self):
        from src.cli.postprocess import main
        with patch("src.cli.postprocess.load_config") as mock_cfg, \
             patch("src.cli.postprocess.ClassRegistry") as MockReg, \
             patch("src.cli.postprocess.MaskPostProcessor"):
            cfg = MagicMock()
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock(class_names=[])
            main(_args("--strategy", "area"))
        assert cfg.post_processing.strategy == "area"


# ---------------------------------------------------------------------------
# Test: sam3-filter
# ---------------------------------------------------------------------------

class TestFilterCLI:
    """Tests for src.cli.filter.main"""

    def test_help_exits_zero(self):
        from src.cli.filter import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.filter import main
        rc = main(["--config", BAD_CONFIG])
        assert rc == 1

    def test_no_label_files_returns_0(self):
        from src.cli.filter import main
        with patch("src.cli.filter.load_config") as mock_cfg, \
             patch("src.cli.filter.ResultFilter") as MockFilt, \
             patch("src.cli.filter.Path.rglob", return_value=iter([])):
            cfg = MagicMock()
            mock_cfg.return_value = cfg
            MockFilt.return_value.neither_dir = Path("/tmp/neither")
            rc = main(_args())
        assert rc == 0

    def test_neither_dir_override(self):
        from src.cli.filter import main
        with patch("src.cli.filter.load_config") as mock_cfg, \
             patch("src.cli.filter.ResultFilter") as MockFilt, \
             patch("src.cli.filter.Path.rglob", return_value=iter([])):
            cfg = MagicMock()
            mock_cfg.return_value = cfg
            MockFilt.return_value.neither_dir = Path("/custom/neither")
            main(_args("--neither-dir", "/custom/neither"))
        assert cfg.pipeline.neither_dir == Path("/custom/neither")


# ---------------------------------------------------------------------------
# Test: sam3-annotate
# ---------------------------------------------------------------------------

class TestAnnotateCLI:
    """Tests for src.cli.annotate.main"""

    def test_help_exits_zero(self):
        from src.cli.annotate import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.annotate import main
        rc = main(["--config", BAD_CONFIG])
        assert rc == 1

    def test_valid_config_writes_yaml(self):
        from src.cli.annotate import main
        with patch("src.cli.annotate.load_config") as mock_cfg, \
             patch("src.cli.annotate.ClassRegistry") as MockReg, \
             patch("src.cli.annotate.AnnotationWriter") as MockWriter:
            cfg = MagicMock()
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock(class_names=["teacher", "student"])
            rc = main(_args())
        assert rc == 0
        MockWriter.return_value.write_data_yaml.assert_called_once()

    def test_write_yaml_failure_returns_1(self):
        from src.cli.annotate import main
        with patch("src.cli.annotate.load_config") as mock_cfg, \
             patch("src.cli.annotate.ClassRegistry") as MockReg, \
             patch("src.cli.annotate.AnnotationWriter") as MockWriter:
            cfg = MagicMock()
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock(class_names=[])
            MockWriter.return_value.write_data_yaml.side_effect = IOError("disk full")
            rc = main(_args())
        assert rc == 1


# ---------------------------------------------------------------------------
# Test: sam3-validate
# ---------------------------------------------------------------------------

class TestValidateCLI:
    """Tests for src.cli.validate.main"""

    def test_help_exits_zero(self):
        from src.cli.validate import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.validate import main
        rc = main(["--config", BAD_CONFIG])
        assert rc == 1

    def test_no_action_returns_0(self):
        from src.cli.validate import main
        with patch("src.cli.validate.load_config") as mock_cfg, \
             patch("src.cli.validate.Validator"):
            mock_cfg.return_value = MagicMock()
            rc = main(_args())
        assert rc == 0

    def test_validate_complete_dataset_returns_0(self):
        from src.cli.validate import main
        with patch("src.cli.validate.load_config") as mock_cfg, \
             patch("src.cli.validate.Validator") as MockVal:
            mock_cfg.return_value = MagicMock()
            inst = MockVal.return_value
            result = MagicMock()
            result.is_complete = True
            result.missing_count = 0
            result.summary.return_value = "complete"
            inst.compare_datasets.return_value = result
            rc = main(_args("--validate"))
        assert rc == 0

    def test_validate_incomplete_dataset_returns_1(self):
        from src.cli.validate import main
        with patch("src.cli.validate.load_config") as mock_cfg, \
             patch("src.cli.validate.Validator") as MockVal:
            mock_cfg.return_value = MagicMock()
            inst = MockVal.return_value
            result = MagicMock()
            result.is_complete = False
            result.missing_count = 5
            result.summary.return_value = "5 missing"
            inst.compare_datasets.return_value = result
            rc = main(_args("--validate"))
        assert rc == 1

    def test_cache_missing_calls_validator(self):
        from src.cli.validate import main
        with patch("src.cli.validate.load_config") as mock_cfg, \
             patch("src.cli.validate.Validator") as MockVal:
            mock_cfg.return_value = MagicMock()
            inst = MockVal.return_value
            result = MagicMock()
            result.is_complete = False
            result.missing_count = 3
            result.summary.return_value = "3 missing"
            inst.compare_datasets.return_value = result
            inst.cache_missing_images.return_value = 3
            rc = main(_args("--validate", "--cache-missing"))
        inst.cache_missing_images.assert_called_once()


# ---------------------------------------------------------------------------
# Test: sam3-upload
# ---------------------------------------------------------------------------

class TestUploadCLI:
    """Tests for src.cli.upload.main"""

    def test_help_exits_zero(self):
        from src.cli.upload import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.upload import main
        rc = main(["--config", BAD_CONFIG, "--job-name", "x"])
        assert rc == 1

    def test_roboflow_disabled_returns_0(self):
        from src.cli.upload import main
        with patch("src.cli.upload.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.roboflow.enabled = False
            mock_cfg.return_value = cfg
            rc = main(_args("--job-name", "myjob"))
        assert rc == 0

    def test_dry_run_unknown_job_returns_1(self):
        from src.cli.upload import main
        with patch("src.cli.upload.load_config") as mock_cfg, \
             patch("src.cli.upload.ProgressTracker") as MockTracker:
            cfg = MagicMock()
            cfg.roboflow.enabled = True
            mock_cfg.return_value = cfg
            MockTracker.return_value.get_job_id.return_value = None
            rc = main(_args("--job-name", "ghost", "--dry-run"))
        assert rc == 1

    def test_dry_run_shows_pending_batches(self):
        from src.cli.upload import main
        with patch("src.cli.upload.load_config") as mock_cfg, \
             patch("src.cli.upload.ProgressTracker") as MockTracker:
            cfg = MagicMock()
            cfg.roboflow.enabled = True
            mock_cfg.return_value = cfg
            tracker = MockTracker.return_value
            tracker.get_job_id.return_value = 1
            tracker.get_pending_batches.return_value = [{"id": 1}, {"id": 2}]
            rc = main(_args("--job-name", "myjob", "--dry-run"))
        assert rc == 0

    def test_no_pending_batches_returns_0(self):
        from src.cli.upload import main
        with patch("src.cli.upload.load_config") as mock_cfg, \
             patch("src.cli.upload.ProgressTracker") as MockTracker, \
             patch("src.cli.upload.DistributedUploader"):
            cfg = MagicMock()
            cfg.roboflow.enabled = True
            mock_cfg.return_value = cfg
            tracker = MockTracker.return_value
            tracker.get_job_id.return_value = 1
            tracker.get_pending_batches.return_value = []
            rc = main(_args("--job-name", "myjob"))
        assert rc == 0


# ---------------------------------------------------------------------------
# Test: sam3-pipeline
# ---------------------------------------------------------------------------

class TestPipelineCLI:
    """Tests for src.cli.pipeline.main"""

    def test_help_exits_zero(self):
        from src.cli.pipeline import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_bad_config_returns_1(self):
        from src.cli.pipeline import main
        rc = main(["--config", BAD_CONFIG, "--job-name", "test"])
        assert rc == 1

    def test_pipeline_run_success(self):
        from src.cli.pipeline import main
        with patch("src.cli.pipeline.load_config") as mock_cfg, \
             patch("src.cli.pipeline.validate_config", return_value=[]), \
             patch("src.cli.pipeline.ClassRegistry") as MockReg, \
             patch("src.cli.pipeline.MaskPostProcessor"), \
             patch("src.cli.pipeline.ProgressTracker"), \
             patch("src.cli.pipeline.DistributedUploader"), \
             patch("src.cli.pipeline.ImagePreprocessor"), \
             patch("src.cli.pipeline.SegmentationPipeline") as MockPipeline:
            cfg = MagicMock()
            cfg.roboflow.enabled = False
            cfg.post_processing.enabled = False
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock()
            pipeline_inst = MockPipeline.return_value
            pipeline_inst.run.return_value = {
                "job_name": "test",
                "total_images": 10,
                "processed": 10,
                "errors": 0,
                "duration": "1.0s",
                "annotations": {},
            }
            rc = main(_args("--job-name", "test"))
        assert rc == 0
        pipeline_inst.cleanup.assert_called_once()

    def test_pipeline_run_exception_returns_1(self):
        from src.cli.pipeline import main
        with patch("src.cli.pipeline.load_config") as mock_cfg, \
             patch("src.cli.pipeline.validate_config", return_value=[]), \
             patch("src.cli.pipeline.ClassRegistry") as MockReg, \
             patch("src.cli.pipeline.MaskPostProcessor"), \
             patch("src.cli.pipeline.ProgressTracker"), \
             patch("src.cli.pipeline.DistributedUploader"), \
             patch("src.cli.pipeline.ImagePreprocessor"), \
             patch("src.cli.pipeline.SegmentationPipeline") as MockPipeline:
            cfg = MagicMock()
            cfg.roboflow.enabled = False
            cfg.post_processing.enabled = False
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock()
            MockPipeline.return_value.run.side_effect = RuntimeError("boom")
            rc = main(_args("--job-name", "test"))
        assert rc == 1

    def test_keyboard_interrupt_returns_1(self):
        from src.cli.pipeline import main
        with patch("src.cli.pipeline.load_config") as mock_cfg, \
             patch("src.cli.pipeline.validate_config", return_value=[]), \
             patch("src.cli.pipeline.ClassRegistry") as MockReg, \
             patch("src.cli.pipeline.MaskPostProcessor"), \
             patch("src.cli.pipeline.ProgressTracker"), \
             patch("src.cli.pipeline.DistributedUploader"), \
             patch("src.cli.pipeline.ImagePreprocessor"), \
             patch("src.cli.pipeline.SegmentationPipeline") as MockPipeline:
            cfg = MagicMock()
            cfg.roboflow.enabled = False
            cfg.post_processing.enabled = False
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock()
            MockPipeline.return_value.run.side_effect = KeyboardInterrupt()
            rc = main(_args("--job-name", "test"))
        assert rc == 1

    def test_device_and_workers_override(self):
        from src.cli.pipeline import main
        with patch("src.cli.pipeline.load_config") as mock_cfg, \
             patch("src.cli.pipeline.validate_config", return_value=[]), \
             patch("src.cli.pipeline.ClassRegistry") as MockReg, \
             patch("src.cli.pipeline.MaskPostProcessor"), \
             patch("src.cli.pipeline.ProgressTracker"), \
             patch("src.cli.pipeline.DistributedUploader"), \
             patch("src.cli.pipeline.ImagePreprocessor"), \
             patch("src.cli.pipeline.SegmentationPipeline") as MockPipeline:
            cfg = MagicMock()
            cfg.roboflow.enabled = False
            cfg.post_processing.enabled = False
            mock_cfg.return_value = cfg
            MockReg.from_config.return_value = MagicMock()
            MockPipeline.return_value.run.return_value = {
                "job_name": "x", "total_images": 0, "processed": 0,
                "errors": 0, "duration": "0s", "annotations": {},
            }
            main(_args("--job-name", "x", "--device", "cpu", "--workers", "2"))
        assert cfg.model.device == "cpu"
        assert cfg.model.parallel_workers == 2
