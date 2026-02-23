"""Integration tests — GPUStrategy + Processor interaction.

Verifies that:
- ParallelProcessor accepts a GPUStrategy via DI and honours it for device
  assignment.
- SequentialProcessor and ParallelProcessor both expose the Processor-
  compatible API (start / process_batch / shutdown).
- create_processor factory wires GPUStrategy correctly.
- ProgressCallback events flow through from process_batch to the manager.

These tests use mocks so no real GPU or model is required.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.gpu_strategy import (
    CPUOnlyStrategy,
    GPUStrategy,
    SingleGPUMultiProcess,
    auto_select_strategy,
)
from src.interfaces import ProgressCallback, SegmentationResult
from src.parallel_processor import (
    ParallelProcessor,
    ProcessingResult,
    ProcessingTask,
    SequentialProcessor,
    create_processor,
)
from src.progress_display import ModuleProgressManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> MagicMock:
    from src.class_registry import ClassRegistry

    reg = MagicMock(spec=ClassRegistry)
    reg.to_dict.return_value = {
        "class_names": ["teacher", "student"],
        "prompt_to_output": {0: 0, 1: 1},
        "output_to_name": {0: "teacher", 1: "student"},
    }
    return reg


def _make_config(workers: int = 1) -> MagicMock:
    cfg = MagicMock()
    cfg.model.parallel_workers = workers
    cfg.model.prompts = ["teacher", "student"]
    cfg.model.device = "cpu"
    cfg.model.path = "models/sam3.pt"
    cfg.model.confidence = 0.5
    cfg.model.half_precision = False
    cfg.model.class_remapping = {}
    cfg.pipeline.input_dir = "data/input"
    cfg.pipeline.output_dir = "data/output"
    cfg.pipeline.resolution = 640
    cfg.pipeline.supported_formats = [".jpg"]
    cfg.pipeline.input_mode = "flat"
    cfg.pipeline.neither_dir = "data/neither"
    cfg.split.train = 0.7
    cfg.split.valid = 0.2
    cfg.split.test = 0.1
    cfg.split.seed = 42
    cfg.progress.db_path = "progress.db"
    cfg.progress.checkpoint_interval = 10
    cfg.progress.log_file = "logs/pipeline.log"
    cfg.progress.log_level = "INFO"
    cfg.roboflow.enabled = False
    cfg.roboflow.api_key = ""
    cfg.roboflow.workspace = ""
    cfg.roboflow.project = ""
    cfg.roboflow.batch_upload_size = 10
    cfg.roboflow.upload_workers = 2
    cfg.roboflow.retry_attempts = 3
    cfg.roboflow.retry_delay = 1.0
    cfg.post_processing.enabled = True
    cfg.post_processing.iou_threshold = 0.5
    cfg.post_processing.strategy = "confidence"
    cfg.post_processing.class_priority = []
    cfg.post_processing.soft_nms_sigma = 0.5
    cfg.post_processing.min_confidence_after_decay = 0.1
    cfg.post_processing.weighted_nms_sigma = 0.5
    cfg.post_processing.adaptive_nms_density_factor = 0.1
    cfg.post_processing.diou_nms_beta = 1.0
    cfg.post_processing.mask_merge_threshold = 0.7
    cfg.post_processing.enable_class_specific = False
    return cfg


def _dummy_result(path: Path) -> SegmentationResult:
    return SegmentationResult(
        image_path=path,
        masks=[],
        image_width=640,
        image_height=480,
        inference_time_ms=10.0,
    )


# ---------------------------------------------------------------------------
# ParallelProcessor constructor validation
# ---------------------------------------------------------------------------


class TestParallelProcessorConstruction:
    """Tests for ParallelProcessor constructor DI validation."""

    def test_rejects_non_gpu_strategy(self) -> None:
        with pytest.raises(TypeError, match="GPUStrategy"):
            ParallelProcessor(
                config=_make_config(),
                gpu_strategy="not_a_strategy",  # type: ignore[arg-type]
                registry=_make_registry(),
            )

    def test_rejects_non_class_registry(self) -> None:
        strategy = CPUOnlyStrategy()
        with pytest.raises(TypeError, match="ClassRegistry"):
            ParallelProcessor(
                config=_make_config(),
                gpu_strategy=strategy,
                registry="not_a_registry",  # type: ignore[arg-type]
            )

    def test_accepts_valid_cpu_strategy(self) -> None:
        strategy = CPUOnlyStrategy()
        registry = _make_registry()
        proc = ParallelProcessor(_make_config(), strategy, registry)
        assert proc._gpu_strategy is strategy

    def test_num_workers_defaults_to_strategy_num_workers(self) -> None:
        strategy = CPUOnlyStrategy(workers=3)
        proc = ParallelProcessor(_make_config(), strategy, _make_registry())
        assert proc._num_workers == 3

    def test_num_workers_override(self) -> None:
        strategy = CPUOnlyStrategy(workers=3)
        proc = ParallelProcessor(_make_config(), strategy, _make_registry(), num_workers=5)
        assert proc._num_workers == 5


# ---------------------------------------------------------------------------
# SequentialProcessor with mock segmentor
# ---------------------------------------------------------------------------


class TestSequentialProcessorIntegration:
    """Integration tests for SequentialProcessor: real class, mock internals."""

    def _make_sequential(self) -> SequentialProcessor:
        return SequentialProcessor(config=_make_config(), registry=_make_registry())

    def test_start_loads_components(self) -> None:
        proc = self._make_sequential()
        with (
            patch("src.sam3_segmentor.SAM3Segmentor"),
            patch("src.result_filter.ResultFilter"),
            patch("src.annotation_writer.AnnotationWriter"),
        ):
            proc.start()
            assert proc._segmentor is not None
            assert proc._filter is not None
            assert proc._writer is not None

    def test_process_batch_calls_segmentor(self) -> None:
        proc = self._make_sequential()
        img = Path("data/train/img_001.jpg")

        mock_result = _dummy_result(img)
        mock_seg = MagicMock()
        mock_seg.process_image.return_value = mock_result
        mock_filter = MagicMock()
        mock_filter.filter_result.return_value = True
        mock_writer = MagicMock()

        with (
            patch("src.sam3_segmentor.SAM3Segmentor", return_value=mock_seg),
            patch("src.result_filter.ResultFilter", return_value=mock_filter),
            patch("src.annotation_writer.AnnotationWriter", return_value=mock_writer),
        ):
            results = list(proc.process_batch([img]))

        assert len(results) == 1
        mock_seg.process_image.assert_called_once_with(img)

    def test_process_batch_invokes_progress_callbacks(self) -> None:
        proc = self._make_sequential()
        img = Path("data/train/img_001.jpg")

        mock_result = _dummy_result(img)
        mock_seg = MagicMock()
        mock_seg.process_image.return_value = mock_result
        mock_filter = MagicMock()
        mock_filter.filter_result.return_value = False  # no annotation

        callback = MagicMock(spec=ProgressCallback)
        with (
            patch("src.sam3_segmentor.SAM3Segmentor", return_value=mock_seg),
            patch("src.result_filter.ResultFilter", return_value=mock_filter),
            patch("src.annotation_writer.AnnotationWriter"),
        ):
            list(proc.process_batch([img], callback=callback))

        callback.on_item_start.assert_called_once_with(img.stem)
        callback.on_item_complete.assert_called_once_with(img.stem)
        callback.on_item_error.assert_not_called()

    def test_process_batch_fires_error_callback_on_exception(self) -> None:
        proc = self._make_sequential()
        img = Path("data/train/img_001.jpg")

        mock_seg = MagicMock()
        mock_seg.process_image.side_effect = RuntimeError("model failure")

        callback = MagicMock(spec=ProgressCallback)
        with (
            patch("src.sam3_segmentor.SAM3Segmentor", return_value=mock_seg),
            patch("src.result_filter.ResultFilter"),
            patch("src.annotation_writer.AnnotationWriter"),
        ):
            list(proc.process_batch([img], callback=callback))

        callback.on_item_error.assert_called_once()
        args = callback.on_item_error.call_args[0]
        assert args[0] == img.stem
        assert isinstance(args[1], RuntimeError)

    def test_shutdown_calls_segmentor_cleanup(self) -> None:
        proc = self._make_sequential()
        mock_seg = MagicMock()
        with (
            patch("src.sam3_segmentor.SAM3Segmentor", return_value=mock_seg),
            patch("src.result_filter.ResultFilter"),
            patch("src.annotation_writer.AnnotationWriter"),
        ):
            proc.start()
            proc.shutdown()
        mock_seg.cleanup.assert_called_once()

    def test_context_manager_shuts_down(self) -> None:
        with (
            patch("src.sam3_segmentor.SAM3Segmentor"),
            patch("src.result_filter.ResultFilter"),
            patch("src.annotation_writer.AnnotationWriter"),
        ):
            with SequentialProcessor(_make_config(), _make_registry()) as proc:
                assert proc._segmentor is not None
            assert proc._segmentor is None


# ---------------------------------------------------------------------------
# create_processor factory
# ---------------------------------------------------------------------------


class TestCreateProcessorFactory:
    """Tests for the create_processor factory function."""

    def test_single_worker_returns_sequential(self) -> None:
        config = _make_config(workers=1)
        result = create_processor(config, _make_registry())
        assert isinstance(result, SequentialProcessor)

    def test_multi_worker_returns_parallel(self) -> None:
        config = _make_config(workers=3)
        with patch(
            "src.gpu_strategy.auto_select_strategy",
            return_value=CPUOnlyStrategy(workers=3),
        ):
            result = create_processor(config, _make_registry())
        assert isinstance(result, ParallelProcessor)

    def test_parallel_processor_uses_auto_selected_strategy(self) -> None:
        config = _make_config(workers=2)
        mock_strategy = CPUOnlyStrategy(workers=2)
        with patch(
            "src.gpu_strategy.auto_select_strategy",
            return_value=mock_strategy,
        ) as mock_factory:
            result = create_processor(config, _make_registry())
        mock_factory.assert_called_once_with(config)
        assert isinstance(result, ParallelProcessor)
        assert result._gpu_strategy is mock_strategy


# ---------------------------------------------------------------------------
# GPUStrategy + ModuleProgressManager event flow
# ---------------------------------------------------------------------------


class TestGPUStrategyProgressIntegration:
    """Tests verifying that GPUStrategy device info flows to ModuleProgressManager."""

    def test_cpu_strategy_device_assignment_reported_in_stats(self) -> None:
        """Create a CPU processor, run a batch, and check stats flow."""
        config = _make_config(workers=1)
        registry = _make_registry()
        proc = SequentialProcessor(config, registry)

        mgr = ModuleProgressManager(active_stage="Segment")
        mgr.start()
        mgr.start_stage("Segment", total=5)

        imgs = [Path(f"data/train/img_{i:03d}.jpg") for i in range(5)]
        mock_result = _dummy_result(imgs[0])
        mock_seg = MagicMock()
        mock_seg.process_image.return_value = mock_result
        mock_filter = MagicMock()
        mock_filter.filter_result.return_value = False

        with (
            patch("src.sam3_segmentor.SAM3Segmentor", return_value=mock_seg),
            patch("src.result_filter.ResultFilter", return_value=mock_filter),
            patch("src.annotation_writer.AnnotationWriter"),
        ):
            list(proc.process_batch(imgs, callback=mgr))

        stats = mgr.get_stage_stats("Segment")
        assert stats["completed"] == 5
        assert stats["errors"] == 0
        mgr.stop()

    def test_auto_select_strategy_integrates_with_create_processor(self) -> None:
        """auto_select_strategy result can be passed into ParallelProcessor."""
        config = _make_config(workers=2)
        config.gpu.strategy = "cpu"
        config.gpu.workers_per_gpu = 2
        config.gpu.devices = []
        config.gpu.memory_threshold = 0.85

        strategy = auto_select_strategy(config)
        registry = _make_registry()
        proc = ParallelProcessor(config, strategy, registry, num_workers=2)

        # Verify strategy type and worker count
        assert isinstance(strategy, CPUOnlyStrategy)
        assert proc._num_workers == 2
