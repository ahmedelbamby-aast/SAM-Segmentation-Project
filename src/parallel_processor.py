"""Multi-process parallel inference processor for SAM 3 Segmentation Pipeline.

Decoupled from ``SAM3Segmentor`` -- receives a :class:`~src.gpu_strategy.GPUStrategy`
instance at construction and uses it to assign devices to worker processes.

Pipeline stage run by this module:

    For each image batch:
      segment() -> raw prompt indices
        remap() using ClassRegistry  (applied in worker, inside process boundary)
        return SegmentationResult with remapped output class IDs

Conforms to the :class:`~src.interfaces.Processor` Protocol.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .interfaces import ProgressCallback, SegmentationResult
from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses (used for IPC via pickling)
# ---------------------------------------------------------------------------


@dataclass
class ProcessingTask:
    """A single-image processing task passed to a worker process.

    Args:
        image_id: Integer index for correlation with the result.
        image_path: Absolute path of the image to process.
        split: Dataset split name -- ``"train"``, ``"valid"``, or ``"test"``.
    """

    image_id: int
    image_path: str
    split: str


@dataclass
class ProcessingResult:
    """Picklable result produced by a worker process.

    Args:
        image_id: Index matching the originating :class:`ProcessingTask`.
        image_path: Absolute path of the processed image.
        split: Dataset split name.
        success: Whether processing completed without an exception.
        has_detections: Whether the image yielded at least one detection.
        error_message: Exception message if ``success`` is ``False``.
        result_data: Reserved for future use.
    """

    image_id: int
    image_path: str
    split: str
    success: bool
    has_detections: bool
    error_message: Optional[str] = None
    result_data: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Worker-process state  (module-level -- safe: each worker is a separate
# process with its own memory space, not a thread)
# ---------------------------------------------------------------------------

_worker_state: Optional[Dict] = None


def _init_worker(
    config_dict: Dict,
    registry_dict: Dict,
    devices_list: List[str],
    worker_counter: object,
) -> None:
    """Initialise a worker process with its own model copy.

    Called once per worker by :class:`multiprocessing.Pool`.

    Args:
        config_dict: Serialised config dict.
        registry_dict: Serialised :class:`~src.class_registry.ClassRegistry`.
        devices_list: Ordered list of device strings.
        worker_counter: Shared atomic counter for unique worker ID assignment.
    """
    global _worker_state  # noqa: PLW0603

    with worker_counter.get_lock():  # type: ignore[union-attr]
        worker_id = worker_counter.value  # type: ignore[union-attr]
        worker_counter.value += 1  # type: ignore[union-attr]

    device = devices_list[worker_id % len(devices_list)] if devices_list else "cpu"

    from .config_manager import load_config_from_dict
    from .class_registry import ClassRegistry

    config = load_config_from_dict(config_dict)
    config.model.device = device
    registry = ClassRegistry.from_dict(registry_dict)

    from .sam3_segmentor import SAM3Segmentor
    from .result_filter import ResultFilter
    from .annotation_writer import AnnotationWriter

    import logging as _std_logging
    _std_logging.getLogger(__name__).info(
        "Worker %d initialised -- device=%s process=%s",
        worker_id, device, mp.current_process().name,
    )

    # Build NMS post-processor in worker (if enabled)
    post_processor = None
    pp_cfg = getattr(config, "post_processing", None)
    if pp_cfg is not None and getattr(pp_cfg, "enabled", False):
        from .post_processor import create_post_processor
        post_processor = create_post_processor(pp_cfg, class_names=registry.class_names)

    _worker_state = {
        "worker_id": worker_id,
        "device": device,
        "segmentor": SAM3Segmentor(config.model, config.pipeline),
        "filter": ResultFilter(config.pipeline),
        "writer": AnnotationWriter(config.pipeline, registry),
        "registry": registry,
        "post_processor": post_processor,
    }


def _process_image_worker(task: ProcessingTask) -> ProcessingResult:
    """Worker function -- processes one image and returns a picklable result.

    Execution order (per copilot-instructions.md "Remap before NMS"):
      1. segment()  → raw prompt indices
      2. remap()    → output class IDs via ClassRegistry
      3. NMS()      → suppress overlapping masks (optional)
      4. filter()   → check if image has valid detections
      5. annotate() → write YOLO label file

    Args:
        task: The image task to process.

    Returns:
        :class:`ProcessingResult` with success/error information.
    """
    global _worker_state  # noqa: PLW0603

    if _worker_state is None:
        return ProcessingResult(
            task.image_id, task.image_path, task.split,
            success=False, has_detections=False,
            error_message="Worker not initialised",
        )

    image_path = Path(task.image_path)
    segmentor = _worker_state["segmentor"]
    registry = _worker_state["registry"]
    post_processor = _worker_state["post_processor"]
    result_filter = _worker_state["filter"]
    writer = _worker_state["writer"]

    try:
        # Step 1: Segment — returns raw SAM3 prompt indices as class_id
        result: Optional[SegmentationResult] = segmentor.process_image(image_path)

        if result is None or len(result.masks) == 0:
            # No detections — still filter (to record the None)
            has_detections = result_filter.filter_result(image_path, result)
            return ProcessingResult(
                task.image_id, task.image_path, task.split,
                success=True, has_detections=has_detections,
            )

        # Step 2: Remap — convert prompt indices → output class IDs
        from .interfaces import MaskData as _MaskData
        remapped_masks = [
            _MaskData(
                mask=md.mask,
                confidence=md.confidence,
                class_id=registry.remap_prompt_index(md.class_id),
                area=md.area,
                bbox=md.bbox,
                polygon=md.polygon,
            )
            for md in result.masks
        ]
        result = SegmentationResult(
            image_path=result.image_path,
            masks=remapped_masks,
            image_width=result.image_width,
            image_height=result.image_height,
            inference_time_ms=result.inference_time_ms,
            device=result.device,
        )

        # Step 3: NMS — suppress overlapping masks (if post_processor exists)
        if post_processor is not None:
            result = post_processor.apply_nms(result)

        # Step 4: Filter — check if image has valid detections
        has_detections = result_filter.filter_result(image_path, result)

        # Step 5: Annotate — write YOLO label file
        if has_detections:
            writer.write_annotation(image_path, result, task.split)

        return ProcessingResult(
            task.image_id, task.image_path, task.split,
            success=True, has_detections=has_detections,
        )
    except Exception as exc:  # noqa: BLE001
        import logging as _std_logging
        _std_logging.getLogger(__name__).error(
            "Error processing %s: %s", image_path.name, exc
        )
        return ProcessingResult(
            task.image_id, task.image_path, task.split,
            success=False, has_detections=False,
            error_message=str(exc),
        )


# ---------------------------------------------------------------------------
# ParallelProcessor -- Processor Protocol implementation
# ---------------------------------------------------------------------------


class ParallelProcessor:
    """Multi-process parallel SAM 3 inference processor.

    Each worker process receives its own model copy and is assigned a device
    by the injected :class:`~src.gpu_strategy.GPUStrategy`.

    Implements the :class:`~src.interfaces.Processor` protocol.

    Args:
        config: Full :class:`~src.config_manager.Config` instance.
        gpu_strategy: Device assignment strategy (injected at construction).
        registry: :class:`~src.class_registry.ClassRegistry` -- serialised
            for IPC and reconstructed in each worker.
        num_workers: Override for worker count; defaults to
            ``gpu_strategy.num_workers``.
    """

    def __init__(
        self,
        config: object,
        gpu_strategy: object,
        registry: object,
        num_workers: Optional[int] = None,
    ) -> None:
        from .gpu_strategy import GPUStrategy
        from .class_registry import ClassRegistry

        if not isinstance(gpu_strategy, GPUStrategy):
            raise TypeError(
                f"gpu_strategy must be a GPUStrategy instance, got {type(gpu_strategy)}"
            )
        if not isinstance(registry, ClassRegistry):
            raise TypeError(
                f"registry must be a ClassRegistry instance, got {type(registry)}"
            )

        self._config = config
        self._gpu_strategy: GPUStrategy = gpu_strategy
        self._registry: ClassRegistry = registry
        self._num_workers: int = max(1, num_workers or gpu_strategy.num_workers)
        self._pool: Optional[mp.pool.Pool] = None  # type: ignore[name-defined]
        self._config_dict: Optional[Dict] = None
        self._registry_dict: Optional[Dict] = None

        _logger.info(
            "ParallelProcessor created -- strategy=%s, workers=%d",
            gpu_strategy.backend, self._num_workers,
        )

    # ------------------------------------------------------------------
    # Processor Protocol
    # ------------------------------------------------------------------

    @trace
    def start(self) -> None:
        """Start the worker pool (idempotent)."""
        if self._pool is not None:
            return
        self._gpu_strategy.initialize()

        devices: List[str] = list(dict.fromkeys(
            self._gpu_strategy.get_device_for_worker(i % self._gpu_strategy.num_workers)
            for i in range(self._num_workers)
        ))

        ctx = mp.get_context("spawn")
        worker_counter = ctx.Value("i", 0)

        _logger.info(
            "Starting pool -- %d workers, devices=%s",
            self._num_workers, devices,
        )
        self._pool = ctx.Pool(
            processes=self._num_workers,
            initializer=_init_worker,
            initargs=(
                self._get_config_dict(),
                self._get_registry_dict(),
                devices,
                worker_counter,
            ),
        )
        _logger.info("Process pool started")

    @trace
    def process_batch(
        self,
        image_paths: List[Path],
        *,
        callback: Optional[ProgressCallback] = None,
    ) -> Iterator[SegmentationResult]:
        """Process images in parallel, yielding results as they arrive.

        Conforms to the :class:`~src.interfaces.Processor` protocol.

        Args:
            image_paths: Absolute image paths.
            callback: Optional progress observer.

        Yields:
            :class:`~src.interfaces.SegmentationResult` for each successful image.
        """
        if self._pool is None:
            self.start()

        tasks = [
            ProcessingTask(i, str(p), self._resolve_split(p))
            for i, p in enumerate(image_paths)
        ]

        if callback is not None:
            for t in tasks:
                callback.on_item_start(Path(t.image_path).stem)

        for proc_result in self._pool.imap_unordered(  # type: ignore[union-attr]
            _process_image_worker, tasks
        ):
            item_id = Path(proc_result.image_path).stem
            if proc_result.success:
                if callback is not None:
                    callback.on_item_complete(item_id)
                yield self._build_segmentation_result(proc_result)
            else:
                err = RuntimeError(proc_result.error_message or "Unknown error")
                if callback is not None:
                    callback.on_item_error(item_id, err)

    @trace
    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool and release GPU resources.

        Args:
            wait: If ``True``, wait for all workers before returning.
        """
        if self._pool is not None:
            _logger.info("Shutting down process pool (wait=%s)...", wait)
            if wait:
                self._pool.close()
                self._pool.join()
            else:
                self._pool.terminate()
            self._pool = None
        self._gpu_strategy.cleanup()
        _logger.info("ParallelProcessor shutdown complete")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ParallelProcessor":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config_dict(self) -> Dict:
        if self._config_dict is None:
            self._config_dict = self._serialise_config()
        return self._config_dict

    def _get_registry_dict(self) -> Dict:
        if self._registry_dict is None:
            self._registry_dict = self._registry.to_dict()
        return self._registry_dict

    def _serialise_config(self) -> Dict:
        cfg = self._config
        p = cfg.pipeline  # type: ignore[attr-defined]
        m = cfg.model   # type: ignore[attr-defined]
        s = cfg.split   # type: ignore[attr-defined]
        pr = cfg.progress  # type: ignore[attr-defined]
        rf = cfg.roboflow  # type: ignore[attr-defined]

        d: Dict = {
            "pipeline": {
                "input_dir": str(p.input_dir),
                "output_dir": str(p.output_dir),
                "resolution": p.resolution,
                "supported_formats": list(p.supported_formats),
                "num_workers": 1,
                "input_mode": p.input_mode,
            },
            "model": {
                "path": str(m.path),
                "confidence": m.confidence,
                "prompts": list(m.prompts),
                "half_precision": m.half_precision,
                "device": m.device,
                "parallel_workers": 1,
                "class_remapping": getattr(m, "class_remapping", None) or {},
            },
            "split": {
                "train": s.train,
                "valid": s.valid,
                "test": s.test,
                "seed": s.seed,
            },
            "progress": {
                "db_path": str(pr.db_path),
                "checkpoint_interval": pr.checkpoint_interval,
                "log_file": str(pr.log_file),
                "log_level": pr.log_level,
            },
            "roboflow": {
                "enabled": rf.enabled,
                "api_key": rf.api_key,
                "workspace": rf.workspace,
                "project": rf.project,
                "batch_upload_size": rf.batch_upload_size,
                "upload_workers": rf.upload_workers,
                "retry_attempts": rf.retry_attempts,
                "retry_delay": rf.retry_delay,
            },
        }
        pp = getattr(cfg, "post_processing", None)
        if pp is not None:
            d["post_processing"] = {
                "enabled": getattr(pp, "enabled", True),
                "iou_threshold": getattr(pp, "iou_threshold", 0.5),
                "strategy": getattr(pp, "strategy", "confidence"),
                "class_priority": list(getattr(pp, "class_priority", []) or []),
                "soft_nms_sigma": getattr(pp, "soft_nms_sigma", 0.5),
                "min_confidence_after_decay": getattr(pp, "min_confidence_after_decay", 0.1),
                "weighted_nms_sigma": getattr(pp, "weighted_nms_sigma", 0.5),
                "adaptive_nms_density_factor": getattr(pp, "adaptive_nms_density_factor", 0.1),
                "diou_nms_beta": getattr(pp, "diou_nms_beta", 1.0),
                "mask_merge_threshold": getattr(pp, "mask_merge_threshold", 0.7),
                "enable_class_specific": getattr(pp, "enable_class_specific", False),
            }
        return d

    @staticmethod
    def _resolve_split(path: Path) -> str:
        """Infer the split name from the image path parents."""
        for part in reversed(path.parts):
            if part in ("train", "valid", "test"):
                return part
        return "train"

    @staticmethod
    def _build_segmentation_result(proc: ProcessingResult) -> SegmentationResult:
        """Build a minimal SegmentationResult from a ProcessingResult."""
        return SegmentationResult(
            image_path=Path(proc.image_path),
            masks=[],
            image_width=0,
            image_height=0,
            inference_time_ms=0.0,
        )


# ---------------------------------------------------------------------------
# SequentialProcessor -- single-process fallback
# ---------------------------------------------------------------------------


class SequentialProcessor:
    """Single-process SAM 3 processor (fallback / testing / CPU-only mode).

    Provides the same API as :class:`ParallelProcessor` for transparent
    substitution.

    Args:
        config: Full :class:`~src.config_manager.Config` instance.
        registry: :class:`~src.class_registry.ClassRegistry` instance.
    """

    def __init__(self, config: object, registry: object) -> None:
        self._config = config
        self._registry = registry
        self._segmentor: Optional[object] = None
        self._filter: Optional[object] = None
        self._writer: Optional[object] = None
        self._post_processor: Optional[object] = None
        _logger.info("SequentialProcessor created")

    # ------------------------------------------------------------------
    # Processor Protocol
    # ------------------------------------------------------------------

    @trace
    def start(self) -> None:
        """Lazily load all pipeline components."""
        self._ensure_loaded()

    @trace
    def process_batch(
        self,
        image_paths: List[Path],
        *,
        callback: Optional[ProgressCallback] = None,
    ) -> Iterator[SegmentationResult]:
        """Process images sequentially.

        Args:
            image_paths: Absolute image paths.
            callback: Optional progress observer.

        Yields:
            :class:`~src.interfaces.SegmentationResult` for successful images.
        """
        self._ensure_loaded()
        for img in image_paths:
            item_id = img.stem
            if callback:
                callback.on_item_start(item_id)
            try:
                # Step 1: Segment — raw prompt indices
                result = self._segmentor.process_image(img)  # type: ignore[union-attr]

                if result is not None and len(result.masks) > 0:
                    # Step 2: Remap — prompt indices → output class IDs
                    from .interfaces import MaskData as _MaskData
                    remapped_masks = [
                        _MaskData(
                            mask=md.mask,
                            confidence=md.confidence,
                            class_id=self._registry.remap_prompt_index(md.class_id),
                            area=md.area,
                            bbox=md.bbox,
                            polygon=md.polygon,
                        )
                        for md in result.masks
                    ]
                    result = SegmentationResult(
                        image_path=result.image_path,
                        masks=remapped_masks,
                        image_width=result.image_width,
                        image_height=result.image_height,
                        inference_time_ms=result.inference_time_ms,
                        device=result.device,
                    )

                    # Step 3: NMS — suppress overlapping masks
                    if self._post_processor is not None:
                        result = self._post_processor.apply_nms(result)

                # Step 4: Filter
                has_det = self._filter.filter_result(img, result)  # type: ignore[union-attr]
                split = ParallelProcessor._resolve_split(img)

                # Step 5: Annotate
                if has_det:
                    self._writer.write_annotation(img, result, split)  # type: ignore[union-attr]
                if callback:
                    callback.on_item_complete(item_id)
                yield result
            except Exception as exc:  # noqa: BLE001
                _logger.error("Error processing %s: %s", img.name, exc)
                if callback:
                    callback.on_item_error(item_id, exc)

    @trace
    def shutdown(self, wait: bool = True) -> None:
        """Release model resources.

        Args:
            wait: Unused for sequential processor; kept for API parity.
        """
        if self._segmentor is not None:
            try:
                self._segmentor.cleanup()  # type: ignore[union-attr]
            except Exception:
                pass
        self._segmentor = None
        self._filter = None
        self._writer = None
        self._post_processor = None
        _logger.info("SequentialProcessor shutdown complete")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "SequentialProcessor":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load pipeline components on first use."""
        if self._segmentor is not None:
            return
        from .sam3_segmentor import SAM3Segmentor
        from .result_filter import ResultFilter
        from .annotation_writer import AnnotationWriter

        cfg = self._config
        self._segmentor = SAM3Segmentor(
            cfg.model, cfg.pipeline  # type: ignore[attr-defined]
        )
        self._filter = ResultFilter(cfg.pipeline)  # type: ignore[attr-defined]
        self._writer = AnnotationWriter(cfg.pipeline, self._registry)  # type: ignore[attr-defined]

        # Build post-processor (NMS) if configured
        pp_cfg = getattr(cfg, "post_processing", None)
        if pp_cfg is not None and getattr(pp_cfg, "enabled", False):
            from .post_processor import create_post_processor
            class_names = getattr(self._registry, "class_names", [])
            self._post_processor = create_post_processor(pp_cfg, class_names=class_names)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_processor(
    config: object,
    registry: object,
) -> "ParallelProcessor | SequentialProcessor":
    """Factory -- return the appropriate processor for current config + hardware.

    Calls :func:`~src.gpu_strategy.auto_select_strategy` to choose the GPU
    strategy, then constructs a :class:`ParallelProcessor` if multiple
    workers are configured, or a :class:`SequentialProcessor` otherwise.

    Args:
        config: Full :class:`~src.config_manager.Config` instance.
        registry: :class:`~src.class_registry.ClassRegistry` instance.

    Returns:
        A :class:`ParallelProcessor` or :class:`SequentialProcessor`.
    """
    from .gpu_strategy import auto_select_strategy

    num_workers = getattr(getattr(config, "model", None), "parallel_workers", 1)

    if num_workers > 1:
        strategy = auto_select_strategy(config)
        _logger.info(
            "create_processor: ParallelProcessor -- %d workers, strategy=%s",
            num_workers, strategy.backend,
        )
        return ParallelProcessor(config, strategy, registry, num_workers=num_workers)

    _logger.info("create_processor: SequentialProcessor")
    return SequentialProcessor(config, registry)
