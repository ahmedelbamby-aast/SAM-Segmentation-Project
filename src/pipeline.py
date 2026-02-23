"""Segmentation pipeline orchestrator.

Thin orchestrator that sequences the pipeline stages:

    scan → split-assign → segment (SAM3) → **remap** (ClassRegistry)
    → NMS → filter → annotate → upload → validate

Critical constraint — remap BEFORE NMS:
    :meth:`_remap_result` is called immediately after ``SAM3Segmentor``
    returns raw prompt indices, before NMS sees any class IDs.
    See ``copilot-instructions.md`` § "Critical constraint — Remap before NMS".

``SegmentationPipeline`` accepts **all** heavyweight dependencies via its
constructor so callers (CLI entry points, tests) can inject mocks or
alternative implementations that satisfy the Protocols in
``src/interfaces.py``.

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from .config_manager import Config, load_config
from .interfaces import (
    Filter,
    PostProcessor,
    Preprocessor,
    Processor,
    SegmentationResult,
    MaskData,
    Tracker,
    Uploader,
    Writer,
)
from .class_registry import ClassRegistry
from .utils import format_duration, estimate_eta
from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


class SegmentationPipeline:
    """
    Main pipeline orchestrating the segmentation workflow.
    
    Coordinates:
    - Image scanning and preprocessing
    - Progress tracking and resume
    - SAM 3 segmentation (parallel or sequential)
    - Result filtering (moves no-detection images to 'neither' folder)
    - Annotation writing
    - Roboflow uploads
    """
    
    def __init__(
        self,
        config: Config,
        *,
        registry: Optional[ClassRegistry] = None,
        preprocessor: Optional[Preprocessor] = None,
        tracker: Optional[Tracker] = None,
        uploader: Optional[Uploader] = None,
        post_processor: Optional[PostProcessor] = None,
        processor: Optional[Processor] = None,
        writer: Optional[Writer] = None,
        filter_: Optional[Filter] = None,
    ) -> None:
        """Initialise pipeline with configuration and optional injected deps.

        All heavy dependencies are created from *config* by default.  Pass
        explicit objects to override them (useful for testing or custom wiring).

        Dependencies are accepted via Protocol types from
        ``src/interfaces.py`` to satisfy DIP — concrete classes are only
        imported inside the factory branches.  Callers may inject any
        object satisfying the corresponding Protocol.

        Args:
            config: Full pipeline configuration.
            registry: Optional :class:`~src.class_registry.ClassRegistry`.
            preprocessor: Optional :class:`~src.interfaces.Preprocessor`.
            tracker: Optional :class:`~src.interfaces.Tracker`.
            uploader: Optional :class:`~src.interfaces.Uploader`.
            post_processor: Optional :class:`~src.interfaces.PostProcessor`.
            processor: Optional :class:`~src.interfaces.Processor`.
            writer: Optional :class:`~src.interfaces.Writer`.
            filter_: Optional :class:`~src.interfaces.Filter`.
        """
        self.config = config

        # Build class registry (single source of truth for class names/IDs)
        self.registry = registry or ClassRegistry.from_config(config.model)
        _logger.info("ClassRegistry: %s", self.registry)

        _logger.info("Initializing pipeline components…")

        # --- Preprocessor (DIP: lazy import of concrete class) ---
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            from .preprocessor import ImagePreprocessor
            self.preprocessor = ImagePreprocessor(config.pipeline)
        self.preprocessor.set_fast_scan(True)  # type: ignore[union-attr]

        # --- Progress tracker (DIP: lazy import) ---
        if tracker is not None:
            self.tracker = tracker
        else:
            from .progress_tracker import ProgressTracker
            self.tracker = ProgressTracker(Path(config.progress.db_path))

        # --- Uploader (DIP: lazy import) ---
        if uploader is not None:
            self.uploader = uploader
        else:
            from .roboflow_uploader import DistributedUploader
            self.uploader = DistributedUploader(config.roboflow, self.tracker)

        # --- Dataset cache ---
        from .dataset_cache import DatasetCache
        self.cache = DatasetCache()

        # --- Processor: segment + remap + filter + annotate in workers ---
        if processor is not None:
            self.processor = processor
        else:
            from .parallel_processor import create_processor
            self.processor = create_processor(config, self.registry)
        num_workers = getattr(config.model, "parallel_workers", 1)

        # --- NMS post-processor (injected or built from config) ---
        if post_processor is not None:
            self._post_processor: Optional[PostProcessor] = post_processor
        elif config.post_processing and config.post_processing.enabled:
            from .post_processor import create_post_processor
            self._post_processor = create_post_processor(
                config.post_processing, class_names=self.registry.class_names
            )
        else:
            self._post_processor = None

        # --- Annotation writer & filter (DIP: lazy imports) ---
        if writer is not None:
            self.writer = writer
        else:
            from .annotation_writer import AnnotationWriter
            self.writer = AnnotationWriter(config.pipeline, self.registry)

        if filter_ is not None:
            self.filter = filter_
        else:
            from .result_filter import ResultFilter
            self.filter = ResultFilter(config.pipeline)

        self.batch_size = config.roboflow.batch_upload_size
        self.checkpoint_interval = config.progress.checkpoint_interval

        _logger.info("Pipeline initialized (parallel_workers=%d)", num_workers)
    
    @staticmethod
    def _remap_result(result: SegmentationResult, registry: ClassRegistry) -> SegmentationResult:
        """**Remap stage** — convert raw SAM3 prompt indices → output class IDs.

        This method MUST be called after ``SAM3Segmentor.process_image()`` and
        BEFORE any NMS stage.  It updates each
        :class:`~src.interfaces.MaskData`'s ``class_id`` in-place using
        ``registry.remap_prompt_index()``.

        Args:
            result: Output of the segmentation stage (raw prompt indices).
            registry: :class:`~src.class_registry.ClassRegistry` instance.

        Returns:
            Same :class:`~src.interfaces.SegmentationResult` with all
            ``class_id`` values replaced by remapped output class IDs.
        """
        remapped_masks = [
            MaskData(
                mask=md.mask,
                confidence=md.confidence,
                class_id=registry.remap_prompt_index(md.class_id),
                area=md.area,
                bbox=md.bbox,
                polygon=md.polygon,
            )
            for md in result.masks
        ]
        return SegmentationResult(
            image_path=result.image_path,
            masks=remapped_masks,
            image_width=result.image_width,
            image_height=result.image_height,
            inference_time_ms=result.inference_time_ms,
            device=result.device,
        )

    def _assign_splits(self, image_paths: List[Path]) -> List[str]:
        """
        Assign train/val/test splits to images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of split assignments
        """
        random.seed(self.config.split.seed)
        
        splits = []
        train_threshold = self.config.split.train
        val_threshold = train_threshold + self.config.split.valid
        
        for _ in image_paths:
            r = random.random()
            if r < train_threshold:
                splits.append('train')
            elif r < val_threshold:
                splits.append('valid')
            else:
                splits.append('test')
        
        # Log split distribution
        train_count = splits.count('train')
        valid_count = splits.count('valid')
        test_count = splits.count('test')
        _logger.info("Split distribution - Train: %d, Valid: %d, Test: %d", train_count, valid_count, test_count)
        
        return splits
    
    def _collect_images(self, job_name: str) -> Tuple[List[Path], List[str]]:
        """Scan inputs and return ``(image_paths, splits)`` for a new job.

        Checks the validator cache first.  If missing images are cached for
        *job_name* those are returned directly.  Otherwise the input directory
        is scanned according to ``input_mode`` (``pre-split`` or ``flat``).

        Args:
            job_name: Job identifier (used to look up cached missing images).

        Returns:
            Tuple of ``(image_paths, splits)`` where every ``image_paths[i]``
            has a corresponding split label in ``splits[i]``.
        """
        import random as _random
        from collections import Counter

        input_dir = Path(self.config.pipeline.input_dir)
        input_mode = getattr(self.config.pipeline, "input_mode", "flat")
        split_pct = {
            "train": getattr(self.config.pipeline, "train_percent", 100),
            "valid": getattr(self.config.pipeline, "valid_percent", 100),
            "test": getattr(self.config.pipeline, "test_percent", 100),
        }
        _logger.info(
            "Input mode: %s — sampling: train=%s%%, valid=%s%%, test=%s%%",
            input_mode, split_pct["train"], split_pct["valid"], split_pct["test"],
        )

        # ── validator cache ────────────────────────────────────────────────
        from .validator import Validator
        validator = Validator(self.config.pipeline, db_path=Path(self.config.progress.db_path))
        cached = validator.get_cached_missing_images(job_name, unprocessed_only=True)
        validator.close()
        if cached:
            _logger.info("Found %d cached missing images for job '%s'", len(cached), job_name)
            _logger.info("Using %d cached missing images from validation", len(cached))
            image_paths = [p for p, _ in cached]
            splits = [s for _, s in cached]
            counts = Counter(splits)
            _logger.info(
                "Cached split breakdown — train: %d, valid: %d, test: %d",
                counts.get("train", 0), counts.get("valid", 0), counts.get("test", 0),
            )
            return image_paths, splits

        # ── pre-split mode ─────────────────────────────────────────────────
        if input_mode == "pre-split":
            splits_list = ["train", "valid", "test"]
            cache_valid, cached_files, cache_reason = self.cache.check_cache(input_dir, splits_list)
            _logger.info("Cache status: %s", cache_reason)
            if cache_valid and cached_files:
                split_images = {s: [Path(p) for p in cached_files.get(s, [])] for s in splits_list}
                _logger.info("Using cached scan results (no changes detected)")
            else:
                _logger.info("Scanning %s for images…", input_dir)
                split_images = self.preprocessor.scan_presplit_directory(input_dir)
                self.cache.save_cache(input_dir, split_images, splits_list)

            _random.seed(self.config.split.seed)
            sampled: Dict[str, List[Path]] = {}
            for sname, paths in split_images.items():
                pct = split_pct.get(sname, 100)
                if pct == 0:
                    sampled[sname] = []
                elif pct >= 100:
                    sampled[sname] = paths
                else:
                    n = max(1, int(len(paths) * pct / 100))
                    sampled[sname] = _random.sample(paths, n) if len(paths) > n else paths
                _logger.info("%s: %d images (%s%%)", sname, len(sampled[sname]), pct)

            image_paths = [p for sname, paths in sampled.items() for p in paths]
            splits = [sname for sname, paths in sampled.items() for _ in paths]
            if not image_paths:
                raise ValueError("No images selected! Check your train/valid/test_percent values.")
            return image_paths, splits

        # ── flat mode ──────────────────────────────────────────────────────
        _logger.info("Scanning %s for images…", input_dir)
        image_paths = self.preprocessor.scan_directory(input_dir)
        if not image_paths:
            raise ValueError(f"No valid images found in {input_dir}")

        avg_pct = sum(split_pct.values()) / 3
        if avg_pct < 100:
            _random.seed(self.config.split.seed)
            n = max(1, int(len(image_paths) * avg_pct / 100))
            image_paths = _random.sample(image_paths, n)
            _logger.info("Sampled %.0f%% (avg): %d images", avg_pct, len(image_paths))
        else:
            _logger.info("Found %d valid images", len(image_paths))

        splits = self._assign_splits(image_paths)
        return image_paths, splits

    def _run_processing_loop(
        self, job_id: int, total_images: int, already_processed: int
    ) -> Tuple[int, int]:
        """Run the main processing while-loop via tqdm.

        Fetches batches of pending images from the tracker, processes them
        through ``self.processor``, updates image status, and queues Roboflow
        upload batches when they reach *batch_size*.

        Args:
            job_id: Active job ID.
            total_images: Total image count registered with the tracker.
            already_processed: Count of images already done (for tqdm offset).

        Returns:
            Tuple of ``(processed_count, error_count)`` accumulated this run.
        """
        processed_count: int = 0
        error_count: int = 0
        batch_num: int = len(self.tracker.get_uploaded_batches(job_id))
        batch_images_processed: int = 0

        with tqdm(
            total=total_images,
            initial=already_processed,
            desc="Processing images",
            unit="img",
        ) as pbar:
            while True:
                pending = self.tracker.get_pending_images(job_id, limit=100)
                if not pending:
                    break

                # Build lookup: image_path → (image_id, split)
                path_to_info: Dict[str, Tuple[int, str]] = {}
                image_paths: List[Path] = []
                image_ids: List[int] = []
                for image_id, image_path, split in pending:
                    p = Path(str(image_path))
                    path_to_info[str(p)] = (image_id, split)
                    image_paths.append(p)
                    image_ids.append(image_id)

                self.tracker.mark_processing(image_ids)
                completed_ids: set = set()

                for result in self.processor.process_batch(image_paths):
                    # Match result back to tracked image via image_path
                    result_key = str(result.image_path)
                    info = path_to_info.get(result_key)
                    if info is None:
                        # Fallback: stem match
                        for p_str, p_info in path_to_info.items():
                            if Path(p_str).stem == result.image_path.stem:
                                info = p_info
                                break
                    if info is None:
                        _logger.warning(
                            "Result for untracked image: %s",
                            result.image_path,
                        )
                        pbar.update(1)
                        continue

                    image_id, _split = info
                    completed_ids.add(image_id)
                    self.tracker.mark_completed(image_id)
                    processed_count += 1

                    has_detections = len(result.masks) > 0
                    # Track stats on pipeline's own filter (processor
                    # already handled file ops — copy_to_neither=False)
                    self.filter.filter_result(
                        result.image_path, result, copy_to_neither=False
                    )
                    if has_detections:
                        batch_images_processed += 1
                    pbar.update(1)

                # Images not yielded by processor are errors
                for img_id in image_ids:
                    if img_id not in completed_ids:
                        self.tracker.mark_error(
                            img_id, "Processing failed"
                        )
                        error_count += 1
                        pbar.update(1)

                pbar.set_postfix(
                    {"detections": batch_images_processed, "errors": error_count}
                )

                if (processed_count + error_count) % self.checkpoint_interval == 0:
                    self.tracker.checkpoint(job_id)

                if batch_images_processed >= self.batch_size:
                    batch_num += 1
                    batch_id = self.tracker.create_batch(job_id, batch_num, batch_images_processed)
                    self.uploader.queue_batch(self.config.pipeline.output_dir, batch_id)
                    batch_images_processed = 0
                    _logger.info("Queued batch %d for upload", batch_num)

        if batch_images_processed > 0:
            batch_num += 1
            batch_id = self.tracker.create_batch(job_id, batch_num, batch_images_processed)
            self.uploader.queue_batch(self.config.pipeline.output_dir, batch_id)
            _logger.info("Queued final batch %d (%d images)", batch_num, batch_images_processed)

        return processed_count, error_count

    def _finalize(
        self,
        job_id: int,
        job_name: str,
        start_time: float,
        processed_count: int,
        error_count: int,
    ) -> Dict[str, Any]:
        """Write YOLO artifacts, wait for uploads, and collect statistics.

        Args:
            job_id: Active job ID.
            job_name: Human-readable job name (for the stats dict).
            start_time: ``time.time()`` snapshot taken at job start.
            processed_count: Images processed successfully in this run.
            error_count: Images that failed in this run.

        Returns:
            Statistics dictionary summarising the completed job.
        """
        self.writer.write_data_yaml()
        self.filter.write_neither_manifest()

        _logger.info("📤 UPLOAD PHASE: Uploading to Roboflow…")
        _logger.info("Waiting for Roboflow uploads to complete…")
        self.uploader.wait_for_uploads()

        neither_dir = self.config.pipeline.neither_dir
        if self.uploader.should_upload_neither():
            _logger.info("Uploading neither folder (upload_neither: true)…")
            self.uploader.upload_neither_folder(neither_dir)
        else:
            neither_count = self.filter.get_neither_count()
            if neither_count > 0:
                _logger.info(
                    "Neither folder preserved for manual review: %s (%d images)",
                    neither_dir, neither_count,
                )

        self.tracker.checkpoint(job_id)

        elapsed = time.time() - start_time
        final_progress = self.tracker.get_progress(job_id)
        filter_stats = self.filter.get_stats()
        stats: Dict[str, Any] = {
            "job_name": job_name,
            "total_images": final_progress.get("total_images", 0),
            "processed": final_progress.get("processed_count", 0),
            "errors": final_progress.get("error_count", 0),
            "duration": format_duration(elapsed),
            "duration_seconds": elapsed,
            "uploads": self.uploader.get_stats(),
            "annotations": self.writer.get_stats(),
            "filtered": filter_stats,
        }
        _logger.info(
            "Pipeline complete! Processed %d images in %s",
            stats["processed"], stats["duration"],
        )
        _logger.info(
            "Filtering: %d with detections, %d moved to 'neither' (%s)",
            filter_stats["with_detections"],
            filter_stats["no_detections"],
            filter_stats["detection_rate"],
        )
        return stats

    @trace
    def run(self, job_name: str, resume: bool = False) -> Dict[str, Any]:
        """Run the segmentation pipeline (thin orchestrator).

        Delegates scanning to :meth:`_collect_images`, image processing to
        :meth:`_run_processing_loop`, and upload/stats to :meth:`_finalize`.

        Args:
            job_name: Unique name for this processing job.
            resume: Whether to resume an existing job.

        Returns:
            Dictionary with final statistics.
        """
        start_time = time.time()

        if not resume:
            image_paths, splits = self._collect_images(job_name)
            job_id = self.tracker.create_job(job_name, image_paths, splits)
            _logger.info("Starting new job '%s' with %d images", job_name, len(image_paths))
        else:
            _logger.info("Resuming job: %s", job_name)
            job_id = self.tracker.get_job_id(job_name)
            if job_id is None:
                raise ValueError(f"Job '{job_name}' not found. Cannot resume.")
            self.tracker.reset_processing_images(job_id)

        progress = self.tracker.get_progress(job_id)
        total_images = progress.get("total_images", 0)
        already_processed = progress.get("processed_count", 0)
        if resume and already_processed > 0:
            _logger.info("Resuming from %d/%d images", already_processed, total_images)

        self.processor.start()
        processed_count, error_count = self._run_processing_loop(
            job_id, total_images, already_processed
        )
        return self._finalize(job_id, job_name, start_time, processed_count, error_count)

    @trace
    def get_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a job.
        
        Args:
            job_name: Job name to check
            
        Returns:
            Status dictionary or None if job not found
        """
        job_id = self.tracker.get_job_id(job_name)
        
        if job_id is None:
            return None
        
        progress = self.tracker.get_progress(job_id)
        split_progress = self.tracker.get_progress_by_split(job_id)
        pending_batches = self.tracker.get_pending_batches(job_id)
        uploaded_batches = self.tracker.get_uploaded_batches(job_id)
        
        return {
            'job_name': job_name,
            'job_id': job_id,
            'progress': progress,
            'by_split': split_progress,
            'pending_uploads': len(pending_batches),
            'completed_uploads': len(uploaded_batches)
        }
    
    @trace
    def cleanup(self) -> None:
        """Release resources."""
        self.processor.shutdown()
        self.uploader.shutdown()
        self.tracker.close()
        _logger.info("Pipeline resources released")

    # ------------------------------------------------------------------
    # Stats pattern
    # ------------------------------------------------------------------

    @trace
    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics from all pipeline sub-components.

        Returns:
            Dict keyed by component name, each value being the
            component's own ``get_stats()`` dict.
        """
        stats: Dict[str, Any] = {
            "registry": self.registry.get_stats(),
        }
        if hasattr(self.filter, "get_stats"):
            stats["filter"] = self.filter.get_stats()
        if hasattr(self.writer, "get_stats"):
            stats["writer"] = self.writer.get_stats()
        if self._post_processor is not None and hasattr(self._post_processor, "get_stats"):
            stats["post_processor"] = self._post_processor.get_stats()
        if hasattr(self.processor, "get_stats"):
            stats["processor"] = self.processor.get_stats()
        return stats

    @trace
    def reset_stats(self) -> None:
        """Reset statistics on all sub-components that support it."""
        self.registry.reset_stats()
        if hasattr(self.filter, "reset_stats"):
            self.filter.reset_stats()
        if hasattr(self.writer, "reset_stats"):
            self.writer.reset_stats()
        if self._post_processor is not None and hasattr(self._post_processor, "reset_stats"):
            self._post_processor.reset_stats()
        if hasattr(self.processor, "reset_stats"):
            self.processor.reset_stats()
