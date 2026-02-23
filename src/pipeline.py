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
Date: 22-02-2026
"""

from __future__ import annotations

import random
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from .config_manager import Config, load_config
from .interfaces import (
    Segmentor,
    PostProcessor,
    Writer,
    Filter,
    Tracker,
    Uploader,
    SegmentationResult,
)
from .class_registry import ClassRegistry
from .preprocessor import ImagePreprocessor
from .progress_tracker import ProgressTracker
from .roboflow_uploader import DistributedUploader
from .dataset_cache import DatasetCache
from .parallel_processor import create_processor
from .utils import format_duration, estimate_eta
from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)
logger = _logger  # backward-compat alias for legacy code in this module


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
        preprocessor: Optional[ImagePreprocessor] = None,
        tracker: Optional[ProgressTracker] = None,
        uploader: Optional[DistributedUploader] = None,
        post_processor: Optional[PostProcessor] = None,
    ) -> None:
        """Initialise pipeline with configuration and optional injected deps.

        All heavy dependencies are created from *config* by default.  Pass
        explicit objects to override them (useful for testing or custom wiring).

        Args:
            config: Full pipeline configuration.
            registry: Optional pre-built :class:`~src.class_registry.ClassRegistry`.
                      Created from *config* when not supplied.
            preprocessor: Optional :class:`~src.preprocessor.ImagePreprocessor`.
            tracker: Optional :class:`~src.progress_tracker.ProgressTracker`.
            uploader: Optional :class:`~src.roboflow_uploader.DistributedUploader`.
            post_processor: Optional :class:`~src.interfaces.PostProcessor`
                            implementation used in the NMS stage.
        """
        self.config = config

        # Build class registry (single source of truth for class names/IDs)
        self.registry = registry or ClassRegistry.from_config(config.model)
        _logger.info("ClassRegistry: %s", self.registry)

        _logger.info("Initializing pipeline components…")

        self.preprocessor = preprocessor or ImagePreprocessor(config)
        self.preprocessor.set_fast_scan(True)
        self.tracker = tracker or ProgressTracker(Path(config.progress.db_path))
        self.uploader = uploader or DistributedUploader(config, self.tracker)
        self.cache = DatasetCache()

        num_workers = getattr(config.model, "parallel_workers", 1)
        self.processor = create_processor(config)

        # NMS post-processor (injected or built from config)
        if post_processor is not None:
            self._post_processor: Optional[PostProcessor] = post_processor
        elif config.post_processing and config.post_processing.enabled:
            from .post_processor import create_post_processor
            self._post_processor = create_post_processor(
                config.post_processing, class_names=self.registry.class_names
            )
        else:
            self._post_processor = None

        from .annotation_writer import AnnotationWriter
        from .result_filter import ResultFilter
        self.writer = AnnotationWriter(config.pipeline, self.registry)
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
        from dataclasses import replace as _replace
        from .interfaces import MaskData

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
        logger.info(f"Split distribution - Train: {train_count}, Valid: {valid_count}, Test: {test_count}")
        
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
        logger.info(
            "Input mode: %s — sampling: train=%s%%, valid=%s%%, test=%s%%",
            input_mode, split_pct["train"], split_pct["valid"], split_pct["test"],
        )

        # ── validator cache ────────────────────────────────────────────────
        from .validator import Validator
        validator = Validator(self.config.pipeline, db_path=Path(self.config.progress.db_path))
        cached = validator.get_cached_missing_images(job_name, unprocessed_only=True)
        validator.close()
        if cached:
            logger.info("Found %d cached missing images for job '%s'", len(cached), job_name)
            print(f"\n📋 Using {len(cached)} cached missing images from validation")
            image_paths = [p for p, _ in cached]
            splits = [s for _, s in cached]
            counts = Counter(splits)
            print(f"   train: {counts.get('train',0)}, valid: {counts.get('valid',0)}, test: {counts.get('test',0)}\n")
            return image_paths, splits

        # ── pre-split mode ─────────────────────────────────────────────────
        if input_mode == "pre-split":
            splits_list = ["train", "valid", "test"]
            cache_valid, cached_files, cache_reason = self.cache.check_cache(input_dir, splits_list)
            logger.info("Cache status: %s", cache_reason)
            if cache_valid and cached_files:
                split_images = {s: [Path(p) for p in cached_files.get(s, [])] for s in splits_list}
                logger.info("Using cached scan results (no changes detected)")
            else:
                logger.info("Scanning %s for images…", input_dir)
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
                logger.info("%s: %d images (%s%%)", sname, len(sampled[sname]), pct)

            image_paths = [p for sname, paths in sampled.items() for p in paths]
            splits = [sname for sname, paths in sampled.items() for _ in paths]
            if not image_paths:
                raise ValueError("No images selected! Check your train/valid/test_percent values.")
            return image_paths, splits

        # ── flat mode ──────────────────────────────────────────────────────
        logger.info("Scanning %s for images…", input_dir)
        image_paths = self.preprocessor.scan_directory(input_dir)
        if not image_paths:
            raise ValueError(f"No valid images found in {input_dir}")

        avg_pct = sum(split_pct.values()) / 3
        if avg_pct < 100:
            _random.seed(self.config.split.seed)
            n = max(1, int(len(image_paths) * avg_pct / 100))
            image_paths = _random.sample(image_paths, n)
            logger.info("Sampled %.0f%% (avg): %d images", avg_pct, len(image_paths))
        else:
            logger.info("Found %d valid images", len(image_paths))

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

                image_ids = [p[0] for p in pending]
                self.tracker.mark_processing(image_ids)
                tasks = [
                    (image_id, str(image_path), split)
                    for image_id, image_path, split in pending
                ]
                results = self.processor.process_batch(tasks)

                for image_id, image_path, split, success, has_detections, error_msg in results:
                    if success:
                        self.tracker.mark_completed(image_id)
                        processed_count += 1
                        if has_detections:
                            batch_images_processed += 1
                    else:
                        self.tracker.mark_error(image_id, error_msg or "Unknown error")
                        error_count += 1
                        logger.error("Error processing %s: %s", Path(image_path).name, error_msg)
                    pbar.update(1)

                pbar.set_postfix({"detections": batch_images_processed, "errors": error_count})

                if (processed_count + error_count) % self.checkpoint_interval == 0:
                    self.tracker.checkpoint(job_id)

                if batch_images_processed >= self.batch_size:
                    batch_num += 1
                    batch_id = self.tracker.create_batch(job_id, batch_num, batch_images_processed)
                    self.uploader.queue_batch(self.config.pipeline.output_dir, batch_id)
                    batch_images_processed = 0
                    logger.info("Queued batch %d for upload", batch_num)

        if batch_images_processed > 0:
            batch_num += 1
            batch_id = self.tracker.create_batch(job_id, batch_num, batch_images_processed)
            self.uploader.queue_batch(self.config.pipeline.output_dir, batch_id)
            logger.info("Queued final batch %d (%d images)", batch_num, batch_images_processed)

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

        print("\n" + "=" * 60)
        print("📤 UPLOAD PHASE: Uploading to Roboflow…")
        print("=" * 60)
        logger.info("Waiting for Roboflow uploads to complete…")
        self.uploader.wait_for_uploads()

        neither_dir = self.config.pipeline.neither_dir
        if self.uploader.should_upload_neither():
            logger.info("Uploading neither folder (upload_neither: true)…")
            self.uploader.upload_neither_folder(neither_dir)
        else:
            neither_count = self.filter.get_neither_count()
            if neither_count > 0:
                logger.info(
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
        logger.info(
            "Pipeline complete! Processed %d images in %s",
            stats["processed"], stats["duration"],
        )
        logger.info(
            "Filtering: %d with detections, %d moved to 'neither' (%s)",
            filter_stats["with_detections"],
            filter_stats["no_detections"],
            filter_stats["detection_rate"],
        )
        return stats

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
            logger.info("Starting new job '%s' with %d images", job_name, len(image_paths))
        else:
            logger.info("Resuming job: %s", job_name)
            job_id = self.tracker.get_job_id(job_name)
            if job_id is None:
                raise ValueError(f"Job '{job_name}' not found. Cannot resume.")
            self.tracker.reset_processing_images(job_id)

        progress = self.tracker.get_progress(job_id)
        total_images = progress.get("total_images", 0)
        already_processed = progress.get("processed_count", 0)
        if resume and already_processed > 0:
            logger.info("Resuming from %d/%d images", already_processed, total_images)

        self.processor.start()
        processed_count, error_count = self._run_processing_loop(
            job_id, total_images, already_processed
        )
        return self._finalize(job_id, job_name, start_time, processed_count, error_count)

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
    
    def cleanup(self):
        """Release resources."""
        self.processor.shutdown()
        self.uploader.shutdown()
        self.tracker.close()
        logger.info("Pipeline resources released")
