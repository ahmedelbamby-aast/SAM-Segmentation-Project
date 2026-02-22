"""Protocol definitions (interfaces) for the SAM 3 Segmentation Pipeline.

Every module in the pipeline depends on these abstractions, never on
concrete implementations.  Wiring of concrete classes to Protocols happens
ONLY in CLI entry points (``src/cli/``) and factory functions.

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from typing_extensions import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class MaskData:
    """Represents a single segmented mask from SAM 3.

    Attributes:
        mask: Boolean 2-D array (H × W).
        confidence: Detection confidence in [0, 1].
        class_id: Output class ID after remapping (int ≥ 0).
        area: Number of ``True`` pixels in *mask*.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` pixel coords.
        polygon: Flattened YOLO polygon ``[x1, y1, x2, y2, …]`` normalised
            to [0, 1], or ``None`` if not yet computed.
    """

    mask: Any  # np.ndarray[bool] — typed as Any to avoid numpy import here
    confidence: float
    class_id: int
    area: int
    bbox: Tuple[int, int, int, int]
    polygon: Optional[List[float]] = None


@dataclass
class SegmentationResult:
    """Output produced by a segmentor for a single image.

    Attributes:
        image_path: Absolute path to the source image.
        masks: All masks returned by SAM 3 for this image.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        inference_time_ms: Wall-clock inference time.
        device: Device used (e.g. ``"cuda:0"``, ``"cpu"``).
    """

    image_path: Path
    masks: List[MaskData]
    image_width: int
    image_height: int
    inference_time_ms: float = 0.0
    device: str = "cpu"


@dataclass
class ProcessingStats:
    """Generic stats container returned by pipeline modules.

    Attributes:
        processed: Number of items successfully processed.
        skipped: Number of items skipped (e.g. cached / already done).
        errors: Number of items that raised an exception.
        total_time_ms: Cumulative wall-clock time in milliseconds.
        extra: Module-specific extra counters.
    """

    processed: int = 0
    skipped: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ProgressCallback protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProgressCallback(Protocol):
    """Observer interface for per-item progress events.

    Implemented by :class:`~src.progress_display.ModuleProgressManager`
    (ephemeral Rich display) and :class:`~src.progress_tracker.ProgressTracker`
    (durable SQLite persistence).
    """

    def on_item_start(self, item_id: str) -> None:
        """Called immediately before an item begins processing.

        Args:
            item_id: Unique identifier for the item (e.g. image path stem).
        """
        ...

    def on_item_complete(self, item_id: str) -> None:
        """Called after an item is processed successfully.

        Args:
            item_id: Unique identifier for the item.
        """
        ...

    def on_item_error(self, item_id: str, error: Exception) -> None:
        """Called when an item raises an exception.

        Args:
            item_id: Unique identifier for the item.
            error: The exception that was raised.
        """
        ...


# ---------------------------------------------------------------------------
# Segmentor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Segmentor(Protocol):
    """Runs SAM 3 inference on images.

    Implemented by :class:`~src.sam3_segmentor.SAM3Segmentor`.

    Note:
        ``process_image`` returns **raw prompt indices** as ``class_id``
        values.  Remapping to output class IDs is the responsibility of the
        pipeline remap stage, NOT the segmentor.
    """

    def process_image(
        self,
        image_path: Path,
        *,
        callback: Optional[ProgressCallback] = None,
    ) -> SegmentationResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.
            callback: Optional progress observer.

        Returns:
            :class:`SegmentationResult` with raw SAM 3 prompt indices as
            ``mask.class_id``.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
            RuntimeError: If the model fails to load or inference errors out.
        """
        ...

    def process_batch(
        self,
        image_paths: List[Path],
        *,
        callback: Optional[ProgressCallback] = None,
    ) -> Iterator[SegmentationResult]:
        """Run inference on a list of images, yielding results one by one.

        Args:
            image_paths: Ordered list of image paths.
            callback: Optional progress observer.

        Yields:
            :class:`SegmentationResult` for each image.
        """
        ...

    def get_device_info(self) -> Dict[str, Any]:
        """Return device and memory information.

        Returns:
            Dict with keys ``device``, ``memory_allocated_mb``,
            ``memory_reserved_mb``, ``is_cuda``.
        """
        ...

    def cleanup(self) -> None:
        """Release model weights and free GPU memory."""
        ...


# ---------------------------------------------------------------------------
# PostProcessor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PostProcessor(Protocol):
    """Applies Non-Maximum Suppression to segmentation results.

    Implemented by :class:`~src.post_processor.MaskPostProcessor`.

    Note:
        Receives :class:`SegmentationResult` objects whose masks already
        have **remapped output class IDs** — never raw SAM 3 prompt indices.
    """

    def apply_nms(
        self,
        result: SegmentationResult,
        *,
        callback: Optional[ProgressCallback] = None,
    ) -> SegmentationResult:
        """Apply NMS to all masks in *result*.

        Args:
            result: Segmentation result with remapped class IDs.
            callback: Optional progress observer.

        Returns:
            New :class:`SegmentationResult` with suppressed masks removed
            (or confidence-decayed for Soft-NMS variants).
        """
        ...

    def get_stats(self) -> ProcessingStats:
        """Return cumulative NMS statistics.

        Returns:
            :class:`ProcessingStats` with ``processed``, ``skipped``
            (masks suppressed), ``errors``, ``total_time_ms``.
        """
        ...

    def reset_stats(self) -> None:
        """Reset all counters returned by :meth:`get_stats`."""
        ...


# ---------------------------------------------------------------------------
# Filter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Filter(Protocol):
    """Filters segmentation results by confidence, area, and class rules.

    Implemented by :class:`~src.result_filter.ResultFilter`.
    """

    def filter_result(
        self,
        result: SegmentationResult,
        *,
        callback: Optional[ProgressCallback] = None,
    ) -> SegmentationResult:
        """Filter masks in *result* according to configured thresholds.

        Args:
            result: Post-NMS segmentation result.
            callback: Optional progress observer.

        Returns:
            New :class:`SegmentationResult` containing only masks that pass
            all filter criteria.
        """
        ...

    def get_stats(self) -> ProcessingStats:
        """Return cumulative filter statistics.

        Returns:
            :class:`ProcessingStats` — ``skipped`` = masks removed.
        """
        ...

    def reset_stats(self) -> None:
        """Reset all counters returned by :meth:`get_stats`."""
        ...


# ---------------------------------------------------------------------------
# Writer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Writer(Protocol):
    """Writes YOLO-format polygon annotation files.

    Implemented by :class:`~src.annotation_writer.AnnotationWriter`.
    """

    def write_annotation(
        self,
        result: SegmentationResult,
        *,
        split: str = "train",
        callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Write a ``.txt`` annotation file for the image in *result*.

        Args:
            result: Filtered segmentation result.
            split: Dataset split (``"train"``, ``"valid"``, ``"test"``).
            callback: Optional progress observer.

        Returns:
            Path to the written annotation file.

        Raises:
            RuntimeError: If writing fails.
        """
        ...

    def write_data_yaml(self) -> Path:
        """Write the YOLO ``data.yaml`` file for the dataset.

        Returns:
            Path to the written ``data.yaml``.
        """
        ...

    def get_stats(self) -> ProcessingStats:
        """Return cumulative write statistics.

        Returns:
            :class:`ProcessingStats` — ``processed`` = files written.
        """
        ...

    def reset_stats(self) -> None:
        """Reset all counters returned by :meth:`get_stats`."""
        ...


# ---------------------------------------------------------------------------
# Tracker protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tracker(Protocol):
    """Persists job-level and per-image progress to SQLite.

    Implemented by :class:`~src.progress_tracker.ProgressTracker`.
    """

    def create_job(self, job_name: str, total_images: int) -> None:
        """Create or resume a job record.

        Args:
            job_name: Unique identifier for the pipeline run.
            total_images: Total number of images in the job.
        """
        ...

    def mark_completed(self, image_path: Path, stage: str) -> None:
        """Record that *image_path* completed *stage* successfully.

        Args:
            image_path: Absolute path to the image.
            stage: Pipeline stage name (e.g. ``"segment"``, ``"annotate"``).
        """
        ...

    def mark_error(self, image_path: Path, stage: str, error: str) -> None:
        """Record that *image_path* failed at *stage*.

        Args:
            image_path: Absolute path to the image.
            stage: Pipeline stage name.
            error: One-line error description for the DB.
        """
        ...

    def get_progress(self, job_name: str) -> Dict[str, Any]:
        """Return progress summary for *job_name*.

        Args:
            job_name: Unique identifier for the pipeline run.

        Returns:
            Dict with keys ``total``, ``completed``, ``errors``,
            ``percent_complete``, ``stage_counts``.
        """
        ...

    def checkpoint(self, job_name: str) -> List[Path]:
        """Return image paths not yet completed for *job_name*.

        Args:
            job_name: Unique identifier for the pipeline run.

        Returns:
            List of :class:`~pathlib.Path` objects awaiting processing.
        """
        ...


# ---------------------------------------------------------------------------
# Uploader protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Uploader(Protocol):
    """Uploads annotated images to Roboflow (or another dataset host).

    Implemented by :class:`~src.roboflow_uploader.DistributedUploader`.
    """

    def queue_batch(self, image_paths: List[Path], annotation_paths: List[Path]) -> None:
        """Add a batch of (image, annotation) pairs to the upload queue.

        Args:
            image_paths: List of image file paths.
            annotation_paths: Corresponding YOLO ``.txt`` annotation paths.
        """
        ...

    def wait_for_uploads(self) -> ProcessingStats:
        """Block until the upload queue is empty.

        Returns:
            :class:`ProcessingStats` with ``processed`` = uploads succeeded,
            ``errors`` = upload failures.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shut down background upload workers.

        Args:
            wait: If ``True`` drain the queue before shutting down.
        """
        ...


# ---------------------------------------------------------------------------
# Processor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Processor(Protocol):
    """Orchestrates parallel / distributed processing of a batch of images.

    Implemented by :class:`~src.parallel_processor.ParallelProcessor`.
    """

    def start(self) -> None:
        """Initialise worker pool and GPU strategy."""
        ...

    def process_batch(
        self,
        image_paths: List[Path],
        *,
        callback: Optional[ProgressCallback] = None,
    ) -> Iterator[SegmentationResult]:
        """Distribute *image_paths* across workers and yield results.

        Args:
            image_paths: Images to process.
            callback: Optional progress observer forwarded to workers.

        Yields:
            :class:`SegmentationResult` objects as workers complete them.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the worker pool.

        Args:
            wait: If ``True`` allow in-flight jobs to finish.
        """
        ...
