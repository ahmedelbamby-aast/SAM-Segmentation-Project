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
    (ephemeral Rich display).
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
# Preprocessor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Preprocessor(Protocol):
    """Validates, scans, and loads images before segmentation.

    Implemented by :class:`~src.preprocessor.ImagePreprocessor`.
    """

    def validate_image(self, image_path: Path) -> bool:
        """Check if *image_path* is a valid, readable image.

        Args:
            image_path: Path to the image file.

        Returns:
            ``True`` if the image is valid, ``False`` otherwise.
        """
        ...

    def set_fast_scan(self, enabled: bool = True) -> None:
        """Toggle fast-scan mode (skips ``cv2.imread`` validation).

        Args:
            enabled: ``True`` to enable fast scanning.
        """
        ...

    def scan_directory(self, input_dir: Path) -> List[Path]:
        """Scan *input_dir* for valid images using parallel validation.

        Args:
            input_dir: Directory to scan.

        Returns:
            Sorted list of valid image paths.

        Raises:
            FileNotFoundError: If *input_dir* does not exist.
        """
        ...

    def scan_presplit_directory(
        self, input_dir: Path,
    ) -> Dict[str, List[Path]]:
        """Scan a pre-split directory with train/valid/test subfolders.

        Args:
            input_dir: Root directory containing split subdirectories.

        Returns:
            Mapping of split name → sorted list of valid image paths.

        Raises:
            FileNotFoundError: If *input_dir* does not exist.
        """
        ...

    def load_image(self, image_path: Path) -> Optional[Any]:
        """Load an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            Image array (BGR ``np.ndarray``) or ``None`` on failure.
        """
        ...

    def detect_input_mode(self, input_dir: Path) -> str:
        """Auto-detect whether *input_dir* is flat or pre-split.

        Args:
            input_dir: Input directory path.

        Returns:
            ``"pre-split"`` if ≥ 2 of train/valid/test exist, else
            ``"flat"``.
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
    ) -> Optional[SegmentationResult]:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            :class:`SegmentationResult` with raw SAM 3 prompt indices as
            ``mask.class_id``, or ``None`` if the image yields no masks.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
            RuntimeError: If the model fails to load or inference errors out.
        """
        ...

    def process_batch(
        self,
        image_paths: List[Path],
    ) -> List[Optional[SegmentationResult]]:
        """Run inference on a list of images.

        Args:
            image_paths: Ordered list of image paths.

        Returns:
            List of :class:`SegmentationResult` (or ``None``) for each image,
            in the same order as *image_paths*.
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

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative NMS statistics.

        Returns:
            Dict with keys ``total_processed``, ``total_suppressed``,
            ``total_time_ms``, etc.
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
        image_path: Path,
        result: Optional[Any],
        copy_to_neither: bool = True,
    ) -> bool:
        """Decide whether *result* contains valid detections.

        If ``copy_to_neither`` is ``True`` and there are no detections, the
        image is copied to a ``neither/`` directory for review.

        Args:
            image_path: Absolute path to the source image.
            result: Post-NMS segmentation result (may be ``None``).
            copy_to_neither: Whether to copy undetected images.

        Returns:
            ``True`` if the image has at least one valid detection after
            filtering, ``False`` otherwise.
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative filter statistics.

        Returns:
            Dict with keys ``total_images``, ``images_with_detections``,
            ``images_no_detections``, etc.
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
        image_path: Path,
        result: Any,
        split: str,
        copy_image: bool = True,
    ) -> Optional[Path]:
        """Write a ``.txt`` annotation file for the image.

        Args:
            image_path: Absolute path to the source image.
            result: Segmentation result (filtered, post-NMS).
            split: Dataset split (``"train"``, ``"valid"``, ``"test"``).
            copy_image: Whether to copy the source image to the output dir.

        Returns:
            Path to the written annotation file, or ``None`` on skip.

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

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative write statistics.

        Returns:
            Dict with keys ``total_annotations``, ``total_images_copied``,
            etc.
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

    def create_job(
        self,
        name: str,
        image_paths: List[Path],
        splits: List[str],
    ) -> int:
        """Create a new processing job and register all images.

        Args:
            name: Human-readable job name.
            image_paths: Absolute paths for every image in the job.
            splits: Parallel list of dataset split names for each image.

        Returns:
            Integer job ID for subsequent calls.
        """
        ...

    def mark_completed(self, image_id: int) -> None:
        """Record that *image_id* completed processing successfully.

        Args:
            image_id: Row ID of the image (returned implicitly during
                job creation).
        """
        ...

    def mark_error(self, image_id: int, error_msg: str) -> None:
        """Record that *image_id* failed during processing.

        Args:
            image_id: Row ID of the image.
            error_msg: One-line error description for the DB.
        """
        ...

    def get_progress(self, job_id: int) -> Dict[str, Any]:
        """Return progress summary for *job_id*.

        Args:
            job_id: Integer job ID returned by :meth:`create_job`.

        Returns:
            Dict with keys ``total``, ``completed``, ``errors``,
            ``percent_complete``, etc.
        """
        ...

    def checkpoint(self, job_id: int) -> None:
        """Persist in-memory progress to SQLite.

        Args:
            job_id: Integer job ID returned by :meth:`create_job`.
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

    def queue_batch(
        self,
        batch_dir: Path,
        batch_id: int,
        split: str = "train",
    ) -> None:
        """Add a batch directory to the upload queue.

        Args:
            batch_dir: Directory containing images and annotations.
            batch_id: Integer batch identifier for tracking.
            split: Dataset split (``"train"``, ``"valid"``, ``"test"``).
        """
        ...

    def wait_for_uploads(self, timeout: Optional[float] = None) -> bool:
        """Block until the upload queue is empty or *timeout* expires.

        Args:
            timeout: Maximum seconds to wait (``None`` = forever).

        Returns:
            ``True`` if all uploads completed, ``False`` on timeout.
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
