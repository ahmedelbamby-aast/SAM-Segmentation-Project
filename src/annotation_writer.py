"""
YOLOv11 segmentation annotation writer (polygon format).

SRP breakdown:
  - :class:`MaskConverter` — binary mask ↔ normalised polygon geometry only
  - :class:`DatasetMetadataWriter` — data.yaml and _classes.txt files only
  - :class:`AnnotationWriter` — per-image TXT annotation writing + directory setup;
    delegates conversion to MaskConverter and metadata to DatasetMetadataWriter

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import yaml

from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


# ---------------------------------------------------------------------------
# MaskConverter — SRP: geometry conversion only
# ---------------------------------------------------------------------------

class MaskConverter:
    """Convert binary masks to normalised YOLOv11 polygon coordinates.

    Single Responsibility: all mask ↔ polygon geometry logic lives here.
    ``AnnotationWriter`` owns an instance and delegates conversion to it.
    """

    @trace
    def mask_to_polygon(
        self,
        mask: np.ndarray,
        simplify_epsilon: float = 0.001,
    ) -> List[float]:
        """Convert a binary mask to normalised polygon coordinates.

        Args:
            mask: Binary mask array (H, W) — uint8 or bool.
            simplify_epsilon: Contour simplification factor
                (0.001 = 0.1 % of perimeter).

        Returns:
            List of normalised ``[x1, y1, x2, y2, …]`` coordinates in
            ``[0, 1]``.  Empty list if the mask contains no valid contour.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < 100:
            return []

        perimeter = cv2.arcLength(contour, True)
        epsilon = simplify_epsilon * perimeter
        contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(contour) < 3:
            return []

        h, w = mask.shape
        polygon: List[float] = []
        for point in contour.squeeze():
            if len(point.shape) == 0 or point.shape == ():
                continue
            x, y = point
            polygon.extend([float(x) / w, float(y) / h])

        return polygon

    @trace
    def masks_to_polygons(
        self,
        masks: np.ndarray,
        class_ids: List[int],
    ) -> List[Tuple[int, List[float]]]:
        """Convert multiple masks to polygons.

        Args:
            masks: Array of masks (N, H, W).
            class_ids: Class ID for each mask.

        Returns:
            List of ``(class_id, polygon)`` tuples where ``polygon`` contains
            at least 3 normalised points (6 floats).
        """
        results: List[Tuple[int, List[float]]] = []
        for mask, class_id in zip(masks, class_ids):
            polygon = self.mask_to_polygon(mask)
            if len(polygon) >= 6:
                results.append((class_id, polygon))
        return results


# ---------------------------------------------------------------------------
# DatasetMetadataWriter — SRP: dataset-level schema files only
# ---------------------------------------------------------------------------

class DatasetMetadataWriter:
    """Write dataset-level metadata files.

    Single Responsibility: ``data.yaml`` and per-split ``_classes.txt``
    generation only.  ``AnnotationWriter`` owns an instance and delegates
    metadata writing to it.
    """

    def __init__(self, output_dir: Path, class_names: List[str]) -> None:
        """Initialise metadata writer.

        Args:
            output_dir: Root output directory for the dataset.
            class_names: Ordered list of output class names (from ClassRegistry).
        """
        self.output_dir = output_dir
        self.class_names = class_names

    @trace
    def write_classes_files(self) -> None:
        """Write ``_classes.txt`` into every split dir and ``classes.txt`` at root."""
        for split in ['train', 'valid', 'test']:
            classes_file = self.output_dir / split / '_classes.txt'
            with open(classes_file, 'w', encoding='utf-8') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")

        root_classes = self.output_dir / 'classes.txt'
        with open(root_classes, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")

        _logger.debug("Wrote class files to all split directories")

    @trace
    def write_data_yaml(self) -> Path:
        """Generate ``data.yaml`` for YOLOv11 training.

        Returns:
            Path to the written ``data.yaml`` file.
        """
        names_dict = {i: name for i, name in enumerate(self.class_names)}

        data = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': names_dict,
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        _logger.info("Generated data.yaml at %s", yaml_path)
        return yaml_path


# ---------------------------------------------------------------------------
# AnnotationWriter — SRP: per-image annotation writing + directory layout
# ---------------------------------------------------------------------------

class AnnotationWriter:
    """Write per-image annotations in YOLOv11 segmentation format.

    Responsibilities:
    - Create the output directory tree.
    - Write per-image label TXT files.
    - Delegate polygon conversion to :class:`MaskConverter`.
    - Delegate metadata files to :class:`DatasetMetadataWriter`.
    """

    def __init__(self, pipeline_config, class_registry) -> None:
        """Initialise annotation writer.

        Args:
            pipeline_config: :class:`~src.config_manager.PipelineConfig` slice
                (ISP — only the pipeline config, not the full Config object).
            class_registry: :class:`~src.class_registry.ClassRegistry` instance
                used as the single source of truth for class names.
        """
        self.output_dir = Path(pipeline_config.output_dir)
        self.class_names = class_registry.class_names

        self._converter = MaskConverter()
        self._metadata = DatasetMetadataWriter(self.output_dir, self.class_names)

        self._setup_directories()
        _logger.info("AnnotationWriter initialised, output: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_directories(self) -> None:
        """Create the ``train/valid/test × images/labels`` directory tree."""
        for split in ['train', 'valid', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        self._metadata.write_classes_files()
        _logger.debug("Created output directory structure")

    # ------------------------------------------------------------------
    # Delegation helpers (keep public API stable for callers & tests)
    # ------------------------------------------------------------------

    @trace
    def mask_to_polygon(
        self,
        mask: np.ndarray,
        simplify_epsilon: float = 0.001,
    ) -> List[float]:
        """Delegate to :class:`MaskConverter`.

        Args:
            mask: Binary mask array (H, W).
            simplify_epsilon: Contour simplification factor.

        Returns:
            Normalised polygon coordinate list.
        """
        return self._converter.mask_to_polygon(mask, simplify_epsilon)

    @trace
    def masks_to_polygons(
        self,
        masks: np.ndarray,
        class_ids: List[int],
    ) -> List[Tuple[int, List[float]]]:
        """Delegate to :class:`MaskConverter`.

        Args:
            masks: Array of masks (N, H, W).
            class_ids: Class ID per mask.

        Returns:
            List of ``(class_id, polygon)`` tuples.
        """
        return self._converter.masks_to_polygons(masks, class_ids)

    # ------------------------------------------------------------------
    # Core annotation writing
    # ------------------------------------------------------------------

    @trace
    def write_annotation(
        self,
        image_path: Path,
        result: Any,
        split: str,
        copy_image: bool = True,
    ) -> Optional[Path]:
        """Write image and label TXT into the appropriate split folder.

        Args:
            image_path: Path to source image.
            result: :class:`~src.interfaces.SegmentationResult` object.
            split: Split name (``'train'``, ``'valid'``, or ``'test'``).
            copy_image: Whether to copy the image to the output directory.

        Returns:
            Path to the written label file, or ``None`` if no valid polygons.
        """
        if result is None or result.num_detections == 0:
            return None

        image_path = Path(image_path)
        dest_image = self.output_dir / split / 'images' / image_path.name
        label_path = self.output_dir / split / 'labels' / f"{image_path.stem}.txt"

        if copy_image:
            shutil.copy2(str(image_path), str(dest_image))

        polygons = self._converter.masks_to_polygons(result.masks, result.class_ids)

        if not polygons:
            _logger.debug("No valid polygons for %s", image_path.name)
            return None

        lines = []
        for class_id, polygon in polygons:
            coords_str = ' '.join(f"{c:.6f}" for c in polygon)
            lines.append(f"{class_id} {coords_str}")

        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        _logger.debug("Wrote %d annotations to %s", len(polygons), label_path.name)
        return label_path

    @trace
    def write_empty_annotation(self, image_path: Path, split: str) -> Path:
        """Write an empty label file for an image with no detections.

        Args:
            image_path: Path to source image.
            split: Split name.

        Returns:
            Path to the empty label file.
        """
        label_path = self.output_dir / split / 'labels' / f"{Path(image_path).stem}.txt"
        label_path.touch()
        return label_path

    @trace
    def write_data_yaml(self) -> Path:
        """Delegate ``data.yaml`` generation to :class:`DatasetMetadataWriter`.

        Returns:
            Path to the written ``data.yaml`` file.
        """
        return self._metadata.write_data_yaml()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @trace
    def get_stats(self) -> Dict[str, Any]:
        """Return counts of images, labels, and annotations per split.

        Returns:
            Dict keyed by split name, each value a dict with keys
            ``'images'``, ``'labels'``, ``'annotations'``.
        """
        stats: Dict[str, Any] = {}

        for split in ['train', 'valid', 'test']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'

            image_count = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
            label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0

            annotation_count = 0
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r', encoding='utf-8') as f:
                        annotation_count += sum(1 for line in f if line.strip())

            stats[split] = {
                'images': image_count,
                'labels': label_count,
                'annotations': annotation_count,
            }

        return stats

    @trace
    def reset_stats(self) -> None:
        """Reset annotation statistics.

        Clears the output directory labels so :meth:`get_stats` returns zeroes.
        Note: This does NOT delete files — it only satisfies the Protocol
        contract.  Statistics are directory-derived, so a no-op is correct.
        """
