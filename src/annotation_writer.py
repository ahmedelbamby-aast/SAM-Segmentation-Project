"""
YOLOv11 segmentation annotation writer (polygon format).

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import yaml

from .logging_system import LoggingSystem

logger = LoggingSystem.get_logger(__name__)


class AnnotationWriter:
    """
    Write annotations in YOLOv11 segmentation format.
    
    Generates:
    - Per-image TXT files with polygon annotations
    - data.yaml for dataset configuration
    """
    
    def __init__(self, pipeline_config, class_registry):
        """
        Initialize annotation writer.

        Args:
            pipeline_config: :class:`~src.config_manager.PipelineConfig` slice
                (ISP — only the pipeline config, not the full Config object).
            class_registry: :class:`~src.class_registry.ClassRegistry` instance
                used as the single source of truth for class names.
        """
        self.output_dir = Path(pipeline_config.output_dir)
        self.class_names = class_registry.class_names
        self._setup_directories()
        
        logger.info(f"Annotation writer initialized, output: {self.output_dir}")
    
    def _setup_directories(self):
        """Create output directory structure."""
        for split in ['train', 'valid', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Write _classes.txt for Roboflow to understand class names
            classes_file = self.output_dir / split / '_classes.txt'
            with open(classes_file, 'w', encoding='utf-8') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
        
        # Also write classes.txt at root level
        root_classes = self.output_dir / 'classes.txt'
        with open(root_classes, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        logger.debug("Created output directory structure with class files")
    
    def mask_to_polygon(
        self, 
        mask: np.ndarray,
        simplify_epsilon: float = 0.001
    ) -> List[float]:
        """
        Convert binary mask to normalized polygon coordinates.
        
        Args:
            mask: Binary mask array (H, W)
            simplify_epsilon: Contour simplification factor (0.001 = 0.1% of perimeter)
            
        Returns:
            List of normalized [x1, y1, x2, y2, ...] coordinates
        """
        # Ensure mask is uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Take largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Check minimum area
        area = cv2.contourArea(contour)
        if area < 100:  # Skip very small regions
            return []
        
        # Simplify contour to reduce points
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplify_epsilon * perimeter
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Need at least 3 points for a valid polygon
        if len(contour) < 3:
            return []
        
        # Normalize coordinates
        h, w = mask.shape
        polygon = []
        
        for point in contour.squeeze():
            if len(point.shape) == 0 or point.shape == ():
                continue
            x, y = point
            # Normalize to [0, 1]
            polygon.extend([float(x) / w, float(y) / h])
        
        return polygon
    
    def masks_to_polygons(
        self, 
        masks: np.ndarray,
        class_ids: List[int]
    ) -> List[Tuple[int, List[float]]]:
        """
        Convert multiple masks to polygons.
        
        Args:
            masks: Array of masks (N, H, W)
            class_ids: Class ID for each mask
            
        Returns:
            List of (class_id, polygon) tuples
        """
        results = []
        
        for mask, class_id in zip(masks, class_ids):
            polygon = self.mask_to_polygon(mask)
            if len(polygon) >= 6:  # Minimum 3 points
                results.append((class_id, polygon))
        
        return results
    
    def write_annotation(
        self, 
        image_path: Path, 
        result,
        split: str,
        copy_image: bool = True
    ) -> Optional[Path]:
        """
        Write image and annotation to appropriate split folder.
        
        Args:
            image_path: Path to source image
            result: SegmentationResult object
            split: Split name ('train', 'valid', 'test')
            copy_image: Whether to copy image to output directory
            
        Returns:
            Path to label file, or None if no valid annotations
        """
        if result is None or result.num_detections == 0:
            return None
        
        image_path = Path(image_path)
        
        # Destination paths
        dest_image = self.output_dir / split / 'images' / image_path.name
        label_path = self.output_dir / split / 'labels' / f"{image_path.stem}.txt"
        
        # Copy image if requested
        if copy_image:
            shutil.copy2(str(image_path), str(dest_image))
        
        # Convert masks to polygons
        polygons = self.masks_to_polygons(result.masks, result.class_ids)
        
        if not polygons:
            # No valid polygons, still copy image but no label needed
            logger.debug(f"No valid polygons for {image_path.name}")
            return None
        
        # Write annotation file
        lines = []
        for class_id, polygon in polygons:
            # YOLOv11 format: class_id x1 y1 x2 y2 ... xn yn
            coords_str = ' '.join(f"{c:.6f}" for c in polygon)
            lines.append(f"{class_id} {coords_str}")
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.debug(f"Wrote {len(polygons)} annotations to {label_path.name}")
        return label_path
    
    def write_empty_annotation(self, image_path: Path, split: str) -> Path:
        """
        Write empty annotation file for image with no detections.
        
        Args:
            image_path: Path to source image
            split: Split name
            
        Returns:
            Path to empty label file
        """
        label_path = self.output_dir / split / 'labels' / f"{Path(image_path).stem}.txt"
        label_path.touch()
        return label_path
    
    def write_data_yaml(self) -> Path:
        """
        Generate data.yaml for YOLOv11 training.
        
        Returns:
            Path to data.yaml file
        """
        # Roboflow requires names as dict: {0: 'class_a', 1: 'class_b'}
        names_dict = {i: name for i, name in enumerate(self.class_names)}
        
        data = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': names_dict  # Dict format: {0: 'teacher', 1: 'student'}
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Generated data.yaml at {yaml_path}")
        return yaml_path
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about generated annotations.
        
        Returns:
            Dictionary with counts per split
        """
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            
            image_count = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
            label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
            
            # Count total annotations
            annotation_count = 0
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        annotation_count += len([l for l in f.readlines() if l.strip()])
            
            stats[split] = {
                'images': image_count,
                'labels': label_count,
                'annotations': annotation_count
            }
        
        return stats
