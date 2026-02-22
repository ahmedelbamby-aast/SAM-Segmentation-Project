"""
Post-segmentation result filter for categorizing images.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Statistics for filtered results."""
    total_processed: int = 0
    with_detections: int = 0
    no_detections: int = 0
    
    @property
    def detection_rate(self) -> float:
        """Percentage of images with detections."""
        if self.total_processed == 0:
            return 0.0
        return (self.with_detections / self.total_processed) * 100


class ResultFilter:
    """
    Post-segmentation filter to categorize images based on detection results.
    
    Images with no teacher or student detections are moved to a 'Neither' folder.
    """
    
    def __init__(self, config):
        """
        Initialize result filter.
        
        Args:
            config: Configuration object with pipeline settings
        """
        self.output_dir = Path(config.pipeline.output_dir)
        # Use configurable neither_dir from config
        self.neither_dir = Path(config.pipeline.neither_dir)
        self._setup_directories()
        
        self.stats = FilterStats()
        self._filtered_images: List[Path] = []
        
        logger.info(f"ResultFilter initialized, neither folder: {self.neither_dir}")
    
    def _setup_directories(self):
        """Create neither directory structure."""
        (self.neither_dir / 'images').mkdir(parents=True, exist_ok=True)
        logger.debug("Created 'neither' directory structure")
    
    def filter_result(
        self, 
        image_path: Path, 
        result: Optional[Any],
        copy_to_neither: bool = True
    ) -> bool:
        """
        Filter a segmentation result.
        
        Args:
            image_path: Path to the source image
            result: SegmentationResult or None
            copy_to_neither: Whether to copy image to neither folder
            
        Returns:
            True if image has valid detections (teacher/student found)
            False if image has no detections (goes to neither)
        """
        self.stats.total_processed += 1
        
        # Check if result has detections
        has_detections = self._has_valid_detections(result)
        
        if has_detections:
            self.stats.with_detections += 1
            return True
        else:
            self.stats.no_detections += 1
            self._filtered_images.append(image_path)
            
            # Move/copy to neither folder
            if copy_to_neither:
                self._move_to_neither(image_path)
            
            logger.debug(f"Filtered (no detections): {image_path.name}")
            return False
    
    def _has_valid_detections(self, result: Optional[Any]) -> bool:
        """
        Check if result contains valid detections.
        
        Args:
            result: SegmentationResult object or None
            
        Returns:
            True if valid detections exist
        """
        if result is None:
            return False
        
        # Check num_detections property
        if hasattr(result, 'num_detections'):
            return result.num_detections > 0
        
        # Fallback: check masks and class_ids
        if hasattr(result, 'masks') and hasattr(result, 'class_ids'):
            if result.masks is None or len(result.masks) == 0:
                return False
            if not result.class_ids:
                return False
            return True
        
        return False
    
    def _move_to_neither(self, image_path: Path):
        """
        Copy image to neither folder.
        
        Args:
            image_path: Path to source image
        """
        try:
            dest_path = self.neither_dir / 'images' / image_path.name
            shutil.copy2(str(image_path), str(dest_path))
            logger.debug(f"Copied to neither: {image_path.name}")
        except Exception as e:
            logger.error(f"Failed to copy {image_path.name} to neither: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get filter statistics.
        
        Returns:
            Dictionary with filter statistics
        """
        return {
            'total_processed': self.stats.total_processed,
            'with_detections': self.stats.with_detections,
            'no_detections': self.stats.no_detections,
            'detection_rate': f"{self.stats.detection_rate:.1f}%",
            'neither_folder': str(self.neither_dir)
        }
    
    def get_filtered_images(self) -> List[Path]:
        """Get list of images that were filtered (no detections)."""
        return self._filtered_images.copy()
    
    def get_neither_count(self) -> int:
        """Get count of images in neither folder."""
        images_dir = self.neither_dir / 'images'
        if images_dir.exists():
            return len(list(images_dir.glob('*')))
        return 0
    
    def write_neither_manifest(self) -> Path:
        """
        Write a manifest file listing all filtered images.
        
        Returns:
            Path to manifest file
        """
        manifest_path = self.neither_dir / 'manifest.txt'
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(f"# Images with no teacher/student detections\n")
            f.write(f"# Total: {len(self._filtered_images)}\n")
            f.write(f"# Detection rate: {self.stats.detection_rate:.1f}%\n\n")
            
            for img_path in sorted(self._filtered_images):
                f.write(f"{img_path.name}\n")
        
        logger.info(f"Wrote neither manifest: {manifest_path}")
        return manifest_path
    
    def reset_stats(self):
        """Reset filter statistics."""
        self.stats = FilterStats()
        self._filtered_images.clear()
