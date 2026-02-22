"""
Image preprocessing: validation, resizing with aspect ratio preservation.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing utilities for validation and resizing.
    
    Supports:
    - Image validation (format, readability)
    - Resize with aspect ratio preservation (letterboxing)
    - Parallel directory scanning
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration object with pipeline settings
        """
        self.target_size = config.pipeline.resolution
        self.supported_formats = [f.lower() for f in config.pipeline.supported_formats]
        self.num_workers = config.pipeline.num_workers
    
    def validate_image(self, image_path: Path) -> bool:
        """
        Check if image is valid and readable.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Check file extension
            if image_path.suffix.lower() not in self.supported_formats:
                return False
            
            # Check file exists and is not empty
            if not image_path.exists() or image_path.stat().st_size == 0:
                return False
            
            # Fast mode: skip the slow cv2.imread validation
            # Images are validated during actual processing anyway
            if getattr(self, '_fast_scan', False):
                return True
            
            # Full validation: Try to read image (SLOW for large datasets)
            img = cv2.imread(str(image_path))
            if img is None or img.size == 0:
                return False
            
            # Check minimum dimensions
            h, w = img.shape[:2]
            if h < 10 or w < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Image validation failed for {image_path}: {e}")
            return False
    
    def set_fast_scan(self, enabled: bool = True):
        """Enable/disable fast scanning (skips cv2.imread validation)."""
        self._fast_scan = enabled
        if enabled:
            logger.info("Fast scan mode enabled (skipping image content validation)")
    
    def resize_with_padding(
        self, 
        image: np.ndarray, 
        target_size: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resize image maintaining aspect ratio with letterboxing.
        
        Args:
            image: Input image array (BGR)
            target_size: Target size (uses config value if not specified)
            
        Returns:
            Tuple of (resized_image, transform_info)
        """
        target = target_size or self.target_size
        h, w = image.shape[:2]
        
        # Calculate scale to fit in target size
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(
            image, 
            (new_w, new_h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create padded image with gray background
        padded = np.full((target, target, 3), 114, dtype=np.uint8)
        
        # Center the resized image
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Store transform info for coordinate conversion
        transform_info = {
            'original_size': (w, h),
            'scale': scale,
            'padding': (pad_x, pad_y),
            'new_size': (new_w, new_h),
            'target_size': target
        }
        
        return padded, transform_info
    
    def reverse_transform_coordinates(
        self, 
        coords: np.ndarray, 
        transform_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Convert coordinates from padded image back to original image space.
        
        Args:
            coords: Array of coordinates (N, 2) in padded image space
            transform_info: Transform info from resize_with_padding
            
        Returns:
            Coordinates in original image space
        """
        pad_x, pad_y = transform_info['padding']
        scale = transform_info['scale']
        
        # Remove padding offset
        coords = coords.copy()
        coords[:, 0] -= pad_x
        coords[:, 1] -= pad_y
        
        # Reverse scale
        coords = coords / scale
        
        return coords
    
    def scan_directory(self, input_dir: Path) -> List[Path]:
        """
        Scan directory for valid images using parallel validation.
        
        Args:
            input_dir: Directory to scan
            
        Returns:
            List of valid image paths
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Collect all potential image files
        all_files = []
        for fmt in self.supported_formats:
            all_files.extend(input_dir.rglob(f"*{fmt}"))
            all_files.extend(input_dir.rglob(f"*{fmt.upper()}"))
        
        # Remove duplicates
        all_files = list(set(all_files))
        logger.info(f"Found {len(all_files)} potential image files")
        
        if not all_files:
            return []
        
        # Parallel validation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            valid_mask = list(executor.map(self.validate_image, all_files))
        
        valid_files = [f for f, valid in zip(all_files, valid_mask) if valid]
        
        invalid_count = len(all_files) - len(valid_files)
        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid/unreadable images")
        
        logger.info(f"Found {len(valid_files)} valid images")
        return sorted(valid_files)
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load image from disk.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array (BGR) or None if failed
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Failed to load image: {image_path}")
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def get_image_info(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get image metadata without loading full image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with width, height, channels, format
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1
            
            return {
                'width': w,
                'height': h,
                'channels': channels,
                'format': image_path.suffix.lower(),
                'size_bytes': image_path.stat().st_size
            }
        except Exception:
            return None
    
    def scan_presplit_directory(self, input_dir: Path) -> Dict[str, List[Path]]:
        """
        Scan pre-split directory with train/val/test subfolders.
        
        Args:
            input_dir: Root directory containing train/val/test subdirectories
            
        Returns:
            Dictionary mapping split names to lists of valid image paths
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        splits = ['train', 'valid', 'test']
        result = {}
        total_images = 0
        
        for split in splits:
            split_dir = input_dir / split
            
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                result[split] = []
                continue
            
            # Scan for images in the split directory
            all_files = []
            for fmt in self.supported_formats:
                all_files.extend(split_dir.rglob(f"*{fmt}"))
                all_files.extend(split_dir.rglob(f"*{fmt.upper()}"))
            
            # Remove duplicates
            all_files = list(set(all_files))
            
            if not all_files:
                logger.warning(f"No images found in {split_dir}")
                result[split] = []
                continue
            
            # Parallel validation
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                valid_mask = list(executor.map(self.validate_image, all_files))
            
            valid_files = sorted([f for f, valid in zip(all_files, valid_mask) if valid])
            
            result[split] = valid_files
            total_images += len(valid_files)
            
            logger.info(f"Found {len(valid_files)} valid images in {split}")
        
        logger.info(f"Total valid images across all splits: {total_images}")
        return result
    
    def detect_input_mode(self, input_dir: Path) -> str:
        """
        Auto-detect input mode based on directory structure.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            "pre-split" if train/val/test folders exist, "flat" otherwise
        """
        input_dir = Path(input_dir)
        splits = ['train', 'valid', 'test']
        
        existing_splits = sum(1 for s in splits if (input_dir / s).exists())
        
        if existing_splits >= 2:  # At least 2 of 3 splits present
            logger.info(f"Detected pre-split structure ({existing_splits}/3 splits found)")
            return "pre-split"
        else:
            logger.info("Detected flat structure (no train/val/test subdirectories)")
            return "flat"

