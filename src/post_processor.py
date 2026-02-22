"""
Post-processing module for handling overlapping annotations.

Implements mask-aware NMS (Non-Maximum Suppression) to resolve overlapping
segmentation masks between different classes (teacher/student).

Best Practices Applied:
1. Mask IoU calculation (not just bounding box)
2. Multi-class NMS with configurable strategies
3. Confidence-based suppression
4. Soft-NMS option for gradual confidence decay

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class OverlapStrategy(Enum):
    """Strategy for resolving overlapping annotations."""
    CONFIDENCE = "confidence"      # Keep higher confidence mask
    AREA = "area"                  # Keep larger mask
    CLASS_PRIORITY = "class_priority"  # Prioritize specific class
    SOFT_NMS = "soft_nms"          # Decay confidence instead of hard removal


@dataclass
class PostProcessConfig:
    """Configuration for post-processing."""
    enabled: bool = True
    iou_threshold: float = 0.5     # IoU threshold for considering overlap
    strategy: OverlapStrategy = OverlapStrategy.CONFIDENCE
    class_priority: List[str] = field(default_factory=lambda: ["teacher", "student"])
    soft_nms_sigma: float = 0.5    # Sigma for Soft-NMS Gaussian decay
    min_confidence_after_decay: float = 0.1  # Min confidence after Soft-NMS


class MaskPostProcessor:
    """
    Post-processor for segmentation masks using NMS algorithms.
    
    Handles overlapping masks between different classes by:
    1. Calculating mask IoU (Intersection over Union)
    2. Applying selected suppression strategy
    3. Optionally using Soft-NMS for gradual confidence decay
    """
    
    def __init__(self, config: Optional[PostProcessConfig] = None):
        """
        Initialize post-processor.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config or PostProcessConfig()
        self._stats = {
            'total_processed': 0,
            'overlaps_detected': 0,
            'masks_suppressed': 0,
            'confidence_decayed': 0
        }
        
        if self.config.enabled:
            logger.info(f"MaskPostProcessor initialized (strategy={self.config.strategy.value}, iou_threshold={self.config.iou_threshold})")
    
    def calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two binary masks.
        
        Args:
            mask1: First binary mask (H, W)
            mask2: Second binary mask (H, W)
            
        Returns:
            IoU value between 0 and 1
        """
        # Ensure masks are boolean
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate overlap percentages between two masks.
        
        Returns:
            Tuple of (overlap_on_mask1, overlap_on_mask2)
            - overlap_on_mask1: percentage of mask1 covered by mask2
            - overlap_on_mask2: percentage of mask2 covered by mask1
        """
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        
        intersection = np.logical_and(mask1, mask2).sum()
        
        mask1_area = mask1.sum()
        mask2_area = mask2.sum()
        
        overlap_on_mask1 = intersection / mask1_area if mask1_area > 0 else 0
        overlap_on_mask2 = intersection / mask2_area if mask2_area > 0 else 0
        
        return overlap_on_mask1, overlap_on_mask2
    
    def apply_nms(
        self,
        masks: List[np.ndarray],
        class_ids: List[int],
        confidences: List[float],
        class_names: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[int], List[float], Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression to segmentation masks.
        
        This is a cross-class NMS that handles overlaps between different classes.
        
        Args:
            masks: List of binary masks (H, W)
            class_ids: List of class IDs for each mask
            confidences: List of confidence scores for each mask
            class_names: Optional list of class names for priority
            
        Returns:
            Tuple of (filtered_masks, filtered_class_ids, filtered_confidences, stats)
        """
        if not self.config.enabled or len(masks) == 0:
            return masks, class_ids, confidences, {'suppressed': 0}
        
        self._stats['total_processed'] += len(masks)
        
        # Convert to numpy arrays for easier manipulation
        n = len(masks)
        confidences = np.array(confidences) if not isinstance(confidences, np.ndarray) else confidences
        class_ids = np.array(class_ids) if not isinstance(class_ids, np.ndarray) else class_ids
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(-confidences)
        
        # Track which masks to keep
        keep = np.ones(n, dtype=bool)
        suppressed_info = []
        
        for i, idx_i in enumerate(sorted_indices):
            if not keep[idx_i]:
                continue
            
            mask_i = masks[idx_i]
            conf_i = confidences[idx_i]
            class_i = class_ids[idx_i]
            
            # Compare with remaining masks
            for j in range(i + 1, n):
                idx_j = sorted_indices[j]
                if not keep[idx_j]:
                    continue
                
                mask_j = masks[idx_j]
                class_j = class_ids[idx_j]
                
                # Calculate IoU
                iou = self.calculate_mask_iou(mask_i, mask_j)
                
                if iou > self.config.iou_threshold:
                    self._stats['overlaps_detected'] += 1
                    
                    # Determine which mask to suppress based on strategy
                    suppress_j = self._should_suppress(
                        mask_i=mask_i, mask_j=mask_j,
                        conf_i=conf_i, conf_j=confidences[idx_j],
                        class_i=class_i, class_j=class_j,
                        class_names=class_names
                    )
                    
                    if suppress_j:
                        if self.config.strategy == OverlapStrategy.SOFT_NMS:
                            # Apply confidence decay instead of hard suppression
                            decay = np.exp(-(iou ** 2) / self.config.soft_nms_sigma)
                            confidences[idx_j] *= decay
                            self._stats['confidence_decayed'] += 1
                            
                            if confidences[idx_j] < self.config.min_confidence_after_decay:
                                keep[idx_j] = False
                                self._stats['masks_suppressed'] += 1
                        else:
                            # Hard suppression
                            keep[idx_j] = False
                            self._stats['masks_suppressed'] += 1
                            suppressed_info.append({
                                'suppressed_class': int(class_j),
                                'kept_class': int(class_i),
                                'iou': float(iou)
                            })
        
        # Filter results
        keep_indices = np.where(keep)[0]
        filtered_masks = [masks[i] for i in keep_indices]
        filtered_class_ids = class_ids[keep_indices].tolist()
        filtered_confidences = confidences[keep_indices].tolist()
        
        stats = {
            'original_count': n,
            'final_count': len(filtered_masks),
            'suppressed': n - len(filtered_masks),
            'suppressed_info': suppressed_info[:5]  # Limit to first 5 for logging
        }
        
        if stats['suppressed'] > 0:
            logger.debug(f"NMS: Suppressed {stats['suppressed']}/{n} overlapping masks")
        
        return filtered_masks, filtered_class_ids, filtered_confidences, stats
    
    def _should_suppress(
        self,
        mask_i: np.ndarray, mask_j: np.ndarray,
        conf_i: float, conf_j: float,
        class_i: int, class_j: int,
        class_names: Optional[List[str]] = None
    ) -> bool:
        """
        Determine if mask_j should be suppressed in favor of mask_i.
        
        Returns:
            True if mask_j should be suppressed, False otherwise
        """
        if self.config.strategy == OverlapStrategy.CONFIDENCE:
            # Higher confidence wins (mask_i should already have higher conf)
            return conf_i >= conf_j
        
        elif self.config.strategy == OverlapStrategy.AREA:
            # Larger mask wins
            area_i = mask_i.sum()
            area_j = mask_j.sum()
            return area_i >= area_j
        
        elif self.config.strategy == OverlapStrategy.CLASS_PRIORITY:
            # Check class priority
            if class_names:
                name_i = class_names[class_i] if class_i < len(class_names) else str(class_i)
                name_j = class_names[class_j] if class_j < len(class_names) else str(class_j)
                
                priority = self.config.class_priority
                if name_i in priority and name_j in priority:
                    # Lower index = higher priority
                    return priority.index(name_i) <= priority.index(name_j)
            
            # Fall back to confidence if class names not available
            return conf_i >= conf_j
        
        elif self.config.strategy == OverlapStrategy.SOFT_NMS:
            # Soft-NMS always returns True but applies decay instead of removal
            return True
        
        return conf_i >= conf_j  # Default to confidence
    
    def apply_class_specific_nms(
        self,
        masks: List[np.ndarray],
        class_ids: List[int],
        confidences: List[float]
    ) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """
        Apply NMS within each class separately.
        
        This preserves overlapping masks from different classes while
        removing duplicates within the same class.
        
        Args:
            masks: List of binary masks
            class_ids: List of class IDs
            confidences: List of confidence scores
            
        Returns:
            Filtered masks, class_ids, and confidences
        """
        if not masks:
            return masks, class_ids, confidences
        
        # Group by class
        unique_classes = set(class_ids)
        final_masks = []
        final_class_ids = []
        final_confidences = []
        
        for cls in unique_classes:
            # Get masks for this class
            indices = [i for i, c in enumerate(class_ids) if c == cls]
            cls_masks = [masks[i] for i in indices]
            cls_confs = [confidences[i] for i in indices]
            cls_ids = [class_ids[i] for i in indices]
            
            # Apply NMS within class
            filtered_masks, filtered_ids, filtered_confs, _ = self.apply_nms(
                cls_masks, cls_ids, cls_confs
            )
            
            final_masks.extend(filtered_masks)
            final_class_ids.extend(filtered_ids)
            final_confidences.extend(filtered_confs)
        
        return final_masks, final_class_ids, final_confidences
    
    def get_stats(self) -> Dict[str, Any]:
        """Get post-processing statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = {
            'total_processed': 0,
            'overlaps_detected': 0,
            'masks_suppressed': 0,
            'confidence_decayed': 0
        }


def create_post_processor(config) -> MaskPostProcessor:
    """
    Factory function to create post-processor from main config.
    
    Args:
        config: Main configuration object
        
    Returns:
        Configured MaskPostProcessor instance
    """
    # Check if post-processing config exists
    post_config_dict = getattr(config, 'post_processing', None)
    
    if post_config_dict and isinstance(post_config_dict, dict):
        strategy_str = post_config_dict.get('strategy', 'confidence')
        strategy = OverlapStrategy(strategy_str)
        
        pp_config = PostProcessConfig(
            enabled=post_config_dict.get('enabled', True),
            iou_threshold=post_config_dict.get('iou_threshold', 0.5),
            strategy=strategy,
            class_priority=post_config_dict.get('class_priority', ['teacher', 'student']),
            soft_nms_sigma=post_config_dict.get('soft_nms_sigma', 0.5),
            min_confidence_after_decay=post_config_dict.get('min_confidence_after_decay', 0.1)
        )
    else:
        # Default configuration
        pp_config = PostProcessConfig()
    
    return MaskPostProcessor(pp_config)
