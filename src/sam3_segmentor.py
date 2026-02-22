"""
SAM 3 segmentation wrapper with CPU/GPU/Multi-GPU support.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import cv2
import gc
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    masks: np.ndarray              # (N, H, W) boolean masks
    class_ids: List[int]           # Class ID for each mask (0=teacher, 1=student)
    confidences: List[float]       # Confidence scores
    boxes: np.ndarray              # (N, 4) bounding boxes [x1, y1, x2, y2]
    image_size: Tuple[int, int]    # (height, width)
    
    @property
    def num_detections(self) -> int:
        """Number of detected objects."""
        return len(self.class_ids)
    
    def filter_by_confidence(self, min_conf: float) -> 'SegmentationResult':
        """Filter results by minimum confidence."""
        mask = np.array(self.confidences) >= min_conf
        return SegmentationResult(
            masks=self.masks[mask] if len(self.masks) > 0 else self.masks,
            class_ids=[c for c, m in zip(self.class_ids, mask) if m],
            confidences=[c for c, m in zip(self.confidences, mask) if m],
            boxes=self.boxes[mask] if len(self.boxes) > 0 else self.boxes,
            image_size=self.image_size
        )


class DeviceManager:
    """Manage device selection for CPU/GPU/Multi-GPU environments."""
    
    @staticmethod
    def get_available_devices() -> Dict[str, Any]:
        """
        Detect available compute devices.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'cuda_available': False,
            'cuda_device_count': 0,
            'cuda_devices': [],
            'recommended_device': 'cpu'
        }
        
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            
            if info['cuda_available']:
                info['cuda_device_count'] = torch.cuda.device_count()
                
                for i in range(info['cuda_device_count']):
                    props = torch.cuda.get_device_properties(i)
                    info['cuda_devices'].append({
                        'id': i,
                        'name': props.name,
                        'memory_gb': props.total_memory / 1e9,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                
                if info['cuda_device_count'] > 0:
                    info['recommended_device'] = 'cuda:0'
        except ImportError:
            logger.warning("PyTorch not found, defaulting to CPU")
        except Exception as e:
            logger.warning(f"Error detecting CUDA devices: {e}")
        
        return info
    
    @staticmethod
    def resolve_device(device_config: str) -> str:
        """
        Resolve device string from config.
        
        Args:
            device_config: Device from config ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            
        Returns:
            Resolved device string
        """
        if device_config == 'auto':
            info = DeviceManager.get_available_devices()
            device = info['recommended_device']
            logger.info(f"Auto-detected device: {device}")
            return device
        
        # Validate CUDA device if specified
        if device_config.startswith('cuda'):
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    return 'cpu'
            except ImportError:
                logger.warning("PyTorch not found, falling back to CPU")
                return 'cpu'
        
        return device_config
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage info."""
        info = {'cpu_ram_gb': 0}
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            info['cpu_ram_gb'] = mem.available / 1e9
            info['cpu_ram_total_gb'] = mem.total / 1e9
            info['cpu_ram_percent'] = mem.percent
        except ImportError:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    info[f'cuda:{i}_allocated_gb'] = allocated
                    info[f'cuda:{i}_reserved_gb'] = reserved
                    info[f'cuda:{i}_total_gb'] = total
                    info[f'cuda:{i}_free_gb'] = total - reserved
        except ImportError:
            pass
        
        return info


class SAM3Segmentor:
    """
    SAM 3 segmentation with text prompts for teacher/student classification.
    
    Supports:
    - CPU mode (optimized for low memory)
    - Single GPU mode (with FP16 optimization)
    - Multi-GPU mode (automatic device selection)
    """
    
    def __init__(self, config):
        """
        Initialize SAM 3 segmentor.
        
        Args:
            config: Configuration object with model settings
        """
        self.prompts = config.model.prompts  # ["teacher", "student"]
        self.confidence = config.model.confidence
        self.model_path = config.model.path
        self.resolution = config.pipeline.resolution
        
        # Resolve device
        self.device = DeviceManager.resolve_device(config.model.device)
        self.is_cpu = self.device == 'cpu'
        
        # Adjust half precision for CPU (not supported on CPU)
        self.half_precision = config.model.half_precision and not self.is_cpu
        if self.is_cpu and config.model.half_precision:
            logger.info("FP16 not supported on CPU, using FP32")
        
        self.predictor = None
        self._model_loaded = False
        
        # Initialize post-processor for handling overlapping annotations
        self.post_processor = None
        if config.post_processing and config.post_processing.enabled:
            from .post_processor import MaskPostProcessor, PostProcessConfig, OverlapStrategy
            pp_config = PostProcessConfig(
                enabled=config.post_processing.enabled,
                iou_threshold=config.post_processing.iou_threshold,
                strategy=OverlapStrategy(config.post_processing.strategy),
                class_priority=config.post_processing.class_priority,
                soft_nms_sigma=config.post_processing.soft_nms_sigma,
                min_confidence_after_decay=config.post_processing.min_confidence_after_decay
            )
            self.post_processor = MaskPostProcessor(pp_config)
            logger.info(f"Post-processing enabled: strategy={config.post_processing.strategy}, iou_threshold={config.post_processing.iou_threshold}")
        
        # Memory optimization settings
        self._gc_interval = 50 if self.is_cpu else 100  # More frequent GC on CPU
        self._process_count = 0
        
        logger.info(f"SAM3Segmentor initialized - Device: {self.device}, FP16: {self.half_precision}")
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system configuration for debugging."""
        device_info = DeviceManager.get_available_devices()
        
        if device_info['cuda_available']:
            for dev in device_info['cuda_devices']:
                logger.info(f"GPU {dev['id']}: {dev['name']} ({dev['memory_gb']:.1f} GB)")
        else:
            logger.info("No GPU available, running on CPU")
        
        mem_info = DeviceManager.get_memory_info()
        if 'cpu_ram_total_gb' in mem_info:
            logger.info(f"System RAM: {mem_info['cpu_ram_total_gb']:.1f} GB "
                       f"(Available: {mem_info['cpu_ram_gb']:.1f} GB)")
    
    def _ensure_loaded(self):
        """Lazy load model on first use."""
        if self._model_loaded:
            return
        
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
            
            overrides = {
                'conf': self.confidence,
                'task': 'segment',
                'mode': 'predict',
                'model': str(self.model_path),
                'half': self.half_precision,
                'imgsz': self.resolution,
                'verbose': False,
                'device': self.device,
            }
            
            logger.info(f"Loading SAM 3 model from {self.model_path}...")
            logger.info(f"Model config: device={self.device}, half={self.half_precision}, imgsz={self.resolution}")
            
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            self.predictor.setup_model()
            self._model_loaded = True
            
            logger.info("SAM 3 model loaded successfully")
            
            # Log memory after model load
            mem_info = DeviceManager.get_memory_info()
            if self.is_cpu:
                logger.info(f"RAM after model load: {mem_info.get('cpu_ram_gb', 0):.1f} GB available")
            else:
                key = f'{self.device}_free_gb'
                if key in mem_info:
                    logger.info(f"GPU memory after model load: {mem_info[key]:.1f} GB free")
            
        except ImportError as e:
            raise ImportError(
                "SAM3SemanticPredictor not found. Ensure ultralytics>=8.3.237 is installed "
                "and CLIP fork is installed: pip install git+https://github.com/ultralytics/CLIP.git"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM 3 model: {e}") from e
    
    def _maybe_gc(self):
        """Periodically run garbage collection to free memory."""
        self._process_count += 1
        
        if self._process_count % self._gc_interval == 0:
            gc.collect()
            
            if not self.is_cpu:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
    
    def process_image(self, image_path: Path) -> Optional[SegmentationResult]:
        """
        Process single image and return segmentation results.
        
        Args:
            image_path: Path to image file
            
        Returns:
            SegmentationResult or None if no detections
        """
        self._ensure_loaded()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image to get dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            h, w = img.shape[:2]
            
            # Set image for feature extraction
            self.predictor.set_image(str(image_path))
            
            # Query with text prompts
            results = self.predictor(text=self.prompts)
            
            if results is None or len(results) == 0:
                logger.debug(f"No detections in {image_path.name}")
                self._maybe_gc()
                return None
            
            result = results[0]
            
            # Check for valid masks
            if result.masks is None or len(result.masks) == 0:
                logger.debug(f"No masks in {image_path.name}")
                self._maybe_gc()
                return None
            
            # Extract data (move to CPU/numpy for memory efficiency)
            masks = result.masks.data.cpu().numpy()
            
            # Get boxes and class info
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
                confidences = result.boxes.conf.cpu().numpy().tolist()
            else:
                boxes = np.array([])
                class_ids = []
                confidences = []
            
            seg_result = SegmentationResult(
                masks=masks,
                class_ids=class_ids,
                confidences=confidences,
                boxes=boxes,
                image_size=(h, w)
            )
            
            # Apply post-processing NMS to handle overlapping annotations
            if self.post_processor and seg_result.num_detections > 1:
                filtered_masks, filtered_class_ids, filtered_confidences, pp_stats = \
                    self.post_processor.apply_nms(
                        list(seg_result.masks),
                        seg_result.class_ids,
                        seg_result.confidences,
                        class_names=self.prompts
                    )
                
                if pp_stats['suppressed'] > 0:
                    logger.debug(f"Post-processing: removed {pp_stats['suppressed']} overlapping masks")
                    
                    # Update result with filtered data
                    seg_result = SegmentationResult(
                        masks=np.array(filtered_masks) if filtered_masks else np.array([]),
                        class_ids=filtered_class_ids,
                        confidences=filtered_confidences,
                        boxes=seg_result.boxes[np.isin(range(len(seg_result.boxes)), 
                               [i for i in range(len(seg_result.masks)) if i < len(filtered_masks)])] 
                               if len(seg_result.boxes) > 0 else seg_result.boxes,
                        image_size=seg_result.image_size
                    )
            
            logger.debug(f"Found {seg_result.num_detections} objects in {image_path.name}")
            
            self._maybe_gc()
            return seg_result
            
        except Exception as e:
            logger.error(f"Segmentation failed for {image_path}: {e}")
            self._maybe_gc()
            raise RuntimeError(f"Segmentation failed for {image_path}: {e}") from e
    
    def process_batch(self, image_paths: List[Path]) -> List[Optional[SegmentationResult]]:
        """
        Process multiple images.
        
        Note: SAM3 processes images sequentially but reuses model features.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of SegmentationResult or None for each image
        """
        results = []
        for path in image_paths:
            try:
                result = self.process_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append(None)
        return results
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        if 0 <= class_id < len(self.prompts):
            return self.prompts[class_id]
        return f"unknown_{class_id}"
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information."""
        return {
            'device': self.device,
            'is_cpu': self.is_cpu,
            'half_precision': self.half_precision,
            'model_loaded': self._model_loaded,
            'memory': DeviceManager.get_memory_info()
        }
    
    def cleanup(self):
        """Release model resources."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
            self._model_loaded = False
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # Force garbage collection
            gc.collect()
            
            logger.info("SAM 3 model resources released")
