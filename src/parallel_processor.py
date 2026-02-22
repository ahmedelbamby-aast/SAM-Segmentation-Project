"""
Multi-process parallel inference processor for SAM 3.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import multiprocessing as mp
from multiprocessing import Pool, Queue
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import logging
import time
import queue

logger = logging.getLogger(__name__)


@dataclass  
class ProcessingTask:
    """A single image processing task."""
    image_id: int
    image_path: str
    split: str


@dataclass
class ProcessingResult:
    """Result from processing an image."""
    image_id: int
    image_path: str
    split: str
    success: bool
    has_detections: bool
    error_message: Optional[str] = None
    result_data: Optional[Dict] = None  # Serializable result data


# Global variables for worker process (initialized once per worker)
_worker_segmentor = None
_worker_filter = None
_worker_writer = None
_worker_config = None


def _init_worker(config_dict: Dict):
    """Initialize worker process with its own SAM 3 model instance."""
    global _worker_segmentor, _worker_filter, _worker_writer, _worker_config
    
    # Reconstruct config from dict
    from .config_manager import load_config_from_dict
    _worker_config = load_config_from_dict(config_dict)
    
    # Initialize components in worker process
    from .sam3_segmentor import SAM3Segmentor
    from .result_filter import ResultFilter
    from .annotation_writer import AnnotationWriter
    
    logger.info(f"Initializing worker process {mp.current_process().name}")
    
    _worker_segmentor = SAM3Segmentor(_worker_config)
    _worker_filter = ResultFilter(_worker_config)
    _worker_writer = AnnotationWriter(_worker_config)
    
    logger.info(f"Worker {mp.current_process().name} ready")


def _process_image_worker(task: Tuple[int, str, str]) -> Tuple[int, str, str, bool, bool, Optional[str]]:
    """
    Worker function to process a single image.
    
    Args:
        task: Tuple of (image_id, image_path, split)
        
    Returns:
        Tuple of (image_id, image_path, split, success, has_detections, error_message)
    """
    global _worker_segmentor, _worker_filter, _worker_writer
    
    image_id, image_path_str, split = task
    image_path = Path(image_path_str)  # Convert string to Path object
    
    try:
        # Process image with SAM 3
        result = _worker_segmentor.process_image(image_path)
        
        # Filter: check if image has detections
        has_detections = _worker_filter.filter_result(image_path, result)
        
        # Write annotation only if detections found
        if has_detections and result and result.num_detections > 0:
            _worker_writer.write_annotation(image_path, result, split)
        
        return (image_id, image_path_str, split, True, has_detections, None)
        
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return (image_id, image_path_str, split, False, False, str(e))


class ParallelProcessor:
    """
    Multi-process parallel processor for SAM 3 inference.
    
    Uses a process pool to run inference on multiple images simultaneously.
    Each worker process has its own copy of the SAM 3 model.
    """
    
    def __init__(self, config, num_workers: int = 2):
        """
        Initialize parallel processor.
        
        Args:
            config: Configuration object
            num_workers: Number of parallel worker processes
        """
        self.config = config
        self.num_workers = max(1, num_workers)
        self.pool = None
        self._config_dict = None
        
        logger.info(f"ParallelProcessor initialized with {self.num_workers} workers")
    
    def _get_config_dict(self) -> Dict:
        """Convert config to serializable dict for worker processes."""
        if self._config_dict is None:
            self._config_dict = {
                'pipeline': {
                    'input_dir': str(self.config.pipeline.input_dir),
                    'output_dir': str(self.config.pipeline.output_dir),
                    'resolution': self.config.pipeline.resolution,
                    'supported_formats': self.config.pipeline.supported_formats,
                    'num_workers': self.config.pipeline.num_workers,
                    'input_mode': self.config.pipeline.input_mode,
                },
                'model': {
                    'path': str(self.config.model.path),
                    'confidence': self.config.model.confidence,
                    'prompts': self.config.model.prompts,
                    'half_precision': self.config.model.half_precision,
                    'device': self.config.model.device,
                    'parallel_workers': 1,  # Workers don't spawn sub-workers
                },
                'split': {
                    'train': self.config.split.train,
                    'valid': self.config.split.valid,
                    'test': self.config.split.test,
                    'seed': self.config.split.seed,
                },
                'progress': {
                    'db_path': str(self.config.progress.db_path),
                    'checkpoint_interval': self.config.progress.checkpoint_interval,
                    'log_file': str(self.config.progress.log_file),
                    'log_level': self.config.progress.log_level,
                },
                'roboflow': {
                    'enabled': self.config.roboflow.enabled,
                    'api_key': self.config.roboflow.api_key,
                    'workspace': self.config.roboflow.workspace,
                    'project': self.config.roboflow.project,
                    'batch_upload_size': self.config.roboflow.batch_upload_size,
                    'upload_workers': self.config.roboflow.upload_workers,
                    'retry_attempts': self.config.roboflow.retry_attempts,
                    'retry_delay': self.config.roboflow.retry_delay,
                }
            }
        return self._config_dict
    
    def start(self):
        """Start the worker pool."""
        if self.pool is not None:
            return
        
        logger.info(f"Starting process pool with {self.num_workers} workers...")
        
        # Use spawn method for cleaner process isolation
        ctx = mp.get_context('spawn')
        
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self._get_config_dict(),)
        )
        
        logger.info("Process pool started")
    
    def process_batch(
        self, 
        tasks: List[Tuple[int, str, str]]
    ) -> List[Tuple[int, str, str, bool, bool, Optional[str]]]:
        """
        Process a batch of images in parallel.
        
        Args:
            tasks: List of (image_id, image_path, split) tuples
            
        Returns:
            List of (image_id, image_path, split, success, has_detections, error) results
        """
        if self.pool is None:
            self.start()
        
        # Process all tasks in parallel
        results = self.pool.map(_process_image_worker, tasks)
        
        return results
    
    def process_batch_async(
        self, 
        tasks: List[Tuple[int, str, str]],
        callback: Optional[Callable] = None
    ):
        """
        Process a batch of images asynchronously.
        
        Args:
            tasks: List of (image_id, image_path, split) tuples
            callback: Optional callback for each completed result
            
        Returns:
            AsyncResult object
        """
        if self.pool is None:
            self.start()
        
        return self.pool.map_async(_process_image_worker, tasks, callback=callback)
    
    def shutdown(self):
        """Shutdown the worker pool."""
        if self.pool is not None:
            logger.info("Shutting down process pool...")
            self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("Process pool shutdown complete")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


class SequentialProcessor:
    """
    Sequential processor for SAM 3 inference (fallback/single-worker mode).
    
    Provides the same interface as ParallelProcessor for consistency.
    """
    
    def __init__(self, config):
        """Initialize with config."""
        self.config = config
        self.segmentor = None
        self.filter = None
        self.writer = None
        
        logger.info("SequentialProcessor initialized (single-process mode)")
    
    def _ensure_loaded(self):
        """Lazy load components."""
        if self.segmentor is None:
            from .sam3_segmentor import SAM3Segmentor
            from .result_filter import ResultFilter
            from .annotation_writer import AnnotationWriter
            
            self.segmentor = SAM3Segmentor(self.config)
            self.filter = ResultFilter(self.config)
            self.writer = AnnotationWriter(self.config)
    
    def start(self):
        """Start (lazy load components)."""
        self._ensure_loaded()
    
    def process_batch(
        self, 
        tasks: List[Tuple[int, str, str]]
    ) -> List[Tuple[int, str, str, bool, bool, Optional[str]]]:
        """Process tasks sequentially."""
        self._ensure_loaded()
        
        results = []
        for task in tasks:
            image_id, image_path_str, split = task
            image_path = Path(image_path_str)  # Convert string to Path object
            
            try:
                result = self.segmentor.process_image(image_path)
                has_detections = self.filter.filter_result(image_path, result)
                
                if has_detections and result and result.num_detections > 0:
                    self.writer.write_annotation(image_path, result, split)
                
                results.append((image_id, image_path_str, split, True, has_detections, None))
                
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                results.append((image_id, image_path_str, split, False, False, str(e)))
        
        return results
    
    def shutdown(self):
        """Cleanup resources."""
        if self.segmentor:
            self.segmentor.cleanup()
        self.segmentor = None
        self.filter = None
        self.writer = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


def create_processor(config) -> 'ParallelProcessor | SequentialProcessor':
    """
    Factory function to create the appropriate processor.
    
    Args:
        config: Configuration object
        
    Returns:
        ParallelProcessor or SequentialProcessor based on config
    """
    num_workers = getattr(config.model, 'parallel_workers', 1)
    
    if num_workers > 1:
        logger.info(f"Creating ParallelProcessor with {num_workers} workers")
        return ParallelProcessor(config, num_workers)
    else:
        logger.info("Creating SequentialProcessor (single-process mode)")
        return SequentialProcessor(config)
