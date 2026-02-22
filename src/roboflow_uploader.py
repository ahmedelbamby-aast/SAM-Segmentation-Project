"""
Async distributed Roboflow uploader with retry logic.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Any
from queue import Queue, Empty
from threading import Thread, Event
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UploadTask:
    """Represents a batch upload task."""
    batch_dir: Path
    batch_id: int
    split: str = "train"


class DistributedUploader:
    """
    Async batch uploader for Roboflow with retry capability.
    
    Runs upload workers in background threads to avoid blocking
    the main segmentation pipeline.
    """
    
    def __init__(self, config, progress_tracker):
        """
        Initialize uploader.
        
        Args:
            config: Configuration object with roboflow settings
            progress_tracker: ProgressTracker instance for status updates
        """
        self.config = config.roboflow
        self.tracker = progress_tracker
        self.enabled = self.config.enabled
        
        self.rf = None
        self.project = None
        
        if self.enabled:
            self._init_roboflow()
        
        self.batch_size = self.config.batch_upload_size
        self.upload_queue: Queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=self.config.upload_workers)
        self.futures: List[Future] = []
        self._stop_event = Event()
        
        # Statistics
        self._completed = 0
        self._failed = 0
        
        # Start background worker
        if self.enabled:
            self._worker_thread = Thread(target=self._upload_worker, daemon=True)
            self._worker_thread.start()
            logger.info(f"Roboflow uploader started with {self.config.upload_workers} workers")
    
    def _init_roboflow(self):
        """Initialize Roboflow connections for all configured workspaces."""
        try:
            from roboflow import Roboflow
            self._roboflow_cls = Roboflow
            
            # Get all upload targets from config
            self.upload_targets = self.config.get_all_targets()
            
            if not self.upload_targets:
                logger.warning("No Roboflow workspaces configured, uploads disabled")
                self.enabled = False
                return
            
            # Log configured targets
            logger.info(f"Configured {len(self.upload_targets)} upload target(s):")
            for api_key, workspace, project, is_pred in self.upload_targets:
                pred_str = "prediction" if is_pred else "ground truth"
                logger.info(f"  → {workspace}/{project} ({pred_str})")
            
        except ImportError:
            logger.error("Roboflow package not installed. Run: pip install roboflow")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Roboflow: {e}")
            self.enabled = False


    
    def queue_batch(self, batch_dir: Path, batch_id: int, split: str = "train"):
        """
        Queue a batch directory for upload.
        
        Args:
            batch_dir: Directory containing the batch data
            batch_id: Batch ID for tracking
            split: Dataset split (train/val/test)
        """
        if not self.enabled:
            logger.debug("Roboflow disabled, skipping upload")
            return
        
        task = UploadTask(batch_dir=Path(batch_dir), batch_id=batch_id, split=split)
        self.upload_queue.put(task)
        logger.info(f"Queued batch {batch_id} for upload")
    
    def _upload_worker(self):
        """Background worker that processes upload queue."""
        while not self._stop_event.is_set():
            try:
                task = self.upload_queue.get(timeout=1.0)
                future = self.executor.submit(
                    self._upload_with_retry, 
                    task.batch_dir, 
                    task.batch_id, 
                    task.split
                )
                self.futures.append(future)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Upload worker error: {e}")
    
    def _upload_with_retry(self, batch_dir: Path, batch_id: int, split: str) -> bool:
        """
        Upload batch with retry logic to ALL configured workspaces/projects.
        
        Args:
            batch_dir: Directory containing batch data
            batch_id: Batch ID
            split: Dataset split
            
        Returns:
            True if ALL uploads succeeded, False if any failed
        """
        all_success = True
        
        # Upload to each configured workspace/project
        for api_key, workspace_name, project_name, is_prediction in self.upload_targets:
            success = self._upload_to_target(
                batch_dir, batch_id, api_key, workspace_name, project_name, is_prediction
            )
            if not success:
                all_success = False
        
        if all_success:
            self.tracker.mark_batch_uploaded(batch_id)
            self._completed += 1
            logger.info(f"Batch {batch_id} uploaded to all {len(self.upload_targets)} target(s)")
        else:
            self._failed += 1
            self.tracker.mark_batch_error(batch_id, "Failed to upload to one or more targets")
        
        return all_success
    
    def _upload_to_target(
        self, batch_dir: Path, batch_id: int, 
        api_key: str, workspace_name: str, project_name: str, is_prediction: bool
    ) -> bool:
        """Upload batch to a single workspace/project target."""
        for attempt in range(self.config.retry_attempts):
            try:
                logger.info(f"Uploading batch {batch_id} to {workspace_name}/{project_name} "
                           f"(attempt {attempt + 1}/{self.config.retry_attempts})")
                
                # Create Roboflow connection for this target
                rf = self._roboflow_cls(api_key=api_key)
                workspace = rf.workspace(workspace_name)
                
                workspace.upload_dataset(
                    dataset_path=str(batch_dir),
                    project_name=project_name,
                    num_workers=self.config.upload_workers,
                    project_license="MIT",
                    project_type="instance-segmentation",
                    batch_name=f"batch_{batch_id}",
                    num_retries=0,  # We handle retries ourselves
                    is_prediction=is_prediction
                )
                
                logger.info(f"✓ Batch {batch_id} uploaded to {workspace_name}/{project_name}")
                return True
                
            except Exception as e:
                logger.warning(f"Batch {batch_id} → {workspace_name}/{project_name} "
                              f"attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
        
        logger.error(f"✗ Batch {batch_id} failed to upload to {workspace_name}/{project_name}")
        return False

    
    def wait_for_uploads(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all queued uploads to complete.
        
        Args:
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            True if all uploads succeeded, False if any failed
        """
        import time as _time
        start_time = _time.time()
        
        # Wait for queue to drain and futures to be created
        # This fixes a race condition where queue_batch() is called but
        # the background worker hasn't yet moved the task into a future
        while not self.upload_queue.empty():
            if timeout is not None:
                elapsed = _time.time() - start_time
                if elapsed >= timeout:
                    logger.warning("Timeout waiting for upload queue to drain")
                    break
            _time.sleep(0.1)  # Small sleep to avoid busy-waiting
        
        # Give the worker a moment to submit any final tasks
        _time.sleep(0.5)
        
        if not self.futures:
            return True
        
        logger.info(f"Waiting for {len(self.futures)} uploads to complete...")
        
        all_success = True
        for future in self.futures:
            try:
                remaining_timeout = None
                if timeout is not None:
                    elapsed = _time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                result = future.result(timeout=remaining_timeout)
                if not result:
                    all_success = False
            except Exception as e:
                logger.error(f"Upload task failed: {e}")
                all_success = False
        
        return all_success
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get upload statistics.
        
        Returns:
            Dictionary with completed, failed, pending counts
        """
        completed = sum(1 for f in self.futures if f.done() and not f.exception())
        failed = sum(1 for f in self.futures if f.done() and f.exception())
        pending = self.upload_queue.qsize()
        in_progress = len([f for f in self.futures if not f.done()])
        
        return {
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'in_progress': in_progress,
            'total': len(self.futures)
        }
    
    def shutdown(self, wait: bool = True):
        """
        Gracefully shutdown uploader.
        
        Args:
            wait: Whether to wait for pending uploads
        """
        logger.info("Shutting down Roboflow uploader...")
        
        self._stop_event.set()
        
        if wait:
            self.wait_for_uploads(timeout=300)  # 5 minute timeout
        
        self.executor.shutdown(wait=wait)
        
        stats = self.get_stats()
        logger.info(f"Uploader shutdown. Stats: {stats}")
    
    def retry_failed_batches(self, job_id: int):
        """
        Retry uploading failed batches for a job.
        
        Args:
            job_id: Job ID to retry failed batches for
        """
        if not self.enabled:
            logger.warning("Roboflow disabled, cannot retry uploads")
            return
        
        pending_batches = self.tracker.get_pending_batches(job_id)
        
        if not pending_batches:
            logger.info("No pending batches to retry")
            return
        
        logger.info(f"Retrying {len(pending_batches)} failed batches")
        
        for batch in pending_batches:
            self.queue_batch(
                batch_dir=Path(batch.get('batch_dir', '')),
                batch_id=batch['id'],
                split=batch.get('split', 'train')
            )
    
    def upload_neither_folder(self, neither_dir: Path) -> bool:
        """
        Upload the 'neither' folder (images with no detections) as a single batch.
        
        This is only called when upload_neither is enabled in config.
        The folder is uploaded as a special batch called "neither" for manual review.
        Uploads to ALL configured workspaces/projects.
        
        Args:
            neither_dir: Path to the neither folder
            
        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.enabled:
            logger.info("Roboflow disabled, skipping neither folder upload")
            return True
        
        # Check if upload_neither is enabled
        upload_neither = getattr(self.config, 'upload_neither', False)
        if not upload_neither:
            logger.info("Neither folder upload disabled (upload_neither: false)")
            logger.info(f"Neither folder preserved for manual review at: {neither_dir}")
            return True
        
        neither_dir = Path(neither_dir)
        images_dir = neither_dir / 'images'
        
        if not images_dir.exists():
            logger.info("No neither folder found, skipping upload")
            return True
        
        # Count images in neither folder
        image_files = list(images_dir.glob('*'))
        if not image_files:
            logger.info("Neither folder is empty, skipping upload")
            return True
        
        logger.info(f"Uploading {len(image_files)} images from neither folder...")
        
        # Upload to ALL configured workspaces/projects
        all_success = True
        for api_key, workspace_name, project_name, is_prediction in self.upload_targets:
            for attempt in range(self.config.retry_attempts):
                try:
                    rf = self._roboflow_cls(api_key=api_key)
                    workspace = rf.workspace(workspace_name)
                    workspace.upload_dataset(
                        dataset_path=str(neither_dir),
                        project_name=project_name,
                        num_workers=self.config.upload_workers,
                        project_license="MIT",
                        project_type="instance-segmentation",
                        batch_name="neither",  # Special batch name for manual review
                        num_retries=0,
                        is_prediction=is_prediction
                    )
                    
                    logger.info(f"Neither folder uploaded to {workspace_name}/{project_name} ({len(image_files)} images)")
                    break  # Success, move to next target
                    
                except Exception as e:
                    logger.warning(f"Neither folder upload to {workspace_name}/{project_name} attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        logger.error(f"Neither folder upload to {workspace_name}/{project_name} failed after {self.config.retry_attempts} attempts")
                        all_success = False
        
        if all_success:
            logger.info(f"Neither folder uploaded to all {len(self.upload_targets)} target(s)")
        return all_success
    
    def should_upload_neither(self) -> bool:
        """Check if neither folder upload is enabled."""
        return getattr(self.config, 'upload_neither', False)

