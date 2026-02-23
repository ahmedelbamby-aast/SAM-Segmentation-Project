"""
Async distributed Roboflow uploader with retry logic.

SRP breakdown:
  - :class:`AsyncWorkerPool`  — generic background queue + thread + executor only
  - :class:`DistributedUploader` — Roboflow-specific upload logic only;
    delegates queue management to :class:`AsyncWorkerPool`

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Any, Callable
from queue import Queue, Empty
from threading import Thread, Event
from dataclasses import dataclass

from .logging_system import LoggingSystem

logger = LoggingSystem.get_logger(__name__)


# ---------------------------------------------------------------------------
# UploadTask — IPC data container
# ---------------------------------------------------------------------------

@dataclass
class UploadTask:
    """Represents a batch upload task."""

    batch_dir: Path
    batch_id: int
    split: str = "train"


# ---------------------------------------------------------------------------
# AsyncWorkerPool — SRP: background queue management only
# ---------------------------------------------------------------------------

class AsyncWorkerPool:
    """Generic background task queue backed by a ``ThreadPoolExecutor``.

    Single Responsibility: manages the queue drain loop, the executor, and
    the futures list.  It knows nothing about Roboflow or upload logic.

    Usage::

        pool = AsyncWorkerPool(max_workers=4, task_fn=my_callable)
        pool.submit(some_task)
        results = pool.wait(timeout=300)
        pool.shutdown()
    """

    def __init__(
        self,
        max_workers: int,
        task_fn: Callable[[Any], Any],
    ) -> None:
        """Initialise the pool.

        Args:
            max_workers: Thread-pool concurrency limit.
            task_fn: Callable invoked for each submitted task.  Must accept a
                single positional argument (the task object) and return any
                value that is stored as the future result.
        """
        self._queue: Queue = Queue()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max_workers
        )
        self._futures: List[Future] = []
        self._stop_event: Event = Event()
        self._task_fn = task_fn

        self._worker_thread = Thread(target=self._drain_queue, daemon=True)
        self._worker_thread.start()
        logger.debug(
            f"AsyncWorkerPool started (max_workers={max_workers})"
        )

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------

    def submit(self, task: Any) -> None:
        """Enqueue a task for background execution.

        Args:
            task: Any picklable object passed verbatim to *task_fn*.
        """
        self._queue.put(task)

    def _drain_queue(self) -> None:
        """Background thread that moves queued tasks into the executor."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=1.0)
                future = self._executor.submit(self._task_fn, task)
                self._futures.append(future)
            except Empty:
                continue
            except Exception as exc:
                logger.error(f"AsyncWorkerPool drain error: {exc}")

    # ------------------------------------------------------------------
    # Wait / stats / shutdown
    # ------------------------------------------------------------------

    def wait(self, timeout: Optional[float] = None) -> List[Any]:
        """Block until all queued tasks finish.

        Args:
            timeout: Maximum seconds to wait.  ``None`` = wait forever.

        Returns:
            List of results (one per submitted task, in completion order).
            Tasks that raised an exception contribute ``None`` to the list.
        """
        start = time.time()

        # Drain the queue first so all tasks become futures.
        while not self._queue.empty():
            if timeout is not None and time.time() - start >= timeout:
                logger.warning("AsyncWorkerPool.wait: queue drain timeout")
                break
            time.sleep(0.1)

        # Brief pause to let the worker submit any last task.
        time.sleep(0.5)

        if not self._futures:
            return []

        logger.info(
            f"AsyncWorkerPool: waiting for {len(self._futures)} future(s)"
        )
        results: List[Any] = []
        for future in self._futures:
            remaining: Optional[float] = None
            if timeout is not None:
                remaining = max(0.0, timeout - (time.time() - start))
            try:
                results.append(future.result(timeout=remaining))
            except Exception as exc:
                logger.error(f"AsyncWorkerPool: task raised {exc}")
                results.append(None)

        return results

    def get_stats(self) -> Dict[str, int]:
        """Return counts of completed, failed, pending and in-progress tasks.

        Returns:
            Dict with keys ``completed``, ``failed``, ``pending``,
            ``in_progress``, ``total``.
        """
        completed = sum(
            1 for f in self._futures if f.done() and f.exception() is None
        )
        failed = sum(
            1 for f in self._futures if f.done() and f.exception() is not None
        )
        in_progress = sum(1 for f in self._futures if not f.done())
        return {
            "completed": completed,
            "failed": failed,
            "pending": self._queue.qsize(),
            "in_progress": in_progress,
            "total": len(self._futures),
        }

    def shutdown(self, wait: bool = True) -> None:
        """Stop the drain thread and shut down the executor.

        Args:
            wait: If ``True``, block until all running tasks complete.
        """
        self._stop_event.set()
        self._executor.shutdown(wait=wait)
        logger.debug("AsyncWorkerPool shut down")


class DistributedUploader:
    """
    Async batch uploader for Roboflow with retry capability.

    Delegates all queue and thread management to ``AsyncWorkerPool``
    (SRP — this class owns only Roboflow-specific upload logic).
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

        self._roboflow_cls = None
        self.upload_targets: List[Any] = []

        if self.enabled:
            self._init_roboflow()

        self.batch_size = self.config.batch_upload_size

        # Delegate queue/thread/executor management to AsyncWorkerPool (SRP)
        self._pool = AsyncWorkerPool(
            max_workers=self.config.upload_workers,
            task_fn=self._execute_task,
        )

        if self.enabled:
            logger.info(f"Roboflow uploader started with {self.config.upload_workers} workers")

    # ------------------------------------------------------------------
    # AsyncWorkerPool adapter
    # ------------------------------------------------------------------

    def _execute_task(self, task: UploadTask) -> bool:
        """Adapter: run one upload task inside the worker pool."""
        return self._upload_with_retry(task.batch_dir, task.batch_id, task.split)

    def queue_batch(self, batch_dir: Path, batch_id: int, split: str = "train") -> None:
        """
        Queue a batch directory for upload.

        Args:
            batch_dir: Directory containing the batch data
            batch_id: Batch ID for tracking
            split: Dataset split (train/valid/test)
        """
        if not self.enabled:
            logger.debug("Roboflow disabled, skipping upload")
            return

        task = UploadTask(batch_dir=Path(batch_dir), batch_id=batch_id, split=split)
        self._pool.submit(task)
        logger.info(f"Queued batch {batch_id} for upload")

    def wait_for_uploads(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all queued uploads to complete.

        Args:
            timeout: Maximum time to wait in seconds (``None`` = wait forever)

        Returns:
            True if all uploads succeeded, False if any failed
        """
        results = self._pool.wait(timeout=timeout)
        return all(r for r in results if r is not None)

    def get_stats(self) -> Dict[str, Any]:
        """Return upload statistics delegated from AsyncWorkerPool."""
        return self._pool.get_stats()

    def shutdown(self, wait: bool = True) -> None:
        """
        Gracefully shut down the uploader.

        Args:
            wait: Whether to wait for pending uploads to finish
        """
        logger.info("Shutting down Roboflow uploader...")
        self._pool.shutdown(wait=wait)
        logger.info(f"Uploader shutdown. Stats: {self.get_stats()}")

    # ------------------------------------------------------------------
    # Roboflow-specific logic
    # ------------------------------------------------------------------

    def _init_roboflow(self) -> None:
        """Initialize Roboflow connections for all configured workspaces."""
        try:
            from roboflow import Roboflow
            self._roboflow_cls = Roboflow

            self.upload_targets = self.config.get_all_targets()

            if not self.upload_targets:
                logger.warning("No Roboflow workspaces configured, uploads disabled")
                self.enabled = False
                return

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
            logger.info(f"Batch {batch_id} uploaded to all {len(self.upload_targets)} target(s)")
        else:
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

