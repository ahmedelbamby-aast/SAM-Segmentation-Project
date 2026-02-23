"""
SQLite-based progress tracking for crash recovery and resume capability.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import threading

from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


class Status(Enum):
    """Status values for images and batches."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    UPLOADED = "uploaded"


class ProgressTracker:
    """
    SQLite-based progress tracking for crash recovery and resume capability.
    
    Tracks processing jobs, individual image status, and batch uploads.
    Thread-safe with per-thread connections.
    """
    
    SCHEMA = """
    -- Jobs table: tracks overall processing jobs
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        status TEXT DEFAULT 'running',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_images INTEGER DEFAULT 0,
        processed_count INTEGER DEFAULT 0,
        error_count INTEGER DEFAULT 0
    );
    
    -- Images table: tracks status of each image
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        path TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        split TEXT,
        processed_at TIMESTAMP,
        error_message TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs(id),
        UNIQUE(job_id, path)
    );
    
    -- Batches table: tracks upload batches
    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        batch_num INTEGER NOT NULL,
        status TEXT DEFAULT 'pending',
        image_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        uploaded_at TIMESTAMP,
        upload_error TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs(id)
    );
    
    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
    CREATE INDEX IF NOT EXISTS idx_images_job ON images(job_id);
    CREATE INDEX IF NOT EXISTS idx_images_job_status ON images(job_id, status);
    CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
    CREATE INDEX IF NOT EXISTS idx_batches_job ON batches(job_id);
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize progress tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        _logger.info("Initialized progress tracker at %s", self.db_path)
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=300.0,  # 5 minute timeout for large datasets
                check_same_thread=False,
                isolation_level='DEFERRED'
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._local.conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn.execute("PRAGMA busy_timeout=300000")  # 5 min busy timeout
        return self._local.conn
    
    def _execute_with_retry(self, operation, *args, max_retries=5, **kwargs):
        """Execute database operation with retry logic."""
        import time
        
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
                    _logger.warning(
                        "Database locked, retrying in %.1fs (attempt %d/%d)",
                        wait_time, attempt + 1, max_retries,
                    )
                    time.sleep(wait_time)
                    # Try to close and reopen connection
                    if hasattr(self._local, 'conn') and self._local.conn:
                        try:
                            self._local.conn.close()
                        except Exception:
                            pass
                        self._local.conn = None
                else:
                    raise
        raise sqlite3.OperationalError("Database still locked after maximum retries")
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        def _do_init():
            self.conn.executescript(self.SCHEMA)
            self.conn.commit()
        self._execute_with_retry(_do_init)
    
    @trace
    def create_job(self, name: str, image_paths: List[Path], splits: List[str]) -> int:
        """
        Create a new processing job with all images.
        
        Args:
            name: Unique job name
            image_paths: List of image file paths
            splits: List of split assignments ('train', 'valid', 'test')
            
        Returns:
            Job ID
        """
        def _create_job_internal():
            cursor = self.conn.cursor()
            
            # Create or get existing job
            cursor.execute(
                "INSERT OR IGNORE INTO jobs (name, total_images) VALUES (?, ?)",
                (name, len(image_paths))
            )
            
            job_id = cursor.execute(
                "SELECT id FROM jobs WHERE name = ?", (name,)
            ).fetchone()[0]
            
            # Update total if job already exists
            cursor.execute(
                "UPDATE jobs SET total_images = ?, updated_at = ? WHERE id = ?",
                (len(image_paths), datetime.now(), job_id)
            )
            self.conn.commit()
            return job_id
        
        # Create job with retry
        job_id = self._execute_with_retry(_create_job_internal)
        
        # Batch insert images for large datasets
        total_images = len(image_paths)
        batch_size = 5000  # Insert 5k at a time for better reliability
        
        _logger.info("Inserting %d images into database...", total_images)
        
        for i in range(0, total_images, batch_size):
            end_idx = min(i + batch_size, total_images)
            batch_data = [
                (job_id, str(p), s) 
                for p, s in zip(image_paths[i:end_idx], splits[i:end_idx])
            ]
            
            def _insert_batch(data=batch_data):
                cursor = self.conn.cursor()
                cursor.executemany(
                    "INSERT OR IGNORE INTO images (job_id, path, split) VALUES (?, ?, ?)",
                    data
                )
                self.conn.commit()
            
            self._execute_with_retry(_insert_batch)
            
            # Log progress for large datasets
            if total_images > 50000:
                progress = (end_idx / total_images) * 100
                _logger.info(
                    "Database insert progress: %d/%d (%.1f%%)",
                    end_idx, total_images, progress,
                )
        
        _logger.info("Created job '%s' with %d images", name, total_images)
        return job_id
    
    @trace
    def get_job_id(self, name: str) -> Optional[int]:
        """Get job ID by name."""
        row = self.conn.execute(
            "SELECT id FROM jobs WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None
    
    @trace
    def get_pending_images(self, job_id: int, limit: int = 100) -> List[Tuple[int, Path, str]]:
        """
        Get unprocessed images for a job.
        
        Args:
            job_id: Job ID
            limit: Maximum number of images to return
            
        Returns:
            List of (image_id, path, split) tuples
        """
        cursor = self.conn.execute(
            """SELECT id, path, split FROM images 
               WHERE job_id = ? AND status = ? 
               LIMIT ?""",
            (job_id, Status.PENDING.value, limit)
        )
        return [(row['id'], Path(row['path']), row['split']) for row in cursor.fetchall()]
    
    @trace
    def get_image_split(self, image_id: int) -> Optional[str]:
        """Get the split assignment for an image."""
        row = self.conn.execute(
            "SELECT split FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return row['split'] if row else None
    
    @trace
    def mark_processing(self, image_ids: List[int]) -> None:
        """Mark images as currently being processed."""
        self.conn.executemany(
            "UPDATE images SET status = ? WHERE id = ?",
            [(Status.PROCESSING.value, i) for i in image_ids]
        )
        self.conn.commit()
    
    @trace
    def reset_stuck_images(self, job_id: int) -> int:
        """
        Reset images stuck in 'processing' state back to 'pending'.
        
        Args:
            job_id: Job ID to reset
            
        Returns:
            Number of images reset
        """
        cursor = self.conn.execute(
            "UPDATE images SET status = ? WHERE job_id = ? AND status = ?",
            (Status.PENDING.value, job_id, Status.PROCESSING.value)
        )
        count = cursor.rowcount
        self.conn.commit()
        if count > 0:
            _logger.info("Reset %d stuck images to pending", count)
        return count

    @trace
    def reset_error_images(self, job_id: int) -> int:
        """
        Reset images with 'error' status back to 'pending' for retry.

        Args:
            job_id: Job ID to reset

        Returns:
            Number of images reset
        """
        cursor = self.conn.execute(
            "UPDATE images SET status = ? WHERE job_id = ? AND status = ?",
            (Status.PENDING.value, job_id, Status.ERROR.value)
        )
        count = cursor.rowcount
        self.conn.commit()
        if count > 0:
            _logger.info("Reset %d error images to pending for retry", count)
        return count
    
    @trace
    def mark_completed(self, image_id: int) -> None:
        """Mark image as successfully processed."""
        self.conn.execute(
            "UPDATE images SET status = ?, processed_at = ? WHERE id = ?",
            (Status.COMPLETED.value, datetime.now(), image_id)
        )
    
    @trace
    def mark_error(self, image_id: int, error_msg: str) -> None:
        """Mark image as failed with error message."""
        self.conn.execute(
            "UPDATE images SET status = ?, error_message = ?, processed_at = ? WHERE id = ?",
            (Status.ERROR.value, error_msg[:500], datetime.now(), image_id)  # Truncate long errors
        )
    
    @trace
    def checkpoint(self, job_id: int) -> None:
        """Commit current progress and update job stats."""
        self.conn.execute("""
            UPDATE jobs SET 
                processed_count = (SELECT COUNT(*) FROM images WHERE job_id = ? AND status = ?),
                error_count = (SELECT COUNT(*) FROM images WHERE job_id = ? AND status = ?),
                updated_at = ?
            WHERE id = ?
        """, (job_id, Status.COMPLETED.value, job_id, Status.ERROR.value, datetime.now(), job_id))
        self.conn.commit()
        _logger.debug("Checkpoint saved for job %d", job_id)
    
    @trace
    def get_progress(self, job_id: int) -> Dict[str, Any]:
        """
        Get current progress statistics.
        
        Returns:
            Dictionary with total_images, processed_count, error_count, pending_count
        """
        row = self.conn.execute(
            "SELECT total_images, processed_count, error_count FROM jobs WHERE id = ?",
            (job_id,)
        ).fetchone()
        
        if not row:
            return {}
        
        result = dict(row)
        result['pending_count'] = result['total_images'] - result['processed_count'] - result['error_count']
        return result
    
    @trace
    def get_progress_by_split(self, job_id: int) -> Dict[str, Dict[str, int]]:
        """Get progress broken down by split."""
        cursor = self.conn.execute("""
            SELECT split, status, COUNT(*) as count 
            FROM images WHERE job_id = ? 
            GROUP BY split, status
        """, (job_id,))
        
        result = {'train': {}, 'valid': {}, 'test': {}}
        for row in cursor.fetchall():
            if row['split'] in result:
                result[row['split']][row['status']] = row['count']
        
        return result
    
    @trace
    def create_batch(self, job_id: int, batch_num: int, image_count: int) -> int:
        """
        Create a new upload batch.
        
        Args:
            job_id: Job ID
            batch_num: Sequential batch number
            image_count: Number of images in batch
            
        Returns:
            Batch ID
        """
        cursor = self.conn.execute(
            "INSERT INTO batches (job_id, batch_num, image_count) VALUES (?, ?, ?)",
            (job_id, batch_num, image_count)
        )
        self.conn.commit()
        _logger.info("Created batch %d with %d images", batch_num, image_count)
        return cursor.lastrowid
    
    @trace
    def mark_batch_uploaded(self, batch_id: int) -> None:
        """Mark batch as successfully uploaded to Roboflow."""
        self.conn.execute(
            "UPDATE batches SET status = ?, uploaded_at = ? WHERE id = ?",
            (Status.UPLOADED.value, datetime.now(), batch_id)
        )
        self.conn.commit()
        _logger.info("Batch %d marked as uploaded", batch_id)
    
    @trace
    def mark_batch_error(self, batch_id: int, error_msg: str) -> None:
        """Mark batch upload as failed."""
        self.conn.execute(
            "UPDATE batches SET status = ?, upload_error = ? WHERE id = ?",
            (Status.ERROR.value, error_msg[:500], batch_id)
        )
        self.conn.commit()
    
    @trace
    def get_pending_batches(self, job_id: int) -> List[Dict[str, Any]]:
        """Get batches that haven't been uploaded yet."""
        cursor = self.conn.execute(
            "SELECT * FROM batches WHERE job_id = ? AND status != ?",
            (job_id, Status.UPLOADED.value)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    @trace
    def get_uploaded_batches(self, job_id: int) -> List[Dict[str, Any]]:
        """Get successfully uploaded batches."""
        cursor = self.conn.execute(
            "SELECT * FROM batches WHERE job_id = ? AND status = ?",
            (job_id, Status.UPLOADED.value)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    @trace
    def reset_processing_images(self, job_id: int) -> None:
        """Reset images stuck in 'processing' state back to 'pending'."""
        cursor = self.conn.execute(
            "UPDATE images SET status = ? WHERE job_id = ? AND status = ?",
            (Status.PENDING.value, job_id, Status.PROCESSING.value)
        )
        self.conn.commit()
        if cursor.rowcount > 0:
            _logger.info("Reset %d stuck images to pending", cursor.rowcount)
    
    @trace
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Stats pattern
    # ------------------------------------------------------------------

    @trace
    def get_stats(self) -> Dict[str, Any]:
        """Return progress-tracking statistics derived from the database.

        Returns:
            Dict with ``total_jobs``, ``total_images``, and
            ``total_batches``.
        """
        row_jobs = self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        row_imgs = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()
        row_bat = self.conn.execute("SELECT COUNT(*) FROM batches").fetchone()
        return {
            "total_jobs": row_jobs[0] if row_jobs else 0,
            "total_images": row_imgs[0] if row_imgs else 0,
            "total_batches": row_bat[0] if row_bat else 0,
            "db_path": str(self.db_path),
        }

    @trace
    def reset_stats(self) -> None:
        """No-op — stats are derived from the SQLite database."""
