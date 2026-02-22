"""
Tests for the progress tracker module.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import pytest
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.progress_tracker import ProgressTracker, Status


@pytest.fixture
def tracker():
    """Create tracker with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        tracker = ProgressTracker(db_path)
        yield tracker
        tracker.close()


@pytest.fixture
def sample_images():
    """Sample image paths for testing."""
    return [
        Path("/path/to/image1.jpg"),
        Path("/path/to/image2.jpg"),
        Path("/path/to/image3.jpg"),
    ]


@pytest.fixture
def sample_splits():
    """Sample splits for testing."""
    return ["train", "train", "val"]


class TestProgressTracker:
    """Tests for ProgressTracker class."""
    
    def test_create_job(self, tracker, sample_images, sample_splits):
        """Test creating a new job."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        
        assert job_id > 0
        assert tracker.get_job_id("test_job") == job_id
    
    def test_create_duplicate_job(self, tracker, sample_images, sample_splits):
        """Test creating job with same name returns same ID."""
        job_id1 = tracker.create_job("test_job", sample_images, sample_splits)
        job_id2 = tracker.create_job("test_job", sample_images, sample_splits)
        
        assert job_id1 == job_id2
    
    def test_get_pending_images(self, tracker, sample_images, sample_splits):
        """Test getting pending images."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        
        pending = tracker.get_pending_images(job_id)
        
        assert len(pending) == 3
        # Each item is (id, path, split)
        paths = [p[1] for p in pending]
        assert all(p in sample_images for p in paths)
    
    def test_mark_completed(self, tracker, sample_images, sample_splits):
        """Test marking image as completed."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        pending = tracker.get_pending_images(job_id)
        
        # Mark first image as completed
        image_id = pending[0][0]
        tracker.mark_completed(image_id)
        tracker.checkpoint(job_id)
        
        # Should have one less pending
        new_pending = tracker.get_pending_images(job_id)
        assert len(new_pending) == 2
        
        # Progress should update
        progress = tracker.get_progress(job_id)
        assert progress['processed_count'] == 1
    
    def test_mark_error(self, tracker, sample_images, sample_splits):
        """Test marking image as error."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        pending = tracker.get_pending_images(job_id)
        
        image_id = pending[0][0]
        tracker.mark_error(image_id, "Test error message")
        tracker.checkpoint(job_id)
        
        progress = tracker.get_progress(job_id)
        assert progress['error_count'] == 1
    
    def test_get_progress(self, tracker, sample_images, sample_splits):
        """Test getting progress statistics."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        
        progress = tracker.get_progress(job_id)
        
        assert progress['total_images'] == 3
        assert progress['processed_count'] == 0
        assert progress['error_count'] == 0
        assert progress['pending_count'] == 3
    
    def test_create_batch(self, tracker, sample_images, sample_splits):
        """Test creating upload batch."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        
        batch_id = tracker.create_batch(job_id, 1, 100)
        
        assert batch_id > 0
    
    def test_mark_batch_uploaded(self, tracker, sample_images, sample_splits):
        """Test marking batch as uploaded."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        batch_id = tracker.create_batch(job_id, 1, 100)
        
        tracker.mark_batch_uploaded(batch_id)
        
        pending = tracker.get_pending_batches(job_id)
        uploaded = tracker.get_uploaded_batches(job_id)
        
        assert len(pending) == 0
        assert len(uploaded) == 1
    
    def test_reset_processing_images(self, tracker, sample_images, sample_splits):
        """Test resetting stuck processing images."""
        job_id = tracker.create_job("test_job", sample_images, sample_splits)
        pending = tracker.get_pending_images(job_id)
        
        # Mark as processing
        image_ids = [p[0] for p in pending]
        tracker.mark_processing(image_ids)
        
        # No pending now
        assert len(tracker.get_pending_images(job_id)) == 0
        
        # Reset stuck images
        tracker.reset_processing_images(job_id)
        
        # Should be pending again
        assert len(tracker.get_pending_images(job_id)) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
