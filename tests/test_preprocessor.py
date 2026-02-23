"""
Tests for the preprocessor module.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
import cv2
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import ImagePreprocessor


class MockConfig:
    """Mock configuration for testing."""
    class Pipeline:
        resolution = 640
        supported_formats = ['.jpg', '.jpeg', '.png']
        num_workers = 2
    
    pipeline = Pipeline()


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return ImagePreprocessor(MockConfig().pipeline)


@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create valid images
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmpdir / f"image_{i}.jpg"), img)
        
        # Create invalid file (not an image)
        (tmpdir / "not_image.txt").write_text("not an image")
        
        yield tmpdir


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""
    
    def test_validate_valid_image(self, preprocessor, temp_image_dir):
        """Test validation of valid image."""
        image_path = temp_image_dir / "image_0.jpg"
        assert preprocessor.validate_image(image_path) is True
    
    def test_validate_invalid_extension(self, preprocessor, temp_image_dir):
        """Test validation rejects invalid extension."""
        text_file = temp_image_dir / "not_image.txt"
        assert preprocessor.validate_image(text_file) is False
    
    def test_validate_nonexistent_file(self, preprocessor):
        """Test validation of non-existent file."""
        assert preprocessor.validate_image(Path("/nonexistent/image.jpg")) is False
    
    def test_resize_with_padding_square(self, preprocessor):
        """Test resizing square image."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized, info = preprocessor.resize_with_padding(img)
        
        assert resized.shape == (640, 640, 3)
        assert info['original_size'] == (100, 100)
        assert info['scale'] == 6.4
    
    def test_resize_with_padding_landscape(self, preprocessor):
        """Test resizing landscape image."""
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        resized, info = preprocessor.resize_with_padding(img)
        
        assert resized.shape == (640, 640, 3)
        assert info['original_size'] == (200, 100)
    
    def test_resize_with_padding_portrait(self, preprocessor):
        """Test resizing portrait image."""
        img = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        resized, info = preprocessor.resize_with_padding(img)
        
        assert resized.shape == (640, 640, 3)
        assert info['original_size'] == (100, 200)
    
    def test_scan_directory(self, preprocessor, temp_image_dir):
        """Test scanning directory for images."""
        images = preprocessor.scan_directory(temp_image_dir)
        
        assert len(images) == 3
        for img in images:
            assert img.suffix.lower() in ['.jpg', '.jpeg', '.png']
    
    def test_scan_empty_directory(self, preprocessor):
        """Test scanning empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = preprocessor.scan_directory(Path(tmpdir))
            assert len(images) == 0
    
    def test_scan_nonexistent_directory(self, preprocessor):
        """Test scanning non-existent directory raises error."""
        with pytest.raises(FileNotFoundError):
            preprocessor.scan_directory(Path("/nonexistent/dir"))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
