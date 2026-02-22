"""
Tests for the annotation writer module.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
import yaml
import cv2
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotation_writer import AnnotationWriter


class MockPipelineConfig:
    """Mock pipeline configuration for testing."""
    output_dir = None  # Set in fixture


class MockClassRegistry:
    """Minimal mock registry providing class_names."""
    class_names = ["teacher", "student"]


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def writer(temp_output_dir):
    """Create annotation writer instance."""
    pipeline_cfg = MockPipelineConfig()
    pipeline_cfg.output_dir = str(temp_output_dir)
    return AnnotationWriter(pipeline_cfg, MockClassRegistry())


@pytest.fixture
def sample_mask():
    """Create sample binary mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Draw a rectangle
    mask[20:80, 20:80] = 1
    return mask


class MockSegmentationResult:
    """Mock segmentation result for testing."""
    def __init__(self, masks, class_ids, confidences):
        self.masks = masks
        self.class_ids = class_ids
        self.confidences = confidences
    
    @property
    def num_detections(self):
        return len(self.class_ids)


class TestAnnotationWriter:
    """Tests for AnnotationWriter class."""
    
    def test_setup_directories(self, writer, temp_output_dir):
        """Test directory structure is created."""
        for split in ['train', 'valid', 'test']:
            assert (temp_output_dir / split / 'images').exists()
            assert (temp_output_dir / split / 'labels').exists()
    
    def test_mask_to_polygon(self, writer, sample_mask):
        """Test converting mask to polygon."""
        polygon = writer.mask_to_polygon(sample_mask)
        
        assert len(polygon) >= 6  # At least 3 points
        assert all(0 <= p <= 1 for p in polygon)  # Normalized
    
    def test_mask_to_polygon_empty(self, writer):
        """Test empty mask returns empty polygon."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        polygon = writer.mask_to_polygon(empty_mask)
        
        assert len(polygon) == 0
    
    def test_write_annotation(self, writer, temp_output_dir, sample_mask):
        """Test writing annotation file."""
        # Create temp image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            image_path = Path(f.name)
        
        try:
            # Create mock result
            result = MockSegmentationResult(
                masks=np.array([sample_mask]),
                class_ids=[0],
                confidences=[0.9]
            )
            
            # Write annotation
            label_path = writer.write_annotation(image_path, result, 'train')
            
            assert label_path is not None
            assert label_path.exists()
            
            # Check content
            content = label_path.read_text()
            lines = content.strip().split('\n')
            assert len(lines) == 1
            
            parts = lines[0].split()
            assert parts[0] == '0'  # class_id
            assert len(parts) >= 7  # class_id + at least 3 points
        finally:
            image_path.unlink()
    
    def test_write_annotation_no_result(self, writer):
        """Test writing annotation with None result."""
        label_path = writer.write_annotation(Path("/test/image.jpg"), None, 'train')
        assert label_path is None
    
    def test_write_data_yaml(self, writer, temp_output_dir):
        """Test generating data.yaml."""
        yaml_path = writer.write_data_yaml()
        
        assert yaml_path.exists()
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert data['nc'] == 2
        assert data['names'] == {0: 'teacher', 1: 'student'}
        assert 'train' in data
        assert 'val' in data
        assert 'test' in data
    
    def test_get_stats(self, writer, temp_output_dir):
        """Test getting annotation statistics."""
        stats = writer.get_stats()
        
        assert 'train' in stats
        assert 'valid' in stats
        assert 'test' in stats
        
        for split_stats in stats.values():
            assert 'images' in split_stats
            assert 'labels' in split_stats
            assert 'annotations' in split_stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
