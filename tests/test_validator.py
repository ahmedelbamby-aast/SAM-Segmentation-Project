"""
Tests for the validator module.

Author: Ahmed Hany ElBamby
Date: 06-02-2026
"""
import pytest
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validator import Validator, ValidationResult
from src.config_manager import (
    Config, PipelineConfig, ModelConfig, SplitConfig, 
    ProgressConfig, RoboflowConfig
)


@pytest.fixture
def temp_dirs():
    """Create temporary input/output directories with test structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create input structure (pre-split mode)
        input_dir = tmpdir / "input"
        for split in ["train", "valid", "test"]:
            (input_dir / split).mkdir(parents=True)
        
        # Create output structure
        output_dir = tmpdir / "output"
        for split in ["train", "valid", "test"]:
            (output_dir / split / "images").mkdir(parents=True)
            (output_dir / split / "labels").mkdir(parents=True)
        
        # Database path
        db_path = tmpdir / "db" / "test.db"
        
        yield {
            "tmpdir": tmpdir,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "db_path": db_path
        }


@pytest.fixture
def sample_config(temp_dirs):
    """Create sample configuration for testing."""
    return Config(
        pipeline=PipelineConfig(
            input_dir=temp_dirs["input_dir"],
            output_dir=temp_dirs["output_dir"],
            resolution=640,
            supported_formats=[".jpg", ".jpeg", ".png"],
            num_workers=1,
            input_mode="pre-split"
        ),
        model=ModelConfig(
            path=Path("./models/sam3.pt"),
            confidence=0.25,
            prompts=["teacher", "student"],
            half_precision=True,
            device="cpu"
        ),
        split=SplitConfig(
            train=0.7,
            valid=0.2,
            test=0.1,
            seed=42
        ),
        progress=ProgressConfig(
            db_path=temp_dirs["db_path"],
            checkpoint_interval=100,
            log_file=Path("./logs/test.log"),
            log_level="WARNING"
        ),
        roboflow=RoboflowConfig(
            enabled=False,
            api_key="test",
            workspace="test",
            project="test",
            batch_upload_size=100,
            upload_workers=1,
            retry_attempts=1,
            retry_delay=1
        )
    )


def create_test_image(path: Path):
    """Create a dummy image file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\x00' * 100)  # Minimal dummy file


def create_test_annotation(labels_dir: Path, image_stem: str):
    """Create a dummy annotation file."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{image_stem}.txt").write_text("0 0.5 0.5 0.1 0.1")


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_missing_count(self):
        """Test missing count property."""
        result = ValidationResult(
            input_count=10,
            output_count=7,
            missing_images=[Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        )
        assert result.missing_count == 3
    
    def test_is_complete_true(self):
        """Test is_complete when no missing images."""
        result = ValidationResult(
            input_count=10,
            output_count=10,
            missing_images=[]
        )
        assert result.is_complete is True
    
    def test_is_complete_false(self):
        """Test is_complete when missing images exist."""
        result = ValidationResult(
            input_count=10,
            output_count=7,
            missing_images=[Path("a.jpg")]
        )
        assert result.is_complete is False
    
    def test_summary(self):
        """Test summary generation."""
        result = ValidationResult(
            input_count=10,
            output_count=7,
            missing_images=[Path("a.jpg"), Path("b.jpg")],
            missing_by_split={"train": [Path("a.jpg")], "valid": [Path("b.jpg")]}
        )
        summary = result.summary()
        assert "Input images:  10" in summary
        assert "Output images: 7" in summary
        assert "Missing:       2" in summary
        assert "train: 1" in summary
        assert "valid: 1" in summary


class TestValidator:
    """Tests for Validator class."""
    
    def test_scan_input_directory_empty(self, sample_config, temp_dirs):
        """Test scanning empty input directory."""
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.scan_input_directory()
            assert "train" in result
            assert "valid" in result
            assert "test" in result
            assert all(len(v) == 0 for v in result.values())
        finally:
            validator.close()
    
    def test_scan_input_directory_with_images(self, sample_config, temp_dirs):
        """Test scanning input directory with images."""
        # Create test images
        create_test_image(temp_dirs["input_dir"] / "train" / "img1.jpg")
        create_test_image(temp_dirs["input_dir"] / "train" / "img2.png")
        create_test_image(temp_dirs["input_dir"] / "valid" / "img3.jpg")
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.scan_input_directory()
            assert len(result["train"]) == 2
            assert len(result["valid"]) == 1
            assert len(result["test"]) == 0
        finally:
            validator.close()
    
    def test_scan_output_directory_empty(self, sample_config, temp_dirs):
        """Test scanning empty output directory."""
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.scan_output_directory()
            assert all(len(v) == 0 for v in result.values())
        finally:
            validator.close()
    
    def test_scan_output_directory_with_annotations(self, sample_config, temp_dirs):
        """Test scanning output directory with annotations."""
        # Create test annotations
        create_test_annotation(temp_dirs["output_dir"] / "train" / "labels", "img1")
        create_test_annotation(temp_dirs["output_dir"] / "train" / "labels", "img2")
        create_test_annotation(temp_dirs["output_dir"] / "valid" / "labels", "img3")
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.scan_output_directory()
            assert len(result["train"]) == 2
            assert len(result["valid"]) == 1
            assert len(result["test"]) == 0
        finally:
            validator.close()
    
    def test_compare_datasets_no_missing(self, sample_config, temp_dirs):
        """Test comparison when all images are processed."""
        # Create matching input/output
        create_test_image(temp_dirs["input_dir"] / "train" / "img1.jpg")
        create_test_annotation(temp_dirs["output_dir"] / "train" / "labels", "img1")
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.compare_datasets()
            assert result.input_count == 1
            assert result.output_count == 1
            assert result.missing_count == 0
            assert result.is_complete
        finally:
            validator.close()
    
    def test_compare_datasets_with_missing(self, sample_config, temp_dirs):
        """Test comparison when some images are missing."""
        # Create input images
        create_test_image(temp_dirs["input_dir"] / "train" / "img1.jpg")
        create_test_image(temp_dirs["input_dir"] / "train" / "img2.jpg")
        create_test_image(temp_dirs["input_dir"] / "valid" / "img3.jpg")
        
        # Only create annotation for img1
        create_test_annotation(temp_dirs["output_dir"] / "train" / "labels", "img1")
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.compare_datasets()
            assert result.input_count == 3
            assert result.output_count == 1
            assert result.missing_count == 2
            assert not result.is_complete
            
            # Check missing by split
            assert "train" in result.missing_by_split
            assert "valid" in result.missing_by_split
            assert len(result.missing_by_split["train"]) == 1  # img2
            assert len(result.missing_by_split["valid"]) == 1  # img3
        finally:
            validator.close()
    
    def test_cache_missing_images(self, sample_config, temp_dirs):
        """Test caching missing images."""
        create_test_image(temp_dirs["input_dir"] / "train" / "img1.jpg")
        create_test_image(temp_dirs["input_dir"] / "train" / "img2.jpg")
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.compare_datasets()
            cached = validator.cache_missing_images(result, "test_job")
            
            assert cached == 2
            
            # Verify we can retrieve them
            retrieved = validator.get_cached_missing_images("test_job")
            assert len(retrieved) == 2
        finally:
            validator.close()
    
    def test_mark_cached_processed(self, sample_config, temp_dirs):
        """Test marking cached images as processed."""
        create_test_image(temp_dirs["input_dir"] / "train" / "img1.jpg")
        img_path = temp_dirs["input_dir"] / "train" / "img1.jpg"
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.compare_datasets()
            validator.cache_missing_images(result, "test_job")
            
            # Mark as processed
            validator.mark_cached_processed("test_job", [img_path])
            
            # Should no longer appear in unprocessed
            unprocessed = validator.get_cached_missing_images("test_job", unprocessed_only=True)
            assert len(unprocessed) == 0
            
            # But should appear when including all
            all_cached = validator.get_cached_missing_images("test_job", unprocessed_only=False)
            assert len(all_cached) == 1
        finally:
            validator.close()
    
    def test_clear_validation_cache(self, sample_config, temp_dirs):
        """Test clearing validation cache."""
        create_test_image(temp_dirs["input_dir"] / "train" / "img1.jpg")
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.compare_datasets()
            validator.cache_missing_images(result, "test_job")
            
            # Clear cache
            deleted = validator.clear_validation_cache("test_job")
            assert deleted == 1
            
            # Should be empty now
            cached = validator.get_cached_missing_images("test_job")
            assert len(cached) == 0
        finally:
            validator.close()
    
    def test_get_validation_jobs(self, sample_config, temp_dirs):
        """Test getting validation job summaries."""
        create_test_image(temp_dirs["input_dir"] / "train" / "img1.jpg")
        create_test_image(temp_dirs["input_dir"] / "train" / "img2.jpg")
        
        validator = Validator(sample_config, db_path=temp_dirs["db_path"])
        try:
            result = validator.compare_datasets()
            validator.cache_missing_images(result, "job1")
            validator.cache_missing_images(result, "job2")
            
            jobs = validator.get_validation_jobs()
            assert len(jobs) == 2
            
            job_names = [j["job_name"] for j in jobs]
            assert "job1" in job_names
            assert "job2" in job_names
        finally:
            validator.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
