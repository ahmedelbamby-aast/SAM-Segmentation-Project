# Tests Documentation

This folder contains the test suite for the SAM 3 Segmentation Pipeline.

## Test Modules

| Test File | Tests For | Coverage |
|-----------|-----------|----------|
| `test_annotation_writer.py` | `AnnotationWriter` class | YOLO format, mask→polygon, data.yaml |
| `test_preprocessor.py` | `ImagePreprocessor` class | Validation, resizing, directory scan |
| `test_progress_tracker.py` | `ProgressTracker` class | SQLite jobs, checkpoints, resumption |
| `test_validator.py` | `Validator` class | Dataset comparison, caching, missing images |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_annotation_writer.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run single test class
pytest tests/test_validator.py::TestValidator -v

# Run single test method
pytest tests/test_validator.py::TestValidator::test_compare_datasets_no_missing -v
```

## Test Structure

All tests use:
- `pytest` as the test framework
- `tempfile.TemporaryDirectory` for isolated test directories
- Mock configuration objects to avoid filesystem dependencies

### Fixture Pattern
```python
@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def writer(temp_output_dir):
    """Create writer with mock config."""
    config = MockConfig()
    config.pipeline.output_dir = str(temp_output_dir)
    return AnnotationWriter(config)
```

### Mock Objects
- `MockConfig` - Minimal configuration for testing
- `MockSegmentationResult` - Fake SAM 3 output

## Key Test Cases

### AnnotationWriter
- `test_setup_directories` - Directory structure creation
- `test_mask_to_polygon` - Mask→polygon conversion
- `test_write_annotation` - YOLO format output
- `test_write_data_yaml` - Dataset configuration

### Preprocessor
- `test_validate_valid_image` - Image validation
- `test_resize_with_padding_*` - Letterbox resizing
- `test_scan_directory` - Image discovery

### ProgressTracker
- Job creation and retrieval
- Image status tracking (pending→processing→complete)
- Checkpoint creation
- Batch management

### Validator
- `test_compare_datasets_*` - Input/output comparison
- `test_cache_missing_images` - Missing image caching
- `test_mark_cached_processed` - Processing state updates
- `test_get_validation_jobs` - Job listing

## Adding New Tests

1. Create `test_<module_name>.py` in `tests/`
2. Add author header:
   ```python
   """
   Tests for the <module> module.
   
   Author: Ahmed Hany ElBamby
   Date: 06-02-2026
   """
   ```
3. Create `MockConfig` if needed
4. Use fixtures for setup/teardown
5. Follow naming: `test_<method>_<scenario>`
