# Source Code Documentation

This folder contains the core Python modules for the SAM 3 Segmentation Pipeline.

## Module Overview

### Core Pipeline
| Module | Description |
|--------|-------------|
| `pipeline.py` | Main orchestrator that coordinates all pipeline stages |
| `config_manager.py` | Configuration dataclasses and YAML loading utilities |
| `utils.py` | Common utilities (logging, formatting, helpers) |

### Processing Modules
| Module | Description |
|--------|-------------|
| `preprocessor.py` | Image validation, resizing with letterboxing, directory scanning |
| `sam3_segmentor.py` | SAM 3 model wrapper with GPU/multi-GPU support |
| `post_processor.py` | Non-Maximum Suppression (NMS) for overlapping masks |
| `annotation_writer.py` | YOLO-format annotation file generation |
| `result_filter.py` | Categorizes images based on detections (neither folder) |

### Infrastructure
| Module | Description |
|--------|-------------|
| `progress_tracker.py` | SQLite-based progress tracking for resumable jobs |
| `roboflow_uploader.py` | Async multi-workspace upload to Roboflow |
| `parallel_processor.py` | Multi-process inference for parallel batches |
| `dataset_cache.py` | Dataset fingerprinting to skip unchanged rescans |
| `validator.py` | Input/output comparison and missing image caching |
| `model_downloader.py` | Hugging Face model download with auth support |

## Key Classes

### `SegmentationPipeline`
Main entry point in `pipeline.py`:
```python
from src.pipeline import SegmentationPipeline
from src.config_manager import load_config

config = load_config("config/config.yaml")
pipeline = SegmentationPipeline(config)
results = pipeline.run("my_job_name")
```

### `SAM3Segmentor`
Model wrapper with automatic device management:
```python
from src.sam3_segmentor import SAM3Segmentor

segmentor = SAM3Segmentor(config)
result = segmentor.process_image(Path("image.jpg"))
# result.masks, result.class_ids, result.confidences
```

### `MaskPostProcessor`
NMS strategies for resolving overlapping masks:
- `confidence` - Keep highest confidence mask
- `area` - Keep largest mask
- `class_priority` - Use class hierarchy (e.g., teacher > student)
- `soft_nms` - Gaussian decay instead of binary suppression

### `DistributedUploader`
Multi-workspace Roboflow uploads:
```python
from src.roboflow_uploader import DistributedUploader

uploader = DistributedUploader(config, tracker)
uploader.queue_batch(batch_dir, batch_id, split="train")
uploader.wait_for_uploads()
```

## Dependencies

All modules depend on:
- `pathlib.Path` for path handling
- `logging` for consistent logging
- `dataclasses` for configuration structures

Key external dependencies:
- `ultralytics` - SAM 3 model (`SAM3SemanticPredictor`)
- `torch` - GPU/CUDA management
- `numpy` / `opencv-python` - Image processing
- `roboflow` - Dataset upload API
- `tqdm` - Progress bars
