# preprocessor

## Purpose

Image preprocessing: validates, resizes (with aspect-ratio-preserving letterbox),
and batch-scans input directories to produce the canonical image list for the
pipeline.

## Public API

### `ImagePreprocessor`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(config)` | Reads `config.pipeline.resolution`, `supported_formats`, `num_workers` |
| `validate_image` | `(image_path: Path) → bool` | Check extension, existance, and optional cv2 readability |
| `resize_image` | `(image: ndarray, target_size: int) → ndarray` | Letterbox resize maintaining aspect ratio |
| `preprocess_image` | `(image_path: Path) → Optional[ndarray]` | Load + validate + resize a single image |
| `scan_directory` | `(directory: Path) → List[Path]` | Recursively find all supported image files |
| `scan_output_directory` | `(output_dir: Path) → Dict[str, Set[Path]]` | Return `{split: set_of_paths}` for all split subdirs |
| `process_batch` | `(image_paths: List[Path]) → List[Tuple[Path, ndarray]]` | Parallel preprocess with `ThreadPoolExecutor` |

## Design

- `_fast_scan` flag (`self._fast_scan = True`) bypasses slow `cv2.imread` during
  large directory scans — images are fully validated at inference time.
- ISP non-compliant (currently accepts full `config`): will be refactored to accept
  only `config.pipeline` in a future phase.
- Parallel scanning uses `ThreadPoolExecutor(max_workers=num_workers)`.

## Dependencies

- Imports: `cv2`, `numpy`, `pathlib`, `concurrent.futures`
- Does NOT implement a Protocol from `src/interfaces.py` yet

## Data Flow

```mermaid
flowchart LR
    DIR[Input directory] -->|scan_directory| PATHS[List[Path]]
    PATHS -->|process_batch| RESIZED[List[Tuple[Path, ndarray]]]
    RESIZED -->|image_path, preprocessed_img| SEG[SAM3Segmentor]
```

## Usage Examples

```python
from src.preprocessor import ImagePreprocessor
from pathlib import Path

preprocessor = ImagePreprocessor(config)

# Scan
all_images = preprocessor.scan_directory(Path("data/train/images"))

# Single image
img = preprocessor.preprocess_image(Path("data/train/images/room01.jpg"))

# Check what's already been annotated
output_sets = preprocessor.scan_output_directory(Path("output"))
# output_sets == {"train": {Path("output/train/images/room01.jpg"), ...}, ...}
```

## Edge Cases

- `validate_image` returns `False` for zero-byte files or unsupported extensions
- `preprocess_image` returns `None` for invalid images; caller must handle `None`
- `scan_output_directory` returns empty sets for missing splits

## Wiring

- Created by: `src/cli/preprocess.py` and `src/cli/pipeline.py`
- Config source: `config.pipeline` (resolution, supported_formats, num_workers)
- Pipeline stage: `[Preprocess]`
