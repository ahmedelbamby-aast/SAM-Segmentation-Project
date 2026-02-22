# Configuration Documentation

This folder contains the YAML configuration files for the pipeline.

## Main Configuration: `config.yaml`

### Pipeline Settings
```yaml
pipeline:
  input_dir: "./input"              # Source images
  output_dir: "./output/dataset"    # Generated annotations
  resolution: 640                   # Model input size (pixels)
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
  num_workers: 4                    # Parallel preprocessing threads
  input_mode: "single-folder"       # or "pre-split"
```

**Input Modes:**
- `single-folder` - All images in one folder, auto-split
- `pre-split` - Images already in train/valid/test folders

---

### Model Settings
```yaml
model:
  path: "./models/sam3.pt"          # Model weights path
  confidence: 0.25                  # Detection threshold
  prompts: ["teacher", "student"]   # Classes to detect
  half_precision: true              # FP16 for faster inference
  device: "auto"                    # auto/0/cpu/0,1 (multi-GPU)
```

**Device Options:**
- `auto` - Automatic GPU detection
- `0` - First GPU
- `cpu` - CPU only
- `0,1` - Multi-GPU (GPUs 0 and 1)

---

### Split Ratios
```yaml
split:
  train: 0.7
  valid: 0.2
  test: 0.1
  seed: 42                          # For reproducibility
```

---

### Post-Processing (NMS)
```yaml
post_process:
  enabled: true
  iou_threshold: 0.5                # IoU threshold for overlap
  strategy: "confidence"            # confidence/area/class_priority/soft_nms
  class_priority: ["teacher", "student"]
  soft_nms_sigma: 0.5               # Gaussian decay for soft NMS
  min_confidence_after_decay: 0.1   # Min confidence after soft NMS
```

**NMS Strategies:**
| Strategy | Description |
|----------|-------------|
| `confidence` | Keep highest confidence mask |
| `area` | Keep largest mask |
| `class_priority` | Use priority order (first class wins) |
| `soft_nms` | Gaussian confidence decay instead of removal |

---

### Progress Tracking
```yaml
progress:
  db_path: "./db/progress.db"       # SQLite database
  checkpoint_interval: 100          # Save every N images
  log_file: "./logs/pipeline.log"
  log_level: "INFO"                 # DEBUG/INFO/WARNING/ERROR
```

---

### Roboflow Upload
```yaml
roboflow:
  enabled: true
  workspaces:
    - workspace: "my-workspace"
      api_key: "your_api_key"       # Per-workspace key
      projects:
        - project: "teacher-detection"
          is_prediction: false
        - project: "predictions-project"
          is_prediction: true
  batch_upload_size: 50
  upload_workers: 3
  retry_attempts: 3
  retry_delay: 5
```

**Multi-Workspace Support:**
- Each workspace can have its own API key
- Multiple projects per workspace
- `is_prediction` flag for prediction projects

---

### Result Filtering
```yaml
filter:
  save_neither: true                # Save images with no detections
  neither_folder: "neither"
```

## Environment Variable Overrides

| Variable | Overrides |
|----------|-----------|
| `ROBOFLOW_API_KEY` | Default API key if not in config |
| `SAM3_MODEL_PATH` | Model path |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

## Example Minimal Config

```yaml
pipeline:
  input_dir: "./images"
  output_dir: "./output"

model:
  path: "./models/sam3.pt"
  prompts: ["teacher", "student"]

roboflow:
  enabled: false
```
