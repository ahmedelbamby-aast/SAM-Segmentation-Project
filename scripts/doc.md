# Scripts Documentation

This folder contains command-line interface (CLI) scripts for running the pipeline.

## Available Scripts

### `run_pipeline.py`
Main entry point for the segmentation pipeline.

**Usage:**
```bash
# Start a new job
python scripts/run_pipeline.py --job-name batch_001

# Resume interrupted job
python scripts/run_pipeline.py --job-name batch_001 --resume

# Check job status
python scripts/run_pipeline.py --job-name batch_001 --status

# Reset stuck images
python scripts/run_pipeline.py --job-name batch_001 --reset-stuck

# Force full dataset rescan
python scripts/run_pipeline.py --job-name batch_001 --force-scan

# Retry failed uploads
python scripts/run_pipeline.py --job-name batch_001 --retry-uploads
```

**Key Options:**
| Flag | Description |
|------|-------------|
| `--config` | Path to config file (default: `config/config.yaml`) |
| `--job-name` | Unique identifier for the processing job |
| `--resume` | Continue from last checkpoint |
| `--status` | Show job progress and exit |
| `--reset-stuck` | Reset images stuck in 'processing' state |
| `--reset-errors` | Reset error images for retry |
| `--force-scan` | Ignore cache, rescan entire dataset |
| `--log-level` | Override log level (DEBUG/INFO/WARNING/ERROR) |

---

### `run_validator.py`
Dataset validation and missing image caching.

**Usage:**
```bash
# Validate input vs output
python scripts/run_validator.py --validate

# Validate and cache missing images
python scripts/run_validator.py --validate --cache --job-name missing_001

# Show cached images
python scripts/run_validator.py --show-cached
python scripts/run_validator.py --show-cached --job-name missing_001 --verbose

# Clear cache for a job
python scripts/run_validator.py --clear-cache --job-name missing_001
```

---

### `download_model.py`
Download SAM 3 model from Hugging Face.

**Usage:**
```bash
# Download with environment variable
export HF_TOKEN=hf_xxxxxxxxxxxx
python scripts/download_model.py

# Download with token argument
python scripts/download_model.py --token hf_xxxxxxxxxxxx

# Check download status
python scripts/download_model.py --status

# List repository files
python scripts/download_model.py --list --token hf_xxxxxxxxxxxx
```

**Required:** Hugging Face token (`HF_TOKEN` env var or `--token`)

---

### `add_class_files.py`
Add class metadata files for Roboflow compatibility.

**Usage:**
```bash
# Add to default output directory
python scripts/add_class_files.py

# Add to custom directory
python scripts/add_class_files.py /path/to/output/dataset
```

Creates:
- `_classes.txt` in each split folder
- `classes.txt` at root
- Updates `data.yaml` with proper class format

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face authentication token |
| `ROBOFLOW_API_KEY` | Default Roboflow API key |
| `CUDA_VISIBLE_DEVICES` | Control GPU visibility |
