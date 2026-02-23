# download_model (scripts)

## Purpose

Legacy entry-point script that downloads the SAM 3 model weights.
Since Phase 6, this is a **thin wrapper** over `src.cli.download.main`.
All download logic lives in `src/model_downloader.py`.

## Usage

```bash
# Via script
python scripts/download_model.py --token hf_...

# Equivalent canonical CLI
sam3-download --token hf_...
```

## Implementation

```python
import sys
from src.cli.download import main

if __name__ == "__main__":
    sys.exit(main())
```

The script delegates entirely to `src.cli.download.main`, which parses CLI
arguments (`--token`, `--model-dir`, `--repo-id`) and calls
`src.model_downloader.ModelDownloader`.

## Design Decision

The thin-wrapper pattern was adopted in Phase 6 as part of the dead-code cleanup.
The original `scripts/download_model.py` had its own `setup_logging()` which
shadowed `src/utils.py`.  Replacing it with a 3-line delegate:
- Eliminates duplicate logging setup
- Ensures all CLI behaviour is tested via `src/cli/download.py`
- Keeps `scripts/` as a backward-compatible convenience layer only

## Wiring

- Delegates to: `src/cli/download.py` → `src/model_downloader.py`
- Registered entry point: `sam3-download` (in `setup.py`)
