# `src/cli/download.py` — `sam3-download` Entry Point

## Purpose

Downloads SAM3 model weights from Hugging Face Hub. Standalone — does not
require a config file. Can also check download status.

## Public API

```
sam3-download [--token HF_TOKEN] [--output-dir DIR] [--status] [--log-level LEVEL]
```

Token is read from `--token` or `HF_TOKEN` environment variable.

## Dependencies

- **Wires**: `HFModelDownloader` via `download_sam3_model(token, output_dir)`
- **Config**: None (standalone)

## Wiring

| Console script | `sam3-download = src.cli.download:main` |
|---|---|
| Config | Not required |
| Exit codes | 0 = success, 1 = no token or download failure |

## Phase 4 — Created by Agent B (23-02-2026)
