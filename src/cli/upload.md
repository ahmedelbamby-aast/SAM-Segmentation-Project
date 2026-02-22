# `src/cli/upload.py` — `sam3-upload` Entry Point

## Purpose

Uploads pending annotation batches to Roboflow via `DistributedUploader.queue_batch()`.
Batches are retrieved from the SQLite progress database and queued for async upload.

## Public API

```
sam3-upload --job-name NAME [--config PATH] [--output-dir DIR]
            [--dry-run] [--log-level LEVEL]
```

`--dry-run` lists pending batches without uploading.

## Dependencies

- **Wires**: `DistributedUploader`, `ProgressTracker`
- **Config slices**: `config.roboflow`, `config.progress`
- **Key methods**: `uploader.queue_batch(batch_dir, batch_id, split)`, `uploader.shutdown(wait=True)`

## Wiring

| Console script | `sam3-upload = src.cli.upload:main` |
|---|---|
| Config slices | `config.roboflow`, `config.progress` |
| Exit codes | 0 = success, 1 = queue errors or config failure |

## Phase 4 — Created by Agent B (23-02-2026)
