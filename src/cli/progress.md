# `src/cli/progress.py` — `sam3-progress` Entry Point

## Purpose

Displays progress statistics for a named pipeline job stored in the SQLite
progress database. Optionally resets images stuck in `processing` state.

## Public API

```
sam3-progress --job-name NAME [--config PATH] [--reset-stuck] [--log-level LEVEL]
```

`--reset-stuck` calls `ProgressTracker.reset_processing_images(job_id)` to
move stuck `processing` images back to `pending`.

## Dependencies

- **Wires**: `ProgressTracker`
- **Config slice**: `config.progress` (db_path)

## Wiring

| Console script | `sam3-progress = src.cli.progress:main` |
|---|---|
| Config slice | `config.progress` |
| Exit codes | 0 = always (status display), 1 = unknown job when --reset-stuck used |

## Phase 4 — Created by Agent B (23-02-2026)
