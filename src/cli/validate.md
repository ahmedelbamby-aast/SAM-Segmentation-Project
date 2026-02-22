# `src/cli/validate.py` — `sam3-validate` Entry Point

## Purpose

Compares input images with generated annotation outputs. Identifies images
missing from the output set and optionally caches them in SQLite for reprocessing.

## Public API

```
sam3-validate [--config PATH] [--validate] [--cache-missing]
              [--input-dir DIR] [--output-dir DIR] [--log-level LEVEL]
```

## Dependencies

- **Wires**: `Validator`
- **Config slices**: `config.pipeline`, `config.progress` (db_path)
- **Key methods**: `validator.compare_datasets()`, `validator.cache_missing_images(result, job_name)`

## Wiring

| Console script | `sam3-validate = src.cli.validate:main` |
|---|---|
| Config slices | `config.pipeline`, `config.progress` |
| Exit codes | 0 = complete dataset, 1 = missing images or config failure |

## Phase 4 — Created by Agent B (23-02-2026)
