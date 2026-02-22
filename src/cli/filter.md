# `src/cli/filter.py` — `sam3-filter` Entry Point

## Purpose

Audits annotation label files in the output directory. Images with empty label
files (no detections) are moved to the `neither` folder via `ResultFilter`.

## Public API

```
sam3-filter [--config PATH] [--output-dir DIR] [--neither-dir DIR]
            [--log-level LEVEL]
```

## Dependencies

- **Wires**: `ResultFilter` (only `config.pipeline` slice)
- **Config slice**: `config.pipeline` (output_dir, neither_dir)

## Wiring

| Console script | `sam3-filter = src.cli.filter:main` |
|---|---|
| Config slice | `config.pipeline` |
| Exit codes | 0 = success, 1 = errors found or config failure |

## Phase 4 — Created by Agent B (23-02-2026)
