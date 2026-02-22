# `src/cli/annotate.py` — `sam3-annotate` Entry Point

## Purpose

Writes YOLOv11 `data.yaml` and directory structure via `AnnotationWriter`.
Per-image `.txt` annotation files are written during `sam3-pipeline` when
`AnnotationWriter.write_annotation()` is called per result.

## Public API

```
sam3-annotate [--config PATH] [--output-dir DIR] [--log-level LEVEL]
```

## Dependencies

- **Wires**: `ClassRegistry`, `AnnotationWriter`
- **Config slices**: `config.pipeline` (output_dir), `config.model` (prompts → class names via ClassRegistry)

## Wiring

| Console script | `sam3-annotate = src.cli.annotate:main` |
|---|---|
| Config slices | `config.pipeline`, `config.model` |
| Exit codes | 0 = success, 1 = write error or config failure |

## Phase 4 — Created by Agent B (23-02-2026)
