# `src/cli/postprocess.py` — `sam3-postprocess` Entry Point

## Purpose

Configures and verifies the NMS post-processor (`MaskPostProcessor`).
Standalone use reports the active strategy + IoU threshold.
Full NMS execution happens inside `sam3-pipeline`.

## Public API

```
sam3-postprocess [--config PATH] [--strategy NAME] [--iou-threshold FLOAT]
                 [--log-level LEVEL]
```

## Dependencies

- **Wires**: `MaskPostProcessor` (only `config.post_processing` slice per ISP)
- **Config slice**: `config.post_processing` (strategy, iou_threshold, class_priority)

## Wiring

| Console script | `sam3-postprocess = src.cli.postprocess:main` |
|---|---|
| Config slice | `config.post_processing` |
| Exit codes | 0 = success, 1 = config failure |

## Phase 4 — Created by Agent B (23-02-2026)
