# `src/cli/segment.py` — `sam3-segment` Entry Point

## Purpose

Runs SAM3 inference on a directory of images via `create_processor()`.
Automatically selects `SequentialProcessor` (1 worker) or `ParallelProcessor`
(N workers) based on config.

## Public API

```
sam3-segment [--config PATH] [--input-dir DIR] [--output-dir DIR]
             [--device DEVICE] [--workers N] [--log-level LEVEL]
```

## Dependencies

- **Wires**: `ClassRegistry`, `create_processor` (factory → `SequentialProcessor` or `ParallelProcessor`)
- **Config slices**: `config.model` (device, prompts), `config.gpu` (strategy, workers)
- **Protocols used**: `Processor` (returned by `create_processor`)

## Data Flow

```mermaid
flowchart LR
    CLI --> |config.model| REG[ClassRegistry.from_config]
    CLI --> |config.gpu| FAC[create_processor]
    FAC --> PROC[SequentialProcessor | ParallelProcessor]
    PROC --> |process_batch with callback| MGR[ModuleProgressManager]
    MGR --> |on_item_complete| SUMMARY
```

## Wiring

| Console script | `sam3-segment = src.cli.segment:main` |
|---|---|
| Config slices | `config.model`, `config.pipeline`, `config.gpu` |
| Factory | `create_processor(config, registry)` |
| Exit codes | 0 = success, 1 = batch errors or config failure |

## Phase 4 — Created by Agent B (23-02-2026)
