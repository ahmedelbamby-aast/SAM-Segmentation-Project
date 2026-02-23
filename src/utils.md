# utils

## Purpose

Shared utility functions for formatting and I/O used across multiple modules.
Does not implement any pipeline protocol — it is a pure collection of stateless
helpers.

## Public API

| Function | Signature | Description |
|----------|-----------|-------------|
| `format_duration` | `(seconds: float) → str` | Human-readable duration: `"2h 3m 14s"` |
| `format_size` | `(bytes: int) → str` | Human-readable file size: `"1.2 MB"` |
| `estimate_eta` | `(done: int, total: int, elapsed_s: float) → str` | Formatted ETA string |
| `get_timestamp` | `() → str` | ISO-8601 timestamp for the current moment |
| `ensure_dir` | `(path: Path) → Path` | `mkdir -p` variant; returns the path |

## Design

- All functions are pure / stateless — no class needed.
- `ensure_dir` always uses `parents=True, exist_ok=True` — cross-platform safe.
- No `setup_logging()` — logging is handled by `LoggingSystem` in each caller
  module.  The shim was removed in the Phase 6 dead-code cleanup.

## Dependencies

- Imports: `pathlib`, `datetime`, `math`, `src.logging_system`
- Does NOT import any other `src/` module

## Usage Examples

```python
from src.utils import format_duration, format_size, estimate_eta, ensure_dir
from pathlib import Path

print(format_duration(3661))        # "1h 1m 1s"
print(format_size(1_048_576))       # "1.0 MB"
print(estimate_eta(50, 200, 30.0))  # "1m 30s"

out_dir = ensure_dir(Path("output/train/labels"))
```

## Edge Cases

- `estimate_eta` returns `"unknown"` if `done == 0`
- `format_size` handles sizes from bytes up to TB
- `ensure_dir` is idempotent — safe to call multiple times

## Wiring

- Imported by: `src/pipeline.py`, `src/cli/`, `src/parallel_processor.py`
- No config source required
