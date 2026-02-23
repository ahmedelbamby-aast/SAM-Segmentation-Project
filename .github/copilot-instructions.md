# SAM 3 Segmentation Pipeline — AI Agent Instructions & Implementation Plan

## Compliance — Binding for All Agents

Every agent working on this project MUST read, understand, and obey ALL instructions in this file — no exceptions. These instructions are not suggestions; they are binding constraints. Violations invalidate the agent's work and require rework before any merge or delivery tag.

---

## MANDATORY — Agent Completion Enforcement

**This is the highest-priority constraint in this document. It overrides all other conventions if there is a conflict.**

### Non-Negotiable Rules

1. **Finish What You Start:** An agent assigned a task MUST complete it fully — including code, tests, integration, wiring, documentation, and README report — before marking it Delivered. Partial work is NEVER acceptable.

2. **No Placeholders, No Stubs, No TODOs in Committed Code:** Every function, class, and method committed to the repository must be fully implemented and tested. Placeholder comments like `# TODO`, `# FIXME`, `pass`, `raise NotImplementedError` in production code are forbidden.

3. **Verification Before Delivery:** Before marking ANY task as Delivered, the agent MUST:
   - Run `pytest tests/ tests/integration/ tests/system/ -v --tb=short` and confirm **zero failures**
   - Run `python -m py_compile src/<module>.py` for every file modified
   - Verify every public method has `@trace` decorator
   - Verify every module uses `LoggingSystem.get_logger(__name__)`
   - Verify Protocol signatures match implementations exactly
   - Verify no unused imports
   - Verify exception chaining (`raise X from e`) in all `except` blocks that re-raise
   - Verify `encoding='utf-8'` on all file I/O
   - Verify `pathlib.Path` used for all filesystem paths (no raw `str` paths)
   - Commit with descriptive message listing all changes

4. **No Skipping Steps:** Agents MUST follow the phase order. Phase N+1 cannot begin until Phase N passes ALL delivery gates.

5. **Audit Compliance:** After every phase delivery, a compliance audit is run against this document. Every finding must be resolved before the next phase begins.

---

## SOLID Principles — Mandatory

- **SRP:** Each class/module has ONE reason to change.
- **OCP:** Extend via new classes, never modify existing ones. NMS strategies use Strategy Pattern.
- **LSP:** Any class implementing a Protocol must be substitutable without breaking callers.
- **ISP:** Components receive ONLY the config/interfaces they need. Pass `config.model` not `config`.
- **DIP:** Depend on Protocols from `src/interfaces.py`, never on concrete classes.

---

## Architecture

**Data flow:** `scan → split-assign → segment (SAM3) → remap (ClassRegistry) → NMS → filter → annotate → upload → validate`

**Critical constraint — Remap before NMS:** Class remapping MUST be applied immediately after SAM3 returns raw prompt indices, BEFORE NMS begins.

**Orchestrator:** `src/pipeline.py` — thin orchestrator, accepts ALL dependencies via constructor injection using Protocol types.

**Interfaces:** `src/interfaces.py` — Protocol definitions for all inter-module communication.

**Class Registry:** `src/class_registry.py` — single source of truth for class names, IDs, and many-to-one remapping.

**Logging:** `src/logging_system.py` — singleton, structured JSON + Rich console, `@trace` decorator.

**NMS:** `src/post_processor.py` — decoupled NMS with 10 strategies via Strategy Pattern.

**GPU Strategy:** `src/gpu_strategy.py` — `GPUStrategy` ABC with CPU/single-GPU/multi-GPU strategies.

---

## Code Conventions

- **Module header:** Google-style docstring with `Author: Ahmed Hany ElBamby`, `Date: DD-MM-YYYY`
- **Imports:** Relative within `src/`. PEP 8 grouping.
- **Type hints:** Full annotations. Use `typing.List`, `typing.Optional`, `typing.Dict`.
- **Paths:** Always `pathlib.Path`. File I/O with `encoding='utf-8'`.
- **Errors:** Stdlib exceptions only. Chain with `raise X from e`.
- **Config:** Components receive ONLY their config slice (ISP).
- **Logging:** `LoggingSystem.get_logger(__name__)`. `@trace` on public methods. No `print()` for operational output.
- **Stats:** `get_stats()` / `reset_stats()` pattern on every module.
- **No dead code:** Never commit unused functions, classes, or imports.

---

## Testing & Delivery Gates

- Unit: `tests/test_<module>.py`
- Integration: `tests/integration/test_<interaction>.py`
- System: `tests/system/test_<scenario>.py`

ALL existing tests must pass after every change. No exceptions.

---

## Environment

- **Development:** Windows  |  **Production:** Linux
- All path handling via `pathlib.Path`

---

## Phase Delivery Audit & Fix Plan

> **TL;DR:** Deep scan of all 22 source files, 11 CLI files, 23 test files, and 3 scripts against the 10-phase plan and this instructions file. Phases 1–6 were delivered correctly with minor residual issues. Phase 7 (audit fix) was delivered but introduced a runtime bug and left several violations unfixed. Overall: 8 Protocols match perfectly, 591 tests pass, 0 TODO/FIXME/NotImplementedError, but 38 residual violations remain across 5 categories.

---

### Phase-by-Phase Delivery Verdict

#### Phase 1 — Logging System + Interfaces ✅ DELIVERED

| Deliverable | Status | Evidence |
|---|---|---|
| `logging_system.py` created | ✅ | 329 lines, singleton `LoggingSystem`, `@trace` decorator, JSON + Rich console, correlation IDs, log rotation |
| `interfaces.py` created | ✅ | 415 lines, 8 Protocols (`Segmentor`, `PostProcessor`, `Filter`, `Writer`, `Tracker`, `Uploader`, `Processor`, `ProgressCallback`), 3 data classes (`MaskData`, `SegmentationResult`, `ProcessingStats`) |
| `setup_logging()` removed from utils | ✅ | `utils.py` has no `setup_logging`, imports `LoggingSystem` |
| `download_model.py` uses `LoggingSystem` | ✅ | Imports `LoggingSystem`, no `setup_logging()` |
| Every module imports `LoggingSystem` | ⚠️ | `config_manager.py` imports `LoggingSystem` but not `trace` — no `@trace` on any function |
| Tests pass | ✅ | `test_logging_system.py` (346 lines), `test_interfaces.py` (369 lines) |

**Residual issues:** `config_manager.py` and `utils.py` import `LoggingSystem` but not `trace` — no `@trace` decorators.

#### Phase 2 — Pipeline Split + NMS + Class Registry ✅ DELIVERED

| Deliverable | Status | Evidence |
|---|---|---|
| Pipeline thin orchestrator | ✅ | `run()` is ~35 lines, delegates to `_collect_images()`, `_run_processing_loop()`, `_finalize()` |
| `_remap_result()` static method | ✅ | `pipeline.py:147` — converts prompt indices → output class IDs |
| Remap-before-NMS order | ✅ | Both `_process_image_worker` and `SequentialProcessor.process_batch` follow: segment → remap → NMS → filter → annotate |
| 10 NMS strategies via Strategy Pattern | ✅ | `NMSStrategy` ABC + `NMSStrategyFactory` registry + 10 concrete strategies |
| `class_registry.py` created | ✅ | 329 lines, `ClassRegistry` with many-to-one remapping, `from_config()`, `to_dict()`/`from_dict()` for IPC |
| `create_post_processor()` factory wired | ✅ | Used in `pipeline.py` constructor and `_ensure_loaded()` |
| Tests pass | ✅ | `test_post_processor.py` (566 lines), `test_class_registry.py` (367 lines), `test_segment_remap_nms.py` (217 lines), `test_nms_strategies.py` (204 lines) |

**Residual issues:** `class_registry.py` has unused `import logging` (line 17).

#### Phase 3 — GPU Strategy + Progress Display ✅ DELIVERED

| Deliverable | Status | Evidence |
|---|---|---|
| `gpu_strategy.py` created | ✅ | 340 lines, `GPUStrategy` ABC + `CPUOnlyStrategy` + `SingleGPUMultiProcess` + `MultiGPUDDP` + `auto_select_strategy()` |
| `progress_display.py` created | ✅ | 404 lines, `ModuleProgressManager` with Rich progress bars, `ProgressCallback` Protocol |
| `ProgressTracker` uses `Status` enum | ✅ | `Status` enum defined and used |
| `_worker_state` dict in workers | ✅ | No global mutable state in workers |
| Tests pass | ✅ | `test_gpu_strategy.py` (379 lines), `test_progress_display.py` (417 lines), `test_gpu_processor.py` (352 lines) |

**Residual issues:** Possibly unused imports in `progress_display.py` (`Columns`, `Table` from Rich).

#### Phase 4 — CLI Entry Points ✅ DELIVERED

| Deliverable | Status | Evidence |
|---|---|---|
| 11 CLI files in `cli/` | ✅ | `__init__.py`, `annotate.py`, `download.py`, `filter.py`, `pipeline.py`, `postprocess.py`, `preprocess.py`, `progress.py`, `segment.py`, `upload.py`, `validate.py` |
| `setup.py` console_scripts registered | ✅ | All 10 entry points: `sam3-pipeline`, `sam3-preprocess`, `sam3-segment`, `sam3-postprocess`, `sam3-filter`, `sam3-annotate`, `sam3-validate`, `sam3-upload`, `sam3-download`, `sam3-progress` |
| Scripts wrapped as thin delegates | ✅ | `run_pipeline.py` delegates to `src.cli.pipeline.main` |
| CLI `.md` docs | ✅ | 10 `.md` files exist in `cli/` |
| Tests pass | ✅ | `test_cli.py` (636 lines), `test_cli_entrypoints.py` (149 lines) |

**Residual issues:** `validate.py` has duplicate `from pathlib import Path` import.

#### Phase 5 — SOLID Fixes ✅ DELIVERED

| Deliverable | Status | Evidence |
|---|---|---|
| `AnnotationWriter` SRP split | ✅ | `MaskConverter` + `DatasetMetadataWriter` + `AnnotationWriter` — 3 classes |
| `Validator` SRP split | ✅ | `ValidationCache` + `Validator` — 2 classes |
| `RoboflowUploader` SRP split | ✅ | `AsyncWorkerPool` + `DistributedUploader` — 2 classes |
| `val` → `valid` directory fix | ✅ | Tests pass with `valid` |
| Tests pass | ✅ | `test_annotation_writer.py` (159 lines), `test_result_filter.py` (366 lines), `test_validator.py` (333 lines), `test_class_registry_writer.py` (288 lines) |

#### Phase 6 — Dead Code Cleanup + Test Fixes ✅ DELIVERED

| Deliverable | Status | Evidence |
|---|---|---|
| `estimate_eta` wired into progress display | ✅ | Used in `StageProgress.eta_str` |
| `create_post_processor` factory wired | ✅ | Used in pipeline constructor and workers |
| `Status` enum wired | ✅ | Used in `ProgressTracker` |
| System tests | ✅ | `test_pipeline_e2e.py` (354 lines) |
| 591 tests passing | ✅ | Confirmed at Phase 6 delivery commit |

#### Phase 7 — Audit Fix ⚠️ PARTIALLY DELIVERED

| Deliverable | Status | Evidence |
|---|---|---|
| Protocol signatures match implementations | ✅ | All 8 Protocols verified — PERFECT match across all pairs |
| `@trace` on all public methods | ⚠️ | 30+ public methods still missing `@trace` (see table below) |
| `_logger` naming convention | ⚠️ | One `logger.debug()` leftover in `annotation_writer.py:288` — runtime `NameError` bug |
| f-string logging → lazy `%s` | ⚠️ | 39 occurrences of `_logger.xxx(f"...")` remain across `validator.py`, `roboflow_uploader.py`, `pipeline.py`, `model_downloader.py`, `annotation_writer.py` |
| Unused imports removed | ⚠️ | `import logging` in `class_registry.py`; `dataclass, field` unused in `post_processor.py`; `field` unused in `result_filter.py` |
| Exception chaining | ✅ | All `raise` inside `except` blocks properly chained |
| `encoding='utf-8'` on all I/O | ✅ | All 10 `open()` calls verified |
| `pathlib.Path` (no `os.path`) | ✅ | Zero `os.path` usage |
| No `print()` in src modules | ✅ | Zero `print()` in `src/*.py` |
| No `TODO`/`FIXME`/`NotImplementedError` | ✅ | Zero in production code |
| Bare `except:` | ⚠️ | `progress_tracker.py:134` — bare `except:` with `pass` |

---

### Detailed Violation Registry

#### Category A — Bugs (Runtime Crashes)

| # | Severity | File | Line | Issue |
|---|---|---|---|---|
| A1 | 🔴 Critical | `annotation_writer.py` | 288 | `logger.debug(f"...")` — unresolved name `logger` (alias was removed in Phase 7, but this reference was missed). Causes `NameError` when `write_annotation` runs on an image with no valid polygons. |

#### Category B — Missing `@trace` Decorators

| # | File | Methods Missing `@trace` |
|---|---|---|
| B1 | `config_manager.py` | `load_config()`, `validate_config()`, `load_config_from_dict()` (also missing `trace` import) |
| B2 | `utils.py` | `format_duration()`, `format_size()`, `estimate_eta()`, `get_timestamp()`, `ensure_dir()` (also missing `trace` import) |
| B3 | `class_registry.py` | `get_yolo_names()`, `get_output_id_for_prompt_name()`, `to_dict()`, `from_dict()`, `from_config()` |
| B4 | `model_downloader.py` | `get_model_info()`, `list_files()`, `get_download_status()` |
| B5 | `post_processor.py` | `calculate_mask_iou()`, `calculate_mask_overlap()` |
| B6 | `preprocessor.py` | `set_fast_scan()` |
| B7 | `progress_tracker.py` | `get_job_id()`, `get_pending_images()`, `get_image_split()`, `mark_processing()`, `reset_stuck_images()`, `reset_error_images()`, `get_progress_by_split()`, `create_batch()`, `mark_batch_uploaded()`, `mark_batch_error()`, `get_pending_batches()`, `get_uploaded_batches()`, `reset_processing_images()`, `close()` |
| B8 | `result_filter.py` | `get_filtered_images()`, `get_neither_count()` |
| B9 | `roboflow_uploader.py` | `retry_failed_batches()`, `upload_neither_folder()`, `should_upload_neither()` |
| B10 | `validator.py` | `cache_missing_images()`, `get_cached_missing_images()`, `mark_cached_processed()`, `clear_validation_cache()`, `get_validation_jobs()`, `close()` (also `ValidationCache.mark_processed()`, `clear()`, `list_jobs()`, `close()`) |
| B11 | `annotation_writer.py` | `AnnotationWriter.mask_to_polygon()` (delegate), `AnnotationWriter.masks_to_polygons()` (delegate), `reset_stats()`; `DatasetMetadataWriter.write_classes_files()`, `write_data_yaml()` |
| B12 | `progress_display.py` | `on_item_start()`, `on_item_complete()`, `on_item_error()`, `on_stage_item_complete()`, `on_stage_item_error()` |

#### Category C — F-String Logging (should use lazy `%s`)

| # | File | Count |
|---|---|---|
| C1 | `roboflow_uploader.py` | 17 occurrences |
| C2 | `validator.py` | 6 occurrences |
| C3 | `model_downloader.py` | 12 occurrences |
| C4 | `annotation_writer.py` | 3 occurrences |
| C5 | `pipeline.py` | 1 occurrence |

#### Category D — Unused Imports

| # | File | Import |
|---|---|---|
| D1 | `class_registry.py:17` | `import logging` — never used |
| D2 | `post_processor.py` | `from dataclasses import dataclass, field` — neither used |
| D3 | `result_filter.py` | `from dataclasses import dataclass, field` — `field` unused |
| D4 | `progress_display.py` | `from rich.columns import Columns`, `from rich.table import Table` — likely unused |
| D5 | `cli/validate.py` | Duplicate `from pathlib import Path` |

#### Category E — Code Style / Design

| # | File | Issue | Severity |
|---|---|---|---|
| E1 | `progress_tracker.py:134` | Bare `except:` — should be `except Exception:` | 🟡 Medium |
| E2 | `roboflow_uploader.py` | Missing `reset_stats()` (has `get_stats()`) | 🟡 Medium |
| E3 | `pipeline.py` | Constructor uses `Optional[object]` instead of Protocol types for `preprocessor`, `tracker`, `uploader` — no type checking | 🟡 Medium |
| E4 | `pipeline.py` | `DatasetCache`, `AnnotationWriter`, `ResultFilter`, `Validator` not injectable — always hard-instantiated | 🟡 Medium |
| E5 | Multiple modules | Missing `get_stats()`/`reset_stats()`: `config_manager`, `dataset_cache`, `gpu_strategy`, `preprocessor`, `progress_tracker`, `sam3_segmentor`, `validator` | 🟢 Low |

---

### Overall Score Card

| Category | Plan Requirement | Actual | Verdict |
|---|---|---|---|
| Protocols defined | 8 Protocols | 8 Protocols ✅ | PASS |
| Protocol signatures match implementations | All must match | All 8 match perfectly ✅ | PASS |
| 10 NMS strategies | 10 strategies | 10 strategies ✅ | PASS |
| ClassRegistry with many-to-one | Full implementation | Full implementation ✅ | PASS |
| Remap-before-NMS | In workers + pipeline | In both ✅ | PASS |
| Thin orchestrator `run()` | ~50 lines | ~35 lines ✅ | PASS |
| 10 CLI entry points | All registered | All 10 registered ✅ | PASS |
| GPU strategy ABC | 3 strategies + factory | 3 strategies + factory ✅ | PASS |
| Rich progress bars | `ModuleProgressManager` | Implemented ✅ | PASS |
| `@trace` on ALL public methods | Mandatory | ~30+ methods missing ❌ | FAIL |
| Lazy `%s` logging | Mandatory | 39 f-string calls remain ❌ | FAIL |
| No unused imports | Mandatory | 5 violations ❌ | FAIL |
| No bare `except:` | Mandatory | 1 violation ❌ | FAIL |
| No runtime bugs | Mandatory | 1 `NameError` bug ❌ | FAIL |
| `encoding='utf-8'` on I/O | Mandatory | All 10 calls OK ✅ | PASS |
| `pathlib.Path` (no `os.path`) | Mandatory | Zero violations ✅ | PASS |
| No `print()` in `src/` | Mandatory | Zero violations ✅ | PASS |
| No `TODO`/`FIXME` | Mandatory | Zero violations ✅ | PASS |
| Exception chaining | Mandatory | All chained ✅ | PASS |
| Tests pass | 591 all green | 591 passing ✅ | PASS |

---

### Fix Plan

| Step | Action | Priority |
|---|---|---|
| 1 | **Fix A1 bug:** In `annotation_writer.py:288`, change `logger.debug` → `_logger.debug` | 🔴 Critical |
| 2 | **Fix D1–D5 unused imports:** Remove `import logging` from `class_registry.py:17`; remove `dataclass, field` from `post_processor.py`; remove unused `field` from `result_filter.py`; remove `Columns`/`Table` from `progress_display.py` (verify first); remove duplicate `Path` from `cli/validate.py` | 🟡 Medium |
| 3 | **Fix E1 bare except:** In `progress_tracker.py:134`, change `except:` → `except Exception:` | 🟡 Medium |
| 4 | **Fix B1–B12 missing `@trace`:** Add `@trace` to all ~30+ public methods listed above. Add `trace` import to `config_manager.py` and `utils.py` | 🟡 Medium |
| 5 | **Fix C1–C5 f-string logging:** Convert all 39 `_logger.xxx(f"...")` calls to lazy `%s` formatting across `roboflow_uploader.py`, `validator.py`, `model_downloader.py`, `annotation_writer.py`, `pipeline.py` | 🟡 Medium |
| 6 | **Fix E2:** Add `reset_stats()` to `DistributedUploader` in `roboflow_uploader.py` | 🟡 Medium |
| 7 | **Fix E3–E4 (stretch):** Change `pipeline.py` constructor params to use Protocol types instead of `object`; make `DatasetCache`, `AnnotationWriter`, `ResultFilter`, `Validator` injectable | 🟢 Low |
| 8 | **Run tests & compile:** Verify 591 tests pass + all modified files compile | 🔴 Critical |
| 9 | **Commit:** Single commit with descriptive message | 🔴 Critical |

### Verification Commands

```bash
python -m pytest tests/ -v --tb=short               # zero failures
python -m py_compile src/<module>.py                  # for every modified file
grep -r "logger\." src/ --include="*.py"              # only _logger. references
grep -rn "f[\"']" src/*.py | grep "_logger"           # zero results after fix
```

### Decisions

- Steps 1–6 are **mandatory** (instructions violations)
- Step 7 is a **stretch improvement** (design quality, not compliance blocker)
- `get_stats()`/`reset_stats()` on infrastructure modules (`config_manager`, `gpu_strategy`, etc.) is **Low priority** — the instruction says "every module" but these are infrastructure, not pipeline stages
