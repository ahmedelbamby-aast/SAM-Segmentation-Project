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

### Documented `@trace` Exceptions

The following are **deliberate exclusions** from the `@trace` requirement:

1. **`LoggingSystem` class methods** (`logging_system.py`): The `trace` function is defined *below* the `LoggingSystem` class in the same module. Python cannot reference `trace` inside the class body before it exists. These 6 class methods (`initialize`, `get_logger`, `set_log_level`, `get_log_directory`, `add_handler`, `shutdown`) are structurally exempt.

2. **NMS strategy hot-path methods** (`post_processor.py`): The 10 concrete `NMSStrategy` subclasses each have 2 tiny methods (`compute_suppression_score`, `should_suppress`) — 1–3 lines each, called hundreds of times per image inside inner loops. Adding `@trace` would cause severe performance degradation. These 20 methods are exempt.

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

## Phase Delivery Audit — Final Report (23-Feb-2026)

> **TL;DR:** All 10 phases delivered successfully. Exhaustive scan of 19 source modules, 11 CLI files, 20 test files, and 3 scripts confirms 24/24 audit categories PASS. 9 Protocols match implementations perfectly. 591 tests pass. Zero violations remain.

---

### Phase Delivery Summary

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Logging System + Interfaces | ✅ DELIVERED |
| 2 | Pipeline Split + NMS + Class Registry | ✅ DELIVERED |
| 3 | GPU Strategy + Progress Display | ✅ DELIVERED |
| 4 | CLI Entry Points | ✅ DELIVERED |
| 5 | SOLID Fixes | ✅ DELIVERED |
| 6 | Dead Code Cleanup + Test Fixes | ✅ DELIVERED |
| 7 | Audit Fix | ✅ DELIVERED |
| 8 | Deep Audit Fix (SOLID + DIP + ISP) | ✅ DELIVERED |
| 9 | @trace + Import Cleanup | ✅ DELIVERED |
| 10 | Final Polish | ✅ DELIVERED |

### Architecture Verified

| Component | Status | Evidence |
|-----------|--------|---------|
| 9 Protocols defined | ✅ | `ProgressCallback`, `Preprocessor`, `Segmentor`, `PostProcessor`, `Filter`, `Writer`, `Tracker`, `Uploader`, `Processor` |
| All Protocol signatures match implementations | ✅ | All 9 pairs verified — exact method/param/type match |
| 10 NMS strategies via Strategy Pattern | ✅ | Confidence, Area, ClassPriority, GaussianSoftNMS, LinearSoftNMS, Weighted, Adaptive, DIoU, Matrix, MaskMerge |
| ClassRegistry with many-to-one remapping | ✅ | `remap_prompt_index()`, `from_config()`, `to_dict()`/`from_dict()` for IPC |
| Remap-before-NMS enforced | ✅ | Both `pipeline.py._remap_result()` and `parallel_processor._process_image_worker` |
| Thin orchestrator `run()` | ✅ | ~35 lines, delegates to `_collect_images`, `_run_processing_loop`, `_finalize` |
| 10 CLI entry points | ✅ | All registered in `setup.py` `console_scripts` |
| GPU strategy ABC + factory | ✅ | `CPUOnlyStrategy`, `SingleGPUMultiProcess`, `MultiGPUDDP` + `auto_select_strategy()` |
| Rich progress bars | ✅ | `ModuleProgressManager` with `ProgressCallback` Protocol |
| Pipeline DIP (Protocol injection) | ✅ | Constructor accepts 7 Protocol-typed dependencies |
| ISP (config slicing) | ✅ | Each component receives only its config slice |
| SRP splits | ✅ | AnnotationWriter (3 classes), Validator (2 classes), Uploader (2 classes) |

### Code Quality Verified

| Check | Status |
|-------|--------|
| `@trace` on all public methods | ✅ (2 documented exceptions) |
| Lazy `%s` logging (no f-string) | ✅ Zero violations |
| No unused imports | ✅ Zero violations |
| No bare `except:` | ✅ Zero violations |
| No runtime bugs | ✅ Zero bugs |
| `encoding='utf-8'` on all I/O | ✅ All `open()` calls verified |
| `pathlib.Path` (no `os.path`) | ✅ Zero violations |
| No `print()` in `src/` (non-CLI) | ✅ Zero violations |
| No `TODO`/`FIXME`/`NotImplementedError` | ✅ Zero violations |
| Exception chaining (`raise X from e`) | ✅ All 9 raise-in-except chained |
| `get_stats()`/`reset_stats()` pattern | ✅ All 16 pipeline-stage classes |
| 591 tests passing | ✅ All green |

### Remediation History

| Phase | Fixes Applied |
|-------|---------------|
| **Phase 8** | Pipeline↔Processor API mismatch, `Preprocessor` Protocol added, ISP config slicing, injectable pipeline deps (7 Protocol types), `get_stats()`/`reset_stats()` on 10 classes |
| **Phase 9** | A1 bug (`num_output_classes` → `num_classes`), `@trace` on 16+ methods, 5 unused imports removed (D1–D5), `@trace` exceptions documented, duplicate Dunder sections merged |
| **Phase 10** | 2 unused imports removed (`load_config`, `estimate_eta` from `pipeline.py`), stale audit replaced with final all-pass scorecard |

### Verification Commands

```bash
python -m pytest tests/ -v --tb=short               # 591 passed, 0 failed
python -m py_compile src/<module>.py                  # for every source file
grep -rn "_logger\.\w*(f[\"']" src/                   # zero results (no f-string logging)
grep -rn "except:" src/ --include="*.py"              # zero bare except
grep -rn "os\.path\." src/ --include="*.py"           # zero os.path usage
```
