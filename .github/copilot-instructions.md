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
