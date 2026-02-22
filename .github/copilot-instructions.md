# SAM 3 Segmentation Pipeline — AI Agent Instructions

## Compliance — Binding for All Agents

Every agent working on this project MUST read, understand, and obey ALL instructions in this file — no exceptions. These instructions are not suggestions; they are binding constraints. Violations invalidate the agent's work and require rework before any merge or delivery tag.

## SOLID Principles — Mandatory

Every change to this codebase MUST comply with all five SOLID principles. Agents must verify compliance before marking any work as complete.

- **SRP (Single Responsibility):** Each class/module has ONE reason to change. Segmentation, NMS, filtering, annotation writing, progress tracking, and uploading are SEPARATE modules. Never combine inference + post-processing + filtering in one method.
- **OCP (Open/Closed):** Extend via new classes, never modify existing ones. NMS strategies use the Strategy Pattern — add new strategies by subclassing `NMSStrategy` and registering via `@NMSStrategyFactory.register()`. Never add elif branches.
- **LSP (Liskov Substitution):** Any class implementing a Protocol (e.g., `Segmentor`, `PostProcessor`, `Writer`) must be substitutable without breaking callers. All Protocol methods must be implemented with matching signatures.
- **ISP (Interface Segregation):** Components receive ONLY the config/interfaces they need. Pass `config.model` not `config`. Depend on specific Protocols, not monolithic objects.
- **DIP (Dependency Inversion):** Depend on Protocols from `src/interfaces.py`, never on concrete classes. Pipeline accepts `Segmentor`, not `SAM3Segmentor`. All wiring happens in CLI entry points or factory functions.

## Architecture

Automated pipeline processing classroom images with SAM 3 (Segment Anything with Concepts) for N-class segmentation with many-to-one class remapping, outputting YOLOv11-format polygon annotations with optional Roboflow upload.

**Data flow:** `scan → split-assign → segment (SAM3) → remap (ClassRegistry) → NMS → filter → annotate → upload → validate`

**Critical constraint — Remap before NMS:** Class remapping MUST be applied immediately after SAM3 returns raw prompt indices, BEFORE NMS begins. NMS operates ONLY on remapped output class IDs, never on raw prompt indices. Pipeline stage order is inviolable:
```
segment() → raw prompt indices (0..N-1)
    ↓
remap()  → output class IDs (0..M-1)  ← ClassRegistry.remap_prompt_index()
    ↓
NMS()    → operates on output class IDs only
    ↓
filter() → annotate() → upload() → validate()
```

**Orchestrator:** `src/pipeline.py` (`SegmentationPipeline`) — thin orchestrator (~50 lines in `run()`), accepts ALL dependencies via constructor injection using Protocol types from `src/interfaces.py`. Zero direct imports of concrete classes.

**Interfaces:** `src/interfaces.py` — Protocol definitions: `Segmentor`, `PostProcessor`, `Writer`, `Filter`, `Tracker`, `Uploader`, `Processor`. Every module implements the relevant Protocol. All inter-module communication goes through these abstractions.

**Class Registry:** `src/class_registry.py` (`ClassRegistry`) — single source of truth for class names, IDs, and many-to-one prompt-to-output remapping. Supports N input prompts merging into M output classes (M ≤ N). All numeric IDs are auto-computed — user only specifies name-to-name mappings. Serializable for IPC via `to_dict()`/`from_dict()`.

**Logging:** `src/logging_system.py` (`LoggingSystem`) — singleton, structured JSON + Rich console output, correlation IDs per job, `@trace` decorator for auto-tracing entry/exit/duration. Initialized FIRST in every entry point.

**NMS:** `src/post_processor.py` (`MaskPostProcessor`) — decoupled from segmentor, runs as its own pipeline stage. 10 NMS strategies via OCP Strategy Pattern (`NMSStrategy` ABC). Strategy selected at construction via `NMSStrategyFactory`. Receives only remapped output class IDs.

**GPU Strategy:** `src/gpu_strategy.py` — `GPUStrategy` ABC with `CPUOnlyStrategy`, `SingleGPUMultiProcess`, `MultiGPUDDP` concrete implementations. `auto_select_strategy()` factory detects hardware: 0 GPUs → CPU, 1 GPU → multi-process sharing, 2+ GPUs → DDP with NCCL.

**Progress Display:** `src/progress_display.py` (`ModuleProgressManager`) — Rich-based per-stage progress bars: `[Scan]`, `[Preprocess]`, `[Segment]`, `[Remap]`, `[NMS]`, `[Filter]`, `[Annotate]`, `[Upload]`, `[Validate]`. Ephemeral display only — durable persistence remains in `ProgressTracker` (SQLite).

**Persistence:** SQLite with WAL mode for progress tracking (`progress_tracker.py`) and validation caching (`validator.py`) — both share the same DB file. Thread-local connections via `threading.local()` with exponential backoff retry on locks.

**Parallelism:** `parallel_processor.py` accepts `GPUStrategy` via DI for device assignment. Uses proper worker classes with instance state (no global mutable state). Config + `ClassRegistry` serialized to dict for IPC — reconstructed via `load_config_from_dict()` + `ClassRegistry.from_dict()` in worker.

**CLI:** `src/cli/` package — one entry point per module (`sam3-preprocess`, `sam3-segment`, `sam3-postprocess`, `sam3-filter`, `sam3-annotate`, `sam3-validate`, `sam3-upload`, `sam3-download`, `sam3-pipeline`, `sam3-progress`). Each initializes `LoggingSystem` first, loads only its config slice, creates its own Rich progress bar.

### Module Dependency Rules

```
LEAF MODULES (zero internal imports — keep them independent):
  preprocessor, post_processor, result_filter, annotation_writer,
  dataset_cache, model_downloader, utils, class_registry

INFRASTRUCTURE (imported by many, imports nothing from src):
  logging_system, interfaces, gpu_strategy, progress_display

INTEGRATION (may import infrastructure + receive leaf modules via DI):
  pipeline, parallel_processor, roboflow_uploader, validator

CLI LAYER (wires everything together — only place concrete classes are imported):
  src/cli/*.py
```

Concrete class instantiation and wiring happens ONLY in CLI entry points and factory functions. Core modules depend on Protocols, never on sibling concrete classes.

## Code Conventions

- **Module header:** Every `.py` file starts with a Google-style docstring containing description, `Author: Ahmed Hany ElBamby`, `Date: DD-MM-YYYY`
- **Imports:** Relative within `src/` (`from .config_manager import Config`), never `from src.X`. PEP 8 grouping (stdlib → third-party → local). Import Protocols from `src/interfaces.py` for type hints.
- **Type hints:** Full annotations on all signatures. Use `typing.List`, `typing.Optional`, `typing.Dict` (not lowercase builtins) for Python 3.9 compat
- **Docstrings:** Google-style with `Args:`, `Returns:`, `Raises:` sections
- **Paths:** Always `pathlib.Path`, never raw strings. File I/O always with `encoding='utf-8'`. No OS-specific path assumptions (dev=Windows, prod=Linux)
- **Errors:** Stdlib exceptions only (`ValueError`, `FileNotFoundError`, `RuntimeError`). Chain with `raise X from e`
- **Config:** Nested `@dataclass` hierarchy with validation in `__post_init__`. Components receive ONLY the config slice they need (ISP): `SAM3Segmentor(config.model, config.post_processing)`, not `SAM3Segmentor(config)`
- **Logging:** Use `LoggingSystem.get_logger(__name__)` at module top. Apply `@trace` decorator to public methods for auto entry/exit/duration logging. Never use `print()` for operational output — use logger. Reserve `print()` with emoji (`📤 ✓ ✗ ⚠️`) for user-facing CLI messages only.
- **Naming:** `CamelCase` classes, `snake_case` methods, `_private` prefix, `UPPER_CASE` constants
- **Lazy loading:** `_ensure_loaded()` + `self._model_loaded` flag pattern for expensive resources (model, DB)
- **Stats:** Internal `_stats` dict or `@dataclass` with `@property` computed fields, exposed via `get_stats()` / `reset_stats()`
- **Cleanup:** Explicit `cleanup()` or `close()` methods, `shutdown(wait=True)` pattern
- **Protocol compliance:** Every module that participates in the pipeline MUST implement its corresponding Protocol from `src/interfaces.py`. Run `mypy` or manual check to verify.
- **No dead code:** Never commit defined-but-unused functions, classes, or imports. Wire them in or delete them. Run dead code analysis before tagging Delivered.

## Key Patterns

### Class Registry Pattern
`ClassRegistry` is the single source of truth for class names and IDs. Never hardcode class names (`["teacher", "student"]`) anywhere. Always use:
- `ClassRegistry.from_config(config)` to create
- `registry.class_names` for output class list
- `registry.remap_prompt_index(idx)` to convert SAM3 prompt index → output class ID
- `registry.get_yolo_names()` for YOLO `data.yaml` names dict
- `registry.num_classes` for class count
- `registry.to_dict()` / `ClassRegistry.from_dict(d)` for IPC serialization

Many-to-one remapping: User config specifies only name-to-name mappings. System auto-computes all numeric IDs. Example: 5 prompts → 2 output classes via `class_remapping` in `config.yaml`.

### NMS Strategy Pattern
NMS uses `NMSStrategy` ABC with 10 concrete implementations. To add a new strategy:
1. Subclass `NMSStrategy` — implement `compute_suppression_score()` and `should_suppress()`
2. Register via `@NMSStrategyFactory.register("strategy_name")` decorator
3. Add config fields to `PostProcessingConfig` if needed
4. Add unit tests

Never modify `_should_suppress()` with new elif branches. Never modify `NMSStrategyFactory.create()` for new strategies. The 10 strategies are: Confidence, Area, ClassPriority, GaussianSoftNMS, LinearSoftNMS, WeightedNMS, AdaptiveNMS, DIoUNMS, MatrixNMS, MaskMergeNMS.

### GPU Strategy Pattern
`GPUStrategy` ABC with concrete strategies: `CPUOnlyStrategy`, `SingleGPUMultiProcess`, `MultiGPUDDP`. Selected via `auto_select_strategy(config)` factory. To add a new GPU strategy: subclass `GPUStrategy`, implement `initialize()`, `get_device_for_worker()`, `cleanup()`.

### Factory Functions
Module-level factories: `create_processor(config)`, `create_post_processor(config)`, `auto_select_strategy(config)`, `ClassRegistry.from_config(config)`. Return concrete instances — callers use Protocol types.

### Dependency Injection
Pipeline and processors accept all dependencies via constructor parameters typed as Protocols:
```python
def __init__(self, segmentor: Segmentor, post_processor: PostProcessor,
             writer: Writer, filter: Filter, tracker: Tracker, ...):
```
Concrete wiring happens in `src/cli/` entry points only.

### Progress Callback
Modules report progress via `ProgressCallback` protocol: `on_item_start(id)`, `on_item_complete(id)`, `on_item_error(id, err)`. `ModuleProgressManager` subscribes and updates Rich bars. `ProgressTracker` subscribes and persists to SQLite.

### Device Management
`DeviceManager` (static methods) in `src/gpu_strategy.py` handles `"auto"` → best GPU detection, CUDA fallback to CPU, memory reporting via `psutil` + `torch.cuda`. `_maybe_gc()` runs `gc.collect()` every 50 (CPU) or 100 (GPU) images, plus `torch.cuda.empty_cache()` on GPU.

### SQLite Patterns
WAL mode, `PRAGMA synchronous=NORMAL`, `cache_size=-64000`, `busy_timeout=300000`, `Row` factory, retry with exponential backoff (0.5s doubling, 5 retries max). Use `Status` enum values, never raw strings.

### Multiprocessing IPC
Worker functions must be free functions (not methods) for pickling. Config → dict → `initargs` → `load_config_from_dict()` in worker. `ClassRegistry` → `to_dict()` → `initargs` → `ClassRegistry.from_dict()` in worker. Remap applied in worker after inference, before returning results.

## Testing & Delivery Gates

### Delivery Policy — MANDATORY
**Nothing is tagged "Delivered" unless ALL of the following are satisfied:**

1. **Unit Tests:** Every module modified or created MUST have comprehensive unit tests. Each test file covers ONE module. Tests must pass independently (`pytest tests/test_<module>.py -v`). 100% of public methods must be tested.

2. **Integration Tests:** After completing 2 or more modules, write and pass integration tests that verify module interactions:
   - Segmentor → Remap → NMS pipeline flow
   - ClassRegistry + AnnotationWriter YOLO output
   - ProgressTracker + ModuleProgressManager event flow
   - Config loading → module initialization → execution
   - CLI entry point → full stage execution

3. **System Tests:** After integration tests pass, run end-to-end system tests:
   - Full pipeline execution with sample images
   - Resume from checkpoint
   - Multi-class remapping (N→M)
   - Each NMS strategy produces valid output
   - CLI standalone execution for each entry point
   - Cross-platform path handling (Windows + Linux)

4. **Regression Gate:** ALL existing tests must continue to pass after every change. Run `pytest tests/ -v` after every modification.

### Test Pyramid
```
System Tests     ← end-to-end pipeline, CLI, cross-platform
Integration Tests ← 2+ modules interacting via Protocols
Unit Tests       ← every public method of every module (MANDATORY)
```

### Test File Naming
- Unit: `tests/test_<module_name>.py` (e.g., `tests/test_class_registry.py`)
- Integration: `tests/integration/test_<interaction>.py` (e.g., `tests/integration/test_segment_remap_nms.py`)
- System: `tests/system/test_<scenario>.py` (e.g., `tests/system/test_full_pipeline.py`)

### Phase Delivery Gates

| Phase | Modules | Unit Tests Required | Integration Tests Required | Tag |
|-------|---------|--------------------|-----------------------------|-----|
| Phase 1 | `logging_system`, `interfaces` | Yes — both modules | N/A (infrastructure only) | Delivered after unit tests pass |
| Phase 2 | `pipeline`, `post_processor`, `sam3_segmentor`, `class_registry` | Yes — all four | segment→remap→NMS flow, ClassRegistry+Writer | Delivered after unit+integration pass |
| Phase 3 | `gpu_strategy`, `progress_display` | Yes — both | GPU+Processor, Progress+Tracker | Delivered after unit+integration pass |
| Phase 4 | `cli/*`, config slicing | Yes — each CLI | CLI→module→output for each entry point | Delivered after unit+integration+system pass |
| Phase 5 | `annotation_writer`, `validator`, `roboflow_uploader`, dead code cleanup | Yes — all modified | Full pipeline system test | Delivered after full test suite green |

## Agent Collaboration Rules

### Multi-Agent Workflow
Agents working on this codebase MUST collaborate, not work in isolation. Follow these rules:

1. **Shared Interfaces Contract:** Before implementing any module, check `src/interfaces.py` for the Protocol definition. If implementing a Protocol, match the signature EXACTLY. If a Protocol needs modification, update `src/interfaces.py` FIRST, then update all implementors.

2. **No Breaking Changes:** Never modify a Protocol's method signature without updating ALL modules that implement or consume it. Search for all usages before changing any Protocol.

3. **Module Boundary Respect:** Each agent owns their assigned module(s). When you need something from another module, depend on its Protocol, not its internals. If you need a new method on another module's Protocol, propose the addition — don't reach into the implementation.

4. **Class Registry is Canonical:** All class name/ID handling goes through `ClassRegistry`. If you need class information in your module, accept `ClassRegistry` (or its Protocol) in your constructor. Never maintain a separate class name list.

5. **Logging is First:** Every module initializes logging via `LoggingSystem` before doing anything else. Every public method is decorated with `@trace`. Every error is logged before being raised.

6. **Config Slice Only:** Your module receives only its relevant config dataclass. If you need config values from another section, that's a design smell — refactor.

7. **Test Before Integrate:** Write your module's unit tests first. Verify they pass. Only then integrate with other modules. After integration, write integration tests covering the interaction.

8. **Progress Reporting:** If your module processes items (images, masks, batches), implement `ProgressCallback` protocol and report `on_item_start`, `on_item_complete`, `on_item_error` for every item.

9. **Handoff Protocol:** When completing a module, document in your commit message:
   - Which Protocol(s) you implemented
   - Which config section(s) your module consumes
   - What factory function creates your module
   - What unit tests you added and their pass status

10. **Conflict Resolution:** If two agents need to modify the same file (e.g., `config_manager.py` for new dataclasses):
    - The agent adding infrastructure (logging, interfaces, config) goes first
    - The agent adding features waits for infrastructure to be committed
    - Both verify all tests pass after merging

11. **Completion Report — MANDATORY:** When an agent finishes its assigned targets, it MUST write a detailed report in `README.md`. The report is appended to the existing README structure — never overwrite existing content. The report must follow the format specified below.

    #### Report Structure
    Each agent's report is added as a new subsection under `## Development Reports` in `README.md`:

    ```markdown
    ## Development Reports

    ### Phase X — [Module Names] (Agent [Letter])

    **Date:** DD-MM-YYYY
    **Author:** Agent [Letter] — [Role description]
    **Status:** Delivered ✅ | In Progress 🔄 | Blocked ❌

    #### Summary
    Brief overview of what was implemented, why, and how it fits in the overall architecture.

    #### Architecture Diagram
    ```mermaid
    graph TD
        A[Module A] -->|Protocol| B[Module B]
        B --> C[Module C]
    ```

    #### Module Dependency Diagram
    ```mermaid
    graph LR
        CLI[src/cli/segment.py] -->|creates| SEG[SAM3Segmentor]
        SEG -->|implements| P[Segmentor Protocol]
        SEG -->|returns| SR[SegmentationResult]
    ```

    #### Files Modified / Created
    | File | Action | Description |
    |------|--------|-------------|
    | `src/module.py` | Created | Description of the module |
    | `tests/test_module.py` | Created | Unit tests — X tests, all passing |

    #### Key Design Decisions
    - Decision 1: Chose X over Y because...
    - Decision 2: ...

    #### Protocol Compliance
    - Implements: `Segmentor` protocol from `src/interfaces.py`
    - Consumes: `config.model` (ModelConfig dataclass)
    - Factory: `create_segmentor(config)` in `src/cli/segment.py`

    #### Test Results
    ```
    pytest tests/test_module.py -v
    ========================= X passed in Y.YYs =========================
    ```

    #### Integration Points
    Describe how this module connects with others — which Protocols it implements,
    which it consumes, and how data flows through it. Include a data flow diagram:

    ```mermaid
    sequenceDiagram
        participant CLI as CLI Entry Point
        participant SEG as Segmentor
        participant RMP as Remap Stage
        participant NMS as PostProcessor
        CLI->>SEG: process_image(path)
        SEG-->>CLI: SegmentationResult (raw indices)
        CLI->>RMP: remap(result, registry)
        RMP-->>CLI: SegmentationResult (output IDs)
        CLI->>NMS: apply_nms(result)
        NMS-->>CLI: filtered result
    ```

    #### Known Limitations / TODOs
    - Any remaining issues or future work items
    ```

    #### Report Rules
    - **Diagrams are MANDATORY** — use Mermaid syntax (`graph TD`, `sequenceDiagram`, `classDiagram`, `flowchart`). Every report must include at least: (1) an architecture/dependency diagram, (2) a data flow diagram.
    - **Merge with existing structure** — read the current `README.md` first. Add your report section under `## Development Reports`. If the section doesn't exist yet, create it at the end of the file before any footer/license section.
    - **Be factual** — report what was actually done, not what was planned. Include real test output, real file paths, real class names.
    - **Cross-reference** — link to relevant sections of this instructions file when explaining design decisions (e.g., "per the Remap-before-NMS constraint in copilot-instructions.md").
    - **Cumulative** — each agent appends; never delete or overwrite another agent's report.
    - **Well-structured and organized** — the report must integrate cleanly with the existing README structure. Use consistent heading levels, formatting, and style.

12. **Module Documentation — MANDATORY:** Every module or script an agent creates or modifies MUST have a companion documentation file in `.md` format, placed alongside the source file (e.g., `src/post_processor.md` for `src/post_processor.py`, `scripts/download_model.md` for `scripts/download_model.py`).

    #### Documentation File Rules
    - **First agent** creates the doc file with these sections:
      - **Purpose:** What the module does and why it exists
      - **Public API:** All public classes, methods, and their signatures
      - **Design:** Architecture choices, patterns used, and rationale
      - **Dependencies:** Which Protocols it implements, which config slices it consumes, what it imports
      - **Data Flow:** How data enters, transforms, and exits the module (include a Mermaid diagram)
      - **Usage Examples:** Code snippets showing how to use the module standalone and within the pipeline
      - **Edge Cases:** Known limitations, error conditions, and how they're handled
      - **Wiring:** Which CLI entry point creates this module, what config it reads, which pipeline stage calls it

    - **Subsequent agents** who modify the same module MUST NOT overwrite existing documentation. Instead, append a new section:
      ```markdown
      ## [Feature/Fix Name] — Improved by Agent [Letter]

      **Date:** DD-MM-YYYY
      **Status:** Delivered ✅

      ### What Changed
      Concise description of what was modified or added.

      ### Why
      The problem or requirement that motivated this change.

      ### Implementation Strategy
      How the change was implemented — design decisions, patterns chosen, alternatives considered and rejected.

      ### How It Works
      Technical explanation of the new/modified behavior with code references and a Mermaid diagram if the data flow changed.

      ### Impact on Existing Behavior
      What existing behavior was preserved, what changed, and why it's backward-compatible (or why a breaking change was necessary).
      ```

    - **Naming:** `<module_name>.md` in the same directory as the source file. For `src/cli/` modules, use `src/cli/<cli_name>.md`.
    - **Keep synchronized:** If you change a module's public API, update its doc file in the same commit. Stale docs are a bug.

13. **System Wiring Verification — MANDATORY:** Every agent MUST wire up their module or script with the entire system and verify the wiring is correct before marking work as complete.

    #### Wiring Checklist
    - [ ] Module implements its Protocol from `src/interfaces.py` — verified via `mypy` or manual signature check
    - [ ] Module is importable from `src/__init__.py` or accessible via its CLI entry point
    - [ ] Factory function (if applicable) creates the module correctly and is called from the appropriate CLI entry point in `src/cli/`
    - [ ] Config dataclass fields consumed by the module exist in `src/config_manager.py` and are loaded from `config/config.yaml`
    - [ ] Module integrates with `LoggingSystem` — logger initialized, `@trace` on public methods
    - [ ] Module integrates with `ProgressCallback` (if it processes items) — events emitted correctly
    - [ ] Module's constructor parameters match what the pipeline orchestrator or CLI entry point passes
    - [ ] Run the full pipeline (`sam3-pipeline --job-name wiring_test`) or the relevant standalone CLI to verify end-to-end connectivity
    - [ ] Run `pytest tests/ -v` to confirm no regressions from wiring changes
    - [ ] Document wiring in the module's `.md` doc file under a **Wiring** section: which CLI creates it, what config it reads, which pipeline stage calls it

    **Unwired modules are incomplete.** A module that passes unit tests but isn't connected to the pipeline or its CLI entry point is NOT Delivered.

### Agent Task Assignment Pattern
```
Agent A: Infrastructure (logging_system, interfaces, config updates)
Agent B: Core pipeline (pipeline, class_registry, NMS decoupling)
Agent C: Strategies (NMS strategies, GPU strategies)
Agent D: UI/CLI (progress_display, cli/*, setup.py)
Agent E: Testing (unit tests, integration tests, system tests, test fixes)
```

Each agent runs ALL existing tests after every change. No exceptions.

## Build & Test

```bash
pip install -r requirements.txt
pip install git+https://github.com/ultralytics/CLIP.git   # required, not in requirements.txt

# Unit tests
pytest tests/ -v                                    # all unit tests
pytest tests/test_<module>.py -v                     # single module
pytest tests/ --cov=src --cov-report=html            # with coverage

# Integration tests (after 2+ modules complete)
pytest tests/integration/ -v

# System tests (after integration tests pass)
pytest tests/system/ -v

# Full test suite (required before any Delivered tag)
pytest tests/ tests/integration/ tests/system/ -v --tb=short

# CLI entry points
sam3-pipeline --job-name batch_001                   # run full pipeline
sam3-pipeline --job-name batch_001 --resume           # resume from checkpoint
sam3-progress --job-name batch_001                    # check progress
sam3-preprocess --config config/config.yaml --help    # standalone preprocessing
sam3-segment --config config/config.yaml --help       # standalone segmentation
sam3-postprocess --config config/config.yaml --help   # standalone NMS
sam3-download --token $HF_TOKEN                       # download SAM3 model
sam3-validate --validate                              # compare input/output

# Legacy scripts (thin wrappers around src/cli/)
python scripts/run_pipeline.py --job-name batch_001
python scripts/run_validator.py --validate
python scripts/download_model.py --token $HF_TOKEN
```

All console_scripts registered in `setup.py`: `sam3-pipeline`, `sam3-preprocess`, `sam3-segment`, `sam3-postprocess`, `sam3-filter`, `sam3-annotate`, `sam3-validate`, `sam3-upload`, `sam3-download`, `sam3-progress`.

## Known Issues & Dead Code

**Policy:** Dead code is a bug. Wire it in or delete it. Never commit unused functions, classes, or imports.

Current issues to resolve during refactoring:
- `config_manager.py` is the only module with **no logger** — add `LoggingSystem.get_logger(__name__)` when editing
- `download_model.py` defines its own `setup_logging()` that shadows `src/utils.py` — replace with `LoggingSystem`
- `test_annotation_writer.py` has bugs: checks `val` but code creates `valid`; asserts `names` is a list but code writes a dict
- `AnnotationWriter` creates `valid/` directory but tests and some docs reference `val/` — standardize to `valid`
- NMS is coupled inside `SAM3Segmentor.process_image()` — must be decoupled to its own pipeline stage
- Class names hardcoded as `["teacher", "student"]` in defaults — must use `ClassRegistry`
- `Status` enum defined but raw strings used — must use enum values
- `post_processing` config not serialized for IPC workers — must include in config dict + `ClassRegistry.to_dict()`

## Configuration

Config is at `config/config.yaml`. Top-level keys: `pipeline`, `model`, `split`, `progress`, `roboflow`, `post_processing`, `gpu`, `logging`.

Environment variables expanded via `${VAR_NAME}` syntax. Input modes: `"pre-split"` (train/valid/test subdirs) or `"flat"` (auto-split by ratios). Device options: `"auto"`, `"cpu"`, `"0"`, `"0,1"`.

### Class Remapping Config (user-facing — names only, no numeric IDs)
```yaml
model:
  prompts:
    - "teacher"
    - "student"
    - "kid"
    - "child"
    - "Adult"
  class_remapping:       # optional, default: null (1:1 identity)
    Adult: "teacher"     # "Adult" prompt → "teacher" output class
    kid: "student"       # "kid" prompt → "student" output class
    child: "student"     # "child" prompt → "student" output class
```

The system auto-computes: output classes `["teacher", "student"]`, `num_classes: 2`, all prompt-to-output ID mappings, YOLO names dict. No numeric IDs in config — ever.

### GPU Config
```yaml
gpu:
  strategy: "auto"          # "auto", "single_gpu", "multi_gpu_ddp", "cpu"
  devices: []               # e.g., ["0", "1"] — auto-detected if empty
  workers_per_gpu: 2
  memory_threshold: 0.85
```

### Logging Config
```yaml
logging:
  level: "INFO"
  log_file: "logs/pipeline.log"
  json_output: true
  max_file_size_mb: 50
  console_rich: true
```

## Environment

- **Development:** Windows
- **Production:** Linux
- All path handling via `pathlib.Path` — no OS-specific assumptions, no hardcoded separators
