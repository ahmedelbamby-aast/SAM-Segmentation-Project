## Plan: Re-Modularize SAM Segmentation System

**TL;DR**: The codebase has a monolithic orchestrator (`src/pipeline.py` — 9 dependencies, 300-line `run()`), no per-module CLIs, inconsistent logging, dead NMS code paths, single-GPU-only parallelism, and multiple SOLID violations. This plan breaks it into independently runnable modules with their own CLI entry points, a centralized logging/tracing system as the highest-priority subsystem, Rich-based per-module progress bars, smart GPU strategy selection (multi-process on single GPU, DDP across multiple GPUs), and clean protocol-based interfaces between components.

**Environment**: Development on Windows, production on Linux.
**Phase delivery policy**: All existing tests must pass before a phase is considered complete. No phase may begin until the previous phase's tests are green.

---

### Constraints

1. **Remapping before NMS**: Class remapping MUST be applied immediately after SAM3 inference returns raw prompt indices, BEFORE the NMS stage begins. NMS operates on remapped output class IDs, never on raw prompt indices. Pipeline stage order: `segment() → remap() → NMS → filter() → ...`
2. **User-facing simplicity**: The end user only specifies name-to-name mappings (`class_remapping`) in `config.yaml`. The system automatically computes all numeric output class IDs — no `class_id_remapping` in the config. The user says "Adult maps to teacher" and the system handles everything else.

---

### Files to Refactor & How

#### A. NEW FILES TO CREATE

| New File | Purpose |
|----------|---------|
| `src/logging_system.py` | Central logging singleton — structured JSON + console formatters, correlation IDs per request, module-level decorators for auto-tracing entry/exit/duration of every public method |
| `src/interfaces.py` | ABCs/Protocols: `Segmentor`, `Processor`, `Writer`, `Filter`, `Tracker`, `Uploader` — all modules depend on abstractions |
| `src/gpu_strategy.py` | `GPUStrategy` ABC with `SingleGPUMultiProcess`, `MultiGPUDDP`, `CPUOnly` concrete strategies + `auto_select_strategy()` factory |
| `src/progress_display.py` | Rich-based `ModuleProgressManager` — creates/updates independent progress bars per module stage |
| `src/class_registry.py` | `ClassRegistry` — single source of truth for class names, IDs, prompt-to-output remapping, many-to-one merging, N-class support |
| `src/cli/__init__.py` | CLI package init |
| `src/cli/preprocess.py` | CLI entry: `sam3-preprocess` — runs `ImagePreprocessor` standalone (scan, validate, resize) |
| `src/cli/segment.py` | CLI entry: `sam3-segment` — runs `SAM3Segmentor` standalone on pre-scanned images |
| `src/cli/postprocess.py` | CLI entry: `sam3-postprocess` — runs NMS standalone on raw segmentation results |
| `src/cli/filter.py` | CLI entry: `sam3-filter` — runs `ResultFilter` standalone |
| `src/cli/annotate.py` | CLI entry: `sam3-annotate` — runs `AnnotationWriter` standalone |
| `src/cli/validate.py` | CLI entry: `sam3-validate` — runs `Validator` standalone |
| `src/cli/upload.py` | CLI entry: `sam3-upload` — runs `DistributedUploader` standalone |
| `src/cli/download.py` | CLI entry: `sam3-download` — runs model download standalone |
| `src/cli/pipeline.py` | CLI entry: `sam3-pipeline` — runs full orchestrated pipeline |
| `src/cli/progress.py` | CLI entry: `sam3-progress` — query/reset progress DB |

---

#### B. FILES TO REFACTOR

**Step 1 — Logging System** (highest priority, everything depends on it)

- **Create** `src/logging_system.py`:
  - `LoggingSystem` singleton class with `initialize(config)`, `get_logger(module_name)`, `set_correlation_id(id)`
  - Structured JSON file handler + Rich console handler with color-coded levels
  - Auto-attach correlation ID (job name/run ID) to every log record via a `logging.Filter`
  - Module-level `@trace` decorator that logs entry args, exit return value, and duration for every public method call
  - Performance metrics logging (GPU memory, throughput images/sec) as structured data
  - Log rotation (by size, configurable in config.yaml)
  - Exception hook (`sys.excepthook`) to capture unhandled exceptions

- **Refactor** `src/utils.py`:
  - Remove `setup_logging()` — replaced by `LoggingSystem`
  - Keep pure utility functions (`format_duration`, `format_size`, `estimate_eta`, `get_timestamp`, `ensure_dir`)
  - Delete dead code: `format_size`, `get_timestamp`, `ensure_dir` if still unused after refactor, or wire them in

- **Refactor** `scripts/download_model.py`:
  - Remove the duplicate local `setup_logging()` function (line ~22)
  - Use `LoggingSystem.initialize()` instead

- **Refactor** `src/config_manager.py`:
  - Add `logger = logging.getLogger(__name__)` — currently the **only** module with zero logging
  - Add log statements for config loading, validation errors, env var expansion

- **Refactor every `src/*.py`**: Replace `from src.utils import setup_logging` calls with `from src.logging_system import LoggingSystem`

---

**Step 2 — Interfaces / Protocols** (enables DIP compliance)

- **Create** `src/interfaces.py`:
  - `class Segmentor(Protocol)`: `process_image()`, `process_batch()`, `cleanup()`, `get_device_info()`
  - `class PostProcessor(Protocol)`: `apply_nms()`, `get_stats()`
  - `class Writer(Protocol)`: `write_annotation()`, `write_data_yaml()`, `get_stats()`
  - `class Filter(Protocol)`: `filter_result()`, `get_stats()`
  - `class Tracker(Protocol)`: `create_job()`, `mark_completed()`, `mark_error()`, `get_progress()`, `checkpoint()`
  - `class Uploader(Protocol)`: `queue_batch()`, `wait_for_uploads()`, `shutdown()`
  - `class Processor(Protocol)`: `start()`, `process_batch()`, `shutdown()` — unifies `ParallelProcessor` and `SequentialProcessor`

---

**Step 3 — Split the God Class** (`src/pipeline.py`)

- **Refactor** `src/pipeline.py`:
  - Extract `_assign_splits()` logic into `ImagePreprocessor` (it's preprocessing concern)
  - Extract scan + cache logic into a `ScanStage` method on `DatasetCache`
  - Extract upload coordination into `DistributedUploader` (it should own batch creation)
  - Reduce `run()` from ~300 lines to ~50 lines: a sequential call to stage methods
  - Pipeline becomes a thin orchestrator with explicit remap stage before NMS:
    `scan() → preprocess() → segment() → **remap()** → NMS() → filter() → annotate() → upload() → validate()`
  - The `remap()` stage calls `ClassRegistry.remap_prompt_index()` on every `class_id` in every `SegmentationResult` — converting raw SAM3 prompt indices to output class IDs before NMS ever sees the data
  - Accept all dependencies via constructor injection (interfaces, not concrete classes)
  - Remove direct imports of concrete classes — use a `ServiceRegistry` or constructor DI

- **Refactor** `scripts/run_pipeline.py`:
  - Split the monolithic `main()` if/elif chain into a command dispatcher
  - Each command (`--status`, `--retry-uploads`, `--reset-stuck`, `--reset-errors`, `--cache-info`) becomes a separate function
  - Move to `src/cli/pipeline.py`

---

**Step 4 — NMS Logic Fix & Strategy Expansion** (`src/post_processor.py` + `src/sam3_segmentor.py`)

##### Step 4a — Wire Dead NMS Code & Decouple from Segmentor

- **Refactor** `src/post_processor.py`:
  - Wire up `create_post_processor()` factory — currently dead code, never called
  - Wire up `apply_class_specific_nms()` — add config toggle `class_specific: true/false` in `PostProcessingConfig`
  - Wire up `calculate_mask_overlap()` — or remove if truly unnecessary
  - Add logging to NMS decisions (which masks suppressed, why, IoU values)
  - Implement `PostProcessor` protocol from interfaces
  - **NMS receives only remapped class IDs** — it never sees raw prompt indices (enforced by pipeline ordering constraint)

- **Refactor** `src/sam3_segmentor.py`:
  - **Decouple NMS**: Remove `MaskPostProcessor` creation from `SAM3Segmentor.__init__` — post-processing should be injected or called externally by the pipeline
  - `process_image()` returns raw `SegmentationResult` with **raw prompt indices** as `class_id` — remapping is NOT done here (it's the pipeline's responsibility in the remap stage)
  - NMS becomes a separate pipeline stage, not embedded in the segmentor
  - Move `DeviceManager` to the new `src/gpu_strategy.py`
  - Implement `Segmentor` protocol

##### Step 4b — Refactor to Strategy Pattern (OCP-Compliant)

Replace the monolithic `_should_suppress()` if/elif chain with a proper Strategy Pattern:

- **Create** `NMSStrategy` ABC in `src/post_processor.py`:
  ```python
  class NMSStrategy(ABC):
      @abstractmethod
      def compute_suppression_score(self, iou: float, mask_a: MaskData, mask_b: MaskData) -> float:
          """Return suppression weight in [0, 1]. 1 = fully suppress, 0 = keep."""
      
      @abstractmethod
      def should_suppress(self, score: float, threshold: float) -> bool:
          """Determine if mask should be suppressed given computed score."""
  ```

- **Migrate existing 4 strategies** to concrete `NMSStrategy` subclasses:
  1. `ConfidenceNMS` — suppresses lower-confidence mask when IoU > threshold (current `OverlapStrategy.CONFIDENCE`)
  2. `AreaNMS` — suppresses smaller-area mask when IoU > threshold (current `OverlapStrategy.AREA`)
  3. `ClassPriorityNMS` — suppresses lower-priority class when IoU > threshold (current `OverlapStrategy.CLASS_PRIORITY`)
  4. `GaussianSoftNMS` — decays confidence by `exp(-iou² / σ²)` (current `OverlapStrategy.SOFT_NMS`)

- **Refactor** `MaskPostProcessor.apply_nms()`:
  - Replace the if/elif chain in `_should_suppress()` with `self.strategy.compute_suppression_score()` + `self.strategy.should_suppress()`
  - Strategy selected at construction via `NMSStrategyFactory.create(config.overlap_strategy)`
  - `NMSStrategyFactory.create(strategy_name: str) -> NMSStrategy` — registry-based, extensible without modifying factory code

- **Update** `OverlapStrategy` enum to include all 10 strategies (add 6 new members)

##### Step 4c — Implement 6 New NMS Strategies

Add 6 new concrete `NMSStrategy` subclasses:

| # | Strategy | Class Name | Algorithm | Use Case |
|---|----------|-----------|-----------|----------|
| 5 | Linear Soft-NMS | `LinearSoftNMS` | Decay confidence linearly: `score *= (1 - iou)` when `iou > threshold` | Simpler soft suppression; less aggressive than Gaussian for moderate overlaps |
| 6 | Weighted NMS | `WeightedNMS` | Instead of suppressing, merge overlapping masks weighted by confidence: `merged_mask = Σ(conf_i × mask_i) / Σ(conf_i)` | Dense predictions where masks should blend rather than compete; produces smoother boundaries |
| 7 | Adaptive NMS | `AdaptiveNMS` | Per-mask dynamic threshold: `adaptive_thresh = max(base_thresh, density_factor)` using local mask density (count of overlapping neighbors) | Crowded scenes (many students close together); relaxes suppression in dense regions to avoid losing detections |
| 8 | DIoU-NMS | `DIoUNMS` | Replace IoU with Distance-IoU: `diou = iou - (center_dist² / diagonal²)`; suppresses when `diou > threshold` | Better spatial awareness; prefers keeping masks whose centers are far apart even if bounding boxes overlap |
| 9 | Matrix NMS | `MatrixNMS` | Compute full IoU matrix, derive suppression probability from "already suppressed" probability (parallel, no sequential loop) | GPU-efficient batched suppression; significant speedup for large mask counts (>50 per image) |
| 10 | Mask-Merge NMS | `MaskMergeNMS` | When `iou > merge_threshold`, combine masks of same class into union mask instead of suppressing either | Fragmented SAM outputs where one object is split into multiple segments; produces complete object coverage |

- **Config additions** in `PostProcessingConfig`:
  ```yaml
  post_processing:
    overlap_strategy: "confidence"   # any of the 10 strategy names
    soft_nms_sigma: 0.5              # existing — for Gaussian Soft-NMS
    linear_decay_threshold: 0.3      # new — for Linear Soft-NMS
    merge_iou_threshold: 0.7         # new — for Mask-Merge NMS
    adaptive_density_radius: 50      # new — pixel radius for density calculation
    enable_class_specific: false     # new — toggle for class-specific NMS
  ```

- **Registration**: Each new strategy self-registers via `@NMSStrategyFactory.register("strategy_name")` decorator

##### Step 4d — Class Registry & N-Class Support (Many-to-One Remapping)

The system currently supports N classes architecturally (via `config.model.prompts` list), but the mapping from prompt index → output class ID is implicit and undocumented. The class remapping module makes this explicit, validated, and extensible — including **many-to-one merging** where multiple prompts collapse into fewer output classes.

**Design principle**: The end user ONLY specifies name-to-name mappings in `config.yaml`. The system automatically computes all numeric class IDs — output ID assignment, prompt-index-to-output-ID mapping, and YOLO format numbering are all handled internally by `ClassRegistry`. No numeric IDs in the config.

**Pipeline integration constraint**: Remapping is applied as a dedicated pipeline stage between `segment()` and `NMS()`. The segmentor returns raw prompt indices. The remap stage converts them to output class IDs. NMS and all downstream stages only ever see remapped output class IDs.

```
segment() → raw prompt indices (0..N-1)
    ↓
remap()  → output class IDs (0..M-1)  ← ClassRegistry.remap_prompt_index()
    ↓
NMS()    → operates on output class IDs only
    ↓
filter() → annotate() → upload() → validate()
```

**Concrete example — 5 prompts → 2 output classes:**
```
prompts: ["teacher", "student", "kid", "child", "Adult"]

User config (name-to-name only):
  class_remapping:
    Adult: "teacher"
    kid: "student"
    child: "student"

System auto-computes:
  Output classes: ["teacher", "student"]  (2 unique, not 5)
  num_classes: 2
  prompt_to_class: {0: "teacher", 1: "student", 2: "student", 3: "student", 4: "teacher"}
  prompt_index_to_output_id: {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}  ← auto-assigned
  YOLO names: {0: "teacher", 1: "student"}  ← auto-assigned
```

When SAM3 returns `class_id=4` (prompt index for "Adult"), the remap stage converts it to output `class_id=0` ("teacher") before NMS runs. When SAM3 returns `class_id=2` (prompt index for "kid"), it becomes output `class_id=1` ("student"). NMS never sees prompt index 4 or 2 — it only sees output IDs 0 and 1.

- **Create** `src/class_registry.py`:
  ```python
  class ClassRegistry:
      """Single source of truth for class names, IDs, and many-to-one remapping.
      
      Supports N input prompts merging into M output classes (M <= N).
      Multiple prompts can map to the same output class name.
      All numeric IDs are auto-computed — the user only provides name mappings.
      
      Responsibilities:
        - Parse class names from config.model.prompts
        - Apply many-to-one class_remapping (multiple prompts → same output class)
        - Auto-compute deduplicated output class list and sequential IDs
        - Auto-compute prompt index → output class ID mapping
        - Generate YOLO-compatible names dict for data.yaml
      
      Properties:
        - input_prompts: List[str] — raw prompts sent to SAM3 (all N of them)
        - class_names: List[str] — deduplicated ordered output class names (M unique)
        - class_ids: List[int] — auto-assigned output class IDs [0, 1, ..., M-1]
        - num_classes: int — M (number of unique output classes, may be < len(prompts))
        - prompt_to_class: Dict[int, str] — SAM3 prompt index → output class name (many-to-one)
        - prompt_index_to_output_id: Dict[int, int] — SAM3 prompt index → output class ID (many-to-one, auto-computed)
        - name_to_id: Dict[str, int] — output class name → output class ID (auto-computed)
        - id_to_name: Dict[int, str] — output class ID → output class name (auto-computed)
      
      Methods:
        - from_config(config) -> ClassRegistry (factory)
        - remap_class_name(raw_name: str) -> str — apply name remapping
        - remap_prompt_index(prompt_idx: int) -> int — SAM3 prompt index → output class ID
        - get_yolo_names() -> Dict[int, str]  # {0: "teacher", 1: "student"} (auto-computed)
        - validate() -> None  # raises ValueError on inconsistencies
        - to_dict() -> Dict — serializable for IPC (multiprocessing)
        - from_dict(data: Dict) -> ClassRegistry — deserialize from IPC
  ```

- **Config additions** in `config.yaml` — user-facing, names only, no numeric IDs:
  ```yaml
  model:
    prompts:
      - "teacher"
      - "student"
      - "kid"
      - "child"
      - "Adult"
    # Optional: many-to-one remapping — merge multiple prompts into fewer output classes
    # Keys are prompt names, values are target output class names  
    # Prompts not listed here keep their name as-is (identity mapping)
    # All numeric class IDs are auto-computed by the system
    class_remapping:          # default: null (all prompts become output classes 1:1)
      Adult: "teacher"        # "Adult" prompt output → "teacher" class
      kid: "student"          # "kid" prompt output → "student" class
      child: "student"        # "child" prompt output → "student" class
      # "teacher" and "student" are not listed → they keep their names unchanged
  ```

  Note: There is NO `class_id_remapping` in the config. The system auto-assigns sequential IDs (0, 1, 2, ...) to deduplicated output class names in first-seen order. This keeps the config simple — the user only thinks in class names, never in numbers.

- **Remapping logic** (executed in `ClassRegistry.from_config()`):
  1. Read `config.model.prompts` → `["teacher", "student", "kid", "child", "Adult"]`
  2. Apply `class_remapping` to each prompt name:
     - `"teacher"` → no entry → `"teacher"` (identity)
     - `"student"` → no entry → `"student"` (identity)
     - `"kid"` → `"student"` (remapped)
     - `"child"` → `"student"` (remapped)
     - `"Adult"` → `"teacher"` (remapped)
  3. Deduplicate output names preserving first-seen order → `["teacher", "student"]`
  4. Auto-assign output IDs sequentially: `{"teacher": 0, "student": 1}`
  5. Auto-build `prompt_index_to_output_id`: `{0: 0, 1: 1, 2: 1, 3: 1, 4: 0}`
  6. Auto-build `get_yolo_names()`: `{0: "teacher", 1: "student"}`

- **Integration points** — replace hardcoded class handling across all modules:

  | Module | Current State | Change |
  |--------|--------------|--------|
  | `config_manager.py` | `class_priority` defaults to `["teacher", "student"]` (L140) | Default to `None`; populate from `ClassRegistry.class_names` at runtime |
  | `post_processor.py` | `class_names` defaults to `["teacher", "student"]` (L39, L353) | Accept `ClassRegistry` in constructor; derive `class_names` and `class_priority` from registry. NMS only sees remapped output IDs. |
  | `sam3_segmentor.py` | Uses `config.model.prompts` directly; returns raw prompt index as `class_id` | Returns raw prompt indices. Does NOT remap — that's the pipeline's remap stage responsibility. |
  | `pipeline.py` | No remap stage exists | Add `remap()` stage between `segment()` and `NMS()`. Calls `ClassRegistry.remap_prompt_index()` on every `class_id` in every `SegmentationResult`. |
  | `annotation_writer.py` | `class_names` from config used for `data.yaml` + `_classes.txt` | Use `ClassRegistry.get_yolo_names()` for YOLO format output; `num_classes` = `registry.num_classes` |
  | `result_filter.py` | Hardcoded "teacher/student" in manifest.txt header comments | Use `ClassRegistry.class_names` for header generation |
  | `scripts/add_class_files.py` | Hardcoded `class_names = ["teacher", "student"]` (L56) | Read from `ClassRegistry.from_config(config)` |
  | `parallel_processor.py` | Receives class info via serialized config dict | Serialize `ClassRegistry.to_dict()` in config for IPC; reconstruct via `ClassRegistry.from_dict()` in worker. Remap applied in worker after inference. |
  | `tests/test_annotation_writer.py` | Hardcodes `nc == 2` and teacher/student list | Parametrize with N input prompts / M output classes via `ClassRegistry` |

- **Validation rules** in `ClassRegistry.validate()`:
  - `len(prompts) > 0` — at least one class required
  - All prompt names are non-empty strings
  - If `class_remapping` provided: all keys must be existing prompt names (typo protection), all values must be non-empty strings
  - Many-to-one is explicitly allowed: multiple keys can map to the same value
  - Remapping target values should either be an existing (unmapped) prompt name OR appear as the target of another remapping — warn otherwise (potential typo, e.g., `kid: "studnet"`)
  - If `class_remapping` provided: at least one output class must exist after deduplication (no empty output)
  - No duplicate output class names after remapping (catches conflicting config)

---

**Step 5 — GPU Strategy System** (`src/gpu_strategy.py` + `src/parallel_processor.py`)

- **Create** `src/gpu_strategy.py`:
  - `GPUStrategy` ABC with methods: `initialize()`, `get_device_for_worker(worker_id)`, `cleanup()`
  - `CPUOnlyStrategy`: Returns `"cpu"` for all workers; uses `ThreadPoolExecutor` for I/O-bound tasks, `ProcessPoolExecutor` for CPU-bound
  - `SingleGPUMultiProcess`: All workers share one GPU; uses `multiprocessing.Pool` with `spawn` context; auto-selects process count based on GPU memory
  - `MultiGPUDDP`: Uses `torch.distributed` with NCCL backend; each rank owns one GPU; wraps model in `DistributedDataParallel`; uses `DistributedSampler` for data sharding
  - `auto_select_strategy(config)`: Detects available GPUs → 0 GPUs = CPU, 1 GPU = SingleGPU, 2+ GPUs = DDP
  - Task-type awareness: I/O tasks (scan, write) → threading; compute tasks (inference) → process/DDP

- **Refactor** `src/parallel_processor.py`:
  - Remove global mutable state (`_worker_segmentor`, `_worker_filter`, `_worker_writer`) — use proper worker class with instance state
  - Accept `GPUStrategy` via DI, use `strategy.get_device_for_worker(worker_id)` in worker init
  - Implement `Processor` protocol
  - Wire up `ProcessingTask` and `ProcessingResult` dataclasses (currently defined but unused — raw tuples used instead)
  - Delete `process_batch_async()` if truly unused, or wire it up

- **Add to** `src/config_manager.py`:
  - New `GPUConfig` dataclass: `strategy: str` (`"auto"`, `"single_gpu"`, `"multi_gpu_ddp"`, `"cpu"`), `devices: list[str]`, `workers_per_gpu: int`, `memory_threshold: float`

---

**Step 6 — Per-Module Progress Bars** (`src/progress_display.py`)

- **Create** `src/progress_display.py`:
  - `ModuleProgressManager` using `rich.progress.Progress` with `rich.live.Live`
  - Each pipeline stage gets its own named progress bar: `[Scan]`, `[Preprocess]`, `[Segment]`, `[Remap]`, `[NMS]`, `[Filter]`, `[Annotate]`, `[Upload]`, `[Validate]`
  - Each bar shows: `[stage] ████████░░ 156/400 images | 12.3 img/s | ETA 00:19 | GPU: 4.2GB`
  - Wire the unused `estimate_eta()` from utils.py into the progress display
  - `ProgressCallback` protocol: modules call `on_item_start(id)`, `on_item_complete(id)`, `on_item_error(id, err)` — display listens
  - Separate from `ProgressTracker` (SQLite persistence) — display is ephemeral, tracker is durable

- **Refactor** `src/progress_tracker.py`:
  - Wire up the `Status` enum (currently defined but raw strings used instead)
  - Add per-module stage tracking columns to the `images` table: `current_stage TEXT`
  - Emit events that `ModuleProgressManager` can subscribe to
  - Keep SQLite persistence, add a `ProgressCallback` hook

- **Refactor** `src/pipeline.py`:
  - Remove the single monolithic `tqdm` bar
  - Replace with `ModuleProgressManager` calls at each stage boundary

---

**Step 7 — Per-Module CLI Entry Points**

- **Create** `src/cli/` package with one file per module (listed in section A above)
- Each CLI module:
  - Uses `argparse` with clear `--help`
  - Initializes `LoggingSystem` first
  - Creates its own `Rich` progress bar
  - Loads only the config sections it needs (ISP compliance)
  - Can run completely standalone or be called by the pipeline orchestrator
- Common CLI args shared via a `add_common_args(parser)` helper: `--config`, `--log-level`, `--device`, `--job-name`

- **Refactor** `setup.py`:
  - Register all console_scripts:
    ```
    sam3-preprocess, sam3-segment, sam3-postprocess, sam3-filter,
    sam3-annotate, sam3-validate, sam3-upload, sam3-download,
    sam3-pipeline, sam3-progress
    ```

- **Refactor** all existing scripts in `scripts/`:
  - `scripts/run_pipeline.py` → thin wrapper calling `src/cli/pipeline.py`
  - `scripts/run_validator.py` → thin wrapper calling `src/cli/validate.py`
  - `scripts/download_model.py` → thin wrapper calling `src/cli/download.py`
  - `scripts/add_class_files.py` → refactor from `sys.argv[1]` to argparse, move to `src/cli/annotate.py` as a subcommand

---

**Step 8 — ISP Fix: Config Slicing**

- **Refactor** `src/config_manager.py`:
  - Each module receives only its relevant config dataclass, not the full `Config` object
  - `ImagePreprocessor(config.pipeline)` instead of `ImagePreprocessor(config)`
  - `ResultFilter(config.pipeline.output_dir, config.pipeline.neither_dir)` instead of `ResultFilter(config)`
  - `SAM3Segmentor(config.model, config.post_processing)` instead of `SAM3Segmentor(config)`
  - Add the new `GPUConfig`, `LoggingConfig` dataclasses

---

**Step 9 — Remaining SOLID Fixes**

- **Refactor** `src/annotation_writer.py`:
  - Extract `mask_to_polygon` / `masks_to_polygons` into a `MaskConverter` utility class (SRP)
  - Extract `write_data_yaml` + `_classes.txt` generation into a `DatasetMetadataWriter` class (SRP)
  - `AnnotationWriter` keeps only `write_annotation()` and `_setup_directories()`
  - Fix the `val` vs `valid` directory naming inconsistency (and fix tests)

- **Refactor** `src/validator.py`:
  - Remove `print()` statements — use logger instead (separates business logic from presentation)
  - Extract SQLite caching into a `ValidationCache` class (SRP)
  - `Validator` becomes pure comparison logic

- **Refactor** `src/roboflow_uploader.py`:
  - Extract background queue management into a generic `AsyncWorkerPool`
  - `DistributedUploader` keeps only upload-specific logic

---

**Step 10 — Dead Code Cleanup & Test Fixes**

- Remove or wire up all dead code identified:
  - Delete: `ProcessingTask`/`ProcessingResult` (if still unused after Step 5), `process_batch_async`, `detect_input_mode`, `get_image_info`, `write_empty_annotation`, `resize_with_padding`/`reverse_transform_coordinates`, `reset_stats` on both classes, duplicate `return` in validator
  - Wire up: `estimate_eta` (into progress display), `create_post_processor` factory (into pipeline), `apply_class_specific_nms` (as config toggle), `Status` enum (replace raw strings)

- **Fix** `tests/test_annotation_writer.py`:
  - Fix `val` → `valid` directory name assertions
  - Fix `data['names']` assertion: expect `dict` not `list`

- **Add tests** for: `post_processor.py`, `result_filter.py`, `config_manager.py`, `gpu_strategy.py`, `logging_system.py`, `progress_display.py`, `class_registry.py`

---

### Execution Order

| Phase | Steps | Rationale | Gate |
|-------|-------|-----------|------|
| **Phase 1** | Steps 1 + 2 | Logging + Interfaces are foundational — everything else depends on them | All existing tests pass |
| **Phase 2** | Steps 3 + 4 (4a–4d) | Pipeline split + NMS fix + Strategy expansion + Class registry — core architectural changes | All existing + new NMS/class tests pass |
| **Phase 3** | Steps 5 + 6 | GPU strategy + Progress display — infrastructure features | All tests pass; GPU smoke test on both CPU and CUDA |
| **Phase 4** | Steps 7 + 8 | CLI entry points + Config slicing — user-facing improvements | All CLIs respond to `--help`; all tests pass |
| **Phase 5** | Steps 9 + 10 | Remaining SOLID fixes + cleanup — polish | Full test suite green; zero dead code warnings |

---

### Verification

- Run all existing tests after each phase to ensure no regressions
- Run each new CLI entry point standalone: `sam3-preprocess --config config/config.yaml --help`
- Test multi-GPU with `CUDA_VISIBLE_DEVICES=0,1` and verify DDP initialization + cleanup
- Test single-GPU multi-process with `--workers 4 --device cuda:0`
- Verify Rich progress bars render correctly in terminal with `sam3-pipeline --job-name test`
- Verify logging output: check JSON log file contains correlation IDs, module names, durations
- Verify NMS: run `sam3-postprocess` standalone on pre-saved segmentation results, confirm correct IoU suppression for all 10 strategies
- Run `sam3-progress --job-name test` to verify per-module stage tracking in SQLite
- Verify class registry: test with 1, 2, 5, and 10 input prompts; test many-to-one merging (5→2); test identity (no remapping); test validation errors on typos
- Verify remap-before-NMS constraint: assert NMS input data contains only output class IDs (0..M-1), never raw prompt indices (0..N-1)

### Decisions

- **Each module gets its own CLI binary** (e.g., `sam3-preprocess`, `sam3-segment`) per user preference
- **GPU strategy**: Adaptive — single GPU uses multi-process sharing; multiple GPUs use DDP with NCCL; I/O tasks use threading instead of multiprocessing
- **Rich library** for progress bars, not tqdm
- **Clean extensible design** with protocol-based interfaces and config-driven behavior, using SQLite for tracking persistence
- **NMS decoupled** from segmentor — becomes its own pipeline stage with standalone CLI
- **10 NMS strategies** via OCP-compliant Strategy Pattern: Confidence, Area, Class Priority, Gaussian Soft-NMS, Linear Soft-NMS, Weighted, Adaptive, DIoU, Matrix, Mask-Merge
- **Post-processing factory** and **class-specific NMS** wired up (currently dead code)
- **Class registry** centralizes all class handling — N-class support with **many-to-one remapping** (e.g., 5 prompts → 2 output classes: Adult→teacher, kid/child→student)
- **Remapping before NMS** — mandatory pipeline ordering: `segment() → remap() → NMS()`. NMS never sees raw prompt indices.
- **No numeric IDs in config** — user only specifies name-to-name `class_remapping`. System auto-computes all class IDs.
- **Phase delivery gated** by passing tests — no phase starts until previous phase is green
- **Dev=Windows, prod=Linux** — all path handling via `pathlib.Path`, no OS-specific assumptions
