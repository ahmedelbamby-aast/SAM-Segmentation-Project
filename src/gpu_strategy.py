"""GPU strategy module for the SAM 3 Segmentation Pipeline.

Provides an OCP-compliant ``GPUStrategy`` ABC with three concrete
implementations and an ``auto_select_strategy()`` factory that picks the
right one based on available hardware.

``DeviceManager`` is **migrated here from** ``src/sam3_segmentor.py`` (Phase 3).
``sam3_segmentor.py`` re-exports it for backward compatibility during the
transition, but new code should import from here.

Strategy selection logic:
  - 0 GPUs → :class:`CPUOnlyStrategy`
  - 1 GPU  → :class:`SingleGPUMultiProcess` (multiple worker processes share GPU)
  - 2+ GPUs → :class:`MultiGPUDDP` (one rank per GPU, NCCL backend)

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""

from __future__ import annotations

import gc
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .logging_system import LoggingSystem, trace

_logger = LoggingSystem.get_logger(__name__)


# ---------------------------------------------------------------------------
# DeviceManager (migrated from sam3_segmentor.py)
# ---------------------------------------------------------------------------


class DeviceManager:
    """Static helpers for device detection and memory reporting.

    This class is migrated to ``src/gpu_strategy.py`` in Phase 3.
    ``src/sam3_segmentor.py`` re-exports it as a backward-compat alias.
    """

    @staticmethod
    @trace
    def get_available_devices() -> Dict[str, Any]:
        """Detect available compute devices.

        Returns:
            Dict with keys ``cuda_available``, ``cuda_device_count``,
            ``cuda_devices`` (list of dicts), ``recommended_device``.
        """
        info: Dict[str, Any] = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "recommended_device": "cpu",
        }
        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                count = torch.cuda.device_count()
                info["cuda_device_count"] = count
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    info["cuda_devices"].append(
                        {
                            "id": i,
                            "name": props.name,
                            "memory_gb": props.total_memory / 1e9,
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )
                if count > 0:
                    info["recommended_device"] = "cuda:0"
        except ImportError:
            _logger.warning("PyTorch not found — defaulting to CPU")
        except Exception as exc:
            _logger.warning("Error detecting CUDA devices: %s", exc)
        return info

    @staticmethod
    @trace
    def resolve_device(device_config: str) -> str:
        """Resolve device string from config (handles ``"auto"``).

        Args:
            device_config: Value from ``config.model.device``.

        Returns:
            Resolved device string (e.g. ``"cpu"``, ``"cuda:0"``).
        """
        if device_config == "auto":
            info = DeviceManager.get_available_devices()
            device = info["recommended_device"]
            _logger.info("Auto-detected device: %s", device)
            return device
        if device_config.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    _logger.warning("CUDA requested but unavailable — falling back to CPU")
                    return "cpu"
            except ImportError:
                _logger.warning("PyTorch not found — falling back to CPU")
                return "cpu"
        return device_config

    @staticmethod
    @trace
    def get_memory_info() -> Dict[str, float]:
        """Return current CPU RAM and per-GPU memory stats.

        Returns:
            Dict with keys ``cpu_ram_gb``, ``cpu_ram_total_gb``,
            ``cpu_ram_percent``,  ``cuda:N_allocated_gb``,
            ``cuda:N_reserved_gb``, ``cuda:N_total_gb``, ``cuda:N_free_gb``.
        """
        info: Dict[str, float] = {"cpu_ram_gb": 0.0}
        try:
            import psutil

            mem = psutil.virtual_memory()
            info["cpu_ram_gb"] = mem.available / 1e9
            info["cpu_ram_total_gb"] = mem.total / 1e9
            info["cpu_ram_percent"] = float(mem.percent)
        except ImportError:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    total = torch.cuda.get_device_properties(i).total_memory / 1e9
                    info[f"cuda:{i}_allocated_gb"] = allocated
                    info[f"cuda:{i}_reserved_gb"] = reserved
                    info[f"cuda:{i}_total_gb"] = total
                    info[f"cuda:{i}_free_gb"] = total - reserved
        except ImportError:
            pass
        return info

    @staticmethod
    @trace
    def maybe_gc(process_count: int, gc_interval: int, is_cpu: bool) -> None:
        """Run garbage collection periodically.

        Args:
            process_count: Number of images processed so far.
            gc_interval: Run GC every *gc_interval* images.
            is_cpu: ``True`` if running on CPU (torch cache not cleared).
        """
        if process_count % gc_interval == 0:
            gc.collect()
            if not is_cpu:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# GPUStrategy ABC
# ---------------------------------------------------------------------------


class GPUStrategy(ABC):
    """Abstract base for GPU/CPU device assignment strategies.

    Concrete subclasses implement different parallelism models:
    - :class:`CPUOnlyStrategy` — CPU-only, no GPU
    - :class:`SingleGPUMultiProcess` — multi-process sharing one GPU
    - :class:`MultiGPUDDP` — torch.distributed DDP across N GPUs

    To add a new strategy: subclass ``GPUStrategy`` and implement
    ``initialize()``, ``get_device_for_worker()``, and ``cleanup()``.
    Register via :class:`GPUStrategyFactory` if auto-selection is desired.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Set up the GPU process group / environment variables."""

    @abstractmethod
    def get_device_for_worker(self, worker_id: int) -> str:
        """Return the device string a given worker should use.

        Args:
            worker_id: Zero-based worker index.

        Returns:
            Device string such as ``"cpu"``, ``"cuda:0"``, ``"cuda:1"``.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Tear down the GPU process group / release resources."""

    @property
    @abstractmethod
    def num_workers(self) -> int:
        """Number of parallel worker processes/ranks this strategy manages."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend descriptor string (e.g. ``"cpu"``, ``"single_gpu"``, ``"ddp_nccl"``)."""


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class CPUOnlyStrategy(GPUStrategy):
    """CPU-only strategy for environments without a GPU.

    All workers run on ``"cpu"``.  Uses ``ProcessPoolExecutor`` for
    CPU-bound inference tasks and ``ThreadPoolExecutor`` for I/O tasks.

    Args:
        workers: Number of parallel CPU worker processes (default: 2).
    """

    def __init__(self, workers: int = 2) -> None:
        self._workers = max(1, workers)

    @trace
    def initialize(self) -> None:
        _logger.info("CPUOnlyStrategy: using %d CPU workers", self._workers)

    @trace
    def get_device_for_worker(self, worker_id: int) -> str:
        return "cpu"

    @trace
    def cleanup(self) -> None:
        _logger.info("CPUOnlyStrategy: cleanup (no resources to release)")

    @property
    def num_workers(self) -> int:
        return self._workers

    @property
    def backend(self) -> str:
        return "cpu"


class SingleGPUMultiProcess(GPUStrategy):
    """Multiple worker processes sharing a single GPU.

    All workers receive the same device string (e.g. ``"cuda:0"``).
    Worker count is bounded by available GPU memory and
    ``config.gpu.workers_per_gpu``.

    Args:
        device: GPU device string, e.g. ``"cuda:0"``.
        workers_per_gpu: Number of worker processes sharing the GPU.
        memory_threshold: Fraction of GPU memory reserved for workers
            (0..1).  Workers are capped so total allocation stays below
            this threshold.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        workers_per_gpu: int = 2,
        memory_threshold: float = 0.85,
    ) -> None:
        self._device = device
        self._workers_per_gpu = max(1, workers_per_gpu)
        self._memory_threshold = memory_threshold

    @trace
    def initialize(self) -> None:
        _logger.info(
            "SingleGPUMultiProcess: device=%s, workers=%d, mem_threshold=%.0f%%",
            self._device, self._workers_per_gpu, self._memory_threshold * 100,
        )

    @trace
    def get_device_for_worker(self, worker_id: int) -> str:
        return self._device

    @trace
    def cleanup(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        _logger.info("SingleGPUMultiProcess: GPU cache cleared")

    @property
    def num_workers(self) -> int:
        return self._workers_per_gpu

    @property
    def backend(self) -> str:
        return "single_gpu"

    @property
    def device(self) -> str:
        """The GPU device shared by all workers."""
        return self._device


class MultiGPUDDP(GPUStrategy):
    """Distributed Data Parallel strategy using ``torch.distributed`` + NCCL.

    One rank per GPU.  Workers receive ``"cuda:rank_id"`` device strings.
    The caller is responsible for spawning worker processes with proper
    ``MASTER_ADDR`` / ``MASTER_PORT`` environment variables before calling
    :meth:`initialize`.

    Args:
        devices: List of GPU device strings, e.g. ``["cuda:0", "cuda:1"]``.
        master_addr: Hostname for the DDP rendezvous (default: ``"localhost"``).
        master_port: Port for the DDP rendezvous (default: ``"12355"``).
    """

    def __init__(
        self,
        devices: Optional[List[str]] = None,
        master_addr: str = "localhost",
        master_port: str = "12355",
    ) -> None:
        self._devices: List[str] = devices or ["cuda:0", "cuda:1"]
        self._master_addr = master_addr
        self._master_port = master_port
        self._initialized = False

    @trace
    def initialize(self) -> None:
        os.environ.setdefault("MASTER_ADDR", self._master_addr)
        os.environ.setdefault("MASTER_PORT", self._master_port)
        try:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
                self._initialized = True
                _logger.info(
                    "MultiGPUDDP: process group initialized — %d ranks, devices=%s",
                    len(self._devices), self._devices,
                )
        except ImportError:
            raise RuntimeError("torch.distributed is required for MultiGPUDDP strategy") from None
        except Exception as exc:
            _logger.error("MultiGPUDDP initialization failed: %s", exc)
            raise RuntimeError(f"DDP init failed: {exc}") from exc

    @trace
    def get_device_for_worker(self, worker_id: int) -> str:
        if worker_id < 0 or worker_id >= len(self._devices):
            raise ValueError(
                f"worker_id {worker_id} out of range — "
                f"available devices: {self._devices}"
            )
        return self._devices[worker_id]

    @trace
    def cleanup(self) -> None:
        if self._initialized:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
                    _logger.info("MultiGPUDDP: process group destroyed")
            except ImportError:
                pass
            self._initialized = False

    @property
    def num_workers(self) -> int:
        return len(self._devices)

    @property
    def backend(self) -> str:
        return "ddp_nccl"

    @property
    def devices(self) -> List[str]:
        """Ordered list of device strings assigned to ranks."""
        return list(self._devices)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@trace
def auto_select_strategy(config: Any) -> GPUStrategy:
    """Factory — choose the best ``GPUStrategy`` based on available hardware.

    Decision logic:
    - ``config.gpu.strategy != "auto"`` → honour explicit strategy name
    - 0 CUDA devices → :class:`CPUOnlyStrategy`
    - 1 CUDA device → :class:`SingleGPUMultiProcess`
    - 2+ CUDA devices → :class:`MultiGPUDDP`

    Args:
        config: Full ``Config`` object (only ``config.gpu`` slice is read).

    Returns:
        A :class:`GPUStrategy` instance.

    Raises:
        ValueError: If an unknown ``strategy`` name is given.
    """
    gpu_cfg = getattr(config, "gpu", None)
    strategy_name: str = getattr(gpu_cfg, "strategy", "auto")
    workers_per_gpu: int = getattr(gpu_cfg, "workers_per_gpu", 2)
    devices_cfg: List[str] = getattr(gpu_cfg, "devices", [])
    mem_threshold: float = getattr(gpu_cfg, "memory_threshold", 0.85)

    # Honour explicit strategy override
    if strategy_name == "cpu":
        _logger.info("GPU strategy: forced CPU")
        return CPUOnlyStrategy(workers=workers_per_gpu)
    if strategy_name == "single_gpu":
        device = devices_cfg[0] if devices_cfg else "cuda:0"
        _logger.info("GPU strategy: forced single_gpu (%s)", device)
        return SingleGPUMultiProcess(device, workers_per_gpu, mem_threshold)
    if strategy_name == "multi_gpu_ddp":
        devs = devices_cfg if devices_cfg else ["cuda:0", "cuda:1"]
        _logger.info("GPU strategy: forced multi_gpu_ddp (%s)", devs)
        return MultiGPUDDP(devs)
    if strategy_name not in ("auto",):
        raise ValueError(
            f"Unknown GPU strategy {strategy_name!r}. "
            "Valid values: 'auto', 'cpu', 'single_gpu', 'multi_gpu_ddp'."
        )

    # Auto-detect
    hw = DeviceManager.get_available_devices()
    n_gpu = hw["cuda_device_count"]

    if n_gpu == 0:
        _logger.info("GPU strategy (auto): no CUDA devices → CPUOnlyStrategy")
        return CPUOnlyStrategy(workers=workers_per_gpu)
    if n_gpu == 1:
        device = devices_cfg[0] if devices_cfg else "cuda:0"
        _logger.info("GPU strategy (auto): 1 GPU → SingleGPUMultiProcess(%s)", device)
        return SingleGPUMultiProcess(device, workers_per_gpu, mem_threshold)

    # 2+ GPUs
    if devices_cfg:
        devs = devices_cfg
    else:
        devs = [f"cuda:{i}" for i in range(n_gpu)]
    _logger.info("GPU strategy (auto): %d GPUs → MultiGPUDDP(%s)", n_gpu, devs)
    return MultiGPUDDP(devs)
