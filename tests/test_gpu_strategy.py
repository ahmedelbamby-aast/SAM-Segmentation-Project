"""Unit tests for src/gpu_strategy.py.

Tests cover:
- DeviceManager static helpers
- CPUOnlyStrategy lifecycle and device assignment
- SingleGPUMultiProcess lifecycle and device assignment
- MultiGPUDDP lifecycle, device assignment, and error cases
- auto_select_strategy factory (all branches)
- GPUStrategy abstractness (cannot instantiate directly)

Author: Ahmed Hany ElBamby
Date: 23-02-2026
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from src.gpu_strategy import (
    CPUOnlyStrategy,
    DeviceManager,
    GPUStrategy,
    MultiGPUDDP,
    SingleGPUMultiProcess,
    auto_select_strategy,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_gpu_config(
    strategy: str = "auto",
    devices: list = None,
    workers_per_gpu: int = 2,
    memory_threshold: float = 0.85,
) -> MagicMock:
    """Return a mock GPUConfig-like object."""
    cfg = MagicMock()
    cfg.strategy = strategy
    cfg.devices = devices or []
    cfg.workers_per_gpu = workers_per_gpu
    cfg.memory_threshold = memory_threshold
    return cfg


def _make_config(gpu_cfg: MagicMock = None) -> MagicMock:
    """Return a mock top-level config with a .gpu attribute."""
    config = MagicMock()
    config.gpu = gpu_cfg or _make_gpu_config()
    return config


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------


class TestDeviceManager:
    """Tests for DeviceManager static helpers."""

    def test_get_available_devices_returns_cpu_when_no_torch(self) -> None:
        """If torch is not importable, result reports CPU-only."""
        with patch.dict("sys.modules", {"torch": None}):
            info = DeviceManager.get_available_devices()
        assert info["recommended_device"] == "cpu"
        assert info["cuda_available"] is False

    def test_get_available_devices_returns_dict_with_required_keys(self) -> None:
        """Result always contains the required keys."""
        info = DeviceManager.get_available_devices()
        for key in ("cuda_available", "cuda_device_count", "cuda_devices", "recommended_device"):
            assert key in info

    def test_resolve_device_auto_returns_string(self) -> None:
        """resolve_device('auto') always returns a string."""
        device = DeviceManager.resolve_device("auto")
        assert isinstance(device, str)
        assert device in ("cpu",) or device.startswith("cuda")

    def test_resolve_device_cpu(self) -> None:
        """resolve_device('cpu') always returns 'cpu'."""
        assert DeviceManager.resolve_device("cpu") == "cpu"

    def test_resolve_device_cuda_fallback_to_cpu_when_unavailable(self) -> None:
        """If CUDA is not available, cuda device string falls back to cpu."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            device = DeviceManager.resolve_device("cuda:0")
        assert device == "cpu"

    def test_resolve_device_cuda_no_torch(self) -> None:
        """resolve_device('cuda:0') falls back to cpu if torch missing."""
        with patch.dict("sys.modules", {"torch": None}):
            device = DeviceManager.resolve_device("cuda:0")
        assert device == "cpu"

    def test_get_memory_info_returns_dict(self) -> None:
        """get_memory_info always returns a dict."""
        info = DeviceManager.get_memory_info()
        assert isinstance(info, dict)
        assert "cpu_ram_gb" in info

    def test_maybe_gc_does_not_raise(self) -> None:
        """maybe_gc should not raise for any inputs."""
        DeviceManager.maybe_gc(process_count=100, gc_interval=100, is_cpu=True)
        DeviceManager.maybe_gc(process_count=99, gc_interval=100, is_cpu=True)
        DeviceManager.maybe_gc(process_count=100, gc_interval=100, is_cpu=False)


# ---------------------------------------------------------------------------
# CPUOnlyStrategy
# ---------------------------------------------------------------------------


class TestCPUOnlyStrategy:
    """Tests for CPUOnlyStrategy."""

    def test_backend_is_cpu(self) -> None:
        s = CPUOnlyStrategy()
        assert s.backend == "cpu"

    def test_workers_default_to_2(self) -> None:
        s = CPUOnlyStrategy()
        assert s.num_workers == 2

    def test_workers_custom(self) -> None:
        s = CPUOnlyStrategy(workers=4)
        assert s.num_workers == 4

    def test_workers_minimum_one(self) -> None:
        s = CPUOnlyStrategy(workers=0)
        assert s.num_workers == 1

    def test_get_device_for_any_worker_is_cpu(self) -> None:
        s = CPUOnlyStrategy(workers=4)
        for i in range(6):
            assert s.get_device_for_worker(i) == "cpu"

    def test_initialize_does_not_raise(self) -> None:
        s = CPUOnlyStrategy()
        s.initialize()  # should not raise

    def test_cleanup_does_not_raise(self) -> None:
        s = CPUOnlyStrategy()
        s.initialize()
        s.cleanup()  # should not raise

    def test_is_subclass_of_gpu_strategy(self) -> None:
        assert issubclass(CPUOnlyStrategy, GPUStrategy)


# ---------------------------------------------------------------------------
# SingleGPUMultiProcess
# ---------------------------------------------------------------------------


class TestSingleGPUMultiProcess:
    """Tests for SingleGPUMultiProcess."""

    def test_backend_is_single_gpu(self) -> None:
        s = SingleGPUMultiProcess("cuda:0")
        assert s.backend == "single_gpu"

    def test_all_workers_get_same_device(self) -> None:
        s = SingleGPUMultiProcess(device="cuda:0", workers_per_gpu=4)
        for i in range(6):
            assert s.get_device_for_worker(i) == "cuda:0"

    def test_num_workers_equals_workers_per_gpu(self) -> None:
        s = SingleGPUMultiProcess("cuda:0", workers_per_gpu=3)
        assert s.num_workers == 3

    def test_minimum_one_worker(self) -> None:
        s = SingleGPUMultiProcess("cuda:0", workers_per_gpu=0)
        assert s.num_workers == 1

    def test_device_property(self) -> None:
        s = SingleGPUMultiProcess("cuda:1")
        assert s.device == "cuda:1"

    def test_initialize_does_not_raise(self) -> None:
        s = SingleGPUMultiProcess("cuda:0")
        s.initialize()

    def test_cleanup_calls_torch_cache_clear(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            s = SingleGPUMultiProcess("cuda:0")
            s.cleanup()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_cleanup_when_no_torch(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            s = SingleGPUMultiProcess("cuda:0")
            s.cleanup()  # must not raise


# ---------------------------------------------------------------------------
# MultiGPUDDP
# ---------------------------------------------------------------------------


class TestMultiGPUDDP:
    """Tests for MultiGPUDDP."""

    def test_backend_is_ddp_nccl(self) -> None:
        s = MultiGPUDDP(["cuda:0", "cuda:1"])
        assert s.backend == "ddp_nccl"

    def test_num_workers_equals_num_devices(self) -> None:
        s = MultiGPUDDP(["cuda:0", "cuda:1", "cuda:2"])
        assert s.num_workers == 3

    def test_device_assignment_round_robin(self) -> None:
        s = MultiGPUDDP(["cuda:0", "cuda:1"])
        assert s.get_device_for_worker(0) == "cuda:0"
        assert s.get_device_for_worker(1) == "cuda:1"

    def test_worker_id_out_of_range_raises(self) -> None:
        s = MultiGPUDDP(["cuda:0", "cuda:1"])
        with pytest.raises(ValueError, match="out of range"):
            s.get_device_for_worker(5)

    def test_negative_worker_id_raises(self) -> None:
        s = MultiGPUDDP(["cuda:0", "cuda:1"])
        with pytest.raises(ValueError):
            s.get_device_for_worker(-1)

    def test_devices_property_returns_copy(self) -> None:
        devs = ["cuda:0", "cuda:1"]
        s = MultiGPUDDP(devs)
        result = s.devices
        # mutating the result should not affect the strategy
        result.append("cuda:99")
        assert "cuda:99" not in s.devices

    def test_initialize_raises_when_torch_distributed_unavailable(self) -> None:
        with patch.dict("sys.modules", {"torch": None, "torch.distributed": None}):
            s = MultiGPUDDP(["cuda:0", "cuda:1"])
            with pytest.raises(RuntimeError):
                s.initialize()

    def test_cleanup_does_nothing_when_not_initialized(self) -> None:
        s = MultiGPUDDP(["cuda:0"])
        s.cleanup()  # should not raise even if never initialized

    def test_default_devices_are_two(self) -> None:
        s = MultiGPUDDP()
        assert len(s.devices) == 2


# ---------------------------------------------------------------------------
# GPUStrategy ABC
# ---------------------------------------------------------------------------


class TestGPUStrategyABC:
    """Tests verifying GPUStrategy cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            GPUStrategy()  # type: ignore[abstract]

    def test_concrete_classes_implement_all_abstract_methods(self) -> None:
        for cls in (CPUOnlyStrategy, SingleGPUMultiProcess, MultiGPUDDP):
            assert not getattr(cls, "__abstractmethods__", set()), (
                f"{cls.__name__} has unimplemented abstract methods"
            )


# ---------------------------------------------------------------------------
# auto_select_strategy factory
# ---------------------------------------------------------------------------


class TestAutoSelectStrategy:
    """Tests for the auto_select_strategy factory function."""

    def _hw(self, n_gpu: int) -> dict:
        return {
            "cuda_available": n_gpu > 0,
            "cuda_device_count": n_gpu,
            "cuda_devices": [{"id": i} for i in range(n_gpu)],
            "recommended_device": "cpu" if n_gpu == 0 else "cuda:0",
        }

    def test_explicit_cpu_strategy(self) -> None:
        config = _make_config(_make_gpu_config(strategy="cpu"))
        result = auto_select_strategy(config)
        assert isinstance(result, CPUOnlyStrategy)

    def test_explicit_single_gpu_strategy(self) -> None:
        config = _make_config(_make_gpu_config(strategy="single_gpu"))
        result = auto_select_strategy(config)
        assert isinstance(result, SingleGPUMultiProcess)

    def test_explicit_multi_gpu_ddp_strategy(self) -> None:
        config = _make_config(_make_gpu_config(strategy="multi_gpu_ddp"))
        result = auto_select_strategy(config)
        assert isinstance(result, MultiGPUDDP)

    def test_unknown_strategy_raises(self) -> None:
        config = _make_config(_make_gpu_config(strategy="unknown_strategy"))
        with pytest.raises(ValueError, match="Unknown GPU strategy"):
            auto_select_strategy(config)

    def test_auto_no_gpu_returns_cpu(self) -> None:
        config = _make_config(_make_gpu_config(strategy="auto"))
        with patch(
            "src.gpu_strategy.DeviceManager.get_available_devices",
            return_value=self._hw(0),
        ):
            result = auto_select_strategy(config)
        assert isinstance(result, CPUOnlyStrategy)

    def test_auto_one_gpu_returns_single_gpu(self) -> None:
        config = _make_config(_make_gpu_config(strategy="auto"))
        with patch(
            "src.gpu_strategy.DeviceManager.get_available_devices",
            return_value=self._hw(1),
        ):
            result = auto_select_strategy(config)
        assert isinstance(result, SingleGPUMultiProcess)

    def test_auto_two_gpus_returns_ddp(self) -> None:
        config = _make_config(_make_gpu_config(strategy="auto"))
        with patch(
            "src.gpu_strategy.DeviceManager.get_available_devices",
            return_value=self._hw(2),
        ):
            result = auto_select_strategy(config)
        assert isinstance(result, MultiGPUDDP)

    def test_auto_four_gpus_returns_ddp_with_four_devices(self) -> None:
        config = _make_config(_make_gpu_config(strategy="auto"))
        with patch(
            "src.gpu_strategy.DeviceManager.get_available_devices",
            return_value=self._hw(4),
        ):
            result = auto_select_strategy(config)
        assert isinstance(result, MultiGPUDDP)
        assert len(result.devices) == 4

    def test_single_gpu_uses_devices_from_config(self) -> None:
        config = _make_config(_make_gpu_config(strategy="single_gpu", devices=["cuda:1"]))
        result = auto_select_strategy(config)
        assert isinstance(result, SingleGPUMultiProcess)
        assert result.device == "cuda:1"

    def test_multi_gpu_uses_devices_from_config(self) -> None:
        config = _make_config(
            _make_gpu_config(strategy="multi_gpu_ddp", devices=["cuda:0", "cuda:1", "cuda:2"])
        )
        result = auto_select_strategy(config)
        assert isinstance(result, MultiGPUDDP)
        assert result.devices == ["cuda:0", "cuda:1", "cuda:2"]

    def test_cpu_workers_per_gpu_forwarded(self) -> None:
        config = _make_config(_make_gpu_config(strategy="cpu", workers_per_gpu=4))
        result = auto_select_strategy(config)
        assert isinstance(result, CPUOnlyStrategy)
        assert result.num_workers == 4

    def test_config_without_gpu_attribute_defaults_to_cpu(self) -> None:
        """If config has no .gpu, strategy must default gracefully."""
        config = MagicMock(spec=[])  # no attributes → getattr returns defaults
        # This internally resolves to strategy_name="auto", no gpu
        # DeviceManager will detect real hardware — just ensure no exception
        result = auto_select_strategy(config)
        assert isinstance(result, GPUStrategy)
