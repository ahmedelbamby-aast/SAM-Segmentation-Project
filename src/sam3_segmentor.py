"""SAM 3 segmentation wrapper for the segmentation pipeline.

Wraps ``SAM3SemanticPredictor`` (ultralytics) in a Protocol-compliant
:class:`Segmentor` interface.  This module does **NOT** perform NMS or class
remapping  those are separate pipeline stages executed after this one.

``process_image()`` returns a :class:`~src.interfaces.SegmentationResult`
whose :class:`~src.interfaces.MaskData` entries carry *raw SAM3 prompt
indices* (0..N-1) as ``class_id``.  The pipeline remap stage calls
``ClassRegistry.remap_prompt_index()`` before NMS starts.

Author: Ahmed Hany ElBamby
Date: 22-02-2026
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .logging_system import LoggingSystem, trace
from .interfaces import Segmentor, SegmentationResult, MaskData

_logger = LoggingSystem.get_logger(__name__)


class DeviceManager:
    """Static helpers for device detection and memory reporting.

    Note:
        This class will migrate to ``src/gpu_strategy.py`` in Phase 3.
        Kept here during Phase 2 to avoid breaking the pipeline.
    """

    @staticmethod
    def get_available_devices() -> Dict[str, Any]:
        """Detect available compute devices.

        Returns:
            Dict with keys ``cuda_available``, ``cuda_device_count``,
            ``cuda_devices`` (list), ``recommended_device``.
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
                info["cuda_device_count"] = torch.cuda.device_count()
                for i in range(info["cuda_device_count"]):
                    props = torch.cuda.get_device_properties(i)
                    info["cuda_devices"].append(
                        {
                            "id": i,
                            "name": props.name,
                            "memory_gb": props.total_memory / 1e9,
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )
                if info["cuda_device_count"] > 0:
                    info["recommended_device"] = "cuda:0"
        except ImportError:
            _logger.warning("PyTorch not found  defaulting to CPU")
        except Exception as exc:
            _logger.warning("Error detecting CUDA devices: %s", exc)
        return info

    @staticmethod
    def resolve_device(device_config: str) -> str:
        """Resolve the device string from config (handles ``"auto"``).

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
                    _logger.warning("CUDA requested but unavailable  falling back to CPU")
                    return "cpu"
            except ImportError:
                _logger.warning("PyTorch not found  falling back to CPU")
                return "cpu"
        return device_config

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Return current CPU RAM and per-GPU memory stats.

        Returns:
            Dict with keys like ``cpu_ram_gb``, ``cuda:0_free_gb``, etc.
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


class SAM3Segmentor:
    """SAM 3 text-prompt segmentor (Protocol: :class:`~src.interfaces.Segmentor`).

    Wraps ``SAM3SemanticPredictor`` with lazy model loading, CPU/GPU support,
    and periodic garbage collection.  NMS and class remapping are **not**
    applied here  they are separate downstream pipeline stages.

    Args:
        model_config: ``config.model`` slice (ModelConfig dataclass).
        pipeline_config: ``config.pipeline`` slice (PipelineConfig dataclass).
    """

    def __init__(self, model_config: Any, pipeline_config: Any) -> None:
        self._prompts: List[str] = model_config.prompts
        self._confidence: float = model_config.confidence
        self._model_path: Path = Path(model_config.path)
        self._resolution: int = pipeline_config.resolution
        self._device: str = DeviceManager.resolve_device(model_config.device)
        self._is_cpu: bool = self._device == "cpu"
        self._half_precision: bool = model_config.half_precision and not self._is_cpu
        if self._is_cpu and model_config.half_precision:
            _logger.info("FP16 not supported on CPU  using FP32")
        self._predictor = None
        self._model_loaded: bool = False
        self._gc_interval: int = 50 if self._is_cpu else 100
        self._process_count: int = 0
        _logger.info(
            "SAM3Segmentor init  device=%s, fp16=%s, prompts=%s",
            self._device, self._half_precision, self._prompts,
        )
        self._log_system_info()

    def _log_system_info(self) -> None:
        info = DeviceManager.get_available_devices()
        if info["cuda_available"]:
            for dev in info["cuda_devices"]:
                _logger.info("GPU %s: %s (%.1f GB)", dev["id"], dev["name"], dev["memory_gb"])
        else:
            _logger.info("No GPU available  running on CPU")
        mem = DeviceManager.get_memory_info()
        if "cpu_ram_total_gb" in mem:
            _logger.info(
                "System RAM: %.1f GB (Available: %.1f GB)",
                mem["cpu_ram_total_gb"], mem["cpu_ram_gb"],
            )

    def _ensure_loaded(self) -> None:
        """Lazy-load SAM3 model on first call."""
        if self._model_loaded:
            return
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = {
                "conf": self._confidence,
                "task": "segment",
                "mode": "predict",
                "model": str(self._model_path),
                "half": self._half_precision,
                "imgsz": self._resolution,
                "verbose": False,
                "device": self._device,
            }
            _logger.info("Loading SAM3 model from %s ", self._model_path)
            self._predictor = SAM3SemanticPredictor(overrides=overrides)
            self._predictor.setup_model()
            self._model_loaded = True
            _logger.info("SAM3 model loaded successfully")
        except ImportError as exc:
            raise ImportError(
                "SAM3SemanticPredictor not found. "
                "Install ultralytics>=8.3.237 and CLIP fork: "
                "pip install git+https://github.com/ultralytics/CLIP.git"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load SAM3 model: {exc}") from exc

    def _maybe_gc(self) -> None:
        """Periodically run garbage collection to free memory."""
        self._process_count += 1
        if self._process_count % self._gc_interval == 0:
            gc.collect()
            if not self._is_cpu:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    @trace
    def process_image(self, image_path: Path) -> Optional[SegmentationResult]:
        """Segment a single image and return raw (pre-remap) results.

        The ``class_id`` field in each :class:`~src.interfaces.MaskData` is
        the **raw SAM3 prompt index** (0..N-1).  The pipeline remap stage
        converts these to output class IDs via
        ``ClassRegistry.remap_prompt_index()`` before NMS.

        Args:
            image_path: Absolute path to the source image.

        Returns:
            :class:`~src.interfaces.SegmentationResult` with raw prompt
            indices, or ``None`` if no masks were detected.

        Raises:
            FileNotFoundError: Image does not exist on disk.
            RuntimeError: Inference failed unexpectedly.
        """
        self._ensure_loaded()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            h, w = img.shape[:2]
            t0 = time.perf_counter()

            self._predictor.set_image(str(image_path))
            raw_results = self._predictor(text=self._prompts)

            inference_ms = (time.perf_counter() - t0) * 1000.0

            if not raw_results or len(raw_results) == 0:
                _logger.debug("No detections in %s", image_path.name)
                self._maybe_gc()
                return None

            result = raw_results[0]
            if result.masks is None or len(result.masks) == 0:
                _logger.debug("No masks in %s", image_path.name)
                self._maybe_gc()
                return None

            masks_np = result.masks.data.cpu().numpy()  # (N, H, W) float/bool
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_np   = result.boxes.xyxy.cpu().numpy()
                cls_list   = result.boxes.cls.cpu().numpy().astype(int).tolist()
                conf_list  = result.boxes.conf.cpu().numpy().tolist()
            else:
                boxes_np  = np.zeros((0, 4), dtype=float)
                cls_list  = []
                conf_list = []

            mask_data_list: List[MaskData] = []
            for i, (mask_arr, cls_id, conf) in enumerate(zip(masks_np, cls_list, conf_list)):
                bool_mask: np.ndarray = mask_arr.astype(bool)
                area = int(bool_mask.sum())
                if i < len(boxes_np):
                    x1, y1, x2, y2 = boxes_np[i]
                    bbox: Tuple[int, int, int, int] = (int(x1), int(y1), int(x2), int(y2))
                else:
                    bbox = (0, 0, w, h)
                mask_data_list.append(
                    MaskData(
                        mask=bool_mask,
                        confidence=float(conf),
                        class_id=int(cls_id),  # raw prompt index  NOT yet remapped
                        area=area,
                        bbox=bbox,
                        polygon=None,
                    )
                )

            _logger.debug(
                "Detected %d masks in %s (%.1f ms)",
                len(mask_data_list), image_path.name, inference_ms,
            )
            self._maybe_gc()
            return SegmentationResult(
                image_path=image_path,
                masks=mask_data_list,
                image_width=w,
                image_height=h,
                inference_time_ms=inference_ms,
                device=self._device,
            )

        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            _logger.error("Segmentation failed for %s: %s", image_path, exc)
            self._maybe_gc()
            raise RuntimeError(f"Segmentation failed for {image_path}: {exc}") from exc

    @trace
    def process_batch(self, image_paths: List[Path]) -> List[Optional[SegmentationResult]]:
        """Process a list of images sequentially, reusing the loaded model.

        Args:
            image_paths: Paths to process.

        Returns:
            One result per path (``None`` for images with no detections or
            that raised errors  errors are logged, not re-raised).
        """
        out: List[Optional[SegmentationResult]] = []
        for path in image_paths:
            try:
                out.append(self.process_image(path))
            except Exception as exc:
                _logger.error("Error processing %s: %s", path, exc)
                out.append(None)
        return out

    @trace
    def get_device_info(self) -> Dict[str, Any]:
        """Return current device and memory information.

        Returns:
            Dict with ``device``, ``is_cpu``, ``half_precision``,
            ``model_loaded``, and ``memory`` sub-dict.
        """
        return {
            "device": self._device,
            "is_cpu": self._is_cpu,
            "half_precision": self._half_precision,
            "model_loaded": self._model_loaded,
            "memory": DeviceManager.get_memory_info(),
        }

    def cleanup(self) -> None:
        """Release model resources and clear CUDA cache."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
            self._model_loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            _logger.info("SAM3 model resources released")


def create_segmentor(model_config: Any, pipeline_config: Any) -> SAM3Segmentor:
    """Factory: build :class:`SAM3Segmentor` from config slices (ISP).

    Args:
        model_config: ``config.model`` slice.
        pipeline_config: ``config.pipeline`` slice.

    Returns:
        A ready-to-use :class:`SAM3Segmentor` instance.
    """
    return SAM3Segmentor(model_config, pipeline_config)
