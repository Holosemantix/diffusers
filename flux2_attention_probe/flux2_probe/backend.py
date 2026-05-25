from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _import_torch():
    import torch

    return torch


def _try_import_torch_npu():
    try:
        import torch_npu  # noqa: F401

        return torch_npu
    except Exception:
        return None


@dataclass
class BackendAdapter:
    backend: str = "npu"
    device_id: int | str | None = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.backend = (self.backend or "npu").lower()
        if self.backend not in {"npu", "cuda", "cpu"}:
            self.warn(f"Unknown backend={self.backend!r}; falling back to cpu")
            self.backend = "cpu"
        self._torch = None
        self._torch_npu = None

    @property
    def torch(self):
        if self._torch is None:
            self._torch = _import_torch()
        return self._torch

    @property
    def torch_npu(self):
        if self._torch_npu is None:
            self._torch_npu = _try_import_torch_npu()
        return self._torch_npu

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def npu_module(self) -> Any | None:
        torch = self.torch
        return getattr(torch, "npu", None)

    def is_available(self) -> bool:
        if self.backend == "cpu":
            return True
        if self.backend == "npu":
            self.torch_npu
            npu = self.npu_module()
            if npu is None or not hasattr(npu, "is_available"):
                return False
            try:
                return bool(npu.is_available())
            except Exception as exc:
                self.warn(f"torch.npu.is_available failed: {exc}")
                return False
        if self.backend == "cuda":
            cuda = getattr(self.torch, "cuda", None)
            return bool(cuda is not None and cuda.is_available())
        return False

    def get_device(self):
        torch = self.torch
        if self.backend == "cpu" or not self.is_available():
            if self.backend != "cpu":
                self.warn(f"{self.backend} is not available; using cpu")
            return torch.device("cpu")
        return torch.device(f"{self.backend}:{self.device_id}" if self.device_id is not None else self.backend)

    @property
    def device(self):
        return self.get_device()

    def set_device(self) -> None:
        if self.backend == "npu" and self.is_available():
            npu = self.npu_module()
            if hasattr(npu, "set_device"):
                npu.set_device(self.get_device())
        elif self.backend == "cuda" and self.is_available():
            self.torch.cuda.set_device(self.get_device())

    def to_device(self, tensor_or_module: Any, dtype: Any | None = None):
        device = self.get_device()
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if hasattr(tensor_or_module, "to"):
            try:
                return tensor_or_module.to(**kwargs)
            except TypeError:
                if dtype is not None:
                    return tensor_or_module.to(device=device, dtype=dtype)
                return tensor_or_module.to(device)
        return tensor_or_module

    def synchronize(self) -> None:
        try:
            if self.backend == "npu":
                npu = self.npu_module()
                if npu is not None and hasattr(npu, "synchronize") and self.is_available():
                    npu.synchronize()
            elif self.backend == "cuda" and self.is_available():
                self.torch.cuda.synchronize()
        except Exception as exc:
            self.warn(f"{self.backend}.synchronize unavailable: {exc}")

    def empty_cache(self) -> None:
        try:
            if self.backend == "npu":
                npu = self.npu_module()
                if npu is not None and hasattr(npu, "empty_cache"):
                    npu.empty_cache()
            elif self.backend == "cuda":
                self.torch.cuda.empty_cache()
        except Exception as exc:
            self.warn(f"{self.backend}.empty_cache unavailable: {exc}")

    def _memory_call(self, name: str) -> int | None:
        try:
            if self.backend == "npu":
                npu = self.npu_module()
                fn = getattr(npu, name, None) if npu is not None else None
                if fn is None:
                    self.warn(f"torch.npu.{name} is unavailable")
                    return None
                return int(fn(self.get_device()))
            if self.backend == "cuda":
                return int(getattr(self.torch.cuda, name)(self.get_device()))
        except Exception as exc:
            self.warn(f"{self.backend}.{name} failed: {exc}")
        return None

    def memory_allocated(self) -> int | None:
        return self._memory_call("memory_allocated")

    def max_memory_allocated(self) -> int | None:
        return self._memory_call("max_memory_allocated")

    def reset_peak_memory_stats(self) -> None:
        try:
            if self.backend == "npu":
                npu = self.npu_module()
                fn = getattr(npu, "reset_peak_memory_stats", None) if npu is not None else None
                if fn is None:
                    self.warn("torch.npu.reset_peak_memory_stats is unavailable")
                    return
                fn(self.get_device())
            elif self.backend == "cuda":
                self.torch.cuda.reset_peak_memory_stats(self.get_device())
        except Exception as exc:
            self.warn(f"{self.backend}.reset_peak_memory_stats failed: {exc}")

    def manual_seed(self, seed: int):
        torch = self.torch
        if self.backend == "npu":
            try:
                npu = self.npu_module()
                if npu is not None and hasattr(npu, "manual_seed"):
                    npu.manual_seed(seed)
            except Exception as exc:
                self.warn(f"torch.npu.manual_seed failed: {exc}")
        elif self.backend == "cuda":
            torch.cuda.manual_seed_all(seed)
        return torch.Generator(device=self.get_device()).manual_seed(seed)

    def time_call(self, fn, *args, **kwargs):
        self.synchronize()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self.synchronize()
        return result, time.perf_counter() - start

    def device_count(self) -> int | None:
        try:
            if self.backend == "npu":
                npu = self.npu_module()
                return int(npu.device_count()) if npu is not None and hasattr(npu, "device_count") else None
            if self.backend == "cuda":
                return int(self.torch.cuda.device_count())
            return 0
        except Exception as exc:
            self.warn(f"{self.backend}.device_count failed: {exc}")
            return None

    def device_name(self, index: int | None = None) -> str | None:
        index = self.device_id if index is None else index
        try:
            if self.backend == "npu":
                npu = self.npu_module()
                for attr in ("get_device_name", "get_device_properties"):
                    fn = getattr(npu, attr, None) if npu is not None else None
                    if fn is None:
                        continue
                    value = fn(index)
                    return getattr(value, "name", str(value))
            if self.backend == "cuda":
                return self.torch.cuda.get_device_name(index)
        except Exception as exc:
            self.warn(f"{self.backend}.device_name failed: {exc}")
        return None

    def mem_get_info(self) -> tuple[int, int] | None:
        try:
            if self.backend == "npu":
                npu = self.npu_module()
                fn = getattr(npu, "mem_get_info", None) if npu is not None else None
                if fn is None:
                    return None
                free, total = fn(self.get_device())
                return int(free), int(total)
            if self.backend == "cuda":
                free, total = self.torch.cuda.mem_get_info(self.get_device())
                return int(free), int(total)
        except Exception as exc:
            self.warn(f"{self.backend}.mem_get_info failed: {exc}")
        return None

    def env_summary(self) -> dict[str, Any]:
        info = {
            "backend": self.backend,
            "requested_device_id": self.device_id,
            "device": str(self.get_device()),
            "available": self.is_available(),
            "device_count": self.device_count(),
            "device_name": self.device_name(),
            "memory_allocated": self.memory_allocated(),
            "max_memory_allocated": self.max_memory_allocated(),
            "mem_get_info": self.mem_get_info(),
            "warnings": list(self.warnings),
            "ASCEND_VISIBLE_DEVICES": os.environ.get("ASCEND_VISIBLE_DEVICES"),
        }
        return info


def dtype_from_string(value: str | None):
    torch = _import_torch()
    if value is None:
        return None
    normalized = str(value).lower().replace("torch.", "")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    attr = aliases.get(normalized, normalized)
    if not hasattr(torch, attr):
        raise ValueError(f"Unsupported torch dtype string: {value}")
    return getattr(torch, attr)


def dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")

