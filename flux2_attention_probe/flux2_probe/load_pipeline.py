from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from .backend import BackendAdapter, dtype_from_string, dtype_name
from .io_utils import add_local_diffusers_to_path

logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def import_flux2_klein_pipeline(diffusers_src: str | None = None):
    add_local_diffusers_to_path(diffusers_src)
    try:
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline, None
    except Exception as exc:
        try:
            module = importlib.import_module("diffusers.pipelines.flux2.pipeline_flux2_klein")
            return getattr(module, "Flux2KleinPipeline"), None
        except Exception as exc2:
            return None, (
                "Cannot import Flux2KleinPipeline. Install diffusers main with:\n"
                "  pip install git+https://github.com/huggingface/diffusers.git\n"
                f"Primary error: {exc!r}\nFallback error: {exc2!r}"
            )


def import_flux2_components(diffusers_src: str | None = None) -> dict[str, Any]:
    add_local_diffusers_to_path(diffusers_src)
    names = {}
    for module_name, object_names in {
        "diffusers": ["AutoencoderKLFlux2", "FlowMatchEulerDiscreteScheduler"],
        "diffusers.models.transformers.transformer_flux2": ["Flux2Transformer2DModel"],
        "transformers": ["Qwen2TokenizerFast", "Qwen3ForCausalLM"],
    }.items():
        try:
            module = importlib.import_module(module_name)
            for object_name in object_names:
                names[object_name] = getattr(module, object_name)
        except Exception as exc:
            names[f"__error__{module_name}"] = repr(exc)
    return names


def choose_dtype(dtype: str | None, backend: BackendAdapter):
    torch = backend.torch
    requested = dtype_from_string(dtype or "bfloat16")
    candidates = [requested]
    for fallback in (torch.float16, torch.float32):
        if fallback not in candidates:
            candidates.append(fallback)

    device = backend.get_device()
    for candidate in candidates:
        if str(device) == "cpu" and candidate == torch.float16:
            continue
        try:
            x = torch.ones((1,), device=device, dtype=candidate)
            y = x + x
            backend.synchronize()
            _ = y.detach().to("cpu")
            return candidate, None
        except Exception as exc:
            backend.warn(f"dtype {dtype_name(candidate)} failed on {device}: {exc}")
            backend.empty_cache()
    return torch.float32, "All requested low precision dtype probes failed; using float32."


def find_transformer(pipe: Any) -> tuple[str | None, Any | None, list[str]]:
    candidates = []
    for name in ("transformer", "unet", "model", "dit"):
        module = getattr(pipe, name, None)
        if module is not None:
            return name, module, candidates
    for name, module in getattr(pipe, "components", {}).items():
        class_name = module.__class__.__name__ if module is not None else ""
        if "Transformer" in class_name or "Flux2" in class_name:
            return name, module, candidates
        candidates.append(f"{name}: {class_name}")
    if hasattr(pipe, "named_modules"):
        for name, module in pipe.named_modules():
            if name and ("transformer" in name.lower() or "attn" in name.lower()):
                candidates.append(f"{name}: {module.__class__.__name__}")
                if len(candidates) >= 50:
                    break
    return None, None, candidates


def list_pipeline_components(pipe: Any) -> dict[str, str]:
    out = {}
    for name, value in getattr(pipe, "components", {}).items():
        out[name] = value.__class__.__name__ if value is not None else "None"
    for name in ("transformer", "text_encoder", "vae", "scheduler", "tokenizer"):
        if hasattr(pipe, name):
            value = getattr(pipe, name)
            out.setdefault(name, value.__class__.__name__ if value is not None else "None")
    return out


def load_flux2_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: str | None = None,
    dtype: str | None = "bfloat16",
    backend: BackendAdapter | None = None,
    device: str | None = None,
    offload: bool = False,
    low_cpu_mem_usage: bool = True,
    local_files_only: bool = False,
    diffusers_src: str | None = None,
    local_paths: dict[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    backend = backend or BackendAdapter(device or "npu")
    report: dict[str, Any] = {"model_id": model_id, "warnings": []}

    Flux2KleinPipeline, import_error = import_flux2_klein_pipeline(diffusers_src)
    if Flux2KleinPipeline is None:
        raise ImportError(import_error)

    torch_dtype, dtype_warning = choose_dtype(dtype, backend)
    if dtype_warning:
        report["warnings"].append(dtype_warning)
    report["torch_dtype"] = dtype_name(torch_dtype)

    kwargs = {
        "torch_dtype": torch_dtype,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    load_errors = []
    path_candidate = Path(model_id).expanduser()
    try:
        pipe = Flux2KleinPipeline.from_pretrained(str(path_candidate if path_candidate.exists() else model_id), **kwargs)
    except Exception as exc:
        load_errors.append(f"Flux2KleinPipeline.from_pretrained failed: {exc!r}")
        pipe = _load_from_component_paths(Flux2KleinPipeline, local_paths or {}, torch_dtype, diffusers_src, load_errors)

    if offload:
        enabled = _try_enable_offload(pipe, backend)
        report["offload_enabled"] = enabled
        if not enabled:
            report["warnings"].append("CPU offload was requested but not enabled; continuing without offload.")
    else:
        backend.set_device()
        pipe = backend.to_device(pipe)
        report["offload_enabled"] = False

    report["load_errors"] = load_errors
    report["components"] = list_pipeline_components(pipe)
    transformer_name, transformer, possible = find_transformer(pipe)
    report["transformer_path"] = transformer_name
    report["transformer_class"] = transformer.__class__.__name__ if transformer is not None else None
    report["possible_transformer_modules"] = possible
    logger.info("Loaded components: %s", report["components"])
    return pipe, report


def _load_from_component_paths(
    pipeline_cls: Any,
    local_paths: dict[str, str],
    torch_dtype: Any,
    diffusers_src: str | None,
    load_errors: list[str],
):
    required = ["flux2_tokenizer_path", "flux2_text_encoder_path", "flux2_model_path", "flux2_ae_path"]
    missing = [key for key in required if not local_paths.get(key)]
    if missing:
        raise RuntimeError(
            "Cannot load FLUX.2 Klein from pipeline directory, and component paths are incomplete. "
            f"Missing {missing}. Previous errors: {load_errors}"
        )
    components = import_flux2_components(diffusers_src)
    component_errors = {k: v for k, v in components.items() if k.startswith("__error__")}
    if component_errors:
        raise RuntimeError(f"Cannot import component classes: {component_errors}. Previous errors: {load_errors}")

    try:
        tokenizer = components["Qwen2TokenizerFast"].from_pretrained(local_paths["flux2_tokenizer_path"])
        text_encoder = components["Qwen3ForCausalLM"].from_pretrained(
            local_paths["flux2_text_encoder_path"], torch_dtype=torch_dtype
        )
        transformer = components["Flux2Transformer2DModel"].from_single_file(
            local_paths["flux2_model_path"], torch_dtype=torch_dtype
        )
        vae = components["AutoencoderKLFlux2"].from_single_file(local_paths["flux2_ae_path"], torch_dtype=torch_dtype)
        scheduler = components["FlowMatchEulerDiscreteScheduler"]()
        return pipeline_cls(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
    except Exception as exc:
        raise RuntimeError(
            "Component-path fallback failed. Prefer a complete FLUX.2 Klein diffusers directory with "
            "model_index.json and component subfolders. "
            f"Fallback error: {exc!r}. Previous errors: {load_errors}"
        ) from exc


def _try_enable_offload(pipe: Any, backend: BackendAdapter) -> bool:
    if backend.backend == "cuda":
        try:
            pipe.enable_model_cpu_offload()
            return True
        except Exception as exc:
            backend.warn(f"enable_model_cpu_offload failed on cuda: {exc}")
            return False
    if backend.backend == "npu":
        # accelerate's built-in offload helpers are CUDA-centric in many versions. Avoid silent wrong behavior.
        backend.warn("NPU CPU offload support is not assumed; skipping enable_model_cpu_offload.")
        return False
    return False

