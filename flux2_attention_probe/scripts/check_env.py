#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flux2_probe.backend import BackendAdapter
from flux2_probe.io_utils import add_local_diffusers_to_path, ensure_dir, load_yaml, save_json
from flux2_probe.load_pipeline import DEFAULT_MODEL_ID, import_flux2_klein_pipeline, load_flux2_pipeline


def pkg_version(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "version": getattr(module, "__version__", "unknown"), "file": getattr(module, "__file__", None)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def run_cmd(cmd: list[str]) -> dict:
    try:
        out = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=10)
        return {"returncode": out.returncode, "stdout": out.stdout[-4000:], "stderr": out.stderr[-4000:]}
    except Exception as exc:
        return {"error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "npu_config_template.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--check-model-load", action="store_true", help="Attempt loading the FLUX.2 Klein pipeline.")
    parser.add_argument("--device-id", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    diffusers_src = cfg.get("diffusers_src") or cfg.get("diffusers", {}).get("src")
    add_local_diffusers_to_path(diffusers_src)

    backend_name = cfg.get("backend", "npu")
    device_id = args.device_id if args.device_id is not None else cfg.get("npu", {}).get("device_id", 0)
    backend = BackendAdapter(backend=backend_name, device_id=device_id)

    report = {
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "packages": {
            "torch": pkg_version("torch"),
            "torch_npu": pkg_version("torch_npu"),
            "diffusers": pkg_version("diffusers"),
            "transformers": pkg_version("transformers"),
            "accelerate": pkg_version("accelerate"),
            "safetensors": pkg_version("safetensors"),
            "pandas": pkg_version("pandas"),
            "pyarrow": pkg_version("pyarrow"),
        },
        "npu_smi": run_cmd(["npu-smi", "info"]),
        "backend": {},
        "flux2_import": {},
        "model_load": {"attempted": False},
        "env": {
            "ASCEND_VISIBLE_DEVICES": os.environ.get("ASCEND_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
        "compatibility_notes": [],
    }

    try:
        report["backend"] = backend.env_summary()
    except Exception as exc:
        report["backend"] = {"error": repr(exc), "warnings": backend.warnings}

    pipe_cls, import_error = import_flux2_klein_pipeline(diffusers_src)
    report["flux2_import"] = {
        "ok": pipe_cls is not None,
        "class": f"{pipe_cls.__module__}.{pipe_cls.__name__}" if pipe_cls is not None else None,
        "error": import_error,
        "install_hint": "pip install git+https://github.com/huggingface/diffusers.git" if pipe_cls is None else None,
    }

    if not report["packages"]["diffusers"]["ok"]:
        report["compatibility_notes"].append("diffusers is not importable in this Python environment.")
    if report["packages"]["torch"]["ok"] and report["packages"]["torch_npu"]["ok"]:
        torch_v = report["packages"]["torch"]["version"].split("+")[0]
        npu_v = report["packages"]["torch_npu"]["version"]
        if not npu_v.startswith(torch_v[:3]):
            report["compatibility_notes"].append(f"Check torch/torch_npu match: torch={torch_v}, torch_npu={npu_v}")
    if not report["backend"].get("available"):
        report["compatibility_notes"].append("Requested backend is not confirmed available; CPU fallback may be used.")
    if backend.backend == "npu":
        report["compatibility_notes"].append("NPU path selected. CUDA APIs are intentionally not used by this probe.")

    if args.check_model_load:
        report["model_load"]["attempted"] = True
        model_cfg = cfg.get("model", {})
        local_paths = cfg.get("local_paths", {})
        try:
            _, load_report = load_flux2_pipeline(
                model_id=model_cfg.get("model_id") or DEFAULT_MODEL_ID,
                cache_dir=model_cfg.get("cache_dir"),
                dtype=model_cfg.get("dtype", "bfloat16"),
                backend=backend,
                offload=bool(model_cfg.get("offload", False)),
                low_cpu_mem_usage=bool(model_cfg.get("low_cpu_mem_usage", True)),
                local_files_only=bool(model_cfg.get("local_files_only", False)),
                diffusers_src=diffusers_src,
                local_paths=local_paths,
            )
            report["model_load"].update({"ok": True, "report": load_report})
        except Exception as exc:
            report["model_load"].update({"ok": False, "error": repr(exc)})

    output = Path(args.output) if args.output else ROOT / "outputs" / "env_report.json"
    ensure_dir(output.parent)
    save_json(report, output)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {output}")
    if not report["flux2_import"]["ok"]:
        print("\nFlux2KleinPipeline import failed. Suggested fix:")
        print("  pip install git+https://github.com/huggingface/diffusers.git")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

