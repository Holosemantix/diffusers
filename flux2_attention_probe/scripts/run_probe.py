#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flux2_probe.attention_hooks import AttentionProbeManager, ProbeConfig
from flux2_probe.backend import BackendAdapter
from flux2_probe.io_utils import Tee, copy_if_exists, ensure_dir, load_yaml, save_json
from flux2_probe.load_pipeline import DEFAULT_MODEL_ID, load_flux2_pipeline
from flux2_probe.metrics_attention import summarize_attention_records
from flux2_probe.token_segments import TokenSegmentInferer, normalize_manual_ranges
from flux2_probe.visualization import make_input_grid, summarize_figures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "probe_multi_ref.yaml"))
    parser.add_argument("--source", default=None)
    parser.add_argument("--refs", nargs="*", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = ensure_dir(args.output_dir or cfg.get("output_dir") or ROOT / "outputs" / "exp001")
    log_file = open(output_dir / "stdout.log", "w", encoding="utf-8")
    with contextlib.redirect_stdout(Tee(sys.stdout, log_file)), contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
        return run(args, cfg, output_dir)


def run(args, cfg, output_dir: Path) -> int:
    print(f"Output dir: {output_dir}")
    copy_if_exists(args.config, output_dir / "config.yaml")
    save_json(cfg, output_dir / "resolved_config.json")

    backend_cfg = cfg.get("backend", "npu")
    backend_name = args.backend or backend_cfg
    device_id = args.device_id if args.device_id is not None else cfg.get("npu", {}).get("device_id", 0)
    backend = BackendAdapter(backend=backend_name, device_id=device_id)

    model_cfg = cfg.get("model", {})
    local_paths = cfg.get("local_paths", {})
    pipe, load_report = load_flux2_pipeline(
        model_id=model_cfg.get("model_id") or DEFAULT_MODEL_ID,
        cache_dir=model_cfg.get("cache_dir"),
        dtype=model_cfg.get("dtype", "bfloat16"),
        backend=backend,
        offload=bool(model_cfg.get("offload", False)),
        low_cpu_mem_usage=bool(model_cfg.get("low_cpu_mem_usage", True)),
        local_files_only=bool(args.local_files_only or model_cfg.get("local_files_only", False)),
        diffusers_src=cfg.get("diffusers_src"),
        local_paths=local_paths,
        prefer_component_paths=bool(model_cfg.get("prefer_component_paths", False)),
    )
    save_json(load_report, output_dir / "load_report.json")

    probe_cfg_raw = dict(cfg.get("probe", {}))
    probe_cfg_raw["output_dir"] = str(output_dir)
    probe_cfg = ProbeConfig(**{k: v for k, v in probe_cfg_raw.items() if k in ProbeConfig.__dataclass_fields__})
    segment_inferer = TokenSegmentInferer(normalize_manual_ranges(cfg.get("segments")))
    manager = AttentionProbeManager(probe_cfg, segment_inferer=segment_inferer)
    manager.register(pipe)

    source = args.source or cfg.get("source")
    refs = args.refs if args.refs is not None else cfg.get("refs", [])
    prompt = args.prompt or cfg.get("prompt") or ""
    gen_cfg = cfg.get("generation", {})
    height = args.height or gen_cfg.get("height", 512)
    width = args.width or gen_cfg.get("width", 512)
    steps = args.num_inference_steps or gen_cfg.get("num_inference_steps", 1)
    guidance_scale = gen_cfg.get("guidance_scale", 1.0)
    seed = args.seed if args.seed is not None else gen_cfg.get("seed", 0)

    images = load_condition_images(source, refs)
    if images:
        make_input_grid(images, output_dir / "input_grid.png", labels=(["S"] if source else []) + [f"R{i+1}" for i in range(len(refs))])

    backend.reset_peak_memory_stats()
    generator = backend.manual_seed(int(seed))
    t0 = time.perf_counter()
    oom = False
    try:
        output = pipe(
            image=images or None,
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            output_type="pil",
        )
        backend.synchronize()
        elapsed = time.perf_counter() - t0
        image = output.images[0] if hasattr(output, "images") else output[0][0]
        image.save(output_dir / "generated.png")
    except RuntimeError as exc:
        elapsed = time.perf_counter() - t0
        oom = "out of memory" in str(exc).lower() or "oom" in str(exc).lower()
        save_json({"error": repr(exc), "oom": oom}, output_dir / "run_error.json")
        raise
    finally:
        manager.remove(pipe)
        manager.save(str(output_dir))
        backend.empty_cache()

    timing = {"elapsed_sec": elapsed, "height": height, "width": width, "steps": steps, "seed": seed}
    memory = {
        "memory_allocated": backend.memory_allocated(),
        "max_memory_allocated": backend.max_memory_allocated(),
        "warnings": backend.warnings,
        "oom": oom,
    }
    save_json(timing, output_dir / "timing_report.json")
    save_json(memory, output_dir / "memory_report.json")

    metrics_report = summarize_attention_records(
        output_dir,
        output_dir,
        reference_roles=cfg.get("reference_roles"),
        region_annotations=cfg.get("region_annotations"),
    )
    summarize_figures(output_dir)
    print("Probe complete.")
    print(json.dumps({"timing": timing, "memory": memory, "metrics": metrics_report}, indent=2))
    return 0


def load_condition_images(source, refs):
    from PIL import Image

    paths = []
    if source:
        paths.append(source)
    paths.extend(refs or [])
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        images.append(img)
    return images


if __name__ == "__main__":
    raise SystemExit(main())

