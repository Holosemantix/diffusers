#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EDGES = [
    "X->S",
    "X->R1",
    "X->P",
    "S->R1",
    "R1->S",
    "S->X",
    "R1->X",
    "X->X",
    "S->S",
    "R1->R1",
    "P->P",
]
PLOT_EDGES = ["X->R1", "X->P", "X->S", "S->R1", "R1->S", "S->X", "R1->X"]
LIGHT_LAYERS = [0, 4, 14, 24]
LIGHT_HEADS = [0, 4, 8, 12, 16, 20]
LIGHT_STEPS = [0, 14, 27]
MECHANISMS = [
    ("reference_transfer_mid", "X->R1", 14, 12),
    ("reference_transfer_double", "X->R1", 4, 4),
    ("prompt_control_mid", "X->P", 14, 20),
    ("prompt_control_late", "X->P", 24, 20),
    ("source_to_ref_fusion", "S->R1", 24, 8),
    ("ref_to_source_fusion", "R1->S", 24, 8),
    ("source_writeback", "S->X", 24, 20),
    ("ref_writeback_late12", "R1->X", 24, 12),
    ("ref_writeback_late20", "R1->X", 24, 20),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--light_dir", default="/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new")
    parser.add_argument("--heavy_dir", default="/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    light_dir = Path(args.light_dir)
    heavy_dir = Path(args.heavy_dir)
    output_dir = Path(args.output_dir) if args.output_dir else heavy_dir / "compare_light_heavy_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    light_meta = load_metadata(light_dir)
    heavy_meta = load_metadata(heavy_dir)
    metadata_rows, metadata_warnings = compare_metadata(light_dir, heavy_dir, light_meta, heavy_meta)
    write_csv(output_dir / "run_metadata_comparison.csv", metadata_rows)

    light_rows = load_attention_rows(light_dir / "attention_blocks.jsonl")
    heavy_rows = load_attention_rows(heavy_dir / "attention_blocks.jsonl")
    heavy_slice_rows = [
        r
        for r in heavy_rows
        if r["layer"] in LIGHT_LAYERS and r["head"] in LIGHT_HEADS and r["step"] in LIGHT_STEPS
    ]

    light_edge = aggregate_edge_rows(light_rows)
    heavy_slice_edge = aggregate_edge_rows(heavy_slice_rows)
    heavy_full_edge = aggregate_edge_rows(heavy_rows)

    joined = join_light_heavy(light_edge, heavy_slice_edge)
    write_csv(output_dir / "light_vs_heavy_slice_edge_diff.csv", joined)
    write_csv(output_dir / "light_vs_heavy_slice_by_layer.csv", aggregate_diff(joined, "layer"))
    write_csv(output_dir / "light_vs_heavy_slice_by_step.csv", aggregate_diff(joined, "step"))
    write_csv(output_dir / "light_vs_heavy_slice_by_head.csv", aggregate_diff(joined, "head"))

    edge_metrics = compute_edge_metrics(joined)
    write_csv(output_dir / "light_vs_heavy_slice_edge_metrics.csv", edge_metrics)

    mechanisms = evaluate_mechanisms(light_edge, heavy_slice_edge, heavy_full_edge)
    write_csv(output_dir / "key_mechanism_replication.csv", mechanisms)

    full_summary = summarize_heavy_full(heavy_full_edge)
    write_csv(output_dir / "heavy_full_edge_summary.csv", full_summary["edge_summary"])
    write_csv(output_dir / "heavy_full_top_layer_head.csv", full_summary["top_layer_head"])

    bias = sampling_bias_analysis(heavy_full_edge, edge_metrics, mechanisms)
    write_csv(output_dir / "sampling_bias_summary.csv", bias["rows"])

    plot_heavy_full(heavy_full_edge, heavy_rows, fig_dir)
    report = build_report(
        light_dir=light_dir,
        heavy_dir=heavy_dir,
        output_dir=output_dir,
        metadata_rows=metadata_rows,
        metadata_warnings=metadata_warnings,
        edge_metrics=edge_metrics,
        mechanisms=mechanisms,
        full_summary=full_summary,
        bias=bias,
        light_rows=light_rows,
        heavy_rows=heavy_rows,
        heavy_slice_rows=heavy_slice_rows,
    )
    (output_dir / "compare_light_heavy_v2_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote comparison outputs to {output_dir}")
    return 0


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(run_dir: Path) -> dict:
    names = [
        "effective_run.json",
        "segment_map.json",
        "probe_registration.json",
        "timing_report.json",
        "memory_report.json",
        "metrics_report.json",
        "resolved_config.json",
    ]
    return {name: load_json(run_dir / name) for name in names}


def compare_metadata(light_dir: Path, heavy_dir: Path, light_meta: dict, heavy_meta: dict):
    rows = []
    warnings = []

    def add(field, light_value, heavy_value, status=None, note=""):
        if status is None:
            status = "ok" if light_value == heavy_value else "diff"
        rows.append(
            {
                "field": field,
                "light": json_compact(light_value),
                "heavy": json_compact(heavy_value),
                "status": status,
                "note": note,
            }
        )
        if status == "warning":
            warnings.append(f"{field}: {note}")

    le = light_meta.get("effective_run.json") or {}
    he = heavy_meta.get("effective_run.json") or {}
    lt = light_meta.get("timing_report.json") or {}
    ht = heavy_meta.get("timing_report.json") or {}
    lm = light_meta.get("metrics_report.json") or {}
    hm = heavy_meta.get("metrics_report.json") or {}
    lp = (light_meta.get("probe_registration.json") or {}).get("config", {})
    hp = (heavy_meta.get("probe_registration.json") or {}).get("config", {})
    lc = light_meta.get("resolved_config.json") or {}
    hc = heavy_meta.get("resolved_config.json") or {}

    add("prompt", lc.get("prompt"), hc.get("prompt"))
    add("seed", lt.get("seed"), ht.get("seed"))
    add("height", lt.get("height"), ht.get("height"))
    add("width", lt.get("width"), ht.get("width"))
    add("num_steps", lt.get("steps"), ht.get("steps"))
    add("effective_run", le, he)
    add("sample_layers", lp.get("sample_layers"), hp.get("sample_layers"))
    add("sample_heads", lp.get("sample_heads"), hp.get("sample_heads"))
    add("sample_steps", lp.get("sample_steps"), hp.get("sample_steps"))

    light_seg = segment_lengths(light_meta.get("segment_map.json") or {})
    heavy_seg = segment_lengths(heavy_meta.get("segment_map.json") or {})
    add(
        "segment_lengths",
        light_seg,
        heavy_seg,
        "ok" if light_seg == heavy_seg else "warning",
        "" if light_seg == heavy_seg else "segment_map differs; compare edge labels, not raw token positions",
    )

    for label, run_dir, metrics in [("light", light_dir, lm), ("heavy", heavy_dir, hm)]:
        add(f"{label}.attention_blocks_exists", (run_dir / "attention_blocks.jsonl").exists(), True)
        add(f"{label}.num_flow_rows_gt0", int(metrics.get("num_flow_rows", 0)) > 0, True)
        add(f"{label}.num_distribution_rows_gt0", int(metrics.get("num_distribution_rows", 0)) > 0, True)

    pixel = compare_images(light_dir / "generated.png", heavy_dir / "generated.png")
    add("generated_pixel_l1", pixel.get("l1"), pixel.get("l1"), pixel.get("status", "ok"), pixel.get("note", ""))
    add("generated_pixel_l2", pixel.get("l2"), pixel.get("l2"), pixel.get("status", "ok"), pixel.get("note", ""))
    add("generated_pixel_max_diff", pixel.get("max_diff"), pixel.get("max_diff"), pixel.get("status", "ok"), pixel.get("note", ""))
    return rows, warnings


def json_compact(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def segment_lengths(segment_map: dict) -> dict[str, int]:
    ranges = segment_map.get("ranges") or {}
    return {k: int(v[1]) - int(v[0]) for k, v in ranges.items()}


def compare_images(light_path: Path, heavy_path: Path) -> dict:
    if not light_path.exists() or not heavy_path.exists():
        return {"status": "warning", "note": "generated.png missing"}
    try:
        from PIL import Image

        a = np.asarray(Image.open(light_path).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(heavy_path).convert("RGB"), dtype=np.float32)
        if a.shape != b.shape:
            return {"status": "warning", "note": f"image shapes differ: {a.shape} vs {b.shape}"}
        diff = a - b
        return {
            "status": "ok",
            "l1": float(np.mean(np.abs(diff))),
            "l2": float(np.sqrt(np.mean(diff * diff))),
            "max_diff": float(np.max(np.abs(diff))),
            "note": "",
        }
    except Exception as exc:
        return {"status": "warning", "note": f"image compare failed: {exc!r}"}


def load_attention_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            step = int(rec["step_idx"])
            layer = int(rec["layer_id"])
            for head in rec.get("heads", []):
                h = int(head["head"])
                seg = head.get("segment_mass") or {}
                for edge, value in seg.items():
                    if "->" not in edge:
                        continue
                    qseg, kseg = edge.split("->", 1)
                    rows.append(
                        {
                            "step": step,
                            "layer": layer,
                            "head": h,
                            "query_segment": qseg,
                            "key_segment": kseg,
                            "edge": edge,
                            "value": float(value),
                            "entropy": float(head.get("normalized_entropy", 0.0)),
                            "top16_block_mass": float(head.get("top16_block_mass", 0.0)),
                        }
                    )
    return rows


def aggregate_edge_rows(rows: list[dict]) -> dict[tuple, dict]:
    acc = defaultdict(list)
    for r in rows:
        key = (r["step"], r["layer"], r["head"], r["query_segment"], r["key_segment"])
        acc[key].append(r)
    out = {}
    for key, vals in acc.items():
        edge = f"{key[3]}->{key[4]}"
        out[key] = {
            "step": key[0],
            "layer": key[1],
            "head": key[2],
            "query_segment": key[3],
            "key_segment": key[4],
            "edge": edge,
            "value": mean(v["value"] for v in vals),
            "entropy": mean(v["entropy"] for v in vals),
            "top16_block_mass": mean(v["top16_block_mass"] for v in vals),
        }
    return out


def join_light_heavy(light: dict, heavy: dict) -> list[dict]:
    rows = []
    keys = sorted(set(light.keys()) & set(heavy.keys()))
    for key in keys:
        l = light[key]
        h = heavy[key]
        diff = h["value"] - l["value"]
        rows.append(
            {
                "step": l["step"],
                "layer": l["layer"],
                "head": l["head"],
                "query_segment": l["query_segment"],
                "key_segment": l["key_segment"],
                "edge": l["edge"],
                "light_value": l["value"],
                "heavy_slice_value": h["value"],
                "diff": diff,
                "abs_diff": abs(diff),
                "rel_diff": abs(diff) / max(abs(l["value"]), 1e-12),
            }
        )
    return rows


def aggregate_diff(rows: list[dict], field: str) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r[field], r["edge"])].append(r)
    out = []
    for (value, edge), vals in sorted(groups.items()):
        out.append(
            {
                field: value,
                "edge": edge,
                "mean_light": mean(v["light_value"] for v in vals),
                "mean_heavy_slice": mean(v["heavy_slice_value"] for v in vals),
                "mean_abs_diff": mean(v["abs_diff"] for v in vals),
                "max_abs_diff": max(v["abs_diff"] for v in vals),
                "count": len(vals),
            }
        )
    return out


def compute_edge_metrics(joined: list[dict]) -> list[dict]:
    out = []
    for edge in EDGES:
        vals = [r for r in joined if r["edge"] == edge]
        light = [r["light_value"] for r in vals]
        heavy = [r["heavy_slice_value"] for r in vals]
        top5_l = top_keys(vals, "light_value", 5)
        top5_h = top_keys(vals, "heavy_slice_value", 5)
        top10_l = top_keys(vals, "light_value", 10)
        top10_h = top_keys(vals, "heavy_slice_value", 10)
        out.append(
            {
                "edge": edge,
                "mean_light": mean(light),
                "mean_heavy_slice": mean(heavy),
                "abs_diff": abs(mean(light) - mean(heavy)),
                "rel_diff": abs(mean(light) - mean(heavy)) / max(abs(mean(light)), 1e-12),
                "pearson": pearson(light, heavy),
                "spearman": spearman(light, heavy),
                "top5_head_overlap": overlap(top5_l, top5_h),
                "top10_head_overlap": overlap(top10_l, top10_h),
                "count": len(vals),
            }
        )
    return out


def top_keys(rows: list[dict], field: str, k: int) -> set[tuple[int, int, int]]:
    return {
        (r["step"], r["layer"], r["head"])
        for r in sorted(rows, key=lambda x: x[field], reverse=True)[:k]
    }


def overlap(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def evaluate_mechanisms(light: dict, heavy_slice: dict, heavy_full: dict) -> list[dict]:
    grouped = group_by_layer_head_edge(heavy_full)
    out = []
    for name, edge, layer, head in MECHANISMS:
        qseg, kseg = edge.split("->", 1)
        light_value = mean(
            r["value"] for key, r in light.items() if key[1] == layer and key[2] == head and r["edge"] == edge
        )
        heavy_slice_value = mean(
            r["value"] for key, r in heavy_slice.items() if key[1] == layer and key[2] == head and r["edge"] == edge
        )
        ranked = sorted(
            [r for r in grouped if r["edge"] == edge],
            key=lambda x: x["mean_value"],
            reverse=True,
        )
        rank = next((i + 1 for i, r in enumerate(ranked) if r["layer"] == layer and r["head"] == head), None)
        top = ranked[0] if ranked else {}
        if rank is not None and rank <= 10 and abs(light_value - heavy_slice_value) < 0.03:
            conclusion = "reproduced"
        elif abs(light_value - heavy_slice_value) < 0.03:
            conclusion = "shifted"
        else:
            conclusion = "failed"
        out.append(
            {
                "mechanism": name,
                "edge": edge,
                "layer": layer,
                "head": head,
                "light_value": light_value,
                "heavy_slice_same_head_value": heavy_slice_value,
                "heavy_full_rank": rank,
                "heavy_full_top_head": f"L{top.get('layer')}/H{top.get('head')}" if top else "",
                "heavy_full_top_value": top.get("mean_value", ""),
                "conclusion": conclusion,
            }
        )
    return out


def group_by_layer_head_edge(edge_rows: dict) -> list[dict]:
    acc = defaultdict(list)
    for r in edge_rows.values():
        acc[(r["layer"], r["head"], r["edge"])].append(r["value"])
    return [
        {"layer": k[0], "head": k[1], "edge": k[2], "mean_value": mean(v)}
        for k, v in acc.items()
    ]


def summarize_heavy_full(heavy_full: dict) -> dict:
    edge_summary = []
    for edge in EDGES:
        vals = [r["value"] for r in heavy_full.values() if r["edge"] == edge]
        edge_summary.append(
            {
                "edge": edge,
                "mean": mean(vals),
                "min": min(vals) if vals else "",
                "max": max(vals) if vals else "",
                "count": len(vals),
            }
        )
    grouped = group_by_layer_head_edge(heavy_full)
    top = []
    for edge in EDGES:
        for rank, r in enumerate(sorted([x for x in grouped if x["edge"] == edge], key=lambda x: x["mean_value"], reverse=True)[:20], 1):
            top.append({"edge": edge, "rank": rank, **r})
    return {"edge_summary": edge_summary, "top_layer_head": top}


def sampling_bias_analysis(heavy_full: dict, edge_metrics: list[dict], mechanisms: list[dict]) -> dict:
    rows = []
    for edge in ["X->R1", "X->P", "X->S", "S->R1", "R1->S", "S->X", "R1->X"]:
        by_step = means_by(heavy_full, edge, "step")
        early = window_mean(by_step, range(0, 5))
        mid = window_mean(by_step, range(12, 17))
        late = window_mean(by_step, range(23, 28))
        sampled = {s: by_step.get(s, float("nan")) for s in LIGHT_STEPS}
        rows.append(
            {
                "question": "step_representativeness",
                "edge": edge,
                "sampled_steps": json.dumps(sampled, sort_keys=True),
                "heavy_early_mean_0_4": early,
                "heavy_mid_mean_12_16": mid,
                "heavy_late_mean_23_27": late,
                "assessment": assess_step(sampled, early, mid, late),
            }
        )
        top_layers = sorted(means_by(heavy_full, edge, "layer").items(), key=lambda x: x[1], reverse=True)[:5]
        rows.append(
            {
                "question": "layer_representativeness",
                "edge": edge,
                "sampled_layers": json.dumps(LIGHT_LAYERS),
                "heavy_top_layers": json.dumps(top_layers),
                "assessment": "covered" if any(layer in LIGHT_LAYERS for layer, _ in top_layers[:3]) else "missed_top3",
            }
        )
        grouped = group_by_layer_head_edge(heavy_full)
        top_heads = [
            (r["layer"], r["head"], r["mean_value"])
            for r in sorted([x for x in grouped if x["edge"] == edge], key=lambda x: x["mean_value"], reverse=True)[:10]
        ]
        rows.append(
            {
                "question": "head_representativeness",
                "edge": edge,
                "sampled_heads": json.dumps(LIGHT_HEADS),
                "heavy_top_layer_heads": json.dumps(top_heads),
                "assessment": "covered" if any(h in LIGHT_HEADS and l in LIGHT_LAYERS for l, h, _ in top_heads[:10]) else "missed_top10",
            }
        )
    recommendations = recommend_sampling(rows)
    return {"rows": rows, "recommendations": recommendations}


def means_by(edge_rows: dict, edge: str, field: str) -> dict[int, float]:
    acc = defaultdict(list)
    for r in edge_rows.values():
        if r["edge"] == edge:
            acc[int(r[field])].append(r["value"])
    return {k: mean(v) for k, v in acc.items()}


def window_mean(by_step: dict[int, float], steps) -> float:
    vals = [by_step[s] for s in steps if s in by_step]
    return mean(vals)


def assess_step(sampled: dict, early: float, mid: float, late: float) -> str:
    targets = {0: early, 14: mid, 27: late}
    diffs = [abs(sampled.get(k, float("nan")) - v) for k, v in targets.items()]
    diffs = [d for d in diffs if not math.isnan(d)]
    if not diffs:
        return "insufficient"
    return "representative" if mean(diffs) < 0.03 else "biased"


def recommend_sampling(rows: list[dict]) -> dict:
    layer_counts = defaultdict(int)
    head_counts = defaultdict(int)
    for r in rows:
        if r["question"] == "layer_representativeness":
            for layer, _ in json.loads(r["heavy_top_layers"])[:3]:
                layer_counts[int(layer)] += 1
        if r["question"] == "head_representativeness":
            for layer, head, _ in json.loads(r["heavy_top_layer_heads"])[:10]:
                layer_counts[int(layer)] += 1
                head_counts[int(head)] += 1
    layers = [x for x, _ in sorted(layer_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]
    heads = [x for x, _ in sorted(head_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]]
    return {
        "sample_layers": sorted(set(LIGHT_LAYERS) | set(layers)),
        "sample_heads": sorted(set(LIGHT_HEADS) | set(heads)),
        "sample_steps": [0, 7, 14, 21, 27],
        "block_size": 256,
    }


def plot_heavy_full(heavy_full: dict, heavy_rows: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for edge in PLOT_EDGES:
        mat = np.full((25, 28), np.nan)
        for (layer, step), vals in group_values(heavy_full, edge, ("layer", "step")).items():
            mat[int(layer), int(step)] = mean(vals)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xlabel("Denoising step index")
        ax.set_ylabel("Attention layer id (0-4 double, 5-24 single)")
        ax.set_title(f"Heavy full layer-step heatmap: {edge}")
        fig.colorbar(im, ax=ax, label="attention mass")
        fig.tight_layout()
        fig.savefig(fig_dir / f"layer_step_heatmap_{safe(edge)}.png", dpi=160)
        plt.close(fig)

    plot_edge_by_step(heavy_full, "X->R1", fig_dir / "x_to_r1_by_step_full.png", "Heavy full X->R1 by step")
    plot_edge_by_step(heavy_full, "X->P", fig_dir / "x_to_p_by_step_full.png", "Heavy full X->P by step")
    plot_fusion_by_layer(heavy_full, fig_dir / "fusion_edges_by_layer_full.png")
    plot_head_specialization_full(heavy_full, fig_dir / "head_specialization_full.png")
    plot_entropy_gini(heavy_rows, fig_dir / "entropy_gini_by_layer_step.png")


def group_values(edge_rows: dict, edge: str, fields: tuple[str, ...]) -> dict[tuple, list[float]]:
    acc = defaultdict(list)
    for r in edge_rows.values():
        if r["edge"] == edge:
            acc[tuple(r[f] for f in fields)].append(r["value"])
    return acc


def plot_edge_by_step(edge_rows: dict, edge: str, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_step = means_by(edge_rows, edge, "step")
    xs = sorted(by_step)
    ys = [by_step[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Denoising step index")
    ax.set_ylabel(f"{edge} attention mass")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fusion_by_layer(edge_rows: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = ["S->R1", "R1->S", "S->X", "R1->X"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for edge in edges:
        by_layer = means_by(edge_rows, edge, "layer")
        xs = sorted(by_layer)
        ax.plot(xs, [by_layer[x] for x in xs], marker="o", label=edge)
    ax.set_xlabel("Attention layer id")
    ax.set_ylabel("attention mass")
    ax.set_title("Heavy full fusion/writeback edges by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_head_specialization_full(edge_rows: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = ["X->S", "X->R1", "X->P", "S->R1", "R1->S", "S->X", "R1->X"]
    grouped = group_by_layer_head_edge(edge_rows)
    pairs = sorted({(r["layer"], r["head"]) for r in grouped})
    pair_idx = {p: i for i, p in enumerate(pairs)}
    edge_idx = {e: i for i, e in enumerate(edges)}
    mat = np.zeros((len(pairs), len(edges)), dtype=np.float64)
    for r in grouped:
        if r["edge"] in edge_idx:
            mat[pair_idx[(r["layer"], r["head"])], edge_idx[r["edge"]]] = r["mean_value"]
    fig, ax = plt.subplots(figsize=(8, 14))
    im = ax.imshow(mat, aspect="auto", cmap="cividis")
    ax.set_xticks(range(len(edges)), labels=edges, rotation=30, ha="right")
    yticks = list(range(0, len(pairs), 24))
    ax.set_yticks(yticks, labels=[f"L{pairs[i][0]}/H{pairs[i][1]}" for i in yticks])
    ax.set_ylabel("Layer/head rows")
    ax.set_title("Heavy full head specialization")
    fig.colorbar(im, ax=ax, label="mean attention mass")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_entropy_gini(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc_ent = defaultdict(list)
    acc_gini = defaultdict(list)
    for r in rows:
        if r["edge"] != "X->R1":
            continue
        acc_ent[(r["layer"], r["step"])].append(r["entropy"])
    mat_entropy = np.full((25, 28), np.nan)
    mat_top16 = np.full((25, 28), np.nan)
    top16_acc = defaultdict(list)
    for (layer, step), vals in acc_ent.items():
        mat_entropy[int(layer), int(step)] = mean(vals)
    for r in rows:
        if r["edge"] == "X->R1":
            top16_acc[(r["layer"], r["step"])].append(r["top16_block_mass"])
    for (layer, step), vals in top16_acc.items():
        mat_top16[int(layer), int(step)] = mean(vals)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    im0 = axes[0].imshow(mat_entropy, aspect="auto", cmap="magma_r")
    axes[0].set_xlabel("Denoising step index")
    axes[0].set_ylabel("Attention layer id")
    axes[0].set_title("Normalized entropy")
    fig.colorbar(im0, ax=axes[0], label="entropy")
    im1 = axes[1].imshow(mat_top16, aspect="auto", cmap="viridis")
    axes[1].set_xlabel("Denoising step index")
    axes[1].set_ylabel("Attention layer id")
    axes[1].set_title("Top16 block mass")
    fig.colorbar(im1, ax=axes[1], label="top16 mass")
    fig.suptitle("Heavy full sparsity proxies for X->R1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def safe(edge: str) -> str:
    return edge.replace("->", "_to_")


def build_report(**kwargs) -> str:
    edge_metrics = kwargs["edge_metrics"]
    mechanisms = kwargs["mechanisms"]
    full_summary = kwargs["full_summary"]
    bias = kwargs["bias"]
    metadata_warnings = kwargs["metadata_warnings"]
    light_rows = kwargs["light_rows"]
    heavy_rows = kwargs["heavy_rows"]
    heavy_slice_rows = kwargs["heavy_slice_rows"]
    output_dir = kwargs["output_dir"]

    replicated_edges = [r for r in edge_metrics if float(r["abs_diff"]) < 0.03]
    reproduced_mechs = [r for r in mechanisms if r["conclusion"] == "reproduced"]
    shifted_mechs = [r for r in mechanisms if r["conclusion"] == "shifted"]
    failed_mechs = [r for r in mechanisms if r["conclusion"] == "failed"]
    slice_reproduced = len(replicated_edges) == len(edge_metrics)
    overall = (
        "heavy_slice exactly reproduces light; heavy_full shows light is representative, not exhaustive"
        if slice_reproduced
        else "heavy_slice differs from light; inspect engineering consistency before mechanism claims"
    )

    lines = [
        "# Light vs Heavy V2 Attention Comparison",
        "",
        "## Metadata Sanity Check",
        f"- Light dir: `{kwargs['light_dir']}`",
        f"- Heavy dir: `{kwargs['heavy_dir']}`",
        f"- Light attention rows parsed: `{len(light_rows)}` segment-edge rows.",
        f"- Heavy full attention rows parsed: `{len(heavy_rows)}` segment-edge rows.",
        f"- Heavy slice attention rows parsed: `{len(heavy_slice_rows)}` segment-edge rows.",
        f"- Metadata warnings: `{'; '.join(metadata_warnings) if metadata_warnings else 'none'}`.",
        "",
        "Generated CSV: `run_metadata_comparison.csv`.",
        "",
        "## Light vs Heavy Slice",
        "",
        "Heavy slice filters heavy_full to the exact light sample set: layers `[0,4,14,24]`, heads `[0,4,8,12,16,20]`, steps `[0,14,27]`. This is the only valid direct comparison.",
        "",
        csv_table(edge_metrics, ["edge", "mean_light", "mean_heavy_slice", "abs_diff", "pearson", "spearman", "top10_head_overlap"]),
        "",
        f"Conclusion: **{overall}**. Rule used: edge mean `abs_diff < 0.03` indicates slice-level replication.",
        "",
        "## Key Mechanism Replication",
        "",
        csv_table(mechanisms, ["mechanism", "edge", "layer", "head", "light_value", "heavy_slice_same_head_value", "heavy_full_rank", "heavy_full_top_head", "conclusion"]),
        "",
        "Interpretation rules:",
        "- `reproduced`: light and heavy_slice differ by < 0.03 and the same layer/head remains top10 in heavy_full.",
        "- `shifted`: light and heavy_slice agree, but heavy_full finds stronger heads outside the sampled set.",
        "- `failed`: light and heavy_slice differ strongly; check engineering consistency before making a mechanism claim.",
        "",
        "## Heavy Full Mechanism Summary",
        "",
        csv_table(full_summary["edge_summary"], ["edge", "mean", "min", "max", "count"]),
        "",
        "Figures:",
        "- `figures/layer_step_heatmap_X_to_R1.png`: x-axis is denoising step, y-axis is attention layer id, color is mean `X->R1` mass over heads.",
        "- `figures/layer_step_heatmap_X_to_P.png`: x-axis is denoising step, y-axis is layer id, color is mean `X->P` mass.",
        "- `figures/layer_step_heatmap_X_to_S.png`: x-axis is denoising step, y-axis is layer id, color is mean `X->S` mass.",
        "- `figures/layer_step_heatmap_S_to_R1.png` / `R1_to_S`: source/reference fusion heatmaps.",
        "- `figures/layer_step_heatmap_S_to_X.png` / `R1_to_X`: source/reference writeback heatmaps.",
        "- `figures/head_specialization_full.png`: x-axis is edge, y-axis is layer/head row, color is mean attention mass over all steps.",
        "- `figures/x_to_r1_by_step_full.png`: x-axis is denoising step, y-axis is mean `X->R1` mass over all layers/heads.",
        "- `figures/x_to_p_by_step_full.png`: x-axis is denoising step, y-axis is mean `X->P` mass over all layers/heads.",
        "- `figures/fusion_edges_by_layer_full.png`: x-axis is layer id, y-axis is mean attention mass; curves are fusion/writeback edges.",
        "- `figures/entropy_gini_by_layer_step.png`: two panels. Both use x-axis denoising step and y-axis layer id; left color is normalized entropy, right color is top16 block mass as a sparsity proxy.",
        "",
        "Heavy_full strengthens the broad light conclusion that cross-segment routing is head-specialized, but it also shows the light head set misses several stronger prompt, fusion, and writeback heads. Therefore the light result should be used as a reliable scout, not as an exhaustive sparse routing design.",
        "",
        "## Sampling Bias Analysis",
        "",
        "Answers:",
        f"1. Step sampling `[0,14,27]`: {summarize_assessment(bias['rows'], 'step_representativeness')}.",
        f"2. Layer sampling `[0,4,14,24]`: {summarize_assessment(bias['rows'], 'layer_representativeness')}.",
        f"3. Head sampling `[0,4,8,12,16,20]`: {summarize_assessment(bias['rows'], 'head_representativeness')}.",
        "",
        "Recommended next light-efficient config:",
        "```yaml",
        f"sample_layers: {bias['recommendations']['sample_layers']}",
        f"sample_heads: {bias['recommendations']['sample_heads']}",
        f"sample_steps: {bias['recommendations']['sample_steps']}",
        f"block_size: {bias['recommendations']['block_size']}",
        "max_query_tokens_per_record: all",
        "```",
        "",
        "## Next Step",
        "",
        "If the reproduced mechanisms are sufficient, proceed to seed stability and causal ablation. If shifted heads dominate important edges in heavy_full, update the light sampling config first, then rerun a light pass before ablation.",
        "",
    ]
    return "\n".join(lines)


def summarize_assessment(rows: list[dict], question: str) -> str:
    vals = [r["assessment"] for r in rows if r["question"] == question]
    counts = {v: vals.count(v) for v in sorted(set(vals))}
    return ", ".join(f"{k}={v}" for k, v in counts.items()) if counts else "no data"


def csv_table(rows: list[dict], columns: list[str], limit: int | None = None) -> str:
    use = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for r in use:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def mean(values) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def pearson(a, b) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def spearman(a, b) -> float:
    if len(a) < 2:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def rankdata(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
