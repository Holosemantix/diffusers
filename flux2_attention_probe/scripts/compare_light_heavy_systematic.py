#!/usr/bin/env python3
"""Systematic light vs heavy slice comparison for mechanism validation."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LIGHT_DIR = Path("/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new")
HEAVY_DIR = Path("/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy")
OUTPUT_DIR = Path("/home/ag/projects_anguo/results/attention_i2i/priority0_systematic")

LIGHT_LAYERS = [0, 4, 14, 24]
LIGHT_HEADS = [0, 4, 8, 12, 16, 20]
LIGHT_STEPS = [0, 14, 27]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_attention_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
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
                    rows.append({
                        "step": step, "layer": layer, "head": h,
                        "edge": edge, "value": float(value),
                    })
    return rows


def aggregate(rows: list[dict]) -> dict[tuple, list[float]]:
    acc = defaultdict(list)
    for r in rows:
        acc[(r["step"], r["layer"], r["head"], r["edge"])].append(r["value"])
    return acc


def mean(vals) -> float:
    v = [float(x) for x in vals]
    return float(sum(v) / len(v)) if v else float("nan")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUTPUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    light_rows = load_attention_rows(LIGHT_DIR / "attention_blocks.jsonl")
    heavy_rows = load_attention_rows(HEAVY_DIR / "attention_blocks.jsonl")
    heavy_slice_rows = [r for r in heavy_rows
                        if r["layer"] in LIGHT_LAYERS and r["head"] in LIGHT_HEADS and r["step"] in LIGHT_STEPS]

    light_agg = aggregate(light_rows)
    heavy_slice_agg = aggregate(heavy_slice_rows)

    # Join and compute diff
    joined_rows = []
    keys = sorted(set(light_agg.keys()) & set(heavy_slice_agg.keys()))
    for key in keys:
        l_val = mean(light_agg[key])
        h_val = mean(heavy_slice_agg[key])
        diff = abs(l_val - h_val)
        rel = diff / max(abs(l_val), 1e-12)
        joined_rows.append({
            "step": key[0], "layer": key[1], "head": key[2], "edge": key[3],
            "light": l_val, "heavy_slice": h_val, "abs_diff": diff, "rel_diff": rel,
        })

    write_csv(table_dir / "light_vs_heavy_slice_joined.csv", joined_rows)

    # Edge-level metrics
    edge_metrics = []
    for edge in ["X->S", "X->R1", "X->P", "S->R1", "R1->S", "S->X", "R1->X", "X->X", "S->S", "R1->R1", "P->P"]:
        vals = [r for r in joined_rows if r["edge"] == edge]
        if not vals:
            continue
        light_vals = [r["light"] for r in vals]
        heavy_vals = [r["heavy_slice"] for r in vals]
        edge_metrics.append({
            "edge": edge,
            "mean_light": mean(light_vals),
            "mean_heavy_slice": mean(heavy_vals),
            "mean_abs_diff": mean([r["abs_diff"] for r in vals]),
            "max_abs_diff": max(r["abs_diff"] for r in vals),
            "pearson": pearson(light_vals, heavy_vals),
            "count": len(vals),
        })
    write_csv(table_dir / "light_vs_heavy_slice_edge_metrics.csv", edge_metrics)

    print(f"Compared {len(joined_rows)} (step,layer,head,edge) tuples")
    print(f"All edges abs_diff < 0.03: {all(r['mean_abs_diff'] < 0.03 for r in edge_metrics)}")
    return 0


def pearson(a, b) -> float:
    if len(a) < 2:
        return float("nan")
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


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
