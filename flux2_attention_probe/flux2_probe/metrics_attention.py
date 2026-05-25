from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, save_json


def iter_jsonl(path: str | Path):
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def gini(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    if np.min(arr) < 0:
        arr = arr - np.min(arr)
    total = arr.sum()
    if total <= 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2 * np.arange(1, n + 1).dot(arr) / total - (n + 1)) / n)


def entropy(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def js_divergence(p, q) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _kl(p, q) -> float:
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))).sum())


def effective_rank(matrix) -> float:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    try:
        s = np.linalg.svd(arr, compute_uv=False)
    except Exception:
        return 0.0
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    return float(np.exp(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum()))


def summarize_attention_records(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    reference_roles: dict[str, str] | None = None,
    region_annotations: dict[str, str] | None = None,
    top_k_blocks: int = 16,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = ensure_dir(output_dir or input_dir)
    block_path = input_dir / "attention_blocks.jsonl"

    summary_rows = []
    flow_rows = []
    dist_rows = []
    wrong_rows = []
    for rec in iter_jsonl(block_path) or []:
        common = {
            "step_idx": rec.get("step_idx"),
            "timestep": rec.get("timestep"),
            "layer_id": rec.get("layer_id"),
            "module_name": rec.get("module_name"),
            "attention_kind": rec.get("attention_kind"),
            "q_len": rec.get("q_len"),
            "k_len": rec.get("k_len"),
            "block_size": rec.get("block_size"),
        }
        for head_rec in rec.get("heads", []):
            head = head_rec.get("head")
            matrix = np.asarray(head_rec.get("matrix", []), dtype=np.float64)
            flat = matrix.reshape(-1)
            if flat.size:
                top_mass = float(np.sort(flat)[-min(top_k_blocks, flat.size) :].sum())
            else:
                top_mass = 0.0
            dist_rows.append(
                {
                    **common,
                    "head": head,
                    "entropy_blocks": entropy(flat),
                    "normalized_entropy_blocks": entropy(flat) / math.log(max(flat.size, 2)),
                    "topk_block_mass": top_mass,
                    "gini_blocks": gini(flat),
                    "effective_rank": effective_rank(matrix),
                    "mean_token_entropy": head_rec.get("mean_entropy"),
                    "normalized_token_entropy": head_rec.get("normalized_entropy"),
                    "top16_block_mass": head_rec.get("top16_block_mass"),
                }
            )
            seg_mass = head_rec.get("segment_mass") or {}
            refs = sorted({edge.split("->")[1] for edge in seg_mass if edge.startswith("X->R")})
            ref_masses = [float(seg_mass.get(f"X->{ref}", 0.0)) for ref in refs]
            if ref_masses:
                order = np.argsort(ref_masses)[::-1]
                top = ref_masses[int(order[0])]
                second = ref_masses[int(order[1])] if len(order) > 1 else 0.0
                summary_rows.append(
                    {
                        **common,
                        "head": head,
                        "reference_concentration": gini(ref_masses),
                        "top1_ref_margin": float(top - second),
                        "top_ref": refs[int(order[0])],
                    }
                )
            for edge, mass in seg_mass.items():
                source, target = edge.split("->", 1)
                flow_rows.append({**common, "head": head, "source": source, "target": target, "mass": float(mass)})
            wrong_rows.extend(
                wrong_reference_proxy(common, head, seg_mass, reference_roles or {}, region_annotations or {})
            )

    _write_table(summary_rows, output_dir / "attention_summary.parquet")
    _write_table(flow_rows, output_dir / "segment_flow_by_step_layer_head.parquet")
    _write_table(dist_rows, output_dir / "distribution_metrics.parquet")
    _write_table(wrong_rows, output_dir / "wrong_reference_activation.parquet")
    report = {
        "num_summary_rows": len(summary_rows),
        "num_flow_rows": len(flow_rows),
        "num_distribution_rows": len(dist_rows),
        "num_wrong_reference_rows": len(wrong_rows),
        "input": str(input_dir),
    }
    save_json(report, output_dir / "metrics_report.json")
    return report


def wrong_reference_proxy(common, head, seg_mass, reference_roles, region_annotations):
    rows = []
    refs = sorted({edge.split("->")[1] for edge in seg_mass if edge.startswith("X->R")})
    if not refs:
        return rows
    total = sum(float(seg_mass.get(f"X->{ref}", 0.0)) for ref in refs)
    if total <= 0:
        return rows
    expected_refs = set(region_annotations.values()) or set(reference_roles.keys())
    if not expected_refs:
        expected_refs = {refs[0]}
    wrong = sum(float(seg_mass.get(f"X->{ref}", 0.0)) for ref in refs if ref not in expected_refs)
    rows.append(
        {
            **common,
            "head": head,
            "scope": "segment_proxy",
            "expected_refs": ",".join(sorted(expected_refs)),
            "wrong_reference_activation_ratio": float(wrong / total),
            "total_target_reference_mass": float(total),
        }
    )
    return rows


def _write_table(rows: list[dict[str, Any]], parquet_path: Path) -> None:
    ensure_dir(parquet_path.parent)
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        if parquet_path.suffix == ".parquet":
            try:
                df.to_parquet(parquet_path, index=False)
            except Exception:
                df.to_csv(parquet_path.with_suffix(".csv"), index=False)
        else:
            df.to_csv(parquet_path, index=False)
    except Exception:
        with open(parquet_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)

