from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import save_json
from .metrics_attention import iter_jsonl, js_divergence


def topk_indices(matrix, k: int) -> set[int]:
    arr = np.asarray(matrix, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return set()
    k = min(k, arr.size)
    return set(np.argpartition(arr, -k)[-k:].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def load_block_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(Path(path) / "attention_blocks.jsonl") or [])


def compare_low_high(low_dir: str | Path, high_dir: str | Path, output_path: str | Path, top_k: int = 64) -> dict[str, Any]:
    low_records = load_block_records(low_dir)
    high_records = load_block_records(high_dir)
    pairs = []
    for low, high in zip(low_records, high_records):
        for low_head, high_head in zip(low.get("heads", []), high.get("heads", [])):
            lm = np.asarray(low_head.get("matrix", []), dtype=np.float64)
            hm = np.asarray(high_head.get("matrix", []), dtype=np.float64)
            if lm.size == 0 or hm.size == 0:
                continue
            hd = downsample_matrix(hm, lm.shape)
            pairs.append(
                {
                    "step_idx": low.get("step_idx"),
                    "layer_id": low.get("layer_id"),
                    "head": low_head.get("head"),
                    "topk_jaccard": jaccard(topk_indices(lm, top_k), topk_indices(hd, top_k)),
                    "js_divergence": js_divergence(lm, hd),
                    "corr": corrcoef_safe(lm.reshape(-1), hd.reshape(-1)),
                    "low_top_ref": max_ref(low_head.get("segment_mass") or {}),
                    "high_top_ref": max_ref(high_head.get("segment_mass") or {}),
                }
            )
    for row in pairs:
        row["reference_binding_consistent"] = row["low_top_ref"] == row["high_top_ref"]
    report = {
        "num_pairs": len(pairs),
        "mean_topk_jaccard": float(np.mean([p["topk_jaccard"] for p in pairs])) if pairs else None,
        "mean_js_divergence": float(np.mean([p["js_divergence"] for p in pairs])) if pairs else None,
        "mean_corr": float(np.mean([p["corr"] for p in pairs])) if pairs else None,
        "reference_binding_consistency": float(np.mean([p["reference_binding_consistent"] for p in pairs])) if pairs else None,
        "pairs": pairs,
    }
    save_json(report, output_path)
    return report


def downsample_matrix(matrix: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src = np.asarray(matrix, dtype=np.float64)
    tq, tk = target_shape
    out = np.zeros((tq, tk), dtype=np.float64)
    for i in range(tq):
        qs = int(i * src.shape[0] / tq)
        qe = int((i + 1) * src.shape[0] / tq)
        for j in range(tk):
            ks = int(j * src.shape[1] / tk)
            ke = int((j + 1) * src.shape[1] / tk)
            block = src[qs:max(qe, qs + 1), ks:max(ke, ks + 1)]
            out[i, j] = block.mean() if block.size else 0.0
    return out


def corrcoef_safe(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def max_ref(segment_mass: dict[str, float]) -> str | None:
    refs = {k.split("->")[1]: v for k, v in segment_mass.items() if k.startswith("X->R")}
    if not refs:
        return None
    return max(refs, key=refs.get)

