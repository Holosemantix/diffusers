#!/usr/bin/env python3
"""
FLUX.2 Klein Priority 0: Full-Attention Anatomy with Spatial Block Analysis.

Reads heavy_v2 attention_blocks.jsonl and performs systematic analysis across:
- timestep x layer x head x query_segment x key_segment x spatial_block

Outputs:
- Block-level heatmaps for key heads
- Spatial overlay maps on source/reference/generated images
- Sparsity analysis tables and figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HEAVY_DIR = Path("/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy")
OUTPUT_DIR = Path("/home/ag/projects_anguo/results/attention_i2i/priority0_systematic")

# Spatial mapping constants
IMG_H, IMG_W = 1248, 832
LATENT_H, LATENT_W = IMG_H // 8, IMG_W // 8  # 156x104
PATCH_H, PATCH_W = 2, 2  # FLUX patch embedding
TOKEN_H, TOKEN_W = LATENT_H // PATCH_H, LATENT_W // PATCH_W  # 78x52 = 4056
PIXELS_PER_TOKEN_H = IMG_H // TOKEN_H  # 16
PIXELS_PER_TOKEN_W = IMG_W // TOKEN_W  # 16
BLOCK_SIZE = 256

# Key heads to analyze spatially (from heavy_full ranking + light findings)
KEY_HEADS = [
    # (name, edge, layer, head, step_range)
    ("ref_transfer_top", "X->R1", 19, 0, range(10, 20)),
    ("ref_transfer_light", "X->R1", 14, 12, range(0, 28)),
    ("ref_transfer_double", "X->R1", 4, 4, range(0, 20)),
    ("prompt_control_top", "X->P", 12, 17, range(0, 28)),
    ("prompt_control_light", "X->P", 14, 20, range(0, 28)),
    ("writeback_top", "R1->X", 5, 22, range(5, 20)),
    ("writeback_late", "R1->X", 24, 20, range(20, 28)),
    ("fusion_top", "S->R1", 10, 1, range(10, 25)),
    ("fusion_late", "S->R1", 24, 8, range(20, 28)),
]

EDGES = [
    "X->S", "X->R1", "X->P", "S->R1", "R1->S", "S->X", "R1->X",
    "X->X", "S->S", "R1->R1", "P->P",
]


def load_segment_map(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data.get("ranges", {})


def token_to_pixel_grid(token_idx: int, token_offset: int, token_h: int, token_w: int) -> tuple[int, int, int, int]:
    """Map a token index within a segment to pixel coordinates [y1, y2, x1, x2]."""
    rel_idx = token_idx - token_offset
    row = rel_idx // token_w
    col = rel_idx % token_w
    y1 = row * PIXELS_PER_TOKEN_H
    y2 = y1 + PIXELS_PER_TOKEN_H
    x1 = col * PIXELS_PER_TOKEN_W
    x2 = x1 + PIXELS_PER_TOKEN_W
    return y1, y2, x1, x2


def block_to_token_range(block_idx: int, block_size: int) -> tuple[int, int]:
    return block_idx * block_size, (block_idx + 1) * block_size


def get_segment_blocks(segment_ranges: dict, segment: str, block_size: int) -> list[int]:
    """Return list of block indices that overlap with given segment."""
    s, e = segment_ranges[segment]
    start_block = s // block_size
    end_block = (e + block_size - 1) // block_size
    return list(range(start_block, end_block))


def load_attention_blocks(path: Path) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(rec)
    return records


def build_head_lookup(records: list[dict]) -> dict:
    """Build lookup: (step, layer, head) -> record with matrix."""
    lookup = {}
    for rec in records:
        step = int(rec["step_idx"])
        layer = int(rec["layer_id"])
        for h in rec.get("heads", []):
            head = int(h["head"])
            lookup[(step, layer, head)] = {
                "step": step,
                "layer": layer,
                "head": head,
                "matrix": np.array(h["matrix"], dtype=np.float32),
                "entropy": float(h.get("normalized_entropy", 0)),
                "top16": float(h.get("top16_block_mass", 0)),
                "segment_mass": h.get("segment_mass", {}),
            }
    return lookup


def render_block_spatial_map(
    matrix: np.ndarray,
    query_segment: str,
    key_segment: str,
    segment_ranges: dict,
    img_shape: tuple[int, int],
    title: str = "",
) -> np.ndarray:
    """
    Render a spatial map showing query-side attention concentration.
    For each query token/block, sum attention to key_segment blocks.
    """
    q_start, q_end = segment_ranges[query_segment]
    k_start, k_end = segment_ranges[key_segment]
    q_blocks = get_segment_blocks(segment_ranges, query_segment, BLOCK_SIZE)
    k_blocks = get_segment_blocks(segment_ranges, key_segment, BLOCK_SIZE)

    # Compute per-query-block attention to key segment
    q_block_values = np.zeros(len(q_blocks), dtype=np.float32)
    for i, qb in enumerate(q_blocks):
        if qb < matrix.shape[0]:
            q_block_values[i] = sum(matrix[qb, kb] for kb in k_blocks if kb < matrix.shape[1])

    # Normalize
    if q_block_values.max() > 0:
        q_block_values = q_block_values / q_block_values.max()

    # Render to image-sized array
    h, w = img_shape
    canvas = np.zeros((h, w), dtype=np.float32)

    for i, qb in enumerate(q_blocks):
        t0, t1 = block_to_token_range(qb, BLOCK_SIZE)
        # Clip to segment range
        t0 = max(t0, q_start)
        t1 = min(t1, q_end)
        for ti in range(t0, t1):
            y1, y2, x1, x2 = token_to_pixel_grid(ti, q_start, TOKEN_H, TOKEN_W)
            if y2 <= h and x2 <= w:
                canvas[y1:y2, x1:x2] = q_block_values[i]

    return canvas


def render_key_spatial_map(
    matrix: np.ndarray,
    query_segment: str,
    key_segment: str,
    segment_ranges: dict,
    img_shape: tuple[int, int],
) -> np.ndarray:
    """Render key-side spatial map: how much each key block is attended by query segment."""
    q_blocks = get_segment_blocks(segment_ranges, query_segment, BLOCK_SIZE)
    k_blocks = get_segment_blocks(segment_ranges, key_segment, BLOCK_SIZE)
    k_start, k_end = segment_ranges[key_segment]

    k_block_values = np.zeros(len(k_blocks), dtype=np.float32)
    for i, kb in enumerate(k_blocks):
        if kb < matrix.shape[1]:
            k_block_values[i] = sum(matrix[qb, kb] for qb in q_blocks if qb < matrix.shape[0])

    if k_block_values.max() > 0:
        k_block_values = k_block_values / k_block_values.max()

    h, w = img_shape
    canvas = np.zeros((h, w), dtype=np.float32)

    for i, kb in enumerate(k_blocks):
        t0, t1 = block_to_token_range(kb, BLOCK_SIZE)
        t0 = max(t0, k_start)
        t1 = min(t1, k_end)
        for ti in range(t0, t1):
            y1, y2, x1, x2 = token_to_pixel_grid(ti, k_start, TOKEN_H, TOKEN_W)
            if y2 <= h and x2 <= w:
                canvas[y1:y2, x1:x2] = k_block_values[i]

    return canvas


def overlay_on_image(
    background_path: Path,
    heatmap: np.ndarray,
    output_path: Path,
    title: str = "",
    alpha: float = 0.6,
    cmap: str = "hot",
) -> None:
    """Overlay heatmap on background image."""
    from matplotlib import cm
    from matplotlib.colors import Normalize

    if not background_path.exists():
        print(f"[WARN] Background image not found: {background_path}")
        return

    bg = np.array(Image.open(background_path).convert("RGB"))
    if bg.shape[:2] != heatmap.shape:
        # Resize heatmap to match background
        from PIL import Image as PILImage
        heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((bg.shape[1], bg.shape[0]), PILImage.Resampling.BILINEAR)
        heatmap = np.array(heatmap_img, dtype=np.float32) / 255.0

    norm = Normalize(vmin=0, vmax=1)
    colormap = cm.get_cmap(cmap)
    colored = colormap(norm(heatmap))[:, :, :3]  # RGB
    colored = (colored * 255).astype(np.uint8)

    # Blend
    blended = (bg * (1 - alpha) + colored * alpha).astype(np.uint8)

    img = Image.fromarray(blended)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255), font=font)
    img.save(output_path)


def plot_block_matrix_heatmap(
    matrix: np.ndarray,
    query_segment: str,
    key_segment: str,
    segment_ranges: dict,
    output_path: Path,
    title: str = "",
) -> None:
    """Plot block matrix heatmap with segment boundaries marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_blocks = get_segment_blocks(segment_ranges, query_segment, BLOCK_SIZE)
    k_blocks = get_segment_blocks(segment_ranges, key_segment, BLOCK_SIZE)

    # Extract submatrix
    submat = matrix[np.ix_(q_blocks, k_blocks)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(submat, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel(f"Key blocks ({key_segment})")
    ax.set_ylabel(f"Query blocks ({query_segment})")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="attention mass")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_priority0() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = OUTPUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    segment_ranges = load_segment_map(HEAVY_DIR / "segment_map.json")
    print(f"Segment ranges: {segment_ranges}")

    records = load_attention_blocks(HEAVY_DIR / "attention_blocks.jsonl")
    print(f"Loaded {len(records)} block records")

    lookup = build_head_lookup(records)
    print(f"Built lookup with {len(lookup)} (step, layer, head) entries")

    # --- Spatial analysis for key heads ---
    input_grid = HEAVY_DIR / "input_grid.png"
    generated = HEAVY_DIR / "generated.png"

    # We need to extract source and ref images from input_grid for overlay
    if input_grid.exists():
        grid_img = Image.open(input_grid)
        grid_w, grid_h = grid_img.size
        # Assuming 3 images in a row: source, ref, generated
        source_img = grid_img.crop((0, 0, grid_w // 3, grid_h))
        ref_img = grid_img.crop((grid_w // 3, 0, 2 * grid_w // 3, grid_h))
        source_img.save(OUTPUT_DIR / "source_crop.png")
        ref_img.save(OUTPUT_DIR / "ref_crop.png")
    else:
        source_img = None
        ref_img = None

    spatial_results = []

    for name, edge, layer, head, step_range in KEY_HEADS:
        qseg, kseg = edge.split("->", 1)
        for step in step_range:
            key = (step, layer, head)
            if key not in lookup:
                continue
            rec = lookup[key]
            mat = rec["matrix"]

            # 1. Block matrix heatmap
            plot_block_matrix_heatmap(
                mat, qseg, kseg, segment_ranges,
                fig_dir / f"blockmat_{name}_s{step}_l{layer}_h{head}.png",
                title=f"{edge} @ L{layer}/H{head}, Step {step}\n(entropy={rec['entropy']:.3f}, top16={rec['top16']:.2f})",
            )

            # 2. Query-side spatial map (overlay on generated image for X queries, on source for S queries)
            if qseg == "X" and generated.exists():
                qmap = render_block_spatial_map(mat, qseg, kseg, segment_ranges, (IMG_H, IMG_W))
                overlay_on_image(
                    generated, qmap,
                    fig_dir / f"spatial_query_{name}_s{step}.png",
                    title=f"{edge} query-side: {qseg} attends to {kseg}\nL{layer}/H{head}, Step {step}",
                    cmap="hot",
                )

            # 3. Key-side spatial map (overlay on source for S keys, on ref for R1 keys)
            if kseg == "R1" and ref_img is not None:
                kmap = render_key_spatial_map(mat, qseg, kseg, segment_ranges, (IMG_H, IMG_W))
                overlay_on_image(
                    OUTPUT_DIR / "ref_crop.png", kmap,
                    fig_dir / f"spatial_key_{name}_s{step}.png",
                    title=f"{edge} key-side: {kseg} attended by {qseg}\nL{layer}/H{head}, Step {step}",
                    cmap="hot",
                )

            # Record statistics
            q_blocks = get_segment_blocks(segment_ranges, qseg, BLOCK_SIZE)
            k_blocks = get_segment_blocks(segment_ranges, kseg, BLOCK_SIZE)
            submat = mat[np.ix_(q_blocks, k_blocks)]
            spatial_results.append({
                "name": name,
                "edge": edge,
                "step": step,
                "layer": layer,
                "head": head,
                "entropy": rec["entropy"],
                "top16_block_mass": rec["top16"],
                "segment_mass": rec["segment_mass"].get(edge, 0),
                "max_block_value": float(submat.max()),
                "mean_block_value": float(submat.mean()),
                "num_blocks_above_01": int((submat > 0.1).sum()),
                "num_blocks_above_02": int((submat > 0.2).sum()),
            })

    write_csv(table_dir / "key_head_spatial_analysis.csv", spatial_results)
    print(f"Generated {len(spatial_results)} spatial analysis records")

    # --- Systematic by-step segment flow ---
    by_step = defaultdict(lambda: defaultdict(list))
    by_layer = defaultdict(lambda: defaultdict(list))
    by_head = defaultdict(lambda: defaultdict(list))
    sparsity_records = []

    for rec in lookup.values():
        step, layer, head = rec["step"], rec["layer"], rec["head"]
        for edge in EDGES:
            val = rec["segment_mass"].get(edge, 0)
            by_step[step][edge].append(val)
            by_layer[layer][edge].append(val)
            by_head[(layer, head)][edge].append(val)
        sparsity_records.append({
            "step": step,
            "layer": layer,
            "head": head,
            "entropy": rec["entropy"],
            "top16": rec["top16"],
            "max_segment_mass": max(rec["segment_mass"].values()) if rec["segment_mass"] else 0,
            "dominant_edge": max(rec["segment_mass"], key=rec["segment_mass"].get) if rec["segment_mass"] else "",
        })

    # Write by-step table
    by_step_rows = []
    for step in sorted(by_step):
        for edge in EDGES:
            by_step_rows.append({
                "step": step,
                "edge": edge,
                "mean": float(np.mean(by_step[step][edge])),
                "std": float(np.std(by_step[step][edge])),
                "max": float(np.max(by_step[step][edge])) if by_step[step][edge] else 0,
            })
    write_csv(table_dir / "segment_flow_by_step.csv", by_step_rows)

    # Write by-layer table
    by_layer_rows = []
    for layer in sorted(by_layer):
        for edge in EDGES:
            by_layer_rows.append({
                "layer": layer,
                "edge": edge,
                "mean": float(np.mean(by_layer[layer][edge])),
                "std": float(np.std(by_layer[layer][edge])),
                "max": float(np.max(by_layer[layer][edge])) if by_layer[layer][edge] else 0,
            })
    write_csv(table_dir / "segment_flow_by_layer.csv", by_layer_rows)

    # Write sparsity table
    write_csv(table_dir / "sparsity_by_head.csv", sparsity_records)

    # --- Generate summary plots ---
    plot_systematic_figures(by_step, by_layer, sparsity_records, fig_dir)

    print(f"All outputs written to {OUTPUT_DIR}")
    return 0


def plot_systematic_figures(by_step, by_layer, sparsity_records, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Cross-segment edges by step
    cross_edges = ["X->S", "X->R1", "X->P", "S->R1", "R1->S", "S->X", "R1->X"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for edge in cross_edges:
        steps = sorted(by_step)
        vals = [np.mean(by_step[s][edge]) for s in steps]
        ax.plot(steps, vals, marker="o", label=edge, linewidth=1.5)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Mean attention mass")
    ax.set_title("Cross-segment edges by step (heavy full)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_edges_by_step.png", dpi=160)
    plt.close(fig)

    # 2. All edges by layer
    edges_for_layer = ["X->S", "X->R1", "X->P", "S->R1", "R1->S", "S->X", "R1->X"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, edge in enumerate(edges_for_layer):
        ax = axes[idx]
        layers = sorted(by_layer)
        vals = [np.mean(by_layer[l][edge]) for l in layers]
        ax.plot(layers, vals, marker="o", linewidth=1.5)
        ax.set_title(edge)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean mass")
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle("Cross-segment edges by layer (heavy full)")
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_edges_by_layer.png", dpi=160)
    plt.close(fig)

    # 3. Sparsity: entropy by layer and head
    df = {}
    for r in sparsity_records:
        df[(r["layer"], r["head"])] = r["entropy"]

    layers = sorted({k[0] for k in df})
    heads = sorted({k[1] for k in df})
    mat = np.full((len(layers), len(heads)), np.nan)
    for i, l in enumerate(layers):
        for j, h in enumerate(heads):
            mat[i, j] = df.get((l, h), np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([str(h) for h in heads])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Normalized entropy by layer x head (lower = sparser)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_by_layer_head.png", dpi=160)
    plt.close(fig)

    # 4. Top sparse heads bar chart with dominant edge
    sparse_sorted = sorted(sparsity_records, key=lambda x: x["entropy"])[:30]
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"L{r['layer']}/H{r['head']}/S{r['step']}" for r in sparse_sorted]
    vals = [r["entropy"] for r in sparse_sorted]
    colors = plt.cm.tab10([hash(r["dominant_edge"]) % 10 for r in sparse_sorted])
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized entropy (lower = sparser)")
    ax.set_title("Top 30 sparsest (layer, head, step) and their dominant edge")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_sparse_heads.png", dpi=160)
    plt.close(fig)


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
    raise SystemExit(analyze_priority0())
