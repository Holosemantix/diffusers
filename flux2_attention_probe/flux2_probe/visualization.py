from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir
from .metrics_attention import iter_jsonl


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def make_input_grid(images, output_path: str | Path, labels: list[str] | None = None) -> None:
    if not images:
        return
    from PIL import Image, ImageDraw

    labels = labels or [str(i) for i in range(len(images))]
    thumb_w = 256
    thumb_h = 256
    canvas = Image.new("RGB", (thumb_w * len(images), thumb_h + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(images):
        im = img.copy()
        im.thumbnail((thumb_w, thumb_h))
        x = idx * thumb_w + (thumb_w - im.width) // 2
        y = 24 + (thumb_h - im.height) // 2
        canvas.paste(im, (x, y))
        draw.text((idx * thumb_w + 8, 6), labels[idx] if idx < len(labels) else str(idx), fill="black")
    ensure_dir(Path(output_path).parent)
    canvas.save(output_path)


def summarize_figures(input_dir: str | Path) -> None:
    input_dir = Path(input_dir)
    fig_dir = ensure_dir(input_dir / "figures")
    records = list(iter_jsonl(input_dir / "attention_blocks.jsonl") or [])
    if not records:
        write_summary_report(input_dir, {"warning": "No attention block records found."})
        return
    plot_segment_flow(records, fig_dir / "segment_flow_heatmap.png")
    plot_x_to_refs(records, fig_dir / "x_to_refs_by_step.png")
    plot_ref_ref(records, fig_dir / "reference_reference_heatmap.png")
    plot_head_specialization(records, fig_dir / "head_specialization.png")
    plot_entropy(records, fig_dir / "entropy_topk_distribution.png")
    write_summary_report(input_dir, {"num_attention_block_records": len(records)})


def collect_segment_mass(records):
    rows = []
    for rec in records:
        for head in rec.get("heads", []):
            common = {"step": rec.get("step_idx"), "layer": rec.get("layer_id"), "head": head.get("head")}
            for edge, mass in (head.get("segment_mass") or {}).items():
                src, dst = edge.split("->", 1)
                rows.append({**common, "source": src, "target": dst, "mass": float(mass)})
    return rows


def plot_segment_flow(records, path: Path) -> None:
    rows = collect_segment_mass(records)
    labels = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    if not labels:
        return
    idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=np.float64)
    counts = np.zeros_like(mat)
    for r in rows:
        i = idx[r["source"]]
        j = idx[r["target"]]
        mat[i, j] += r["mass"]
        counts[i, j] += 1
    mat = mat / np.maximum(counts, 1)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Segment Flow")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_x_to_refs(records, path: Path) -> None:
    rows = [r for r in collect_segment_mass(records) if r["source"] == "X" and r["target"].startswith("R")]
    if not rows:
        return
    refs = sorted({r["target"] for r in rows})
    steps = sorted({r["step"] for r in rows if r["step"] is not None})
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for ref in refs:
        ys = []
        for step in steps:
            vals = [r["mass"] for r in rows if r["step"] == step and r["target"] == ref]
            ys.append(float(np.mean(vals)) if vals else 0.0)
        ax.plot(steps, ys, marker="o", label=f"X->{ref}")
    ax.set_xlabel("step")
    ax.set_ylabel("attention mass")
    ax.legend()
    ax.set_title("Target to References")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ref_ref(records, path: Path) -> None:
    rows = [r for r in collect_segment_mass(records) if r["source"].startswith("R") and r["target"].startswith("R")]
    if not rows:
        return
    labels = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)))
    counts = np.zeros_like(mat)
    for r in rows:
        i = idx[r["source"]]
        j = idx[r["target"]]
        mat[i, j] += r["mass"]
        counts[i, j] += 1
    mat = mat / np.maximum(counts, 1)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="magma")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("Reference-Reference Flow")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_head_specialization(records, path: Path) -> None:
    wanted = ["X->S", "X->R1", "X->R2", "X->P"]
    rows = []
    for rec in records:
        for h in rec.get("heads", []):
            seg = h.get("segment_mass") or {}
            rows.append([float(seg.get(edge, 0.0)) for edge in wanted])
    if not rows:
        return
    mat = np.asarray(rows)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, max(3, min(12, mat.shape[0] * 0.15))))
    im = ax.imshow(mat, aspect="auto", cmap="cividis")
    ax.set_xticks(range(len(wanted)), labels=wanted, rotation=30)
    ax.set_ylabel("sampled layer/head records")
    ax.set_title("Head Specialization")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_entropy(records, path: Path) -> None:
    ent = []
    top = []
    for rec in records:
        for h in rec.get("heads", []):
            ent.append(h.get("normalized_entropy", 0.0))
            top.append(h.get("top16_block_mass", 0.0))
    if not ent:
        return
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].hist(ent, bins=30)
    axes[0].set_title("Normalized Entropy")
    axes[1].hist(top, bins=30)
    axes[1].set_title("Top-k Block Mass")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary_report(input_dir: str | Path, report: dict[str, Any] | None = None) -> None:
    input_dir = Path(input_dir)
    report = report or {}
    lines = [
        "# FLUX.2 Attention Probe Summary",
        "",
        "## Outputs",
        "- `attention_blocks.jsonl`: streamed block-level attention summaries",
        "- `segment_map.json`: inferred or manual token segment ranges",
        "- `segment_flow_by_step_layer_head.parquet`: directional segment flow table",
        "- `distribution_metrics.parquet`: entropy, top-k, Gini, effective-rank metrics",
        "- `figures/`: diagnostic plots",
        "",
        "## Interpretation Notes",
        "- High `X->S` means stronger source preservation.",
        "- High `X->Rk` means target tokens directly read reference k.",
        "- High `Ri->Rj` means direct reference-reference mixing or contamination risk.",
        "- High `X->P` means prompt/text tokens remain influential at that stage.",
        "- Low entropy and high top-k mass suggest sparse routing may be viable.",
        "- High low/high top-k overlap supports low-res scout as a high-res routing prior.",
        "",
        "## Run Report",
        "```json",
        json.dumps(report, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    with open(input_dir / "summary_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

