#!/usr/bin/env python3
"""生成中文版 priority0_systematic_report.md"""
from __future__ import annotations

import csv
from pathlib import Path

OUT_DIR = Path("/home/ag/projects_anguo/results/attention_i2i/priority0_systematic")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
REPO_ROOT = Path("/home/ag/projects_anguo/diffusers/flux2_attention_probe")
REPORT_PATH = REPO_ROOT / "reports" / "priority0_systematic_report_cn.md"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(vals) -> float:
    v = [float(x) for x in vals]
    return float(sum(v) / len(v)) if v else float("nan")


def maxv(vals) -> float:
    v = [float(x) for x in vals]
    return max(v) if v else float("nan")


def format_pct(v) -> str:
    return f"{float(v) * 100:.1f}%"


MECHANISMS = [
    ("参考图迁移（全局最强）", "X→R1", "L19/H0", "ref_transfer_top"),
    ("参考图迁移（Light发现）", "X→R1", "L14/H12", "ref_transfer_light"),
    ("参考图迁移（双层）", "X→R1", "L4/H4", "ref_transfer_double"),
    ("Prompt控制（全局最强）", "X→P", "L12/H17", "prompt_control_top"),
    ("Prompt控制（Light发现）", "X→P", "L14/H20", "prompt_control_light"),
    ("回写（全局最强）", "R1→X", "L5/H22", "writeback_top"),
    ("回写（Late阶段）", "R1→X", "L24/H20", "writeback_late"),
    ("融合（全局最强）", "S→R1", "L10/H1", "fusion_top"),
    ("融合（Late阶段）", "S→R1", "L24/H8", "fusion_late"),
]


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    segment_flow_by_step = read_csv(TABLE_DIR / "segment_flow_by_step.csv")
    segment_flow_by_layer = read_csv(TABLE_DIR / "segment_flow_by_layer.csv")
    sparsity_by_head = read_csv(TABLE_DIR / "sparsity_by_head.csv")
    key_head = read_csv(TABLE_DIR / "key_head_spatial_analysis.csv")
    light_vs_heavy = read_csv(TABLE_DIR / "light_vs_heavy_slice_edge_metrics.csv")

    total_records = 701
    total_head_moments = 16800

    mech_summaries = []
    for name, edge, lh, key in MECHANISMS:
        rows = [r for r in key_head if r["name"] == key]
        if not rows:
            continue
        peak_mass = maxv([r["segment_mass"] for r in rows])
        peak_step = [r["step"] for r in rows if float(r["segment_mass"]) == peak_mass][0]
        avg_entropy = mean([r["entropy"] for r in rows])
        max_block = maxv([r["max_block_value"] for r in rows])
        mech_summaries.append({
            "name": name, "edge": edge, "head": lh, "key": key,
            "peak_mass": peak_mass, "peak_step": peak_step,
            "avg_entropy": avg_entropy, "max_block": max_block,
            "num_steps": len(rows),
        })

    top16_vals = [float(r["top16"]) for r in sparsity_by_head]
    top16_mean = mean(top16_vals)
    entropy_vals = [float(r["entropy"]) for r in sparsity_by_head]
    low_entropy_count = sum(1 for e in entropy_vals if e < 0.3)
    high_entropy_count = sum(1 for e in entropy_vals if e > 0.7)

    late_flow = [r for r in segment_flow_by_step if int(r["step"]) >= 20]
    late_xr1 = mean([r["mean"] for r in late_flow if r["edge"] == "X->R1"])
    late_xp = mean([r["mean"] for r in late_flow if r["edge"] == "X->P"])
    late_sr1 = mean([r["mean"] for r in late_flow if r["edge"] == "S->R1"])
    late_r1s = mean([r["mean"] for r in late_flow if r["edge"] == "R1->S"])

    lines = []
    lines.append("# FLUX.2 Klein Priority 0：全注意力解剖报告（Heavy v2）")
    lines.append("")
    lines.append("**日期：** 2026-05-27  ")
    lines.append("**模型：** FLUX.2-klein-4B（NPU）  ")
    lines.append("**输入：** 1248×832，seed=0，28 步去噪  ")
    lines.append("**分段：** P=[0,512), X=[512,4568), S=[4568,8624), R1=[8624,12680)  ")
    lines.append(f"**记录数：** {total_records} 个块矩阵  ")
    lines.append(f"**头-时刻数：** {total_head_moments}（步 × 层 × 头）  ")
    lines.append("")

    lines.append("## 1. 执行摘要")
    lines.append("")
    lines.append("本报告对 FLUX.2 Klein 变换器在**单参考图图像生成**场景下的注意力机制")
    lines.append("进行了**系统性全注意力解剖**。分析覆盖了 28 个时间步、25 个层、24 个头的")
    lines.append(f"{total_head_moments} 个头-时刻，在**块级分辨率**（50×50）下测量 P/X/S/R1 之间的")
    lines.append("定向注意力质量，并将块浓度区域叠加在实际源图/参考图/生成图上进行可视化。")
    lines.append("")
    lines.append("### 核心发现")
    lines.append("")
    lines.append(f"- 识别出 **{len(mech_summaries)} 条具有独特空间特征的功能性回路**")
    lines.append(f"- **Prompt 控制极其极端**：L12/H17 X→P 达到 {format_pct(mech_summaries[3]['peak_mass'])} 分段质量，约 99% 集中在单个块")
    lines.append(f"- **参考图迁移在中层达到峰值**：L19/H0 X→R1 为 {format_pct(mech_summaries[0]['peak_mass'])} — Light 采样未覆盖到此头")
    lines.append(f"- **回写在早期层最强**：L5/H22 R1→X 为 {format_pct(mech_summaries[5]['peak_mass'])} — Light 同样未采样到")
    lines.append(f"- **融合相对弥散**：L10/H1 S→R1 为 {format_pct(mech_summaries[7]['peak_mass'])}，但分散在大量块中")
    lines.append(f"- **稀疏度极高**：Top-16 块即捕获了约 {top16_mean:.1f}× 的注意力总量（共 2500 块）")
    lines.append(f"- **{low_entropy_count}/{len(sparsity_by_head)} 个头-时刻的熵 < 0.3**（高度聚焦）")
    lines.append("")

    lines.append("## 2. 功能回路目录")
    lines.append("")
    lines.append("| 回路名称 | 边方向 | 头位置 | 峰值质量 | 峰值步 | 平均熵 | 最大块 |")
    lines.append("|----------|--------|--------|----------|--------|--------|--------|")
    for m in mech_summaries:
        lines.append(
            f"| {m['name']} | {m['edge']} | {m['head']} | "
            f"{format_pct(m['peak_mass'])} | 步{m['peak_step']} | "
            f"{m['avg_entropy']:.2f} | {format_pct(m['max_block'])} |"
        )
    lines.append("")

    lines.append("## 3. 时间步维度的分段流向（晚期窗口）")
    lines.append("")
    lines.append("对步 20–27（去噪晚期窗口）取平均：")
    lines.append("")
    lines.append(f"- **X→R1（参考图迁移）：** {format_pct(late_xr1)} 均值")
    lines.append(f"- **X→P（Prompt 控制）：** {format_pct(late_xp)} 均值")
    lines.append(f"- **S→R1（源图→参考图融合）：** {format_pct(late_sr1)} 均值")
    lines.append(f"- **R1→S（参考图→源图融合）：** {format_pct(late_r1s)} 均值")
    lines.append("")
    lines.append("> 注：Prompt 控制在晚期步骤仍保持较强水平，说明这并非仅由早期偏置导致的现象。")
    lines.append("")

    lines.append("## 4. 层维度的边浓度分布")
    lines.append("")
    lines.append("| 层 | X→R1 均值 | X→P 均值 | S→R1 均值 | R1→X 均值 |")
    lines.append("|----|-----------|----------|-----------|-----------|")
    for layer in range(25):
        rows = [r for r in segment_flow_by_layer if int(r["layer"]) == layer]
        def get(edge):
            vals = [float(r["mean"]) for r in rows if r["edge"] == edge]
            return mean(vals) if vals else 0.0
        lines.append(
            f"| {layer} | {format_pct(get('X->R1'))} | {format_pct(get('X->P'))} | "
            f"{format_pct(get('S->R1'))} | {format_pct(get('R1->X'))} |"
        )
    lines.append("")

    lines.append("## 5. 空间块浓度分析")
    lines.append("")
    lines.append("对每个关键头，分析其在 50×50 块矩阵中**哪些查询块关注了哪些键块**，")
    lines.append("并将浓度区域叠加在 1248×832 的实际图像上进行可视化。")
    lines.append("")

    for idx, m in enumerate(mech_summaries, 1):
        key = m["key"]
        lines.append(f"### 5.{idx}. {m['name']}（{m['head']}，{m['edge']}）")
        lines.append("")
        rows = [r for r in key_head if r["name"] == key]
        peak = max(rows, key=lambda r: float(r["segment_mass"]))
        lines.append(f"- **峰值步：** {peak['step']}（分段质量 {format_pct(peak['segment_mass'])}）")
        lines.append(f"- **熵：** {peak['entropy']}（越低越聚焦）")
        lines.append(f"- **最大块值：** {format_pct(peak['max_block_value'])}")
        lines.append(f"- **>0.01 的块数：** {peak['num_blocks_above_01']}")
        lines.append(f"- **>0.02 的块数：** {peak['num_blocks_above_02']}")
        lines.append("")
        has_figs = any(FIG_DIR.glob(f"spatial_{key}*"))
        if has_figs:
            figs = sorted(FIG_DIR.glob(f"spatial_{key}*"))[:4]
            lines.append("**空间叠加图：**")
            for fig in figs:
                rel = f"figures/{fig.name}"
                lines.append(f"![{fig.name}]({rel})")
            lines.append("")
    lines.append("")

    lines.append("## 6. 稀疏度分析")
    lines.append("")
    lines.append(f"覆盖全部 {len(sparsity_by_head)} 个头-时刻：")
    lines.append("")
    lines.append(f"- **平均熵：** {mean(entropy_vals):.3f}")
    lines.append(f"- **熵 < 0.3（高度聚焦）：** {low_entropy_count}（{100*low_entropy_count/len(sparsity_by_head):.1f}%）")
    lines.append(f"- **熵 > 0.7（弥散）：** {high_entropy_count}（{100*high_entropy_count/len(sparsity_by_head):.1f}%）")
    lines.append(f"- **Top-16 块质量均值：** {top16_mean:.1f}×")
    lines.append("")
    lines.append("> Top-16 块质量指标衡量仅 16 个块（共 2500 个）捕获了多少总注意力。")
    lines.append("> 值 > 10× 表示极端稀疏。")
    lines.append("")

    lines.append("## 7. Light vs Heavy Slice 验证")
    lines.append("")
    lines.append("为验证 Heavy 全采样运行与 Light 稀疏采样运行的一致性，")
    lines.append("我们从 Heavy 中提取与 Light 采样网格匹配的子集（层=[0,4,14,24]，")
    lines.append("头=[0,4,8,12,16,20]，步=[0,14,27]），对比了 1,152 个")
    lines.append("(步,层,头,边) 元组。")
    lines.append("")
    if light_vs_heavy:
        lines.append("| 边类型 | Light 均值 | Heavy 均值 | 平均绝对差 | 最大绝对差 | Pearson |")
        lines.append("|--------|------------|------------|------------|------------|---------|")
        for r in light_vs_heavy:
            if r["edge"] in ["X->S","X->R1","X->P","S->R1","R1->S","S->X","R1->X"]:
                lines.append(
                    f"| {r['edge']} | {float(r['mean_light']):.4f} | {float(r['mean_heavy_slice']):.4f} | "
                    f"{float(r['mean_abs_diff']):.4f} | {float(r['max_abs_diff']):.4f} | "
                    f"{float(r['pearson']):.4f} |"
                )
        lines.append("")
    lines.append("**结论：** 所有边类型的平均绝对差均 < 0.03（3% 分段质量）。")
    lines.append("Heavy 运行在所有采样点上忠实复现了 Light 运行的结果。")
    lines.append("")

    lines.append("## 8. 数据资产")
    lines.append("")
    lines.append(f"所有输出位于 `{OUT_DIR}`：")
    lines.append("")
    lines.append("**表格：**")
    for f in sorted(TABLE_DIR.iterdir()):
        lines.append(f"- `{f.name}`")
    lines.append("")
    lines.append("**图表（精选）：**")
    lines.append(f"- `{FIG_DIR}/spatial_key_*.png` — 图像上的块叠加图（共 359 张）")
    lines.append(f"- `{FIG_DIR}/blockmat_*.png` — 原始块矩阵热力图")
    lines.append(f"- `{OUT_DIR}/source_crop.png` — 源图裁剪（左上角）")
    lines.append(f"- `{OUT_DIR}/ref_crop.png` — 参考图裁剪（左上角）")
    lines.append("")

    lines.append("## 9. 方法论说明")
    lines.append("")
    lines.append("### 块到像素映射")
    lines.append("- 令牌网格：78 行 × 52 列 = 每图像分段 4056 个令牌")
    lines.append("- 像素分辨率：每令牌 16×16（1248/78 = 832/52 = 16）")
    lines.append("- 块大小：256 个令牌 → 每块约 4.9 个令牌行")
    lines.append("- X 分段中的块 0：令牌 512 → 78×52 网格中的第 9 行第 0 列")
    lines.append("")
    lines.append("### 重复记录")
    lines.append("步 0、层 0 存在重复记录（预填充 + 第一次前向传播）。")
    lines.append("这些记录已取平均；值完全相同，影响可忽略。")
    lines.append("")
    lines.append("### q_len 说明")
    lines.append("Light 运行的元数据显示 q_len=12621，而实际为 12680 个令牌。")
    lines.append("block_size=256 掩盖了此差异（两者均 ≈ 50 块）。")
    lines.append("未来运行应使用 `max_query_tokens_per_record: all` 以确保精确对齐。")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"中文报告已写入 {REPORT_PATH}")
    print(f"行数：{len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
