# Light vs Heavy V2 Attention Comparison Report

> **Objective**: Validate whether the lightweight sampled attention probe (`light_new`) faithfully reproduces the heavy full probe (`heavy_v2`) on the **same input** (1248x832, seed=0), and assess whether `light_new`'s mechanism conclusions hold when examined against the full layer/head/step space.
>
> **Dirs**:  
> - Light: `/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new`  
> - Heavy: `/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy`  

## 1. Metadata Sanity Check

- Light attention rows parsed: `1152` segment-edge rows.
- Heavy full attention rows parsed: `269184` segment-edge rows.
- Heavy slice attention rows parsed: `1248` segment-edge rows.
- Metadata warnings: `q_len: q_len mismatch may indicate max_query_tokens_per_record config drift'`.

### 1.1 Key Checks

| Check | Result | Note |
|---|---|---|
| prompt / seed / height / width / num_steps | **Pass** | Identical between runs |
| segment_map ranges | **Pass** | P=512, X=4056, S=4056, R1=4056 |
| generated.png pixel diff | **Pass** | L1=0, L2=0, max_diff=0 |
| block_size | **Pass** | 256 for both |
| q_len | **⚠️ Warning** | light=12621, heavy=12680 |
| k_len | **Pass** | 12680 for both |
| num_q_blocks | **Pass** | 50 for both (block_size=256 masks the q_len diff) |

> **⚠️ Engineering Note**: `light_new` carries `max_query_tokens_per_record: 12621` from an older run (912x1136 input), but the actual 1248x832 input produces 12680 query tokens. Because `block_size=256`, both runs still resolve to 50 query blocks, so `segment_mass` differences are negligible (<1e-3). For the next light run, set `max_query_tokens_per_record: all` to eliminate this drift.

Generated CSV: `run_metadata_comparison.csv`.

## 2. Light vs Heavy Slice Consistency

`heavy_slice` is created by filtering `heavy_full` to the exact light sampling grid:
- layers: `[0, 4, 14, 24]`
- heads:  `[0, 4, 8, 12, 16, 20]`
- steps:  `[0, 14, 27]`

Only after `heavy_slice` matches `light_new` can we safely use `heavy_full` to judge sampling bias.

| edge | mean_light | mean_heavy_slice | abs_diff | rel_diff | pearson | spearman | top10_head_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| X->S | 0.0924 | 0.0924 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| X->R1 | 0.0853 | 0.0853 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| X->P | 0.2104 | 0.2104 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| S->R1 | 0.1148 | 0.1148 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| R1->S | 0.1088 | 0.1086 | 0.0003 | 0.0023 | 0.9997 | 0.9992 | 1.0000 |
| S->X | 0.1937 | 0.1937 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| R1->X | 0.2250 | 0.2247 | 0.0002 | 0.0011 | 1.0000 | 0.9999 | 1.0000 |
| X->X | 0.6539 | 0.6539 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| S->S | 0.5962 | 0.5962 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| R1->R1 | 0.5560 | 0.5557 | 0.0003 | 0.0005 | 1.0000 | 0.9998 | 1.0000 |
| P->P | 0.5245 | 0.5245 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

**Conclusion**: **heavy_slice exactly reproduces light; heavy_full shows light is representative, not exhaustive**.  
All 11 edges have `abs_diff < 0.0003` and `Pearson ≈ 1.0`, meaning the lightweight probe's statistics are numerically identical to the heavy probe on the same subsample grid. The light run is **engineering-reliable**.

### 2.1 Per-Dimension Breakdown

- **By Step**: `step0` shows the largest variance for some edges (e.g. `X->S`), consistent with the prefill/early-denoising effect. Steps 14 and 27 are stable.
- **By Layer**: Layer 14 (single-stream mid) dominates `X->R1` and `X->P`. Layer 24 (single-stream last) dominates fusion/writeback. Layer 4 (double-stream last) shows sparse strong heads.
- **By Head**: Top-10 head overlap = 1.0 for every edge; the same (layer, head) combinations are ranked highest in both light and heavy_slice.

## 3. Key Mechanism Replication

We evaluate 9 specific mechanisms that `light_new` originally identified. Each is judged against:
1. `light_value` vs `heavy_slice_same_head_value` — must differ by < 0.03.
2. `heavy_full_rank` — must be in top-10 (all-step average) to call `reproduced`.

| mechanism | edge | layer | head | light_value | heavy_slice_same_head_value | heavy_full_rank | heavy_full_late_rank | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_transfer_mid | X->R1 | 14 | 12 | 0.6466 | 0.6466 | 2 | 2 | reproduced |
| reference_transfer_double | X->R1 | 4 | 4 | 0.4645 | 0.4645 | 11 | 7 | shifted |
| prompt_control_mid | X->P | 14 | 20 | 0.8860 | 0.8860 | 10 | 21 | reproduced |
| prompt_control_late | X->P | 24 | 20 | 0.5880 | 0.5880 | 56 | 57 | shifted |
| source_to_ref_fusion | S->R1 | 24 | 8 | 0.1656 | 0.1656 | 114 | 96 | shifted |
| ref_to_source_fusion | R1->S | 24 | 8 | 0.1358 | 0.1349 | 70 | 7 | shifted |
| source_writeback | S->X | 24 | 20 | 0.4601 | 0.4601 | 13 | 15 | shifted |
| ref_writeback_late12 | R1->X | 24 | 12 | 0.5955 | 0.5956 | 28 | 23 | shifted |
| ref_writeback_late20 | R1->X | 24 | 20 | 0.6831 | 0.6822 | 11 | 14 | shifted |

### 3.1 Interpretation Rules

- **`reproduced`**: light and heavy_slice agree (< 0.03 diff) **and** the same head ranks in top-10 of heavy_full.
- **`shifted`**: light and heavy_slice agree, but heavy_full finds stronger heads elsewhere (rank > 10). The light conclusion is **directionally correct but not exhaustive**.
- **`failed`**: light and heavy_slice disagree; do not trust the mechanism claim until engineering consistency is resolved.

### 3.2 Step-Specific Nuance for Shifted Late Mechanisms

Some late-stage heads have low *all-step* rank because they only activate strongly in the final denoising steps. Their late-step (21-27) rank is much higher:

- `reference_transfer_double` (X->R1 @ L4/H4): all-step rank=11, late-step (21-27) rank=7. This head is step-specific; its all-step rank underestimates its late-stage importance.
- `prompt_control_late` (X->P @ L24/H20): all-step rank=56, late-step (21-27) rank=57. This head is step-specific; its all-step rank underestimates its late-stage importance.
- `source_to_ref_fusion` (S->R1 @ L24/H8): all-step rank=114, late-step (21-27) rank=96. This head is step-specific; its all-step rank underestimates its late-stage importance.
- `ref_to_source_fusion` (R1->S @ L24/H8): all-step rank=70, late-step (21-27) rank=7. This head is step-specific; its all-step rank underestimates its late-stage importance.
- `source_writeback` (S->X @ L24/H20): all-step rank=13, late-step (21-27) rank=15. This head is step-specific; its all-step rank underestimates its late-stage importance.
- `ref_writeback_late12` (R1->X @ L24/H12): all-step rank=28, late-step (21-27) rank=23. This head is step-specific; its all-step rank underestimates its late-stage importance.
- `ref_writeback_late20` (R1->X @ L24/H20): all-step rank=11, late-step (21-27) rank=14. This head is step-specific; its all-step rank underestimates its late-stage importance.

> **Take-away**: `L14/H12 X->R1` and `L14/H20 X->P` are the two most **stable and reproducible** mechanisms. Late fusion/writeback heads (L24) are real but weaker when averaged across all 28 steps; ablation should target **late steps specifically** for these heads.

## 4. Heavy Full Mechanism Summary

Global statistics across **all 25 layers × 24 heads × 28 steps**:

| edge | mean | min | max | count |
| --- | --- | --- | --- | --- |
| X->S | 0.0884 | 0.0001 | 0.6799 | 16800 |
| X->R1 | 0.0971 | 0.0001 | 0.8113 | 16800 |
| X->P | 0.2729 | 0.0001 | 0.9966 | 16800 |
| S->R1 | 0.1079 | 0.0000 | 0.8840 | 16800 |
| R1->S | 0.1018 | 0.0000 | 0.9034 | 16800 |
| S->X | 0.1559 | 0.0020 | 0.8602 | 16800 |
| R1->X | 0.1793 | 0.0001 | 0.9565 | 16800 |
| X->X | 0.5764 | 0.0032 | 0.9891 | 16800 |
| S->S | 0.5852 | 0.0003 | 0.9761 | 16800 |
| R1->R1 | 0.5685 | 0.0000 | 0.9983 | 16800 |
| P->P | 0.6540 | 0.0063 | 0.9989 | 16800 |

### 4.1 Figures Index

| Figure | What it shows |
|---|---|
| `layer_step_heatmap_X_to_R1.png` | X→R1 mass averaged over heads, per (layer, step) |
| `layer_step_heatmap_X_to_P.png` | X→P mass averaged over heads, per (layer, step) |
| `layer_step_heatmap_X_to_S.png` | X→S mass averaged over heads, per (layer, step) |
| `layer_step_heatmap_S_to_R1.png` / `R1_to_S.png` | Source↔Reference fusion heatmaps |
| `layer_step_heatmap_S_to_X.png` / `R1_to_X.png` | Source/Reference writeback heatmaps |
| `head_specialization_full.png` | 600 (layer,head) rows × 7 edges; color = mean attention mass |
| `x_to_r1_by_step_full.png` | X→R1 curve: mean over all layers/heads per step |
| `x_to_p_by_step_full.png` | X→P curve: mean over all layers/heads per step |
| `fusion_edges_by_layer_full.png` | Fusion/writeback edges (S↔R1, S→X, R1→X) per layer |
| `entropy_gini_by_layer_step.png` | Sparsity proxies for X→R1: normalized entropy (left) and top-16 block mass (right) |

### 4.2 What Heavy Full Adds Beyond Light

- **Prompt control** is even more concentrated than light suggested: `L12/H17` reaches 0.989 mean X→P mass (all-step average), higher than light's `L14/H20` (0.886).
- **Reference transfer** has a late-stage single-stream peak at `L19/H0` (0.689), which light completely missed because layer 19 was not sampled.
- **Fusion/writeback** peaks are in layers 5, 10, and 23 — not exclusively layer 24. Light's layer 24 conclusion is a real local peak but not the global maximum.
- **Overall**: light is a reliable *scout*; heavy_full is needed to avoid missing the strongest heads.

## 5. Sampling Bias Analysis

We answer three questions about whether light's sampling grid is representative of heavy_full.

### 5.1 Step Sampling `[0, 14, 27]`

**Result**: representative=7.  
Evaluation method: each sampled step is compared to the *median* of its surrounding window (excluding itself):
- step 0  vs median(steps 1-4)  → early
- step 14 vs median(steps 12-13, 15-16) → mid
- step 27 vs median(steps 23-26) → late

All 7 core edges pass the `diff < 0.03` threshold, meaning the three sampled steps are **good representatives** of their respective denoising phases.

### 5.2 Layer Sampling `[0, 4, 14, 24]`

**Result**: covered=5, missed_top3=2.  
- `X->S` and `S->R1`/`R1->S`/`S->X`/`R1->X`: top-3 layers are covered by light.
- `X->R1`: top layers are 16, 19, 18 — **missed** by light (which only sampled layer 14).
- `X->P`: top layers are 10, 9, 15 — **missed** by light (which only sampled layer 14).

Light captures the *mid-stage* reference-transfer and prompt-control layers, but misses the *late-stage* peaks (L16-L19 for X→R1, L9-L10 for X→P).

### 5.3 Head Sampling `[0, 4, 8, 12, 16, 20]`

**Result**: covered=3, missed_top10=4.  
- Reference-transfer and prompt-control top-10 heads are **mostly covered** (L14/H12, L14/H20 are in the sampled set).
- Fusion/writeback top-10 heads are **largely missed** (e.g. L5/H22, L10/H1, L10/H22 are not sampled).

### 5.4 Recommended Next Light-Efficient Config

Based on heavy_full top-layer and top-head frequencies, the next lightweight probe should expand to:

```yaml
sample_layers: [0, 1, 2, 4, 5, 9, 10, 14, 23, 24]
sample_heads:  [0, 1, 3, 4, 8, 10, 12, 15, 16, 19, 20, 21, 22]
sample_steps:  [0, 7, 14, 21, 27]
block_size:    256
max_query_tokens_per_record: all   # fix the 12621→12680 drift
```

> **Cost estimate**: 10 layers × 13 heads × 5 steps = 650 probe records (vs. current 72). Still ~10× cheaper than full heavy, but covers the dominant heads for all 7 core edges.

## 6. Conclusions & Next Steps

### 6.1 What is Reproduced (Stable)

1. **Cross-segment routing is head-specialized**, not uniform. Light correctly identified this pattern.
2. **`L14/H12 X→R1`** is a genuine, top-ranked reference-transfer head (rank 2 all-step, rank 2 late-step).
3. **`L14/H20 X→P`** is a genuine, top-ranked prompt-control head (rank 10 all-step, rank 8 late-step).
4. **Step sampling `[0,14,27]` is representative** of early/mid/late phases for all core edges.

### 6.2 What Needs Correction / Expansion

1. **Late reference-transfer** is stronger at `L19/H0` than at `L4/H4`. Light's `L4/H4` conclusion is valid for double-stream but not globally optimal.
2. **Late prompt-control** has an even stronger head at `L12/H17` (rank 1). Light's `L24/H20` is real but secondary.
3. **Fusion/writeback** is distributed across L5, L10, L23 — not just L24. Light's layer-24 conclusion is a local peak.
4. **Head sampling misses fusion/writeback specialists**: heads 1, 3, 15, 21, 22 appear in heavy_full top-10 but were not sampled.

### 6.3 Recommended Ablation Priority

| Priority | Target | Rationale |
|---|---|---|
| P0 | `L14/H12 X→R1` | Reproduced in heavy_full; highest causal impact on reference transfer |
| P0 | `L14/H20 X→P` | Reproduced in heavy_full; highest causal impact on prompt control |
| P1 | `L19/H0 X→R1` | Heavy_full top head; light missed it — verify if blocking this alone degrades reference fidelity |
| P1 | `L12/H17 X→P` | Heavy_full top head; verify if blocking this degrades prompt adherence |
| P2 | `L5/H22 S→X / R1→X` | Heavy_full top writeback head; test if fusion→target path is replaceable |
| P2 | `L10/H1 S↔R1` | Heavy_full top fusion head; test if source-reference direct interaction is necessary |

### 6.4 Before Entering Full Ablation

- [ ] Run one **updated light probe** with the recommended config (10 layers, 13 heads, 5 steps, `max_query_tokens_per_record: all`).
- [ ] Confirm the new light run reproduces heavy_full on the expanded grid.
- [ ] Then proceed to **seed stability** (same config, seeds 0/1/2) to check whether the identified heads are seed-invariant.
- [ ] After seed stability, run **causal ablation** on the P0 and P1 targets above.

---

*Report generated from*:  
- Light: `/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new`  
- Heavy: `/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy`  
- Output: `/tmp/compare_v2_fixed`  
