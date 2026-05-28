# Light vs Heavy V2 Attention Comparison

## Metadata Sanity Check
- Light dir: `/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new`
- Heavy dir: `/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy`
- Light attention rows parsed: `1152` segment-edge rows.
- Heavy full attention rows parsed: `269184` segment-edge rows.
- Heavy slice attention rows parsed: `1248` segment-edge rows.
- Metadata warnings: `none`.

Generated CSV: `run_metadata_comparison.csv`.

## Light vs Heavy Slice

Heavy slice filters heavy_full to the exact light sample set: layers `[0,4,14,24]`, heads `[0,4,8,12,16,20]`, steps `[0,14,27]`. This is the only valid direct comparison.

| edge | mean_light | mean_heavy_slice | abs_diff | pearson | spearman | top10_head_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| X->S | 0.0924 | 0.0924 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| X->R1 | 0.0853 | 0.0853 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| X->P | 0.2104 | 0.2104 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| S->R1 | 0.1148 | 0.1148 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| R1->S | 0.1088 | 0.1086 | 0.0003 | 0.9997 | 0.9992 | 1.0000 |
| S->X | 0.1937 | 0.1937 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| R1->X | 0.2250 | 0.2247 | 0.0002 | 1.0000 | 0.9999 | 1.0000 |
| X->X | 0.6539 | 0.6539 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| S->S | 0.5962 | 0.5962 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| R1->R1 | 0.5560 | 0.5557 | 0.0003 | 1.0000 | 0.9998 | 1.0000 |
| P->P | 0.5245 | 0.5245 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

Conclusion: **heavy_slice exactly reproduces light; heavy_full shows light is representative, not exhaustive**. Rule used: edge mean `abs_diff < 0.03` indicates slice-level replication.

## Key Mechanism Replication

| mechanism | edge | layer | head | light_value | heavy_slice_same_head_value | heavy_full_rank | heavy_full_top_head | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_transfer_mid | X->R1 | 14 | 12 | 0.6466 | 0.6466 | 2 | L19/H0 | reproduced |
| reference_transfer_double | X->R1 | 4 | 4 | 0.4645 | 0.4645 | 11 | L19/H0 | shifted |
| prompt_control_mid | X->P | 14 | 20 | 0.8860 | 0.8860 | 10 | L12/H17 | reproduced |
| prompt_control_late | X->P | 24 | 20 | 0.5880 | 0.5880 | 56 | L12/H17 | shifted |
| source_to_ref_fusion | S->R1 | 24 | 8 | 0.1656 | 0.1656 | 114 | L10/H1 | shifted |
| ref_to_source_fusion | R1->S | 24 | 8 | 0.1358 | 0.1349 | 70 | L10/H1 | shifted |
| source_writeback | S->X | 24 | 20 | 0.4601 | 0.4601 | 13 | L5/H22 | shifted |
| ref_writeback_late12 | R1->X | 24 | 12 | 0.5955 | 0.5956 | 28 | L5/H22 | shifted |
| ref_writeback_late20 | R1->X | 24 | 20 | 0.6831 | 0.6822 | 11 | L5/H22 | shifted |

Interpretation rules:
- `reproduced`: light and heavy_slice differ by < 0.03 and the same layer/head remains top10 in heavy_full.
- `shifted`: light and heavy_slice agree, but heavy_full finds stronger heads outside the sampled set.
- `failed`: light and heavy_slice differ strongly; check engineering consistency before making a mechanism claim.

## Heavy Full Mechanism Summary

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

Figures:
- `figures/layer_step_heatmap_X_to_R1.png`: x-axis is denoising step, y-axis is attention layer id, color is mean `X->R1` mass over heads.
- `figures/layer_step_heatmap_X_to_P.png`: x-axis is denoising step, y-axis is layer id, color is mean `X->P` mass.
- `figures/layer_step_heatmap_X_to_S.png`: x-axis is denoising step, y-axis is layer id, color is mean `X->S` mass.
- `figures/layer_step_heatmap_S_to_R1.png` / `R1_to_S`: source/reference fusion heatmaps.
- `figures/layer_step_heatmap_S_to_X.png` / `R1_to_X`: source/reference writeback heatmaps.
- `figures/head_specialization_full.png`: x-axis is edge, y-axis is layer/head row, color is mean attention mass over all steps.
- `figures/x_to_r1_by_step_full.png`: x-axis is denoising step, y-axis is mean `X->R1` mass over all layers/heads.
- `figures/x_to_p_by_step_full.png`: x-axis is denoising step, y-axis is mean `X->P` mass over all layers/heads.
- `figures/fusion_edges_by_layer_full.png`: x-axis is layer id, y-axis is mean attention mass; curves are fusion/writeback edges.
- `figures/entropy_gini_by_layer_step.png`: two panels. Both use x-axis denoising step and y-axis layer id; left color is normalized entropy, right color is top16 block mass as a sparsity proxy.

Heavy_full strengthens the broad light conclusion that cross-segment routing is head-specialized, but it also shows the light head set misses several stronger prompt, fusion, and writeback heads. Therefore the light result should be used as a reliable scout, not as an exhaustive sparse routing design.

## Sampling Bias Analysis

Answers:
1. Step sampling `[0,14,27]`: representative=7.
2. Layer sampling `[0,4,14,24]`: covered=5, missed_top3=2.
3. Head sampling `[0,4,8,12,16,20]`: covered=3, missed_top10=4.

Recommended next light-efficient config:
```yaml
sample_layers: [0, 1, 2, 4, 5, 9, 10, 14, 23, 24]
sample_heads: [0, 1, 3, 4, 8, 10, 12, 15, 16, 19, 20, 21, 22]
sample_steps: [0, 7, 14, 21, 27]
block_size: 256
max_query_tokens_per_record: all
```

## Next Step

If the reproduced mechanisms are sufficient, proceed to seed stability and causal ablation. If shifted heads dominate important edges in heavy_full, update the light sampling config first, then rerun a light pass before ablation.
