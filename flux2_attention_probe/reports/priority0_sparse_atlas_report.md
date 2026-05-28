# Sparse Attention Atlas for FLUX.2 Klein Priority 0 (Heavy v2)

**Date:** 2026-05-27  
**Model:** FLUX.2-klein-4B (NPU)  
**Input:** 1248×832, seed=0, 28 denoising steps  
**Block size:** 256 tokens (50×50 block matrix)  

## 1. Atlas Philosophy

This atlas converts the **full-attention teacher** (heavy_v2 run) into an
**interpretable sparse routing map**. Instead of listing top heads, we classify
every (layer, head, edge) combination into three categories:

- **SPARSE** — Safe to approximate with a sparse mask; high concentration, stable spatial position.
- **DENSE** — Must remain fully dense; diffuse attention or globally required coverage.
- **FALLBACK** — Conditionally sparse; needs a density fallback mechanism for certain steps or spatial regions.

### Decision Summary

| Category | Count | Percentage |
|----------|-------|------------|
| SPARSE | 5143 | 53.6% |
| DENSE | 4432 | 46.2% |
| FALLBACK | 25 | 0.3% |

## 2. Top Sparse Candidates

These edges have the highest confidence for sparsification, with recommended density and expected retained mass.

| Edge | Layer | Head | Confidence | Density | Retained Mass | Rationale |
|------|-------|------|------------|---------|---------------|-----------|
| X->X | 0 | 0 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.837>0.80... |
| X->X | 0 | 1 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.952>0.80... |
| R1->R1 | 0 | 1 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.849>0.80... |
| X->X | 0 | 2 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.969>0.80... |
| S->S | 0 | 2 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.852>0.80... |
| R1->R1 | 0 | 2 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.865>0.80... |
| X->X | 0 | 3 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.859>0.80... |
| X->X | 0 | 5 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.949>0.80... |
| S->S | 0 | 5 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.807>0.80... |
| R1->R1 | 0 | 5 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.802>0.80... |
| X->X | 0 | 6 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.932>0.80... |
| X->X | 0 | 7 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.908>0.80... |
| X->X | 0 | 8 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.931>0.80... |
| X->X | 0 | 9 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.943>0.80... |
| X->X | 0 | 11 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.939>0.80... |
| X->X | 0 | 12 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.964>0.80... |
| S->S | 0 | 12 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.814>0.80... |
| X->X | 0 | 13 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.873>0.80... |
| P->X | 0 | 14 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.840>0.80... |
| X->X | 0 | 14 | 0.95 | N/A | N/A | SPARSE: segment_dominant=0.839>0.80... |

## 3. Dense-Required Combinations

These (layer, head) pairs have >=50% of their edges classified as DENSE and must stay fully dense:

| Layer | Head | Dense Edges | Total Edges | Ratio |
|-------|------|-------------|-------------|-------|
| 4 | 1 | 16 | 16 | 1.00 |
| 5 | 13 | 15 | 16 | 0.94 |
| 5 | 16 | 15 | 16 | 0.94 |
| 6 | 2 | 15 | 16 | 0.94 |
| 6 | 21 | 15 | 16 | 0.94 |
| 6 | 23 | 15 | 16 | 0.94 |
| 7 | 11 | 15 | 16 | 0.94 |
| 7 | 12 | 15 | 16 | 0.94 |
| 8 | 9 | 15 | 16 | 0.94 |
| 8 | 10 | 15 | 16 | 0.94 |
| 8 | 21 | 15 | 16 | 0.94 |
| 9 | 2 | 15 | 16 | 0.94 |
| 9 | 17 | 15 | 16 | 0.94 |
| 9 | 18 | 15 | 16 | 0.94 |
| 10 | 2 | 15 | 16 | 0.94 |
| 10 | 6 | 15 | 16 | 0.94 |
| 10 | 8 | 15 | 16 | 0.94 |
| 12 | 19 | 15 | 16 | 0.94 |
| 13 | 3 | 15 | 16 | 0.94 |
| 18 | 22 | 15 | 16 | 0.94 |

### Steps Requiring Full Density

No single step was flagged as globally diffuse across all edges. Step-specific fallback is sufficient.

## 4. Fallback Mechanisms

Fallback edges exhibit moderate concentration but high step-to-step variance.
Recommended strategy: start sparse at low density, but if retained mass falls below
the 90% threshold at runtime, revert to dense for that specific (step, layer, head, edge).

| Edge | Layer | Head | Confidence | Fallback Density | Retained Mass |
|------|-------|------|------------|------------------|---------------|
| X->P | 3 | 11 | 0.67 | 15.0% | 0.77 |
| X->P | 4 | 15 | 0.67 | 15.0% | 0.63 |
| X->S | 1 | 1 | 0.50 | 15.0% | 0.54 |
| X->S | 1 | 10 | 0.50 | 15.0% | 0.60 |
| X->S | 1 | 11 | 0.50 | 15.0% | 0.68 |
| X->S | 1 | 18 | 0.50 | 15.0% | 0.61 |
| S->X | 4 | 18 | 0.50 | 15.0% | 0.56 |
| X->S | 5 | 7 | 0.50 | 15.0% | 0.59 |
| S->X | 10 | 0 | 0.50 | 15.0% | 0.55 |
| R1->S | 10 | 0 | 0.50 | 15.0% | 0.65 |
| S->X | 10 | 18 | 0.50 | 15.0% | 0.47 |
| S->R1 | 11 | 22 | 0.50 | 15.0% | 0.64 |
| X->S | 13 | 16 | 0.50 | 15.0% | 0.72 |
| X->S | 14 | 8 | 0.50 | 15.0% | 0.61 |
| S->P | 14 | 21 | 0.50 | 15.0% | 0.51 |

## 5. Block Size Limitation & 2D Tile Rerun Proposal

### Current Limitation (block_size = 256)

- Token grid: 78 rows × 52 cols = 4,056 tokens per image segment
- Block size: 256 tokens → ~4.9 token rows × 52 cols per block
- Pixel coverage: ~78 px tall × 832 px wide per block
- **Problem:** A single block spans almost the entire image width and ~5 token rows vertically.
  This resolution is too coarse to resolve object boundaries, fine textures, or localized regions.
- **Impact on sparse masks:** Any 'sparse' block we retain still covers ~10% of the image area.
  True spatial sparsity (e.g., attending to just a face, a hand, or a specific object)
  is impossible at this block granularity.

### Proposed 2D Tile Rerun

To achieve true spatial-region sparsity, rerun with a 2D tile decomposition:

| Parameter | Current | Proposed Tile Rerun | Rationale |
|-----------|---------|---------------------|-----------|
| block_size | 256 (1D) | 64 (2D 8×8 tiles) | Finer spatial resolution |
| num_q_blocks | 50 | ~300 (78×52 / 8×8) | 2D tiling across image |
| num_k_blocks | 50 | ~300 | Same for keys |
| tile_shape | N/A (1D strips) | 8×8 tokens = 128×128 px | Near-square spatial patches |
| segment_aware | No | Yes | Tiles should respect segment boundaries |

**Implementation steps:**
1. Modify `compute_block_attention_summary` in `attention_hooks.py` to accept 2D tile dimensions.
2. For each segment (X, S, R1), compute 2D tile indices: `tile_row = token_idx // tokens_per_row // tile_h`, `tile_col = (token_idx % tokens_per_row) // tile_w`.
3. Build per-head block matrices where blocks are 2D tiles instead of 1D strips.
4. Re-run heavy_v2 with tile_size = 64 (8×8 tokens) for steps [0, 5, 10, 15, 20, 25, 27] as pilot.
5. Compare tile-level concentration vs strip-level concentration to quantify the resolution loss.

## 6. Figures

- `figures/decision_distribution.png` — Pie chart of SPARSE/DENSE/FALLBACK classification
- `figures/retained_mass_vs_density.png` — Retained mass curves for representative sparse edges
- `figures/sparse_ratio_layer_head.png` — Heatmap of sparse-safe ratio per layer-head
- `figures/entropy_by_step_key_edges.png` — Step-wise entropy for key edge types

## 7. Ablation Priority

Based on this atlas, the next ablations should prioritize:

1. **X→R1 (reference transfer)** — Strongest sparse candidate. Test 1-3% density masks
   at L19/H0 and L14/H12. Verify image quality retention.
2. **X→P (prompt control)** — Extremely concentrated (99%+ in single block).
   Even 0.5% density may suffice. Test at L12/H17.
3. **R1→X (writeback)** — Early layer (L5/H22) is highly concentrated.
   Late layer (L24/H20) is more diffuse; needs fallback.
4. **S→R1 / R1→S (fusion)** — Diffuse; do NOT sparsify without 2D tile rerun.
   Current block resolution cannot resolve fusion patterns.
5. **X→X (self-attention)** — Not analyzed as a directed edge here, but
   segment-internal self-attention is typically dense. Keep dense.
