# FLUX.2 Klein Priority 0: Full-Attention Anatomy Report (Heavy v2)

**Date:** 2026-05-27  
**Model:** FLUX.2-klein-4B (NPU)  
**Input:** 1248×832, seed=0, 28 denoising steps  
**Segments:** P=[0,512), X=[512,4568), S=[4568,8624), R1=[8624,12680)  
**Records:** 701 block matrices  
**Head-moments:** 16800 (step × layer × head)  

## 1. Executive Summary

This report presents a **systematic full-attention anatomy** of FLUX.2 Klein's transformer
during an image-to-image generation with a single reference image. The analysis covers
16800 head-moments across 28 timesteps, 25 layers, and 24 heads,
measuring directed attention mass between P/X/S/R1 segments at **block-level** (50×50)
resolution overlaid on actual source/reference/generated images.

### Key Findings

- **9 mechanistic circuits identified** with distinct spatial signatures
- **Prompt control is extreme**: L12/H17 X→P reaches 99.7% segment mass, ~99% in single block
- **Reference transfer peaks mid**: L19/H0 X→R1 at 68.8% — unknown to light run
- **Writeback is early**: L5/H22 R1→X at 95.6% — also unknown to light run
- **Fusion is diffuse**: L10/H1 S→R1 at 79.8% but spread across many blocks
- **Sparsity is high**: top-16 blocks capture ~8.9× (out of 50×50 = 2500)
- **2596/16800 head-moments have entropy < 0.3** (highly focused)

## 2. Mechanism Catalog

| Mechanism | Edge | Head | Peak Mass | Peak Step | Avg Entropy | Max Block |
|-----------|------|------|-----------|-----------|-------------|-----------|
| Reference Transfer (Top) | X→R1 | L19/H0 | 68.8% | step 16 | 0.10 | 65.1% |
| Reference Transfer (Light) | X→R1 | L14/H12 | 70.5% | step 11 | 0.35 | 71.9% |
| Reference Transfer (Double) | X→R1 | L4/H4 | 64.0% | step 17 | 0.38 | 61.1% |
| Prompt Control (Top) | X→P | L12/H17 | 99.7% | step 5 | 0.24 | 99.5% |
| Prompt Control (Light) | X→P | L14/H20 | 93.5% | step 12 | 0.52 | 91.4% |
| Writeback (Top) | R1→X | L5/H22 | 95.6% | step 17 | 0.46 | 71.4% |
| Writeback (Late) | R1→X | L24/H20 | 78.9% | step 24 | 0.58 | 50.8% |
| Fusion (Top) | S→R1 | L10/H1 | 79.8% | step 10 | 0.67 | 27.1% |
| Fusion (Late) | S→R1 | L24/H8 | 19.1% | step 26 | 0.73 | 6.5% |

## 3. Step-wise Segment Flow (Late Window)

Averaging steps 20–27 (late denoising window):

- **X→R1 (reference transfer):** 7.2% mean
- **X→P (prompt control):** 27.2% mean
- **S→R1 (source→ref fusion):** 10.6% mean
- **R1→S (ref→source fusion):** 9.9% mean

> Note: Prompt control remains strong even in late steps, confirming it is not
> just an early-bias phenomenon.

## 4. Layer-wise Edge Concentration

| Layer | X→R1 Mean | X→P Mean | S→R1 Mean | R1→X Mean |
|-------|-----------|----------|-----------|-----------|
| 0 | 2.6% | 7.4% | 6.3% | 15.7% |
| 1 | 6.1% | 20.4% | 9.1% | 25.2% |
| 2 | 7.8% | 7.1% | 10.6% | 16.9% |
| 3 | 6.4% | 10.1% | 8.7% | 7.1% |
| 4 | 15.7% | 12.1% | 11.8% | 9.3% |
| 5 | 9.9% | 31.5% | 7.0% | 21.9% |
| 6 | 8.1% | 23.6% | 10.1% | 16.3% |
| 7 | 12.5% | 27.9% | 11.4% | 19.3% |
| 8 | 9.7% | 37.2% | 9.7% | 19.6% |
| 9 | 6.8% | 51.6% | 6.6% | 25.6% |
| 10 | 4.3% | 60.0% | 12.2% | 11.1% |
| 11 | 3.7% | 25.8% | 9.8% | 8.9% |
| 12 | 10.6% | 29.0% | 8.9% | 20.9% |
| 13 | 6.2% | 28.3% | 11.3% | 9.5% |
| 14 | 9.7% | 35.0% | 8.3% | 18.0% |
| 15 | 7.6% | 40.6% | 8.2% | 15.3% |
| 16 | 23.6% | 26.2% | 12.1% | 6.4% |
| 17 | 14.1% | 20.2% | 13.1% | 9.7% |
| 18 | 16.7% | 33.2% | 10.8% | 18.4% |
| 19 | 17.5% | 31.1% | 9.0% | 16.7% |
| 20 | 15.0% | 30.1% | 9.6% | 21.2% |
| 21 | 9.2% | 24.3% | 9.3% | 14.6% |
| 22 | 4.7% | 22.6% | 17.7% | 20.7% |
| 23 | 9.7% | 26.8% | 17.9% | 27.6% |
| 24 | 4.3% | 20.1% | 20.3% | 52.7% |

## 5. Spatial Block Concentration

For each key head, we analyze **which query blocks attend to which key blocks**
in the 50×50 block matrix, overlaid on the actual 1248×832 images.

### 5.1. Reference Transfer (Top) (L19/H0, X→R1)

- **Peak step:** 16 (segment mass 68.8%)
- **Entropy:** 0.09989175773386516 (lower = more focused)
- **Max block:** 64.3%
- **Blocks above 0.01:** 29
- **Blocks above 0.02:** 26

### 5.2. Reference Transfer (Light) (L14/H12, X→R1)

- **Peak step:** 11 (segment mass 70.5%)
- **Entropy:** 0.3225126548910422 (lower = more focused)
- **Max block:** 69.3%
- **Blocks above 0.01:** 30
- **Blocks above 0.02:** 26

### 5.3. Reference Transfer (Double) (L4/H4, X→R1)

- **Peak step:** 17 (segment mass 64.0%)
- **Entropy:** 0.3606578797699605 (lower = more focused)
- **Max block:** 60.4%
- **Blocks above 0.01:** 31
- **Blocks above 0.02:** 20

### 5.4. Prompt Control (Top) (L12/H17, X→P)

- **Peak step:** 5 (segment mass 99.7%)
- **Entropy:** 0.225395975118451 (lower = more focused)
- **Max block:** 99.5%
- **Blocks above 0.01:** 16
- **Blocks above 0.02:** 16

### 5.5. Prompt Control (Light) (L14/H20, X→P)

- **Peak step:** 12 (segment mass 93.5%)
- **Entropy:** 0.5087357180797384 (lower = more focused)
- **Max block:** 90.3%
- **Blocks above 0.01:** 23
- **Blocks above 0.02:** 16

### 5.6. Writeback (Top) (L5/H22, R1→X)

- **Peak step:** 17 (segment mass 95.6%)
- **Entropy:** 0.4270379668866273 (lower = more focused)
- **Max block:** 69.2%
- **Blocks above 0.01:** 41
- **Blocks above 0.02:** 32

### 5.7. Writeback (Late) (L24/H20, R1→X)

- **Peak step:** 24 (segment mass 78.9%)
- **Entropy:** 0.5848782627030484 (lower = more focused)
- **Max block:** 50.5%
- **Blocks above 0.01:** 35
- **Blocks above 0.02:** 24

### 5.8. Fusion (Top) (L10/H1, S→R1)

- **Peak step:** 10 (segment mass 79.8%)
- **Entropy:** 0.6755995937944584 (lower = more focused)
- **Max block:** 22.2%
- **Blocks above 0.01:** 43
- **Blocks above 0.02:** 4

### 5.9. Fusion (Late) (L24/H8, S→R1)

- **Peak step:** 26 (segment mass 19.1%)
- **Entropy:** 0.7189931399273614 (lower = more focused)
- **Max block:** 6.3%
- **Blocks above 0.01:** 0
- **Blocks above 0.02:** 0


## 6. Sparsity Analysis

Across all 16800 head-moments:

- **Mean entropy:** 0.509
- **Entropy < 0.3 (highly focused):** 2596 (15.5%)
- **Entropy > 0.7 (diffuse):** 2207 (13.1%)
- **Mean top-16 block mass:** 8.9×

> The top-16 block mass metric measures how much of the total attention is captured
> by just 16 out of 2500 blocks. Values > 10× indicate extreme sparsity.

## 7. Light vs Heavy Slice Validation

To validate that the heavy run is consistent with the light (sparse-sampled) run,
we extracted the heavy subset matching the light sampling grid (layers=[0,4,14,24],
heads=[0,4,8,12,16,20], steps=[0,14,27]) and compared 1,152 (step,layer,head,edge)
tuples.

| Edge | Mean Light | Mean Heavy | Mean Abs Diff | Max Abs Diff | Pearson |
|------|------------|------------|---------------|--------------|---------|
| X->S | 0.0924 | 0.0924 | 0.0000 | 0.0000 | 1.0000 |
| X->R1 | 0.0853 | 0.0853 | 0.0000 | 0.0000 | 1.0000 |
| X->P | 0.2104 | 0.2104 | 0.0000 | 0.0000 | 1.0000 |
| S->R1 | 0.1148 | 0.1148 | 0.0000 | 0.0000 | 1.0000 |
| R1->S | 0.1088 | 0.1086 | 0.0008 | 0.0042 | 0.9997 |
| S->X | 0.1937 | 0.1937 | 0.0000 | 0.0000 | 1.0000 |
| R1->X | 0.2250 | 0.2247 | 0.0006 | 0.0028 | 1.0000 |

**Result:** All edge types have mean absolute difference < 0.03 (3% segment mass).
The heavy run faithfully reproduces the light run at all sampled points.

## 8. Data Assets

All outputs are in `/home/ag/projects_anguo/results/attention_i2i/priority0_systematic`:

**Tables:**
- `key_head_spatial_analysis.csv`
- `light_vs_heavy_slice_edge_metrics.csv`
- `light_vs_heavy_slice_joined.csv`
- `segment_flow_by_layer.csv`
- `segment_flow_by_step.csv`
- `sparsity_by_head.csv`

**Figures (selected):**
- `/home/ag/projects_anguo/results/attention_i2i/priority0_systematic/figures/spatial_key_*.png` — Block overlays on images (359 total)
- `/home/ag/projects_anguo/results/attention_i2i/priority0_systematic/figures/blockmat_*.png` — Raw block matrix heatmaps
- `/home/ag/projects_anguo/results/attention_i2i/priority0_systematic/source_crop.png` — Source image crop (upper-left)
- `/home/ag/projects_anguo/results/attention_i2i/priority0_systematic/ref_crop.png` — Reference image crop (upper-left)

## 9. Methodology Notes

### Block-to-pixel mapping
- Token grid: 78 rows × 52 cols = 4056 tokens per image segment
- Pixel resolution: 16×16 per token (1248/78 = 832/52 = 16)
- Block size: 256 tokens → ~4.9 token rows per block
- Block 0 in X segment: token 512 → row 9, col 0 in the 78×52 grid

### Duplicate records
Step 0, layer 0 has duplicate records (prefill + first forward pass).
These are averaged; values are identical so impact is negligible.

### q_len note
Light run metadata shows q_len=12621 vs actual 12680 tokens.
Block_size=256 masks this (both ≈ 50 blocks). Use `max_query_tokens_per_record: all`
in future runs for exact alignment.
