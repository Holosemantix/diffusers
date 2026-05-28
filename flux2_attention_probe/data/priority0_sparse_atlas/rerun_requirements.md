# Rerun Requirements for True Spatial Sparse Attention

## Why a Rerun is Needed

The current heavy_v2 run uses `block_size=256` with 1D token-strip blocking.
This yields 50×50 block matrices where each block spans ~78×832 pixels.
At this resolution, 'sparse' masks still retain ~10% of the image per block,
making true spatial-region sparsity impossible.

## Proposed Rerun Configuration

```yaml
run_name: mechanism_single_ref_heavy_v2_tile
model: flux2-klein-4b
input:
  resolution: [1248, 832]
  seed: 0
  steps: 28
attention_probe:
  block_size: 64          # 2D tile: 8×8 tokens
  tile_shape: [8, 8]      # tokens per tile
  tile_2d: true
  max_query_tokens_per_record: all
  heads: all
  layers: all
  steps: all
  segments: [P, X, S, R1]
```

## Expected Outputs

- Block matrices: ~300×300 per head (vs current 50×50)
- Tile-to-pixel mapping: 128×128 px per tile (vs current 78×832 px per block)
- Spatial resolution improvement: ~50× in area granularity
- Memory: ~36× more block entries per head; manageable with NPU tiling

## Validation Plan

1. Run tile rerun for steps [0, 5, 10, 15, 20, 25, 27] as pilot (7 steps).
2. Compare tile-level vs strip-level concentration scores for same heads.
3. If concentration drops significantly with tiles, current strip-based 'sparse'
   candidates are overestimating sparsity and should be downgraded to FALLBACK.
4. Generate true 2D spatial masks overlaid on source/ref/generated images.
