# FLUX.2 Attention Probe Summary

## Outputs
- `attention_blocks.jsonl`: streamed block-level attention summaries
- `segment_map.json`: inferred or manual token segment ranges
- `segment_flow_by_step_layer_head.parquet`: directional segment flow table
- `distribution_metrics.parquet`: entropy, top-k, Gini, effective-rank metrics
- `figures/`: diagnostic plots

## Interpretation Notes
- High `X->S` means stronger source preservation.
- High `X->Rk` means target tokens directly read reference k.
- High `Ri->Rj` means direct reference-reference mixing or contamination risk.
- High `X->P` means prompt/text tokens remain influential at that stage.
- Low entropy and high top-k mass suggest sparse routing may be viable.
- High low/high top-k overlap supports low-res scout as a high-res routing prior.

## Run Report
```json
{
  "num_attention_block_records": 701
}
```
