# FLUX.2 Klein Attention Probe

NPU-first tooling for multi-reference FLUX.2 Klein image-editing inference diagnostics. The goal is not training and not sparse attention yet. The first target is full-attention teacher analysis: how noisy target tokens, source/reference image tokens, and prompt tokens exchange information across layer, head, and timestep.

## Compatibility Notes

The provided target platform is Ascend 910B2 with 64 GB HBM, CANN/npu-smi 25.5.1, `torch==2.1.0`, and `torch-npu==2.1.0.post8.dev20241029`. This project defaults to `backend: npu` and does not call CUDA APIs unless `--backend cuda` is explicitly selected.

`diffusers` must contain `Flux2KleinPipeline`. If your pip build does not:

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

This repository can also import the local checkout at `../diffusers/src` via `diffusers_src` in the YAML configs.

## Install

Install PyTorch, CANN, and `torch_npu` using your NPU platform instructions first. Then install the Python packages:

```bash
pip install -r requirements.txt
pip install git+https://github.com/huggingface/diffusers.git
```

On your current ModelArts-style image, many packages are already present. If `python` is unavailable, use `python3`.

## Environment Check

```bash
python3 scripts/check_env.py --config configs/npu_config_template.yaml
```

To also attempt model loading:

```bash
python3 scripts/check_env.py --config configs/npu_config_template.yaml --check-model-load
```

Expected output:

- package versions for Python, PyTorch, `torch_npu`, diffusers, transformers, accelerate
- `Flux2KleinPipeline` import status
- NPU availability, device count/name, memory API results when supported
- `outputs/env_report.json`

If `Flux2KleinPipeline` import fails, install diffusers main or point `diffusers_src` to a local source checkout that contains `pipelines/flux2/pipeline_flux2_klein.py`.

## Minimal Single-Reference Probe

Edit `configs/probe_single_ref.yaml` and set a real image path:

```yaml
refs:
  - /path/to/ref1.png
```

Run a small smoke test:

```bash
python3 scripts/run_probe.py --config configs/probe_single_ref.yaml
```

This defaults to 512x512, one denoising step, one layer, one head, and `max_query_tokens_per_record: 512`.

## Multi-Reference Probe

Edit `configs/probe_multi_ref.yaml`:

```yaml
source: /path/to/source.png
refs:
  - /path/to/ref1.png
  - /path/to/ref2.png
  - /path/to/ref3.png
prompt: "..."
reference_roles:
  R1: identity
  R2: clothing
  R3: background
region_annotations:
  face: R1
  clothing: R2
  background: R3
```

Run:

```bash
python3 scripts/run_probe.py --config configs/probe_multi_ref.yaml
```

Override from CLI:

```bash
python3 scripts/run_probe.py \
  --config configs/probe_multi_ref.yaml \
  --source /path/source.png \
  --refs /path/ref1.png /path/ref2.png /path/ref3.png \
  --prompt "..." \
  --height 1024 --width 1024 \
  --num_inference_steps 4 \
  --seed 0 \
  --backend npu \
  --output_dir outputs/exp001
```

## Low/High Resolution Consistency

Start with 512 vs 768. Move to 768 vs 1024 or 1024 vs 1536 only after memory is confirmed.

```bash
python3 scripts/run_low_high_res_probe.py --config configs/probe_low_high_res.yaml
```

Output:

- `low_high_res_consistency.json`
- low/high subdirectories with normal probe outputs
- figures created by each probe

Metrics include segment-level correlation, block top-k overlap/Jaccard, JS divergence, reference binding consistency, and seed-stability scaffolding.

## Reference Ablation

The first implementation is diagnostic mode: it does not mutate model attention, but creates the ablation plan and uses full-attention edge importance as teacher signal.

```bash
python3 scripts/run_ref_ablation.py --config configs/probe_multi_ref.yaml --probe_dir outputs/exp_multi_ref
```

Masking-mode TODO is recorded in `ref_ablation/ref_ablation_diagnostic_report.json`. This is deliberate because the exact processor mask insertion point must be validated against the current `Flux2KleinPipeline` and NPU attention kernel behavior.

## Outputs

Each probe writes:

- `generated.png`
- `input_grid.png`
- `load_report.json`
- `config.yaml` and `resolved_config.json`
- `segment_map.json`
- `shape_log.json`
- `attention_shapes.jsonl`
- `attention_records.jsonl`
- `attention_blocks.jsonl`
- `timing_report.json`
- `memory_report.json`
- `attention_summary.parquet`
- `segment_flow_by_step_layer_head.parquet`
- `distribution_metrics.parquet`
- `wrong_reference_activation.parquet`
- `figures/*.png`
- `summary_report.md`

If parquet write fails, the code falls back to CSV or JSON.

## Interpreting Results

- High `X->S`: source preservation is strong.
- High `X->Rk`: target/noisy tokens directly read reference k.
- High `Ri->Rj`: reference images directly mix, which may indicate useful fusion or contamination risk.
- High `X->P`: prompt/text remains influential at that step/layer/head.
- Low attention entropy and high top-k mass: full attention teacher suggests sparse routing may be viable.
- High low/high top-k overlap: low-res scout may work as a high-res routing prior.
- High wrong-reference activation: multi-reference entanglement risk is high.

## Token Segment Logic

For FLUX.2 Klein in the inspected diffusers source:

- target noisy latent IDs use image ID time coordinate `T=0`
- condition images use `T=10,20,30,...`
- attention summary layout is normalized as `[P, X, S, R1, R2, ...]`

`segment_map.json` is inferred from `hidden_states`, `encoder_hidden_states`, and `img_ids` observed at transformer forward. If this fails, add manual ranges:

```yaml
segments:
  manual_token_ranges:
    P: [0, 512]
    X: [512, 4608]
    S: [4608, 8704]
    R1: [8704, 12800]
```

## NPU OOM Mitigation

If an experiment OOMs:

- lower `height` / `width`
- reduce `sample_layers`
- reduce `sample_heads`
- keep `save_full_attention: false`
- increase `block_size`
- reduce `max_query_tokens_per_record`
- reduce `num_inference_steps`
- avoid full 1024/1536 until 512/768 probes are stable

Every run writes `memory_report.json`; unsupported NPU memory APIs return `null` with warnings instead of failing the run.

## Known Risks

- FLUX.2 Klein support in diffusers is moving quickly.
- NPU attention kernels may not expose attention weights.
- Saving full attention maps can OOM even on 64 GB HBM.
- Segment ranges may need manual calibration if the pipeline changes input layout.
- Multi-reference input semantics are based on the current inspected `Flux2KleinPipeline`, where `image=[source, ref1, ref2...]` becomes conditioning image tokens.

