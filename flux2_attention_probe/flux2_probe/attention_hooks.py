from __future__ import annotations

import math
import inspect
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import append_jsonl, ensure_dir, save_json
from .token_segments import TokenSegmentInferer


def _torch():
    import torch

    return torch


def _flux2_module():
    import diffusers.models.transformers.transformer_flux2 as flux2

    return flux2


def _as_list_shape(x):
    return list(x.shape) if x is not None and hasattr(x, "shape") else None


@dataclass
class ProbeConfig:
    output_dir: str
    save_full_attention: bool = False
    save_block_attention: bool = True
    block_size: int = 32
    sample_layers: list[int] | str = field(default_factory=lambda: [0, 4, 8, 12, 16, 20])
    sample_heads: list[int] | str = "all"
    sample_steps: list[int] | str = field(default_factory=lambda: [0, 1, 2, 3])
    stream_to_disk: bool = True
    max_query_tokens_per_record: int | None = None
    record_shapes: bool = True


class AttentionProbeManager:
    def __init__(self, config: ProbeConfig | dict[str, Any], segment_inferer: TokenSegmentInferer | None = None):
        if isinstance(config, dict):
            config = ProbeConfig(**{k: v for k, v in config.items() if k in ProbeConfig.__dataclass_fields__})
        self.config = config
        self.output_dir = ensure_dir(config.output_dir)
        self.records_path = self.output_dir / "attention_records.jsonl"
        self.block_path = self.output_dir / "attention_blocks.jsonl"
        self.shape_path = self.output_dir / "attention_shapes.jsonl"
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.original_processors: dict[str, Any] = {}
        self.original_forward = None
        self.step_idx: int | None = None
        self.timestep: Any | None = None
        self._last_forward_timestep: str | None = None
        self._auto_step_idx = -1
        self.segment_inferer = segment_inferer or TokenSegmentInferer()
        self.enabled = True

    def register(self, pipe: Any) -> None:
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            raise RuntimeError("Cannot register attention probes: pipe.transformer not found.")
        self._patch_transformer_forward(transformer)
        layer_idx = 0
        for name, module in transformer.named_modules():
            if hasattr(module, "processor") and hasattr(module, "set_processor"):
                attn_type = module.__class__.__name__
                if "Flux2" not in attn_type and "Attention" not in attn_type:
                    continue
                original = module.processor
                self.original_processors[name] = original
                wrapper_cls = ProbedFlux2ParallelProcessor if hasattr(module, "to_qkv_mlp_proj") else ProbedFlux2DoubleProcessor
                wrapped = wrapper_cls(original, self, name, layer_idx, attn_type)
                module.set_processor(wrapped)
                layer_idx += 1
        save_json(
            {
                "num_attention_modules": len(self.original_processors),
                "modules": list(self.original_processors.keys()),
                "config": self.config.__dict__,
            },
            self.output_dir / "probe_registration.json",
        )

    def _patch_transformer_forward(self, transformer: Any) -> None:
        self.original_forward = transformer.forward
        manager = self

        def wrapped_forward(this, *args, **kwargs):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            img_ids = kwargs.get("img_ids")
            txt_ids = kwargs.get("txt_ids")
            timestep = kwargs.get("timestep")
            ts_key = str(timestep.detach().to("cpu").flatten()[0].item()) if hasattr(timestep, "detach") else str(timestep)
            if ts_key != manager._last_forward_timestep:
                manager._auto_step_idx += 1
                manager._last_forward_timestep = ts_key
            if manager.step_idx is None or manager.timestep != ts_key:
                manager.set_step(manager._auto_step_idx, ts_key)
            manager.segment_inferer.observe_transformer_inputs(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                img_ids=img_ids,
                txt_ids=txt_ids,
                step_idx=manager.step_idx,
                timestep=timestep,
            )
            return manager.original_forward(*args, **kwargs)

        transformer.forward = types.MethodType(wrapped_forward, transformer)

    def remove(self, pipe: Any | None = None) -> None:
        if pipe is not None:
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None:
                for name, module in transformer.named_modules():
                    if name in self.original_processors and hasattr(module, "set_processor"):
                        module.set_processor(self.original_processors[name])
                if self.original_forward is not None:
                    transformer.forward = self.original_forward
        self.original_processors.clear()
        self.segment_inferer.save(str(self.output_dir))

    def set_step(self, step_idx: int, timestep: Any) -> None:
        self.step_idx = int(step_idx)
        self.timestep = str(timestep)

    def should_sample(self, layer_id: int) -> bool:
        if not self.enabled:
            return False
        steps = self.config.sample_steps
        if steps != "all" and self.step_idx is not None and self.step_idx not in set(int(x) for x in steps):
            return False
        layers = self.config.sample_layers
        if layers != "all" and layer_id not in set(int(x) for x in layers):
            return False
        return True

    def select_heads(self, num_heads: int) -> list[int]:
        heads = self.config.sample_heads
        if heads == "all":
            return list(range(num_heads))
        return [h for h in (int(x) for x in heads) if 0 <= h < num_heads]

    def collect(self, record: dict[str, Any], block_record: dict[str, Any] | None = None) -> None:
        record = {"step_idx": self.step_idx, "timestep": self.timestep, **record}
        if self.config.stream_to_disk:
            append_jsonl(record, self.shape_path if record.get("kind") == "shape" else self.records_path)
            if block_record is not None:
                append_jsonl({"step_idx": self.step_idx, "timestep": self.timestep, **block_record}, self.block_path)
        else:
            self.records.append(record)

    def save(self, output_dir: str | None = None) -> None:
        out = ensure_dir(output_dir or self.output_dir)
        if self.records:
            save_json(self.records, out / "attention_records_buffered.json")
        self.segment_inferer.save(str(out))


class _ProbedFlux2ProcessorBase:
    def __init__(
        self,
        original_processor: Any,
        manager: AttentionProbeManager,
        module_name: str,
        layer_id: int,
        attention_kind: str,
    ):
        self.original_processor = original_processor
        self.manager = manager
        self.module_name = module_name
        self.layer_id = layer_id
        self.attention_kind = attention_kind

    def __getattr__(self, name: str):
        return getattr(self.original_processor, name)

    def _before_call(self, attn, hidden_states, args, kwargs):
        if self.manager.config.record_shapes:
            self.manager.collect(
                {
                    "kind": "shape",
                    "module_name": self.module_name,
                    "layer_id": self.layer_id,
                    "attention_kind": self.attention_kind,
                    "hidden_states_shape": _as_list_shape(hidden_states),
                    "args_shapes": [_as_list_shape(x) for x in args],
                    "kwargs_shapes": {k: _as_list_shape(v) for k, v in kwargs.items()},
                }
            )
        if self.manager.should_sample(self.layer_id):
            try:
                record, block_record = self._probe(attn, hidden_states, *args, **kwargs)
                self.manager.collect(record, block_record)
            except Exception as exc:
                self.manager.collect(
                    {
                        "kind": "probe_error",
                        "module_name": self.module_name,
                        "layer_id": self.layer_id,
                        "attention_kind": self.attention_kind,
                        "error": repr(exc),
                    }
                )

    def _probe(self, attn, hidden_states, *args, **kwargs):
        torch = _torch()
        flux2 = _flux2_module()
        with torch.no_grad():
            if hasattr(attn, "to_qkv_mlp_proj"):
                query, key, value = self._parallel_qkv(attn, hidden_states, args, kwargs, flux2)
                text_len = kwargs.get("num_txt_tokens", 0) or 0
            else:
                encoder_hidden_states = args[0] if len(args) > 0 else kwargs.get("encoder_hidden_states")
                query, key, value = self._double_qkv(attn, hidden_states, encoder_hidden_states, args, kwargs, flux2)
                text_len = int(encoder_hidden_states.shape[1]) if encoder_hidden_states is not None else 0

            selected_heads = self.manager.select_heads(int(query.shape[2]))
            base = {
                "kind": "attention_probe",
                "module_name": self.module_name,
                "layer_id": self.layer_id,
                "attention_kind": self.attention_kind,
                "query_shape": _as_list_shape(query),
                "key_shape": _as_list_shape(key),
                "value_shape": _as_list_shape(value),
                "selected_heads": selected_heads,
                "text_len": text_len,
                "num_ref_tokens": int(kwargs.get("num_ref_tokens", 0) or 0),
                "kv_cache_mode": kwargs.get("kv_cache_mode"),
            }
            block_record = None
            if self.manager.config.save_block_attention and selected_heads:
                block_record = compute_block_attention_summary(
                    query=query,
                    key=key,
                    heads=selected_heads,
                    block_size=int(self.manager.config.block_size),
                    max_query_tokens=self.manager.config.max_query_tokens_per_record,
                    segment_ranges=self.manager.segment_inferer.segment_map.ranges,
                )
                block_record.update(base)
                block_record["kind"] = "block_attention"
            return base, block_record

    def _double_qkv(self, attn, hidden_states, encoder_hidden_states, args, kwargs, flux2):
        image_rotary_emb = args[2] if len(args) > 2 else kwargs.get("image_rotary_emb")
        query, key, value, encoder_query, encoder_key, encoder_value = flux2._get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch_cat([encoder_query, query], dim=1)
            key = torch_cat([encoder_key, key], dim=1)
            value = torch_cat([encoder_value, value], dim=1)
        if image_rotary_emb is not None:
            query = flux2.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = flux2.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        return query, key, value

    def _parallel_qkv(self, attn, hidden_states, args, kwargs, flux2):
        image_rotary_emb = args[1] if len(args) > 1 else kwargs.get("image_rotary_emb")
        hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
        qkv, _ = hidden_states_proj.split([3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = flux2.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = flux2.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        return query, key, value

    def _call_original(self, attn, hidden_states, *args, **kwargs):
        params = set(inspect.signature(self.original_processor.__call__).parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return self.original_processor(attn, hidden_states, *args, **filtered)


class ProbedFlux2DoubleProcessor(_ProbedFlux2ProcessorBase):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        kv_cache=None,
        kv_cache_mode=None,
        num_ref_tokens: int = 0,
    ):
        args = (encoder_hidden_states, attention_mask, image_rotary_emb)
        kwargs = {"kv_cache": kv_cache, "kv_cache_mode": kv_cache_mode, "num_ref_tokens": num_ref_tokens}
        self._before_call(attn, hidden_states, args, kwargs)
        return self._call_original(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            num_ref_tokens=num_ref_tokens,
        )


class ProbedFlux2ParallelProcessor(_ProbedFlux2ProcessorBase):
    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
        kv_cache=None,
        kv_cache_mode=None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
    ):
        args = (attention_mask, image_rotary_emb)
        kwargs = {
            "kv_cache": kv_cache,
            "kv_cache_mode": kv_cache_mode,
            "num_txt_tokens": num_txt_tokens,
            "num_ref_tokens": num_ref_tokens,
        }
        self._before_call(attn, hidden_states, args, kwargs)
        return self._call_original(
            attn,
            hidden_states,
            attention_mask,
            image_rotary_emb,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            num_txt_tokens=num_txt_tokens,
            num_ref_tokens=num_ref_tokens,
        )


def torch_cat(values, dim=0):
    return _torch().cat(values, dim=dim)


def compute_block_attention_summary(query, key, heads: list[int], block_size: int, max_query_tokens: int | None, segment_ranges):
    torch = _torch()
    q = query[:, :, heads, :].detach()
    k = key[:, :, heads, :].detach()
    batch, q_len, num_heads, dim = q.shape
    k_len = k.shape[1]
    if max_query_tokens is not None and q_len > max_query_tokens:
        q_len_eff = int(max_query_tokens)
        q = q[:, :q_len_eff]
        q_len = q_len_eff

    num_q_blocks = math.ceil(q_len / block_size)
    num_k_blocks = math.ceil(k_len / block_size)
    per_head = []
    scale = 1.0 / math.sqrt(dim)
    for hidx, head in enumerate(heads):
        head_matrix = torch.zeros((num_q_blocks, num_k_blocks), device="cpu", dtype=torch.float32)
        entropy_sum = 0.0
        row_count = 0
        for qb in range(num_q_blocks):
            qs = qb * block_size
            qe = min(q_len, qs + block_size)
            q_chunk = q[:, qs:qe, hidx, :].float()
            k_head = k[:, :, hidx, :].float()
            scores = torch.matmul(q_chunk, k_head.transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)
            probs_cpu = probs.detach().to("cpu", dtype=torch.float32)
            entropy = -(probs_cpu.clamp_min(1e-12) * probs_cpu.clamp_min(1e-12).log()).sum(dim=-1)
            entropy_sum += float(entropy.sum().item())
            row_count += int(entropy.numel())
            for kb in range(num_k_blocks):
                ks = kb * block_size
                ke = min(k_len, ks + block_size)
                head_matrix[qb, kb] = probs_cpu[:, :, ks:ke].sum(dim=-1).mean()
            del scores, probs, probs_cpu
        arr = head_matrix.numpy()
        top_blocks = np.sort(arr.reshape(-1))[-min(16, arr.size) :].sum().item() if arr.size else 0.0
        per_head.append(
            {
                "head": int(head),
                "matrix": arr.tolist(),
                "mean_entropy": entropy_sum / max(row_count, 1),
                "normalized_entropy": (entropy_sum / max(row_count, 1)) / math.log(max(k_len, 2)),
                "top16_block_mass": float(top_blocks),
                "segment_mass": segment_mass_from_block_matrix(arr, block_size, segment_ranges, q_len, k_len),
            }
        )
    return {
        "block_size": block_size,
        "q_len": q_len,
        "k_len": k_len,
        "num_q_blocks": num_q_blocks,
        "num_k_blocks": num_k_blocks,
        "heads": per_head,
        "segment_ranges": segment_ranges,
    }


def segment_mass_from_block_matrix(matrix: np.ndarray, block_size: int, ranges: dict[str, list[int]], q_len: int, k_len: int):
    if not ranges:
        return {}
    out = {}
    for qa, (qs, qe) in ranges.items():
        if qs >= q_len:
            continue
        qbs = range(max(0, qs // block_size), min(matrix.shape[0], math.ceil(min(qe, q_len) / block_size)))
        for kb_label, (ks, ke) in ranges.items():
            if ks >= k_len:
                continue
            kbs = range(max(0, ks // block_size), min(matrix.shape[1], math.ceil(min(ke, k_len) / block_size)))
            vals = matrix[np.ix_(list(qbs), list(kbs))] if qbs and kbs else np.array([])
            out[f"{qa}->{kb_label}"] = float(vals.sum() / max(len(list(qbs)), 1)) if vals.size else 0.0
    return out
