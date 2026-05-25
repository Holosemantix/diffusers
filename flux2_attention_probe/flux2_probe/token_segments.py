from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .io_utils import save_json


@dataclass
class SegmentMap:
    ranges: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def labels(self) -> list[str]:
        return list(self.ranges.keys())

    def to_dict(self) -> dict[str, Any]:
        return {"ranges": self.ranges, "metadata": self.metadata}

    def save(self, path: str) -> None:
        save_json(self.to_dict(), path)


def _to_cpu_list(x):
    try:
        return x.detach().to("cpu").tolist()
    except Exception:
        return None


class TokenSegmentInferer:
    """Infer FLUX.2 Klein sequence segments from transformer inputs.

    Diffusers FLUX.2 Klein uses text tokens separately in double blocks, then concatenates
    ``[text, image]`` in single blocks. When image conditioning is present, image tokens are
    ``[target noisy latents, reference image latents...]``. The image IDs' T coordinate is
    0 for target and 10, 20, ... for conditioning images.
    """

    def __init__(self, manual_ranges: dict[str, list[int]] | None = None):
        self.manual_ranges = manual_ranges or {}
        self.segment_map = SegmentMap(ranges=dict(self.manual_ranges), metadata={"source": "manual" if manual_ranges else "auto"})
        self.shape_log: list[dict[str, Any]] = []

    def observe_transformer_inputs(
        self,
        *,
        hidden_states=None,
        encoder_hidden_states=None,
        img_ids=None,
        txt_ids=None,
        step_idx: int | None = None,
        timestep: Any | None = None,
    ) -> SegmentMap:
        record = {
            "step_idx": step_idx,
            "timestep": str(timestep),
            "hidden_states_shape": list(hidden_states.shape) if hidden_states is not None and hasattr(hidden_states, "shape") else None,
            "encoder_hidden_states_shape": list(encoder_hidden_states.shape)
            if encoder_hidden_states is not None and hasattr(encoder_hidden_states, "shape")
            else None,
            "img_ids_shape": list(img_ids.shape) if img_ids is not None and hasattr(img_ids, "shape") else None,
            "txt_ids_shape": list(txt_ids.shape) if txt_ids is not None and hasattr(txt_ids, "shape") else None,
        }
        self.shape_log.append(record)
        if self.manual_ranges:
            return self.segment_map

        text_len = int(encoder_hidden_states.shape[1]) if encoder_hidden_states is not None else 0
        image_len = int(hidden_states.shape[1]) if hidden_states is not None else 0
        ranges = {}
        if text_len > 0:
            ranges["P"] = [0, text_len]

        image_base = text_len
        if img_ids is not None:
            ranges.update(self._infer_image_segments_from_ids(img_ids, image_base=image_base, image_len=image_len))
        elif image_len > 0:
            ranges["X"] = [image_base, image_base + image_len]

        self.segment_map = SegmentMap(
            ranges=ranges,
            metadata={
                "source": "auto",
                "text_len": text_len,
                "image_len": image_len,
                "layout": "[P, X, S/R...] for attention summaries",
                "shape_log_tail": self.shape_log[-5:],
            },
        )
        return self.segment_map

    def _infer_image_segments_from_ids(self, img_ids, image_base: int, image_len: int) -> dict[str, list[int]]:
        ids = img_ids[0] if getattr(img_ids, "ndim", 0) == 3 else img_ids
        t_values = None
        try:
            t_values = ids[:, 0].detach().to("cpu").tolist()
        except Exception:
            pass
        if not t_values:
            return {"X": [image_base, image_base + image_len]}

        runs = []
        start = 0
        current = t_values[0]
        for idx, value in enumerate(t_values[1:], start=1):
            if value != current:
                runs.append((current, start, idx))
                start = idx
                current = value
        runs.append((current, start, len(t_values)))

        ranges = {}
        ref_count = 0
        for t_value, start, end in runs:
            label = "X" if int(t_value) == 0 else None
            if label is None:
                ref_count += 1
                label = "S" if ref_count == 1 else f"R{ref_count - 1}"
            ranges[label] = [image_base + start, image_base + end]
        return ranges

    def save(self, output_dir: str) -> None:
        self.segment_map.save(f"{output_dir}/segment_map.json")
        save_json(self.shape_log, f"{output_dir}/shape_log.json")


def normalize_manual_ranges(config: dict[str, Any] | None) -> dict[str, list[int]]:
    if not config:
        return {}
    raw = config.get("manual_token_ranges") or config.get("token_ranges") or {}
    out = {}
    for key, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            out[str(key)] = [int(value[0]), int(value[1])]
    return out

