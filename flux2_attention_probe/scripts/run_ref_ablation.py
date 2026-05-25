#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flux2_probe.io_utils import load_json, load_yaml
from flux2_probe.ref_ablation import run_diagnostic_ablation_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "probe_multi_ref.yaml"))
    parser.add_argument("--probe_dir", default=None, help="Existing full-attention probe directory.")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    probe_dir = Path(args.probe_dir or cfg.get("output_dir") or ROOT / "outputs" / "exp001")
    output_dir = Path(args.output_dir or probe_dir / "ref_ablation")
    segment_map_path = probe_dir / "segment_map.json"
    segment_map = load_json(segment_map_path) if segment_map_path.exists() else None
    report = run_diagnostic_ablation_summary(probe_dir, output_dir, segment_map=segment_map)
    print(f"Wrote diagnostic ablation report: {output_dir}")
    print(report["mode"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

