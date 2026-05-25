#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flux2_probe.io_utils import load_yaml
from flux2_probe.metrics_attention import summarize_attention_records
from flux2_probe.visualization import summarize_figures, write_summary_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    report = summarize_attention_records(
        args.input_dir,
        args.input_dir,
        reference_roles=cfg.get("reference_roles"),
        region_annotations=cfg.get("region_annotations"),
    )
    summarize_figures(args.input_dir)
    write_summary_report(args.input_dir, report)
    print(f"Wrote summary under {args.input_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

