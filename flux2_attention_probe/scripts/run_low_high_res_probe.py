#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flux2_probe.io_utils import ensure_dir, load_yaml
from flux2_probe.metrics_resolution import compare_low_high


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "probe_low_high_res.yaml"))
    args, extra = parser.parse_known_args()
    cfg = load_yaml(args.config)
    out = ensure_dir(cfg.get("output_dir") or ROOT / "outputs" / "low_high_res")
    low = cfg.get("low_res", 512)
    high = cfg.get("high_res", 768)
    script = ROOT / "scripts" / "run_probe.py"
    low_dir = out / f"low_{low}"
    high_dir = out / f"high_{high}"
    base = [sys.executable, str(script), "--config", args.config]
    subprocess.run(base + ["--height", str(low), "--width", str(low), "--output_dir", str(low_dir)] + extra, check=True)
    subprocess.run(base + ["--height", str(high), "--width", str(high), "--output_dir", str(high_dir)] + extra, check=True)
    compare_low_high(low_dir, high_dir, out / "low_high_res_consistency.json")
    print(f"Wrote {out / 'low_high_res_consistency.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

