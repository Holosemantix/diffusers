from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, save_json


ABLATION_GROUPS = {
    "A_full_all_to_all": "No masking; teacher/full attention diagnostic.",
    "B_block_ref_to_ref": "Diagnostic placeholder for blocking R_i -> R_j, i != j.",
    "C_ref_self_only": "Diagnostic placeholder for reference queries attending only own K/V.",
    "D_block_target_to_ref": "Diagnostic placeholder for blocking X -> R_k.",
    "E_keep_target_ref_block_ref_ref": "Diagnostic placeholder for keeping X -> R_k and blocking R_i -> R_j.",
    "F_block_irrelevant_ref_for_region": "Diagnostic placeholder for region/entity-specific irrelevant reference masking.",
}


def run_diagnostic_ablation_summary(probe_dir: str | Path, output_dir: str | Path, segment_map: dict[str, Any] | None = None):
    """Create an ablation plan/report without mutating model attention.

    The actual masking processors are intentionally separated because FLUX.2 Klein attention layouts
    can change quickly. This diagnostic mode is executable and provides teacher edge importance from
    the full-attention run as the first validation step.
    """

    output_dir = ensure_dir(output_dir)
    report = {
        "mode": "diagnostic",
        "probe_dir": str(probe_dir),
        "segment_map": segment_map,
        "groups": ABLATION_GROUPS,
        "todo_for_masking_mode": [
            "Wrap selected Flux2 processors and add a block mask before softmax for sampled layer/head/step.",
            "Validate shape layout [P, X, S/R...] from segment_map.json before applying masks.",
            "Run group A first and compare hidden-state deviation for each masking group.",
        ],
    }
    save_json(report, output_dir / "ref_ablation_diagnostic_report.json")
    return report

