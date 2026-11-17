#!/usr/bin/env python3
"""Enforce the pre-target GLM PPSD complete-pair fidelity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Text_only.GLM.PPSD.protocol import build_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-complete-pairs", type=int, default=6)
    args = parser.parse_args()
    path = args.output_dir / "analysis" / "visual_fidelity_shadow.json"
    fidelity = json.loads(path.read_text(encoding="utf-8"))
    shadows, _ = build_specs()
    passed = {
        row["anonymous_model_id"] for row in fidelity["per_model"] if row["passed"]
    }
    complete_pairs = []
    for left, right in zip(shadows[::2], shadows[1::2]):
        if left.model_id in passed and right.model_id in passed:
            complete_pairs.append(left.pair_id)
    result = {
        "gate": "pre_target_complete_counterfactual_shadow_fidelity",
        "target_data_queried": False,
        "minimum_complete_pairs": args.minimum_complete_pairs,
        "num_shadow_updates": len(shadows),
        "num_updates_passed": len(passed),
        "num_complete_pairs": len(complete_pairs),
        "complete_pairs": complete_pairs,
        "accepted": len(complete_pairs) >= args.minimum_complete_pairs,
    }
    output = args.output_dir / "shadow_gate.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
