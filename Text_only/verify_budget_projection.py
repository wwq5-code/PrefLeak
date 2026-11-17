#!/usr/bin/env python3
"""Verify that an exact-budget transcript reproduces legacy predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTACK_LOCATIONS = {
    "paired_selected_residualized": "paired",
    "paired_selected_no_residual": "paired",
    "post_only_matched_budget": "post_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = []
    for attack_name, location in ATTACK_LOCATIONS.items():
        legacy = json.loads(
            (args.run_root / "predictions" / f"{attack_name}.json").read_text(
                encoding="utf-8"
            )
        )["predictions"]
        exact_dir = (
            args.run_root
            / "exact_budget"
            / ("predictions_paired" if location == "paired" else "predictions_post_only")
        )
        exact = json.loads(
            (exact_dir / f"{attack_name}.json").read_text(encoding="utf-8")
        )["predictions"]
        legacy_by_id = {row["anonymous_model_id"]: row for row in legacy}
        exact_by_id = {row["anonymous_model_id"]: row for row in exact}
        if set(legacy_by_id) != set(exact_by_id):
            raise ValueError(f"Target IDs changed for {attack_name}")
        max_score_difference = max(
            abs(
                legacy_by_id[model_id]["positive_class_score"]
                - exact_by_id[model_id]["positive_class_score"]
            )
            for model_id in legacy_by_id
        )
        predictions_identical = all(
            legacy_by_id[model_id]["predicted_preference"]
            == exact_by_id[model_id]["predicted_preference"]
            for model_id in legacy_by_id
        )
        checks.append(
            {
                "attack_name": attack_name,
                "num_targets": len(legacy_by_id),
                "predictions_identical": predictions_identical,
                "max_absolute_score_difference": max_score_difference,
            }
        )
    ledgers = {}
    for location in ("paired", "post_only"):
        ledger = json.loads(
            (
                args.run_root
                / "exact_budget"
                / f"{location}_transcript"
                / "query_ledger.json"
            ).read_text(encoding="utf-8")
        )
        if ledger["serialized_rows"] != ledger["expected_serialized_rows"]:
            raise ValueError(f"Incomplete {location} exact-budget transcript")
        ledgers[location] = ledger
    passed = all(
        row["predictions_identical"] and row["max_absolute_score_difference"] == 0.0
        for row in checks
    )
    result = {
        "audit": "exact_budget_projection_equivalence",
        "passed": passed,
        "scope_note": (
            "The projection proves that reported predictions use only selected "
            "rows; it does not retroactively undo extra calls in the legacy run."
        ),
        "checks": checks,
        "paired_effective_online_calls_per_target": ledgers["paired"][
            "effective_online_calls_per_target"
        ],
        "post_only_effective_online_calls_per_target": ledgers["post_only"][
            "effective_online_calls_per_target"
        ],
    }
    if not passed:
        raise RuntimeError(f"Exact-budget equivalence failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result}, indent=2))


if __name__ == "__main__":
    main()
