#!/usr/bin/env python3
"""Summarize visual preference rates for preference-absent control updates."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--control-kind", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODEL_ID=KIND, got {value}")
        model_id, kind = value.split("=", 1)
        if kind not in {"neutral", "label_shuffled"}:
            raise ValueError(f"Unknown control kind: {kind}")
        result[model_id] = kind
    return result


def main() -> None:
    args = parse_args()
    kinds = mapping(args.control_kind)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with args.generations.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parsed = row.get("parsed_choice")
            if parsed not in {"A", "B"}:
                continue
            counts[row["model_version"]][1] += 1
            if parsed == row["reference_option"]:
                counts[row["model_version"]][0] += 1
    expected_ids = {"M0_reference", *kinds}
    if set(counts) != expected_ids:
        raise ValueError(
            f"Generation IDs differ; missing={sorted(expected_ids-set(counts))}, "
            f"extra={sorted(set(counts)-expected_ids)}"
        )
    base_rate = counts["M0_reference"][0] / counts["M0_reference"][1]
    per_update = []
    for model_id, kind in sorted(kinds.items()):
        reference, parsed = counts[model_id]
        rate = reference / parsed
        per_update.append(
            {
                "anonymous_model_id": model_id,
                "control_kind": kind,
                "magenta_choice_rate": rate,
                "change_from_base": rate - base_rate,
                "absolute_preference_from_half": abs(rate - 0.5),
                "num_parsed_queries": parsed,
            }
        )
    summary = {}
    for kind in ("neutral", "label_shuffled"):
        rows = [row for row in per_update if row["control_kind"] == kind]
        rates = [row["magenta_choice_rate"] for row in rows]
        summary[kind] = {
            "num_updates": len(rows),
            "mean_magenta_choice_rate": statistics.fmean(rates),
            "std_magenta_choice_rate": statistics.stdev(rates),
            "mean_absolute_preference_from_half": statistics.fmean(
                abs(value - 0.5) for value in rates
            ),
            "mean_change_from_base": statistics.fmean(
                value - base_rate for value in rates
            ),
        }
    result = {
        "evaluation": "preference_absent_visual_diagnostic",
        "reference_treatment": "magenta",
        "base_magenta_choice_rate": base_rate,
        "summary": summary,
        "per_update": per_update,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result}, indent=2))


if __name__ == "__main__":
    main()
