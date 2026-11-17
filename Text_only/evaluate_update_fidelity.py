#!/usr/bin/env python3
"""Privately verify that each visual DPO update encodes its intended preference."""

import argparse
import json
from pathlib import Path

import numpy as np

from Text_only.choice_attack_features import (
    available_choice_queries,
    group_choice_generations,
    read_choice_generations,
    reference_rate,
)


def parse_mapping(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--target-label must be MODEL_ID=LABEL, got: {value}")
        model_id, label = value.split("=", 1)
        mapping[model_id] = label
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--target-label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-post-accuracy", type=float, default=0.75)
    parser.add_argument("--min-improvement", type=float, default=0.20)
    parser.add_argument("--min-parsed-rate", type=float, default=0.95)
    parser.add_argument(
        "--reference-preference",
        default="warm",
        help="Preference represented by reference_option in the probe bank.",
    )
    return parser.parse_args()


def aggregate_reference_rate(model_queries: dict[int, list[dict]], queries: list[int]) -> tuple[float, float]:
    rates = []
    parsed = []
    for query_index in queries:
        rate, parse_rate = reference_rate(model_queries[query_index])
        rates.append(rate)
        parsed.append(parse_rate)
    return float(np.mean(rates)), float(np.mean(parsed))


def main() -> None:
    args = parse_args()
    labels = parse_mapping(args.target_label)
    records = read_choice_generations(args.generations)
    grouped = group_choice_generations(records)
    queries = available_choice_queries(grouped)
    missing = set(labels) - set(grouped)
    if missing:
        raise ValueError(f"Missing target generations: {sorted(missing)}")

    base_warm_rate, base_parse_rate = aggregate_reference_rate(
        grouped["M0_reference"], queries
    )
    per_model = []
    for model_id in sorted(labels):
        preference = labels[model_id]
        post_warm_rate, post_parse_rate = aggregate_reference_rate(grouped[model_id], queries)
        is_reference = preference == args.reference_preference
        base_accuracy = base_warm_rate if is_reference else 1.0 - base_warm_rate
        post_accuracy = post_warm_rate if is_reference else 1.0 - post_warm_rate
        improvement = post_accuracy - base_accuracy
        passed = (
            post_accuracy >= args.min_post_accuracy
            and improvement >= args.min_improvement
            and post_parse_rate >= args.min_parsed_rate
        )
        per_model.append(
            {
                "anonymous_model_id": model_id,
                "private_preference": preference,
                "base_preference_accuracy": base_accuracy,
                "post_preference_accuracy": post_accuracy,
                "preference_accuracy_improvement": improvement,
                "post_reference_choice_rate": post_warm_rate,
                "parsed_response_rate": post_parse_rate,
                "passed": passed,
            }
        )

    result = {
        "evaluation": "private_update_fidelity_gate",
        "private_labels_used_only_for_update_validation": True,
        "num_queries": len(queries),
        "thresholds": {
            "min_post_accuracy": args.min_post_accuracy,
            "min_improvement": args.min_improvement,
            "min_parsed_response_rate": args.min_parsed_rate,
        },
        "reference_preference": args.reference_preference,
        "base_reference_choice_rate": base_warm_rate,
        "base_parsed_response_rate": base_parse_rate,
        "num_models": len(per_model),
        "num_passed": sum(row["passed"] for row in per_model),
        "all_passed": all(row["passed"] for row in per_model),
        "mean_post_preference_accuracy": float(
            np.mean([row["post_preference_accuracy"] for row in per_model])
        ),
        "mean_preference_accuracy_improvement": float(
            np.mean([row["preference_accuracy_improvement"] for row in per_model])
        ),
        "per_model": per_model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
