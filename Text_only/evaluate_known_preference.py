#!/usr/bin/env python3
"""Evaluate held-out preference predictions after the black-box attack is fixed."""

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def parse_mapping(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Target label must be MODEL_ID=LABEL, got: {value}")
        model_id, label = value.split("=", 1)
        mapping[model_id] = label
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--target-label", action="append", required=True, metavar="MODEL_ID=LABEL")
    parser.add_argument(
        "--target-pair",
        action="append",
        default=[],
        metavar="MODEL_ID=PAIR_ID",
        help="Optional matched-pair IDs for cluster-bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--valid-target-id",
        action="append",
        default=[],
        help="Optional fidelity-valid target subset for a conditional estimand.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def one_run_metrics(
    rows: list[dict],
    private_labels: dict[str, str],
    target_pairs: dict[str, str] | None = None,
) -> dict[str, float | list | str | None]:
    y_true = [private_labels[row["anonymous_model_id"]] for row in rows]
    y_pred = [row["predicted_preference"] for row in rows]
    positive_class = rows[0]["positive_class"]
    y_binary = np.array([label == positive_class for label in y_true], dtype=np.int64)
    scores = np.array([row["positive_class_score"] for row in rows])
    accuracy = float(accuracy_score(y_true, y_pred))
    correctness = np.asarray(
        [truth == predicted for truth, predicted in zip(y_true, y_pred)],
        dtype=np.float64,
    )
    matched_pair_pvalue = None
    if target_pairs and all(row["anonymous_model_id"] in target_pairs for row in rows):
        clustered: dict[str, list[float]] = defaultdict(list)
        for row, correct_value in zip(rows, correctness):
            clustered[target_pairs[row["anonymous_model_id"]]].append(float(correct_value))
        # Treat the matched pair as the effective independent sampling unit.
        # A percentile cluster bootstrap degenerates to [1, 1] when every pair
        # is correct, so use a Wilson score interval with n equal to the number
        # of independent pairs.  This is deliberately more conservative than a
        # target-level binomial interval for the matched design.
        effective_n = len(clustered)
        z = 1.959963984540054
        denominator = 1.0 + z * z / effective_n
        center = (accuracy + z * z / (2.0 * effective_n)) / denominator
        half_width = (
            z
            * np.sqrt(
                accuracy * (1.0 - accuracy) / effective_n
                + z * z / (4.0 * effective_n * effective_n)
            )
            / denominator
        )
        accuracy_interval = [
            float(max(0.0, center - half_width)),
            float(min(1.0, center + half_width)),
        ]
        interval_method = "matched_pair_effective_n_wilson"
        # Under the paired null, the two private labels within every matched
        # pair are exchangeable.  Enumerating all within-pair label swaps gives
        # an exact one-sided randomization test without treating the two
        # members of a pair as independent observations.
        pair_accuracies = [float(np.mean(values)) for values in clustered.values()]
        if effective_n <= 20:
            null_accuracies = [
                float(
                    np.mean(
                        [
                            value if keep else 1.0 - value
                            for value, keep in zip(pair_accuracies, assignment)
                        ]
                    )
                )
                for assignment in itertools.product((False, True), repeat=effective_n)
            ]
            matched_pair_pvalue = float(
                np.mean(np.asarray(null_accuracies) >= accuracy - 1e-12)
            )
    else:
        correct = int(np.sum(correctness))
        interval = binomtest(correct, len(rows)).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        accuracy_interval = [float(interval.low), float(interval.high)]
        interval_method = "wilson"
    auc = float(roc_auc_score(y_binary, scores)) if len(set(y_binary)) == 2 else None

    def optional_mean(key: str) -> float | None:
        values = [row.get(key) for row in rows]
        return (
            float(np.mean(values))
            if values and all(value is not None for value in values)
            else None
        )

    return {
        "accuracy": accuracy,
        "accuracy_95ci": accuracy_interval,
        "accuracy_95ci_method": interval_method,
        "accuracy_effective_sample_size": len(set(target_pairs.values()))
        if target_pairs
        else len(rows),
        "chance_level": 0.5,
        "matched_pair_randomization_pvalue_one_sided": matched_pair_pvalue,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": auc,
        "mean_exact_response_change_rate": optional_mean(
            "exact_response_change_rate"
        ),
        "mean_parsed_response_rate": optional_mean("parsed_response_rate"),
        "mean_reference_choice_rate_delta": optional_mean(
            "mean_reference_choice_rate_delta"
        ),
        "mean_absolute_reference_choice_rate_delta": optional_mean(
            "mean_absolute_reference_choice_rate_delta"
        ),
        "mean_raw_feature_l2_norm": float(
            np.mean([row["raw_feature_l2_norm"] for row in rows])
        ),
    }


def summarize(
    rows: list[dict],
    private_labels: dict[str, str],
    target_pairs: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["query_budget"], row["budget_repeat"])].append(row)
    individual_runs = []
    by_budget = defaultdict(list)
    for (budget, repeat), rows in sorted(grouped.items()):
        metrics = one_run_metrics(rows, private_labels, target_pairs)
        individual_runs.append({"query_budget": budget, "budget_repeat": repeat, **metrics})
        by_budget[budget].append(metrics)

    budget_summary = []
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "mean_exact_response_change_rate",
        "mean_parsed_response_rate",
        "mean_reference_choice_rate_delta",
        "mean_absolute_reference_choice_rate_delta",
        "mean_raw_feature_l2_norm",
    )
    for budget, runs in sorted(by_budget.items()):
        summary = {"query_budget": budget, "num_budget_subsets": len(runs)}
        for metric_name in metric_names:
            values = [run[metric_name] for run in runs if run[metric_name] is not None]
            summary[f"{metric_name}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric_name}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            )
        if len(runs) == 1:
            summary["accuracy_95ci"] = runs[0]["accuracy_95ci"]
            summary["accuracy_95ci_method"] = runs[0]["accuracy_95ci_method"]
            summary["accuracy_effective_sample_size"] = runs[0][
                "accuracy_effective_sample_size"
            ]
        budget_summary.append(summary)
    return individual_runs, budget_summary


def main() -> None:
    args = parse_args()
    attack = json.loads(args.predictions.read_text(encoding="utf-8"))
    private_labels = parse_mapping(args.target_label)
    target_pairs = parse_mapping(args.target_pair) if args.target_pair else None
    predicted_ids = {row["anonymous_model_id"] for row in attack["predictions"]}
    if predicted_ids != set(private_labels):
        raise ValueError("Private target labels do not exactly match predicted target IDs.")

    if target_pairs is not None and set(target_pairs) != set(private_labels):
        raise ValueError("Matched-pair IDs do not exactly match target labels.")
    individual_runs, budget_summary = summarize(
        attack["predictions"], private_labels, target_pairs
    )

    result = {
        "evaluation": "held_out_closed_set_preference_inference",
        "private_labels_used_only_after_prediction": True,
        "access": attack["access"],
        "query_modality": attack["query_modality"],
        "generation": attack.get("generation"),
        "num_shadow_updates": attack["num_shadow_updates"],
        "num_held_out_target_updates": attack["num_held_out_target_updates"],
        "individual_runs": individual_runs,
        "budget_summary": budget_summary,
    }
    if args.valid_target_id:
        valid_ids = set(args.valid_target_id)
        if not valid_ids <= predicted_ids:
            raise ValueError("A fidelity-valid target ID is absent from the predictions.")
        valid_labels = {model_id: private_labels[model_id] for model_id in valid_ids}
        if len(set(valid_labels.values())) != 2:
            raise ValueError("The fidelity-valid target subset must contain both classes.")
        valid_rows = [
            row for row in attack["predictions"] if row["anonymous_model_id"] in valid_ids
        ]
        valid_pairs = None
        if target_pairs is not None:
            candidate_pairs = {
                model_id: target_pairs[model_id] for model_id in valid_ids
            }
            pair_sizes = defaultdict(int)
            for pair_id in candidate_pairs.values():
                pair_sizes[pair_id] += 1
            # A within-pair label swap is defined only when both opposite-label
            # members survive fidelity conditioning.  If selection leaves any
            # singleton pair, use the ordinary target-level interval and do not
            # report the paired randomization p-value for this secondary subset.
            if all(size == 2 for size in pair_sizes.values()):
                valid_pairs = candidate_pairs
        conditional_runs, conditional_summary = summarize(
            valid_rows, valid_labels, valid_pairs
        )
        result["fidelity_conditioned"] = {
            "num_held_out_target_updates": len(valid_ids),
            "coverage": len(valid_ids) / len(predicted_ids),
            "individual_runs": conditional_runs,
            "budget_summary": conditional_summary,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "access": result["access"],
                "query_modality": result["query_modality"],
                "num_held_out_target_updates": result[
                    "num_held_out_target_updates"
                ],
                "accuracy_by_budget": {
                    str(row["query_budget"]): row["accuracy_mean"]
                    for row in result["budget_summary"]
                },
                "fidelity_conditioned_accuracy_by_budget": {
                    str(row["query_budget"]): row["accuracy_mean"]
                    for row in result.get("fidelity_conditioned", {}).get(
                        "budget_summary", []
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
