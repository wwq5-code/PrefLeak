#!/usr/bin/env python3
"""Infer held-out preferences from structured forced-choice shadow features."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Text_only.choice_attack_features import (
    available_choice_queries,
    build_choice_features,
    group_choice_generations,
    read_choice_generations,
)


def parse_mapping(values: list[str], argument_name: str) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{argument_name} must be MODEL_ID=LABEL, got: {value}")
        model_id, label = value.split("=", 1)
        mapping[model_id] = label
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--shadow-label", action="append", required=True)
    parser.add_argument(
        "--shadow-pair",
        action="append",
        default=[],
        help="Optional MODEL_ID=PAIR_ID groups for leakage-safe OOF calibration.",
    )
    parser.add_argument("--target-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--access", choices=("paired", "post_only"), default="paired")
    parser.add_argument("--budgets", default="6,12,24,48")
    parser.add_argument("--budget-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--classifier-c", type=float, default=0.2)
    return parser.parse_args()


def calibrated_threshold(y_binary: np.ndarray, scores: np.ndarray) -> float:
    unique = sorted(set(float(value) for value in scores))
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.extend(unique)
    best = (float("-inf"), -1.0, 0.5)
    for threshold in candidates:
        predicted = (scores >= threshold).astype(np.int64)
        balanced = balanced_accuracy_score(y_binary, predicted)
        # Prefer thresholds close to 0.5 when OOF performance ties.
        key = (balanced, -abs(threshold - 0.5), threshold)
        if key > best:
            best = key
    return float(best[2])


def main() -> None:
    args = parse_args()
    shadow_labels = parse_mapping(args.shadow_label, "--shadow-label")
    shadow_pairs = (
        parse_mapping(args.shadow_pair, "--shadow-pair") if args.shadow_pair else None
    )
    shadow_ids = sorted(shadow_labels)
    if shadow_pairs is not None and set(shadow_pairs) != set(shadow_ids):
        raise ValueError("Shadow pair IDs must exactly match shadow labels.")
    target_ids = sorted(set(args.target_id))
    if set(shadow_ids) & set(target_ids):
        raise ValueError("Shadow and target IDs must be disjoint.")
    classes = sorted(set(shadow_labels.values()))
    if len(classes) != 2:
        raise ValueError("Exactly two candidate preferences are required.")

    records = read_choice_generations(args.generations)
    grouped = group_choice_generations(records)
    # Query identities are public, but a strict post-only attack must not even
    # derive its available-query list from M0_reference.
    all_queries = (
        sorted(grouped[shadow_ids[0]])
        if args.access == "post_only"
        else available_choice_queries(grouped)
    )
    requested = sorted({int(value) for value in args.budgets.split(",")})
    budgets = [value for value in requested if 0 < value <= len(all_queries)]
    if not budgets:
        raise ValueError("No requested query budget is available.")

    predictions = []
    run_summaries = []
    positive_class = classes[1]
    for budget in budgets:
        repeats = 1 if budget == len(all_queries) else args.budget_repeats
        for repeat in range(repeats):
            rng = random.Random(args.seed + budget * 1009 + repeat)
            query_indices = sorted(rng.sample(all_queries, budget))
            model_ids = [*shadow_ids, *target_ids]
            features, observability = build_choice_features(
                grouped, model_ids, query_indices, access=args.access
            )
            x_shadow = features[: len(shadow_ids)]
            x_target = features[len(shadow_ids) :]
            y_shadow = np.asarray([shadow_labels[value] for value in shadow_ids])
            y_binary = (y_shadow == positive_class).astype(np.int64)
            minimum_class_count = min(np.sum(y_shadow == value) for value in classes)

            if np.allclose(x_shadow, x_shadow[0]):
                classifier = DummyClassifier(strategy="prior")
                classifier.fit(x_shadow, y_shadow)
                target_scores = classifier.predict_proba(x_target)[
                    :, list(classifier.classes_).index(positive_class)
                ]
                threshold = 0.5
                oof_balanced = 0.5
                oof_auc = 0.5
                classifier_name = "DummyClassifier(prior; constant signal)"
            else:
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=args.classifier_c,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=args.seed,
                        solver="liblinear",
                    ),
                )
                if shadow_pairs is not None:
                    groups = np.asarray([shadow_pairs[value] for value in shadow_ids])
                    folds = StratifiedGroupKFold(
                        n_splits=min(5, len(set(groups))),
                        shuffle=True,
                        random_state=args.seed + repeat,
                    )
                else:
                    groups = None
                    folds = StratifiedKFold(
                        n_splits=min(5, int(minimum_class_count)),
                        shuffle=True,
                        random_state=args.seed + repeat,
                    )
                oof_probabilities = cross_val_predict(
                    clone(classifier),
                    x_shadow,
                    y_shadow,
                    cv=folds,
                    groups=groups,
                    method="predict_proba",
                )
                positive_index = list(np.unique(y_shadow)).index(positive_class)
                oof_scores = oof_probabilities[:, positive_index]
                threshold = calibrated_threshold(y_binary, oof_scores)
                oof_predictions = (oof_scores >= threshold).astype(np.int64)
                oof_balanced = float(balanced_accuracy_score(y_binary, oof_predictions))
                oof_auc = float(roc_auc_score(y_binary, oof_scores))
                classifier.fit(x_shadow, y_shadow)
                target_scores = classifier.predict_proba(x_target)[
                    :, list(classifier.classes_).index(positive_class)
                ]
                classifier_name = "StandardScaler+LogisticRegression"

            target_predictions = np.where(
                target_scores >= threshold, positive_class, classes[0]
            )
            for index, model_id in enumerate(target_ids):
                predictions.append(
                    {
                        "anonymous_model_id": model_id,
                        "query_budget": budget,
                        "budget_repeat": repeat,
                        "predicted_preference": str(target_predictions[index]),
                        "positive_class": positive_class,
                        "positive_class_score": float(target_scores[index]),
                        **observability[model_id],
                    }
                )
            run_summaries.append(
                {
                    "query_budget": budget,
                    "budget_repeat": repeat,
                    "query_indices": query_indices,
                    "classifier": classifier_name,
                    "shadow_oof_threshold": threshold,
                    "shadow_oof_balanced_accuracy": oof_balanced,
                    "shadow_oof_auroc": oof_auc,
                }
            )

    metadata_path = args.generations.parent / "public_metadata.json"
    generation = (
        json.loads(metadata_path.read_text(encoding="utf-8")).get("generation")
        if metadata_path.is_file()
        else None
    )
    result = {
        "attack": "known_candidate_forced_choice_inference",
        "access": args.access,
        "query_modality": records[0]["modality"],
        "generation": generation,
        "attacker_knows_candidate_set": True,
        "attacker_used_target_private_labels": False,
        "candidate_preferences": classes,
        "threshold_calibration": "shadow_matched_pair_grouped_out_of_fold_only"
        if shadow_pairs is not None
        else "shadow_out_of_fold_only",
        "num_shadow_updates": len(shadow_ids),
        "num_held_out_target_updates": len(target_ids),
        "available_queries": len(all_queries),
        "run_summaries": run_summaries,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "access": args.access,
                "query_modality": result["query_modality"],
                "available_queries": len(all_queries),
                "num_attack_runs": len(run_summaries),
                "num_predictions": len(predictions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
