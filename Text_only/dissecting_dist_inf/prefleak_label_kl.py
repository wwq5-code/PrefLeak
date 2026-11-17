#!/usr/bin/env python3
"""Adapt Dissecting Distribution Inference's label-only KL attack to PrefLeak.

Only public hard task labels derived from generated strings are exposed to the
attack: one means that the parsed A/B response equals the public task answer,
and zero means an incorrect or unparseable response.  The implementation uses
the official ``label_only_KLAttack._get_kl_preds`` pairwise KL voting routine.

Calibration and inference are separate so target preference labels cannot be
provided before the prediction file is serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


REPO_DIR = Path(__file__).resolve().parent
WORKSPACE = REPO_DIR.parents[1]
REBASE_DIR = REPO_DIR / "rebase"
OFFICIAL_ATTACK = (
    REBASE_DIR
    / "distribution_inference"
    / "attacks"
    / "blackbox"
    / "label_KL.py"
)
for import_root in (WORKSPACE, REBASE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from Text_only.remote_attack import (  # noqa: E402
    load_bank,
    load_generations,
    parse_mapping,
    parsed_choice,
)


POSITIVE_CLASS = "magenta"
NEGATIVE_CLASS = "cyan"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_official_attack(config: SimpleNamespace):
    """Load the official attack without its unused legacy config dependency.

    The upstream package imports ``simple_parsing`` and ``lib2to3`` through a
    type-only configuration import.  Neither is needed by label_KL.py, and
    lib2to3 was removed in Python 3.13.  A process-local type stub lets us call
    the upstream attack implementation unchanged.
    """

    if "distribution_inference.config" not in sys.modules:
        stub = types.ModuleType("distribution_inference.config")
        stub.BlackBoxAttackConfig = object
        sys.modules["distribution_inference.config"] = stub
    from distribution_inference.attacks.blackbox.label_KL import (  # noqa: PLC0415
        label_only_KLAttack,
    )

    return label_only_KLAttack(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--shadow-generations", type=Path, required=True)
    calibrate.add_argument("--probe-bank", type=Path, required=True)
    calibrate.add_argument("--shadow-label", action="append", required=True)
    calibrate.add_argument("--shadow-pair", action="append", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--num-probes", type=int, default=24)
    calibrate.add_argument("--repetitions", type=int, default=8)
    calibrate.add_argument("--attack-runs", type=int, default=10)
    calibrate.add_argument("--kl-frac", type=float, default=0.8)
    calibrate.add_argument("--seed", type=int, default=2020)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--target-generations", type=Path, required=True)
    infer.add_argument("--probe-bank", type=Path, required=True)
    infer.add_argument("--frozen-attack", type=Path, required=True)
    infer.add_argument("--target-id", action="append", required=True)
    infer.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_grouped_models(grouped: dict, model_ids: list[str]) -> None:
    expected = {"M0_reference", *model_ids}
    if set(grouped) != expected:
        raise ValueError(
            "Transcript model IDs do not match the requested anonymous IDs: "
            f"expected={sorted(expected)}, observed={sorted(grouped)}"
        )


def select_parseable_balanced_probes(
    rows: list[dict],
    grouped: dict,
    model_ids: list[str],
    num_probes: int,
    repetitions: int,
) -> tuple[list[str], dict[str, float]]:
    if num_probes < 2 or num_probes % 2:
        raise ValueError("--num-probes must be a positive even number.")
    per_answer = num_probes // 2
    candidates: dict[str, list[tuple[float, int, str]]] = {"A": [], "B": []}
    rates = {}
    for bank_index, probe in enumerate(rows):
        if probe["kind"] != "choice":
            continue
        probe_id = probe["probe_id"]
        observations = []
        for model_id in model_ids:
            records = grouped[model_id].get(probe_id, [])[:repetitions]
            if len(records) != repetitions:
                raise ValueError(
                    f"Need {repetitions} shadow repetitions for {model_id}/{probe_id}."
                )
            observations.extend(
                parsed_choice(record["response"]) is not None for record in records
            )
        rate = float(np.mean(observations))
        rates[probe_id] = rate
        expected = probe["expected_answer"]
        if expected not in candidates:
            raise ValueError(f"Choice probe {probe_id} has no public A/B answer.")
        candidates[expected].append((-rate, bank_index, probe_id))

    selected = []
    for answer in ("A", "B"):
        ranked = sorted(candidates[answer])
        if len(ranked) < per_answer:
            raise ValueError(f"Not enough public choice probes with answer {answer}.")
        selected.extend(row[2] for row in ranked[:per_answer])
    selected.sort(key=lambda probe_id: next(
        index for index, row in enumerate(rows) if row["probe_id"] == probe_id
    ))
    return selected, {probe_id: rates[probe_id] for probe_id in selected}


def correctness_probabilities(
    grouped: dict,
    bank: dict[str, dict],
    model_ids: list[str],
    probe_ids: list[str],
    repetitions: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    parse_rates = []
    for model_id in model_ids:
        per_probe = []
        parsed = []
        for probe_id in probe_ids:
            records = grouped[model_id].get(probe_id, [])[:repetitions]
            if len(records) != repetitions:
                raise ValueError(
                    f"Need {repetitions} repetitions for {model_id}/{probe_id}."
                )
            expected = bank[probe_id]["expected_answer"]
            choices = [parsed_choice(record["response"]) for record in records]
            parsed.extend(choice is not None for choice in choices)
            # Invalid strings are projected to the public incorrect label.  No
            # lexical content beyond the strict A/B parser reaches the attack.
            per_probe.append(float(np.mean([choice == expected for choice in choices])))
        probabilities.append(per_probe)
        parse_rates.append(float(np.mean(parsed)))
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.shape != (len(model_ids), len(probe_ids)):
        raise ValueError(f"Unexpected label-only matrix shape: {matrix.shape}")
    return matrix, np.asarray(parse_rates, dtype=np.float64)


def score_targets(
    cyan_templates: dict[str, list[list[float]]],
    magenta_templates: dict[str, list[list[float]]],
    target_probabilities: np.ndarray,
    probe_ids: list[str],
    query_groups: dict[str, list[str]],
    kl_frac: float,
    seed: int,
) -> np.ndarray:
    config = SimpleNamespace(
        multi2=None,
        multi=None,
        multi_class=False,
        kl_frac=kl_frac,
        kl_voting=True,
    )
    attack = load_official_attack(config)
    index = {probe_id: position for position, probe_id in enumerate(probe_ids)}
    vote_features = []
    np.random.seed(seed)
    for group_name in ("answer_A", "answer_B"):
        positions = [index[probe_id] for probe_id in query_groups[group_name]]
        cyan = np.asarray(cyan_templates[group_name], dtype=np.float64)
        magenta = np.asarray(magenta_templates[group_name], dtype=np.float64)
        hard_targets = (target_probabilities[:, positions] >= 0.5).astype(np.float64)
        # The helper returns the official pairwise KL comparisons. Passing the
        # same target matrix twice avoids the upstream evaluation-only need to
        # arrange victim models by their unknown private property.
        votes, _ = attack._get_kl_preds(  # noqa: SLF001
            cyan,
            magenta,
            hard_targets.copy(),
            hard_targets.copy(),
        )
        vote_features.append(votes.astype(np.float64))
    return np.concatenate(vote_features, axis=1).mean(axis=1)


def templates_for(
    matrix: np.ndarray,
    model_ids: list[str],
    labels: dict[str, str],
    probe_ids: list[str],
    query_groups: dict[str, list[str]],
    negative_class: str = NEGATIVE_CLASS,
    positive_class: str = POSITIVE_CLASS,
) -> tuple[dict[str, list[list[float]]], dict[str, list[list[float]]]]:
    index = {probe_id: position for position, probe_id in enumerate(probe_ids)}
    by_id = {model_id: matrix[position] for position, model_id in enumerate(model_ids)}

    def collect(preference: str) -> dict[str, list[list[float]]]:
        result = {}
        for group_name, group_probes in query_groups.items():
            positions = [index[probe_id] for probe_id in group_probes]
            result[group_name] = [
                by_id[model_id][positions].tolist()
                for model_id in model_ids
                if labels[model_id] == preference
            ]
        return result

    return collect(negative_class), collect(positive_class)


def calibrate(args: argparse.Namespace) -> None:
    if not 0 < args.kl_frac <= 1:
        raise ValueError("--kl-frac must be in (0, 1].")
    if args.attack_runs < 1 or args.repetitions < 1:
        raise ValueError("Attack runs and repetitions must be positive.")
    rows, bank = load_bank(args.probe_bank)
    grouped = load_generations(args.shadow_generations)
    labels = parse_mapping(args.shadow_label, "--shadow-label")
    pair_ids = parse_mapping(args.shadow_pair, "--shadow-pair")
    model_ids = sorted(labels)
    validate_grouped_models(grouped, model_ids)

    # Probe selection is completed without consulting preference labels.
    probe_ids, parseability = select_parseable_balanced_probes(
        rows, grouped, model_ids, args.num_probes, args.repetitions
    )
    if set(model_ids) != set(pair_ids):
        raise ValueError("Shadow labels and pair IDs must describe the same models.")
    classes = sorted(set(labels.values()))
    if len(classes) != 2:
        raise ValueError(f"Expected exactly two shadow preference classes: {classes}")
    negative_class, positive_class = classes
    pairs = sorted(set(pair_ids.values()))
    for pair_id in pairs:
        members = [model_id for model_id in model_ids if pair_ids[model_id] == pair_id]
        if len(members) != 2 or {labels[model_id] for model_id in members} != {
            negative_class,
            positive_class,
        }:
            raise ValueError(f"Invalid opposite-preference shadow pair: {pair_id}")

    query_groups = {
        "answer_A": [probe_id for probe_id in probe_ids if bank[probe_id]["expected_answer"] == "A"],
        "answer_B": [probe_id for probe_id in probe_ids if bank[probe_id]["expected_answer"] == "B"],
    }
    matrix, parse_rates = correctness_probabilities(
        grouped, bank, model_ids, probe_ids, args.repetitions
    )
    runs = []
    for run_index in range(args.attack_runs):
        seed = args.seed + run_index
        score_by_id = {}
        for held_pair in pairs:
            train_ids = [
                model_id for model_id in model_ids if pair_ids[model_id] != held_pair
            ]
            test_ids = [
                model_id for model_id in model_ids if pair_ids[model_id] == held_pair
            ]
            train_positions = [model_ids.index(model_id) for model_id in train_ids]
            test_positions = [model_ids.index(model_id) for model_id in test_ids]
            cyan, magenta = templates_for(
                matrix[train_positions], train_ids, labels, probe_ids, query_groups,
                negative_class, positive_class,
            )
            scores = score_targets(
                cyan,
                magenta,
                matrix[test_positions],
                probe_ids,
                query_groups,
                args.kl_frac,
                seed,
            )
            score_by_id.update(dict(zip(test_ids, map(float, scores))))
        y_true = np.asarray(
            [labels[model_id] == positive_class for model_id in model_ids], dtype=int
        )
        oof_scores = np.asarray([score_by_id[model_id] for model_id in model_ids])
        runs.append(
            {
                "run_index": run_index,
                "seed": seed,
                "threshold": 0.5,
                "shadow_oof_balanced_accuracy": float(
                    balanced_accuracy_score(y_true, oof_scores >= 0.5)
                ),
                "shadow_oof_auroc": float(roc_auc_score(y_true, oof_scores)),
                "shadow_oof_scores": {
                    model_id: float(score_by_id[model_id]) for model_id in model_ids
                },
            }
        )

    cyan, magenta = templates_for(
        matrix, model_ids, labels, probe_ids, query_groups,
        negative_class, positive_class,
    )
    document = {
        "attack": "label_only_distribution_inference_kl_voting",
        "adaptation_status": "modified_official_implementation",
        "selection_boundary": "shadow_only_before_target_inference",
        "target_data_opened": False,
        "official_repository": "iamgroot42/dissecting_dist_inf",
        "official_commit": official_commit(),
        "official_code_reused": "rebase/distribution_inference/attacks/blackbox/label_KL.py::label_only_KLAttack._get_kl_preds",
        "official_source_sha256": sha256(OFFICIAL_ATTACK),
        "official_attack_principle": "shadow_property_groups_pairwise_label_only_KL_voting",
        "prefleak_code_reused": [
            "Text_only.remote_attack.load_bank",
            "Text_only.remote_attack.load_generations",
            "Text_only.remote_attack.parsed_choice",
            "Text_only.evaluate_known_preference",
        ],
        "access": "post_only",
        "threat_model_relation": "valid_subset_of_paired_pre_post_access",
        "query_modality": "text_only_remote_label_only",
        "positive_class": positive_class,
        "negative_class": negative_class,
        "probe_bank_sha256": sha256(args.probe_bank),
        "shadow_generations_sha256": sha256(args.shadow_generations),
        "num_shadow_updates": len(model_ids),
        "num_shadow_pairs": len(pairs),
        "num_probes": len(probe_ids),
        "probe_ids": probe_ids,
        "probe_selection": "top_shadow_parseability_balanced_by_public_answer_without_preference_labels",
        "selected_shadow_parseability": parseability,
        "query_groups": query_groups,
        "repetitions": args.repetitions,
        "online_generation_calls": args.repetitions * len(probe_ids),
        "hard_label_projection": {
            "one": "strictly_parsed_choice_equals_public_expected_answer",
            "zero": "incorrect_or_unparseable_response",
            "target_probability_to_label": "empirical_correct_rate_greater_than_or_equal_to_0.5",
            "uses_response_text_beyond_strict_choice_parser": False,
        },
        "official_configuration": {
            "kl_frac": args.kl_frac,
            "kl_voting": True,
            "multi_class": False,
            "threshold": 0.5,
        },
        "shadow_mean_parse_rate": float(np.mean(parse_rates)),
        "negative_templates": cyan,
        "positive_templates": magenta,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "frozen_attack": str(args.output),
                "query_budget": document["online_generation_calls"],
                "selected_probes": len(probe_ids),
                "shadow_mean_parse_rate": document["shadow_mean_parse_rate"],
                "shadow_oof_balanced_accuracy_mean": float(np.mean([
                    run["shadow_oof_balanced_accuracy"] for run in runs
                ])),
                "shadow_oof_auroc_mean": float(np.mean([
                    run["shadow_oof_auroc"] for run in runs
                ])),
            },
            indent=2,
        )
    )


def infer(args: argparse.Namespace) -> None:
    frozen = json.loads(args.frozen_attack.read_text(encoding="utf-8"))
    if frozen["target_data_opened"] is not False:
        raise ValueError("The label-only attack was not frozen before inference.")
    if frozen["probe_bank_sha256"] != sha256(args.probe_bank):
        raise ValueError("The public probe bank changed after calibration.")
    if frozen["official_source_sha256"] != sha256(OFFICIAL_ATTACK):
        raise ValueError("The official label-only KL source changed after calibration.")
    target_ids = sorted(set(args.target_id))
    _, bank = load_bank(args.probe_bank)
    grouped = load_generations(args.target_generations)
    validate_grouped_models(grouped, target_ids)
    matrix, parse_rates = correctness_probabilities(
        grouped,
        bank,
        target_ids,
        frozen["probe_ids"],
        frozen["repetitions"],
    )
    predictions = []
    positive_class = frozen.get("positive_class", POSITIVE_CLASS)
    negative_class = frozen.get("negative_class", NEGATIVE_CLASS)
    negative_templates = frozen.get("negative_templates", frozen.get("cyan_templates"))
    positive_templates = frozen.get("positive_templates", frozen.get("magenta_templates"))
    if negative_templates is None or positive_templates is None:
        raise ValueError("Frozen attack is missing its two class templates.")
    for run in frozen["runs"]:
        scores = score_targets(
            negative_templates,
            positive_templates,
            matrix,
            frozen["probe_ids"],
            frozen["query_groups"],
            frozen["official_configuration"]["kl_frac"],
            run["seed"],
        )
        for index, model_id in enumerate(target_ids):
            predictions.append(
                {
                    "anonymous_model_id": model_id,
                    "query_budget": frozen["online_generation_calls"],
                    "budget_repeat": run["run_index"],
                    "predicted_preference": positive_class
                    if scores[index] >= 0.5
                    else negative_class,
                    "positive_class": positive_class,
                    "positive_class_score": float(scores[index]),
                    "raw_feature_l2_norm": float(np.linalg.norm(matrix[index])),
                    "exact_response_change_rate": None,
                    "parsed_response_rate": float(parse_rates[index]),
                    "mean_reference_choice_rate_delta": None,
                    "mean_absolute_reference_choice_rate_delta": None,
                }
            )
    document = {
        "attack": frozen["attack"],
        "adaptation_status": frozen["adaptation_status"],
        "access": frozen["access"],
        "query_modality": frozen["query_modality"],
        "attacker_used_target_private_labels": False,
        "num_shadow_updates": frozen["num_shadow_updates"],
        "num_held_out_target_updates": len(target_ids),
        "target_generations_sha256": sha256(args.target_generations),
        "target_hard_label_diagnostics": {
            "uses_private_preference_labels": False,
            "num_unique_hard_label_vectors": int(
                len(np.unique((matrix >= 0.5).astype(np.uint8), axis=0))
            ),
            "per_probe_positive_fraction": {
                probe_id: float(value)
                for probe_id, value in zip(
                    frozen["probe_ids"], (matrix >= 0.5).mean(axis=0)
                )
            },
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_targets": len(target_ids),
                "num_attack_runs": len(frozen["runs"]),
                "query_budget_per_run": frozen["online_generation_calls"],
                "mean_target_parse_rate": float(np.mean(parse_rates)),
                "target_labels_used": False,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.phase == "calibrate":
        calibrate(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
