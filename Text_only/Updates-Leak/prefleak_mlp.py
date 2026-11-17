#!/usr/bin/env python3
"""Adapt Updates-Leak's posterior-difference MLP to PrefLeak string outputs.

The adaptation deliberately keeps the Updates-Leak attack structure simple:
a fixed public probing set is queried before and after an update, the output
differences are flattened, and the official Updates-Leak MLP predicts a
property of the update.  It reuses PrefLeak's public bounded string feature
map so that the target-side view remains text-only and contains no logits.

Calibration and inference are separate commands.  Target preference labels
are not accepted by either command; they are opened only by PrefLeak's shared
evaluation script after predictions have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn


REPO_DIR = Path(__file__).resolve().parent
WORKSPACE = REPO_DIR.parents[1]
ATTACKS_DIR = REPO_DIR / "Attacks"
for import_root in (WORKSPACE, ATTACKS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# Reuse the official attack architecture rather than introducing a new MLP.
from network import labelPredMNIST  # noqa: E402

# Reuse the exact feature extraction and transcript parsing used by PrefLeak.
from Text_only.remote_attack import (  # noqa: E402
    FEATURE_DIM,
    build_features,
    calibrated_threshold,
    load_bank,
    load_generations,
    parse_mapping,
)


POSITIVE_CLASS = "magenta"
NEGATIVE_CLASS = "cyan"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--shadow-generations", type=Path, required=True)
    calibrate.add_argument("--probe-bank", type=Path, required=True)
    calibrate.add_argument("--shadow-label", action="append", required=True)
    calibrate.add_argument("--shadow-pair", action="append", required=True)
    calibrate.add_argument("--output-dir", type=Path, required=True)
    calibrate.add_argument("--num-probes", type=int, default=24)
    calibrate.add_argument("--repetitions", type=int, default=4)
    calibrate.add_argument("--attack-runs", type=int, default=10)
    calibrate.add_argument("--epochs", type=int, default=50)
    calibrate.add_argument("--learning-rate", type=float, default=1e-3)
    calibrate.add_argument("--weight-decay", type=float, default=1e-5)
    calibrate.add_argument("--seed", type=int, default=2020)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--target-generations", type=Path, required=True)
    infer.add_argument("--probe-bank", type=Path, required=True)
    infer.add_argument("--frozen-attack", type=Path, required=True)
    infer.add_argument("--weights", type=Path, required=True)
    infer.add_argument("--target-id", action="append", required=True)
    infer.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_matrix(
    generations: Path,
    probe_bank: Path,
    model_ids: list[str],
    probe_ids: list[str],
    repetitions: int,
) -> tuple[np.ndarray, dict | None]:
    _, all_probes = load_bank(probe_bank)
    missing = sorted(set(probe_ids) - set(all_probes))
    if missing:
        raise ValueError(f"Frozen probes are absent from the public bank: {missing}")
    frozen_bank = {probe_id: all_probes[probe_id] for probe_id in probe_ids}
    grouped = load_generations(generations)
    expected = {"M0_reference", *model_ids}
    if set(grouped) != expected:
        raise ValueError(
            "Generation model IDs do not match the requested anonymous IDs: "
            f"expected={sorted(expected)}, observed={sorted(grouped)}"
        )
    features = build_features(
        grouped,
        frozen_bank,
        model_ids,
        access="paired",
        repetitions=repetitions,
    )
    matrix = np.stack(
        [
            np.concatenate([features[model_id][probe_id] for probe_id in probe_ids])
            for model_id in model_ids
        ]
    ).astype(np.float32)
    expected_dimension = len(probe_ids) * FEATURE_DIM
    if matrix.shape != (len(model_ids), expected_dimension):
        raise ValueError(f"Unexpected Updates-Leak attack matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("The Updates-Leak attack matrix contains non-finite values.")

    metadata_path = generations.parent / "public_metadata.json"
    generation = None
    if metadata_path.is_file():
        generation = json.loads(metadata_path.read_text(encoding="utf-8")).get(
            "generation"
        )
    return matrix, generation


def train_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> labelPredMNIST:
    set_seed(seed)
    model = labelPredMNIST(matrix.shape[1], numOfClasses=2)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.NLLLoss()
    x = torch.from_numpy(matrix)
    y = torch.from_numpy(labels.astype(np.int64, copy=False))
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def positive_scores(model: labelPredMNIST, matrix: np.ndarray) -> np.ndarray:
    return torch.exp(model(torch.from_numpy(matrix)))[:, 1].cpu().numpy()


def calibrate(args: argparse.Namespace) -> None:
    rows, _ = load_bank(args.probe_bank)
    if not 1 <= args.num_probes <= len(rows):
        raise ValueError("--num-probes must be within the public probe-bank size.")
    if args.attack_runs < 1 or args.epochs < 1 or args.repetitions < 1:
        raise ValueError("Attack runs, epochs, and repetitions must be positive.")

    # Updates-Leak uses a fixed probing set and slices the first N probe outputs.
    # Preserve that choice to avoid importing PrefLeak's supervised probe search.
    probe_ids = [row["probe_id"] for row in rows[: args.num_probes]]
    labels = parse_mapping(args.shadow_label, "--shadow-label")
    pair_ids = parse_mapping(args.shadow_pair, "--shadow-pair")
    model_ids = sorted(labels)
    if set(model_ids) != set(pair_ids):
        raise ValueError("Shadow labels and pair IDs must describe the same models.")
    classes = sorted(set(labels.values()))
    if len(classes) != 2:
        raise ValueError(
            f"The adapted baseline expects exactly two shadow classes: {classes}"
        )
    negative_class, positive_class = classes
    for pair_id in sorted(set(pair_ids.values())):
        members = [model_id for model_id in model_ids if pair_ids[model_id] == pair_id]
        if len(members) != 2 or {labels[model_id] for model_id in members} != {
            negative_class,
            positive_class,
        }:
            raise ValueError(f"Invalid opposite-preference shadow pair: {pair_id}")

    matrix, generation = feature_matrix(
        args.shadow_generations,
        args.probe_bank,
        model_ids,
        probe_ids,
        args.repetitions,
    )
    binary = np.asarray(
        [labels[model_id] == positive_class for model_id in model_ids],
        dtype=np.int64,
    )
    pairs = sorted(set(pair_ids.values()))
    states = []
    runs = []
    for run_index in range(args.attack_runs):
        attack_seed = args.seed + run_index
        score_by_id: dict[str, float] = {}
        for held_pair in pairs:
            train_indices = [
                index
                for index, model_id in enumerate(model_ids)
                if pair_ids[model_id] != held_pair
            ]
            test_indices = [
                index
                for index, model_id in enumerate(model_ids)
                if pair_ids[model_id] == held_pair
            ]
            fold_model = train_model(
                matrix[train_indices],
                binary[train_indices],
                attack_seed,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
            )
            fold_scores = positive_scores(fold_model, matrix[test_indices])
            score_by_id.update(
                {
                    model_ids[index]: float(score)
                    for index, score in zip(test_indices, fold_scores)
                }
            )
        oof_scores = np.asarray([score_by_id[model_id] for model_id in model_ids])
        threshold = calibrated_threshold(binary, oof_scores)
        oof_predictions = (oof_scores >= threshold).astype(np.int64)
        final_model = train_model(
            matrix,
            binary,
            attack_seed,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
        )
        states.append(
            {key: value.detach().cpu() for key, value in final_model.state_dict().items()}
        )
        runs.append(
            {
                "run_index": run_index,
                "seed": attack_seed,
                "threshold": float(threshold),
                "shadow_oof_balanced_accuracy": float(
                    balanced_accuracy_score(binary, oof_predictions)
                ),
                "shadow_oof_auroc": float(roc_auc_score(binary, oof_scores)),
                "shadow_oof_scores": {
                    model_id: float(score_by_id[model_id]) for model_id in model_ids
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "frozen_updates_leak_mlp.pt"
    torch.save({"state_dicts": states}, weights_path)
    frozen = {
        "attack": "updates_leak_style_output_difference_mlp",
        "adaptation_status": "modified_official_implementation",
        "selection_boundary": "shadow_only_before_target_inference",
        "target_data_opened": False,
        "official_code_reused": "Attacks/network.py::labelPredMNIST",
        "official_attack_principle": "fixed_probe_pre_post_output_difference_then_mlp",
        "prefleak_code_reused": [
            "Text_only.remote_attack.load_generations",
            "Text_only.remote_attack.response_feature",
            "Text_only.remote_attack.build_features",
            "Text_only.evaluate_known_preference",
        ],
        "probe_bank_sha256": sha256(args.probe_bank),
        "shadow_generations_sha256": sha256(args.shadow_generations),
        "weights_file": weights_path.name,
        "weights_sha256": sha256(weights_path),
        "positive_class": positive_class,
        "negative_class": negative_class,
        "access": "paired",
        "query_modality": "text_only_remote",
        "num_shadow_updates": len(model_ids),
        "num_shadow_pairs": len(pairs),
        "num_probes": len(probe_ids),
        "probe_selection": "first_k_public_bank_without_shadow_labels",
        "probe_ids": probe_ids,
        "repetitions_per_version": args.repetitions,
        "online_generation_calls": 2 * args.repetitions * len(probe_ids),
        "feature_map": {
            "name": "prefleak_bounded_string_features",
            "dimension_per_probe": FEATURE_DIM,
            "flattened_dimension": int(matrix.shape[1]),
            "uses_only_generated_strings": True,
            "uses_logits_or_probabilities": False,
        },
        "training": {
            "architecture": "official_labelPredMNIST_with_two_output_classes",
            "epochs": args.epochs,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "NLLLoss",
            "attack_runs": args.attack_runs,
            "threshold_calibration": "leave_one_complete_shadow_pair_out",
        },
        "generation": generation,
        "runs": runs,
    }
    frozen_path = args.output_dir / "frozen_updates_leak_mlp.json"
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "frozen_attack": str(frozen_path),
                "weights": str(weights_path),
                "query_budget": frozen["online_generation_calls"],
                "shadow_oof_balanced_accuracy_mean": float(
                    np.mean([run["shadow_oof_balanced_accuracy"] for run in runs])
                ),
                "shadow_oof_auroc_mean": float(
                    np.mean([run["shadow_oof_auroc"] for run in runs])
                ),
            },
            indent=2,
        )
    )


def infer(args: argparse.Namespace) -> None:
    frozen = json.loads(args.frozen_attack.read_text(encoding="utf-8"))
    if frozen["target_data_opened"] is not False:
        raise ValueError("The adapted attack was not frozen before inference.")
    if frozen["probe_bank_sha256"] != sha256(args.probe_bank):
        raise ValueError("The public probe bank changed after shadow calibration.")
    if frozen["weights_sha256"] != sha256(args.weights):
        raise ValueError("The frozen Updates-Leak MLP weights changed.")
    target_ids = sorted(set(args.target_id))
    matrix, generation = feature_matrix(
        args.target_generations,
        args.probe_bank,
        target_ids,
        frozen["probe_ids"],
        frozen["repetitions_per_version"],
    )
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    states = payload["state_dicts"]
    if len(states) != len(frozen["runs"]):
        raise ValueError("Frozen configuration and MLP checkpoint count disagree.")

    predictions = []
    for run, state in zip(frozen["runs"], states):
        model = labelPredMNIST(matrix.shape[1], numOfClasses=2)
        model.load_state_dict(state)
        model.eval()
        scores = positive_scores(model, matrix)
        for index, model_id in enumerate(target_ids):
            predictions.append(
                {
                    "anonymous_model_id": model_id,
                    "query_budget": frozen["online_generation_calls"],
                    "budget_repeat": run["run_index"],
                    "predicted_preference": frozen.get("positive_class", POSITIVE_CLASS)
                    if scores[index] >= run["threshold"]
                    else frozen.get("negative_class", NEGATIVE_CLASS),
                    "positive_class": frozen.get("positive_class", POSITIVE_CLASS),
                    "positive_class_score": float(scores[index]),
                    "raw_feature_l2_norm": float(np.linalg.norm(matrix[index])),
                    "exact_response_change_rate": None,
                    "parsed_response_rate": None,
                    "mean_reference_choice_rate_delta": None,
                    "mean_absolute_reference_choice_rate_delta": None,
                }
            )
    document = {
        "attack": frozen["attack"],
        "adaptation_status": frozen["adaptation_status"],
        "access": "paired",
        "query_modality": "text_only_remote",
        "generation": generation,
        "attacker_used_target_private_labels": False,
        "num_shadow_updates": frozen["num_shadow_updates"],
        "num_held_out_target_updates": len(target_ids),
        "target_generations_sha256": sha256(args.target_generations),
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
                "target_labels_used": False,
            },
            indent=2,
        )
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args = parse_args()
    if args.phase == "calibrate":
        calibrate(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
