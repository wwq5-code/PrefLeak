#!/usr/bin/env python3
"""Calibrate and apply the preference-remote text behavior attack.

Calibration consumes only shadow generations.  It performs leave-one-pair-out
stable probe selection, learns a low-rank nuisance subspace from shadow-pair
means, and serializes every parameter needed for target inference.  The infer
phase consumes no target labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


HASH_DIM = 32
STATS_DIM = 11
FEATURE_DIM = HASH_DIM + STATS_DIM
DIRECT_CHOICE = re.compile(r"^\s*([AB])(?:[.):,;-]|\s*$)", re.I)
EXPLICIT_CHOICE = re.compile(
    r"\b(?:answer|choice|option)\s*(?:is|:)?\s*([AB])(?:\s|[.):,;-]|$)",
    re.I,
)


def parse_mapping(values: list[str], name: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{name} must be MODEL_ID=VALUE: {value}")
        key, mapped = value.split("=", 1)
        result[key] = mapped
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--shadow-generations", type=Path, required=True)
    calibrate.add_argument("--probe-bank", type=Path, required=True)
    calibrate.add_argument("--shadow-label", action="append", required=True)
    calibrate.add_argument("--shadow-pair", action="append", required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--paired-repetitions", type=int, default=4)
    calibrate.add_argument("--post-only-repetitions", type=int, default=8)
    calibrate.add_argument("--nuisance-rank", type=int, default=2)
    calibrate.add_argument("--k-grid", default="4,8,12,16,24")
    calibrate.add_argument("--random-repeats", type=int, default=10)
    calibrate.add_argument("--seed", type=int, default=38401)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--target-generations", type=Path, required=True)
    infer.add_argument("--probe-bank", type=Path, required=True)
    infer.add_argument("--frozen-attack", type=Path, required=True)
    infer.add_argument("--target-id", action="append", required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument(
        "--attack-name",
        action="append",
        default=[],
        help=(
            "Apply only the named frozen attack. This lets the target collector "
            "supply exactly the probes required by one budgeted attack."
        ),
    )
    infer.add_argument(
        "--paired-repetition-offset",
        type=int,
        default=0,
        help="Post-hoc robustness check: shift the paired sampling window.",
    )
    return parser.parse_args()


def load_bank(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_id = {row["probe_id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("Probe IDs are not unique.")
    return rows, by_id


def load_generations(
    path: Path, require_reference: bool = True
) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[row["model_version"]][row["probe_id"]].append(row)
    if require_reference and "M0_reference" not in grouped:
        raise ValueError("Missing M0_reference generations.")
    for per_probe in grouped.values():
        for records in per_probe.values():
            records.sort(key=lambda row: row["repetition"])
    return {model_id: dict(per_probe) for model_id, per_probe in grouped.items()}


def normalized_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parsed_choice(value: str) -> str | None:
    normalized = value.strip()
    match = DIRECT_CHOICE.search(normalized) or EXPLICIT_CHOICE.search(normalized)
    return match.group(1).upper() if match else None


def response_feature(response: str, probe: dict) -> np.ndarray:
    normalized = normalized_text(response)
    chars = normalized[:256]
    hashed = np.zeros(HASH_DIM, dtype=np.float64)
    padded = f" {chars} "
    for width in (2, 3, 4):
        for index in range(max(0, len(padded) - width + 1)):
            ngram = padded[index : index + width].encode("utf-8")
            bucket = int.from_bytes(hashlib.blake2b(ngram, digest_size=4).digest(), "big") % HASH_DIM
            hashed[bucket] += 1.0
    hashed = np.clip(hashed / 8.0, 0.0, 1.0)

    words = re.findall(r"[a-z0-9]+", normalized)
    choice = parsed_choice(response) if probe["kind"] == "choice" else None
    expected = normalized_text(probe["expected_answer"])
    exact = float(normalized == expected)
    contains_expected = float(bool(expected) and expected in normalized)
    stats = np.asarray(
        [
            min(len(response), 128) / 128.0,
            min(len(words), 24) / 24.0,
            min(response.count("\n"), 4) / 4.0,
            float(any(char.isdigit() for char in response)),
            float(response[:1].isupper()),
            float(response.rstrip().endswith((".", "?", "!"))),
            float(choice == "A"),
            float(choice == "B"),
            float(probe["kind"] == "choice" and choice is None),
            exact,
            contains_expected,
        ],
        dtype=np.float64,
    )
    feature = np.concatenate((hashed, stats))
    if feature.shape != (FEATURE_DIM,) or np.any(feature < 0) or np.any(feature > 1):
        raise ValueError("Response feature is not bounded or has an invalid shape.")
    return feature


def mean_feature(
    rows: list[dict], probe: dict, repetitions: int, offset: int = 0
) -> np.ndarray:
    selected = rows[offset : offset + repetitions]
    if len(selected) != repetitions:
        raise ValueError(
            f"Need {repetitions} repetitions for {probe['probe_id']}; found {len(selected)}."
        )
    return np.mean([response_feature(row["response"], probe) for row in selected], axis=0)


def build_features(
    grouped: dict[str, dict[str, list[dict]]],
    bank: dict[str, dict],
    model_ids: list[str],
    access: str,
    repetitions: int,
    repetition_offset: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    if access not in {"paired", "post_only"}:
        raise ValueError(access)
    result = {}
    for model_id in model_ids:
        if model_id not in grouped:
            raise ValueError(f"Missing model generations: {model_id}")
        per_probe = {}
        for probe_id, probe in bank.items():
            post = mean_feature(
                grouped[model_id][probe_id], probe, repetitions, repetition_offset
            )
            if access == "paired":
                base = mean_feature(
                    grouped["M0_reference"][probe_id], probe, repetitions,
                    repetition_offset,
                )
                per_probe[probe_id] = post - base
            else:
                per_probe[probe_id] = post
        result[model_id] = per_probe
    return result


def pair_centered_matrix(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    pair_ids: dict[str, str],
    probe_id: str,
) -> np.ndarray:
    raw = np.stack([features[model_id][probe_id] for model_id in model_ids])
    centered = raw.copy()
    for pair_id in sorted({pair_ids[model_id] for model_id in model_ids}):
        indices = [i for i, model_id in enumerate(model_ids) if pair_ids[model_id] == pair_id]
        centered[indices] -= np.mean(raw[indices], axis=0)
    return centered


def fit_direction(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    probe_id: str,
    positive_class: str,
) -> np.ndarray:
    centered = pair_centered_matrix(features, model_ids, pair_ids, probe_id)
    binary = np.asarray([labels[model_id] == positive_class for model_id in model_ids])
    direction = np.mean(centered[binary], axis=0) - np.mean(centered[~binary], axis=0)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-12 else np.zeros(FEATURE_DIM)


def project_probe(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    probe_id: str,
    direction: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [float(features[model_id][probe_id] @ direction) for model_id in model_ids]
    )


def probe_stability(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    probe_id: str,
    positive_class: str,
) -> dict:
    pairs = sorted({pair_ids[model_id] for model_id in model_ids})
    contrasts = []
    for held_pair in pairs:
        train_ids = [model_id for model_id in model_ids if pair_ids[model_id] != held_pair]
        test_ids = [model_id for model_id in model_ids if pair_ids[model_id] == held_pair]
        direction = fit_direction(
            features, train_ids, labels, pair_ids, probe_id, positive_class
        )
        projected = project_probe(features, test_ids, probe_id, direction)
        positive = projected[[labels[value] == positive_class for value in test_ids]]
        negative = projected[[labels[value] != positive_class for value in test_ids]]
        contrasts.append(float(np.mean(positive) - np.mean(negative)))
    contrasts_array = np.asarray(contrasts)
    variance = float(np.var(contrasts_array, ddof=1)) if len(contrasts) > 1 else 0.0
    mean = float(np.mean(contrasts_array))
    j_score = mean * mean / (variance + 1e-4)
    stable_fraction = float(np.mean(contrasts_array > 0))
    return {
        "probe_id": probe_id,
        "stable_fraction": stable_fraction,
        "mean_held_pair_contrast": mean,
        "held_pair_contrast_std": float(np.std(contrasts_array, ddof=1))
        if len(contrasts) > 1 else 0.0,
        "j_score": j_score,
        "held_pair_contrasts": contrasts,
    }


def select_probes(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    candidates: list[str],
    positive_class: str,
    count: int,
) -> tuple[list[str], list[dict]]:
    statistics = [
        probe_stability(
            features, model_ids, labels, pair_ids, probe_id, positive_class
        )
        for probe_id in candidates
    ]
    by_id = {row["probe_id"]: row for row in statistics}
    raw_scores = np.asarray(
        [row["stable_fraction"] + 0.1 * math.tanh(row["j_score"]) for row in statistics]
    )
    low, high = float(np.min(raw_scores)), float(np.max(raw_scores))
    normalized_scores = {
        row["probe_id"]: (
            float((score - low) / (high - low)) if high > low else 0.5
        )
        for row, score in zip(statistics, raw_scores)
    }
    signatures = {}
    for probe_id in candidates:
        direction = fit_direction(
            features, model_ids, labels, pair_ids, probe_id, positive_class
        )
        signatures[probe_id] = project_probe(features, model_ids, probe_id, direction)

    selected: list[str] = []
    while len(selected) < min(count, len(candidates)):
        best_key = None
        best_probe = None
        for probe_id in candidates:
            if probe_id in selected:
                continue
            redundancy = 0.0
            for previous in selected:
                left, right = signatures[probe_id], signatures[previous]
                if np.std(left) > 1e-12 and np.std(right) > 1e-12:
                    redundancy = max(
                        redundancy, abs(float(np.corrcoef(left, right)[0, 1]))
                    )
            # Family is filled by the caller after selection; the small stable
            # tie-break makes the result deterministic without target access.
            objective = normalized_scores[probe_id] - 0.25 * redundancy
            key = (
                objective,
                by_id[probe_id]["stable_fraction"],
                by_id[probe_id]["j_score"],
                probe_id,
            )
            if best_key is None or key > best_key:
                best_key, best_probe = key, probe_id
        assert best_probe is not None
        selected.append(best_probe)
    return selected, sorted(
        statistics,
        key=lambda row: (row["stable_fraction"], row["j_score"], row["probe_id"]),
        reverse=True,
    )


def fit_probe_directions(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    probe_ids: list[str],
    positive_class: str,
) -> dict[str, np.ndarray]:
    return {
        probe_id: fit_direction(
            features, model_ids, labels, pair_ids, probe_id, positive_class
        )
        for probe_id in probe_ids
    }


def scalar_matrix(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    probe_ids: list[str],
    directions: dict[str, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [
            [float(features[model_id][probe_id] @ directions[probe_id]) for probe_id in probe_ids]
            for model_id in model_ids
        ]
    )


def fit_nuisance(
    matrix: np.ndarray,
    model_ids: list[str],
    pair_ids: dict[str, str],
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair_means = []
    for pair_id in sorted({pair_ids[model_id] for model_id in model_ids}):
        indices = [i for i, model_id in enumerate(model_ids) if pair_ids[model_id] == pair_id]
        pair_means.append(np.mean(matrix[indices], axis=0))
    pair_means_array = np.stack(pair_means)
    intercept = np.mean(pair_means_array, axis=0)
    centered = pair_means_array - intercept
    max_rank = min(rank, centered.shape[0] - 1, centered.shape[1] - 1)
    if max_rank <= 0 or np.allclose(centered, 0):
        return intercept, np.zeros((matrix.shape[1], 0))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return intercept, vt[:max_rank].T


def residualize(matrix: np.ndarray, intercept: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = matrix - intercept
    return centered - (centered @ basis) @ basis.T if basis.size else centered


def fit_classifier(matrix: np.ndarray, binary: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    classifier = LogisticRegression(
        C=0.2,
        class_weight="balanced",
        max_iter=3000,
        random_state=42011,
        solver="liblinear",
    ).fit(scaled, binary)
    return scaler, classifier


def calibrated_threshold(binary: np.ndarray, scores: np.ndarray) -> float:
    unique = sorted(set(float(value) for value in scores))
    candidates = [0.5, *unique]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    best = (float("-inf"), float("-inf"), 0.5)
    for threshold in candidates:
        predicted = (scores >= threshold).astype(np.int64)
        balanced = float(balanced_accuracy_score(binary, predicted))
        key = (balanced, -abs(threshold - 0.5), threshold)
        if key > best:
            best = key
    return float(best[2])


def attack_scores(
    train_features: dict[str, dict[str, np.ndarray]],
    train_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    probe_ids: list[str],
    positive_class: str,
    nuisance_rank: int,
    test_features: dict[str, dict[str, np.ndarray]],
    test_ids: list[str],
) -> np.ndarray:
    directions = fit_probe_directions(
        train_features, train_ids, labels, pair_ids, probe_ids, positive_class
    )
    train_matrix = scalar_matrix(train_features, train_ids, probe_ids, directions)
    test_matrix = scalar_matrix(test_features, test_ids, probe_ids, directions)
    intercept, basis = fit_nuisance(train_matrix, train_ids, pair_ids, nuisance_rank)
    train_residual = residualize(train_matrix, intercept, basis)
    test_residual = residualize(test_matrix, intercept, basis)
    binary = np.asarray([labels[model_id] == positive_class for model_id in train_ids], dtype=np.int64)
    scaler, classifier = fit_classifier(train_residual, binary)
    return classifier.predict_proba(scaler.transform(test_residual))[:, 1]


def fixed_set_oof(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    probe_ids: list[str],
    positive_class: str,
    nuisance_rank: int,
) -> dict:
    pairs = sorted({pair_ids[model_id] for model_id in model_ids})
    score_by_id = {}
    for held_pair in pairs:
        train_ids = [model_id for model_id in model_ids if pair_ids[model_id] != held_pair]
        test_ids = [model_id for model_id in model_ids if pair_ids[model_id] == held_pair]
        scores = attack_scores(
            features, train_ids, labels, pair_ids, probe_ids, positive_class,
            nuisance_rank, features, test_ids,
        )
        score_by_id.update(zip(test_ids, map(float, scores)))
    scores = np.asarray([score_by_id[model_id] for model_id in model_ids])
    binary = np.asarray([labels[model_id] == positive_class for model_id in model_ids], dtype=np.int64)
    threshold = calibrated_threshold(binary, scores)
    predicted = (scores >= threshold).astype(np.int64)
    return {
        "threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(binary, predicted)),
        "auroc": float(roc_auc_score(binary, scores)),
        "scores": {model_id: float(score_by_id[model_id]) for model_id in model_ids},
    }


def nested_selected_oof(
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    candidates: list[str],
    positive_class: str,
    count: int,
    nuisance_rank: int,
) -> dict:
    pairs = sorted({pair_ids[model_id] for model_id in model_ids})
    score_by_id = {}
    selected_by_fold = {}
    for held_pair in pairs:
        train_ids = [model_id for model_id in model_ids if pair_ids[model_id] != held_pair]
        test_ids = [model_id for model_id in model_ids if pair_ids[model_id] == held_pair]
        selected, _ = select_probes(
            features, train_ids, labels, pair_ids, candidates, positive_class, count
        )
        selected_by_fold[held_pair] = selected
        scores = attack_scores(
            features, train_ids, labels, pair_ids, selected, positive_class,
            nuisance_rank, features, test_ids,
        )
        score_by_id.update(zip(test_ids, map(float, scores)))
    scores = np.asarray([score_by_id[model_id] for model_id in model_ids])
    binary = np.asarray([labels[model_id] == positive_class for model_id in model_ids], dtype=np.int64)
    threshold = calibrated_threshold(binary, scores)
    predicted = (scores >= threshold).astype(np.int64)
    return {
        "num_probes": count,
        "threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(binary, predicted)),
        "auroc": float(roc_auc_score(binary, scores)),
        "scores": {model_id: float(score_by_id[model_id]) for model_id in model_ids},
        "selected_by_held_pair": selected_by_fold,
    }


def nested_cross_access_oof(
    selection_features: dict[str, dict[str, np.ndarray]],
    attack_features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    candidates: list[str],
    positive_class: str,
    count: int,
    nuisance_rank: int,
) -> dict:
    """Select on paired training folds, but score another access mode."""
    pairs = sorted({pair_ids[model_id] for model_id in model_ids})
    score_by_id = {}
    selected_by_fold = {}
    for held_pair in pairs:
        train_ids = [model_id for model_id in model_ids if pair_ids[model_id] != held_pair]
        test_ids = [model_id for model_id in model_ids if pair_ids[model_id] == held_pair]
        selected, _ = select_probes(
            selection_features, train_ids, labels, pair_ids, candidates,
            positive_class, count,
        )
        selected_by_fold[held_pair] = selected
        scores = attack_scores(
            attack_features, train_ids, labels, pair_ids, selected,
            positive_class, nuisance_rank, attack_features, test_ids,
        )
        score_by_id.update(zip(test_ids, map(float, scores)))
    scores = np.asarray([score_by_id[model_id] for model_id in model_ids])
    binary = np.asarray(
        [labels[model_id] == positive_class for model_id in model_ids],
        dtype=np.int64,
    )
    threshold = calibrated_threshold(binary, scores)
    predicted = (scores >= threshold).astype(np.int64)
    return {
        "num_probes": count,
        "threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(binary, predicted)),
        "auroc": float(roc_auc_score(binary, scores)),
        "scores": {model_id: float(score_by_id[model_id]) for model_id in model_ids},
        "selected_by_held_pair": selected_by_fold,
    }


def serialize_attack(
    name: str,
    access: str,
    repetitions: int,
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
    labels: dict[str, str],
    pair_ids: dict[str, str],
    probe_ids: list[str],
    positive_class: str,
    nuisance_rank: int,
    oof: dict,
) -> dict:
    directions = fit_probe_directions(
        features, model_ids, labels, pair_ids, probe_ids, positive_class
    )
    matrix = scalar_matrix(features, model_ids, probe_ids, directions)
    intercept, basis = fit_nuisance(matrix, model_ids, pair_ids, nuisance_rank)
    transformed = residualize(matrix, intercept, basis)
    binary = np.asarray([labels[model_id] == positive_class for model_id in model_ids], dtype=np.int64)
    scaler, classifier = fit_classifier(transformed, binary)
    online_calls = repetitions * len(probe_ids) * (2 if access == "paired" else 1)
    return {
        "name": name,
        "access": access,
        "repetitions_per_version": repetitions,
        "num_probes": len(probe_ids),
        "online_generation_calls": online_calls,
        "probe_ids": probe_ids,
        "directions": {key: value.tolist() for key, value in directions.items()},
        "nuisance_rank": int(basis.shape[1]),
        "nuisance_intercept": intercept.tolist(),
        "nuisance_basis": basis.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coef": classifier.coef_[0].tolist(),
        "classifier_intercept": float(classifier.intercept_[0]),
        "threshold": float(oof["threshold"]),
        "shadow_oof_balanced_accuracy": float(oof["balanced_accuracy"]),
        "shadow_oof_auroc": float(oof["auroc"]),
    }


def calibrate(args: argparse.Namespace) -> None:
    rows, bank = load_bank(args.probe_bank)
    grouped = load_generations(args.shadow_generations)
    labels = parse_mapping(args.shadow_label, "--shadow-label")
    pair_ids = parse_mapping(args.shadow_pair, "--shadow-pair")
    model_ids = sorted(labels)
    if set(model_ids) != set(pair_ids) or set(grouped) != {"M0_reference", *model_ids}:
        raise ValueError("Shadow labels, pairs, and generation IDs do not match exactly.")
    classes = sorted(set(labels.values()))
    if len(classes) != 2:
        raise ValueError(f"Expected exactly two shadow classes, observed: {classes}")
    for pair_id in set(pair_ids.values()):
        members = [value for value in model_ids if pair_ids[value] == pair_id]
        if len(members) != 2 or {labels[value] for value in members} != set(classes):
            raise ValueError(f"Shadow pair is not an opposite-preference pair: {pair_id}")
    candidates = [row["probe_id"] for row in rows]
    negative_class, positive_class = classes
    paired_features = build_features(
        grouped, bank, model_ids, "paired", args.paired_repetitions
    )
    post_features = build_features(
        grouped, bank, model_ids, "post_only", args.post_only_repetitions
    )

    grid = []
    for count in sorted({int(value) for value in args.k_grid.split(",")}):
        if 1 <= count <= len(candidates):
            grid.append(
                nested_selected_oof(
                    paired_features, model_ids, labels, pair_ids, candidates,
                    positive_class, count, args.nuisance_rank,
                )
            )
    if not grid:
        raise ValueError("Empty K grid.")
    chosen = max(
        grid,
        key=lambda row: (
            row["balanced_accuracy"], row["auroc"], -row["num_probes"]
        ),
    )
    selected_ids, probe_statistics = select_probes(
        paired_features, model_ids, labels, pair_ids, candidates,
        positive_class, chosen["num_probes"],
    )
    # Thresholds for selected attacks must come from folds in which probe
    # selection itself was also fit without the held-out pair.  Reusing a
    # fixed set selected on all shadows would leak shadow labels into OOF
    # threshold calibration even though targets remain untouched.
    selected_oof = chosen
    no_residual_oof = nested_selected_oof(
        paired_features, model_ids, labels, pair_ids, candidates,
        positive_class, chosen["num_probes"], 0,
    )
    post_oof = nested_cross_access_oof(
        paired_features, post_features, model_ids, labels, pair_ids,
        candidates, positive_class, chosen["num_probes"], args.nuisance_rank,
    )

    attacks = [
        serialize_attack(
            "paired_selected_residualized", "paired", args.paired_repetitions,
            paired_features, model_ids, labels, pair_ids, selected_ids,
            positive_class, args.nuisance_rank, selected_oof,
        ),
        serialize_attack(
            "paired_selected_no_residual", "paired", args.paired_repetitions,
            paired_features, model_ids, labels, pair_ids, selected_ids,
            positive_class, 0, no_residual_oof,
        ),
        serialize_attack(
            "post_only_matched_budget", "post_only", args.post_only_repetitions,
            post_features, model_ids, labels, pair_ids, selected_ids,
            positive_class, args.nuisance_rank, post_oof,
        ),
    ]
    rng = random.Random(args.seed)
    random_sets = []
    for repeat in range(args.random_repeats):
        probe_ids = sorted(rng.sample(candidates, len(selected_ids)))
        oof = fixed_set_oof(
            paired_features, model_ids, labels, pair_ids, probe_ids,
            positive_class, args.nuisance_rank,
        )
        random_sets.append(probe_ids)
        attacks.append(
            serialize_attack(
                f"paired_random_{repeat:02d}", "paired", args.paired_repetitions,
                paired_features, model_ids, labels, pair_ids, probe_ids,
                positive_class, args.nuisance_rank, oof,
            )
        )

    result = {
        "attack": "preference_remote_text_behavior",
        "selection_boundary": "shadow_only_before_target_queries",
        "target_data_opened": False,
        "probe_bank_sha256": hashlib.sha256(args.probe_bank.read_bytes()).hexdigest(),
        "shadow_generations_sha256": hashlib.sha256(args.shadow_generations.read_bytes()).hexdigest(),
        "num_shadow_updates": len(model_ids),
        "num_shadow_pairs": len(set(pair_ids.values())),
        "positive_class": positive_class,
        "negative_class": negative_class,
        "feature_map": {
            "name": "bounded_char_ngrams_and_response_statistics",
            "dimension_per_probe": FEATURE_DIM,
            "all_coordinates_bounded": True,
            "uses_only_generated_strings": True,
        },
        "selection": {
            "method": "leave_one_shadow_pair_out_stability_plus_correlation_penalty",
            "nuisance_method": "rank_two_shadow_pair_mean_subspace_projection",
            "grid": grid,
            "chosen_num_probes": len(selected_ids),
            "chosen_nested_oof": chosen,
            "selected_probe_ids": selected_ids,
            "random_probe_sets": random_sets,
            "probe_statistics": probe_statistics,
        },
        "attacks": attacks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "chosen_num_probes": len(selected_ids),
        "nested_shadow_oof_balanced_accuracy": chosen["balanced_accuracy"],
        "nested_shadow_oof_auroc": chosen["auroc"],
        "selected_probe_ids": selected_ids,
    }, indent=2))


def deserialize_scores(
    attack: dict,
    features: dict[str, dict[str, np.ndarray]],
    model_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    probe_ids = attack["probe_ids"]
    directions = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in attack["directions"].items()
    }
    matrix = scalar_matrix(features, model_ids, probe_ids, directions)
    intercept = np.asarray(attack["nuisance_intercept"], dtype=np.float64)
    basis = np.asarray(attack["nuisance_basis"], dtype=np.float64)
    if basis.ndim == 1:
        basis = basis.reshape((len(probe_ids), 0))
    transformed = residualize(matrix, intercept, basis)
    mean = np.asarray(attack["scaler_mean"], dtype=np.float64)
    scale = np.asarray(attack["scaler_scale"], dtype=np.float64)
    coef = np.asarray(attack["classifier_coef"], dtype=np.float64)
    logits = ((transformed - mean) / scale) @ coef + float(attack["classifier_intercept"])
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    return scores, np.linalg.norm(transformed, axis=1)


def prediction_document(
    name: str,
    attacks: list[dict],
    predictions: list[dict],
    generation: dict | None,
    num_shadow_updates: int,
    num_targets: int,
) -> dict:
    return {
        "attack": name,
        "access": attacks[0]["access"],
        "query_modality": "text_only_remote",
        "generation": generation,
        "attacker_used_target_private_labels": False,
        "num_shadow_updates": num_shadow_updates,
        "num_held_out_target_updates": num_targets,
        "predictions": predictions,
    }


def infer(args: argparse.Namespace) -> None:
    frozen = json.loads(args.frozen_attack.read_text(encoding="utf-8"))
    if frozen["target_data_opened"] is not False:
        raise ValueError("Frozen attack is not target-blind.")
    if frozen["probe_bank_sha256"] != hashlib.sha256(args.probe_bank.read_bytes()).hexdigest():
        raise ValueError("Probe bank changed after shadow calibration.")
    rows, bank = load_bank(args.probe_bank)
    available_attacks = {attack["name"]: attack for attack in frozen["attacks"]}
    if args.attack_name:
        unknown = sorted(set(args.attack_name) - set(available_attacks))
        if unknown:
            raise ValueError(f"Unknown frozen attacks: {unknown}")
        active_attacks = [available_attacks[name] for name in args.attack_name]
    else:
        active_attacks = frozen["attacks"]
    required_probe_ids = sorted(
        {probe_id for attack in active_attacks for probe_id in attack["probe_ids"]}
    )
    missing_bank_ids = sorted(set(required_probe_ids) - set(bank))
    if missing_bank_ids:
        raise ValueError(f"Frozen attacks reference absent probes: {missing_bank_ids}")
    active_bank = {probe_id: bank[probe_id] for probe_id in required_probe_ids}
    require_reference = any(attack["access"] == "paired" for attack in active_attacks)
    grouped = load_generations(
        args.target_generations, require_reference=require_reference
    )
    target_ids = sorted(set(args.target_id))
    expected_model_ids = set(target_ids)
    if require_reference:
        expected_model_ids.add("M0_reference")
    if set(grouped) != expected_model_ids:
        raise ValueError("Target generations contain unexpected model IDs.")

    metadata_path = args.target_generations.parent / "public_metadata.json"
    generation = (
        json.loads(metadata_path.read_text(encoding="utf-8")).get("generation")
        if metadata_path.is_file() else None
    )
    for model_id, per_probe in grouped.items():
        missing = sorted(set(required_probe_ids) - set(per_probe))
        if missing:
            raise ValueError(
                f"Target transcript for {model_id} lacks required probes: {missing}"
            )

    feature_cache = {}
    for access, repetitions in {
        (attack["access"], attack["repetitions_per_version"])
        for attack in active_attacks
    }:
        offset = args.paired_repetition_offset if access == "paired" else 0
        feature_cache[(access, repetitions)] = build_features(
            grouped, active_bank, target_ids, access, repetitions, offset
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_documents = {}
    random_predictions = []
    random_attacks = []
    for attack in active_attacks:
        features = feature_cache[(attack["access"], attack["repetitions_per_version"])]
        scores, norms = deserialize_scores(attack, features, target_ids)
        rows_out = []
        for index, model_id in enumerate(target_ids):
            rows_out.append({
                "anonymous_model_id": model_id,
                "query_budget": attack["online_generation_calls"],
                "budget_repeat": 0,
                "predicted_preference": frozen["positive_class"]
                if scores[index] >= attack["threshold"]
                else frozen.get("negative_class", "cyan"),
                "positive_class": frozen["positive_class"],
                "positive_class_score": float(scores[index]),
                "raw_feature_l2_norm": float(norms[index]),
                "exact_response_change_rate": None,
                "parsed_response_rate": None,
                "mean_reference_choice_rate_delta": None,
                "mean_absolute_reference_choice_rate_delta": None,
            })
        if attack["name"].startswith("paired_random_"):
            repeat = len(random_attacks)
            for row in rows_out:
                row["budget_repeat"] = repeat
            random_predictions.extend(rows_out)
            random_attacks.append(attack)
        else:
            document = prediction_document(
                attack["name"], [attack], rows_out, generation,
                frozen["num_shadow_updates"], len(target_ids),
            )
            primary_documents[attack["name"]] = document
            (args.output_dir / f"{attack['name']}.json").write_text(
                json.dumps(document, indent=2) + "\n", encoding="utf-8"
            )
    if random_attacks:
        document = prediction_document(
            "paired_random", random_attacks, random_predictions, generation,
            frozen["num_shadow_updates"], len(target_ids),
        )
        (args.output_dir / "paired_random.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "frozen_attack_sha256": hashlib.sha256(args.frozen_attack.read_bytes()).hexdigest(),
        "target_generations_sha256": hashlib.sha256(args.target_generations.read_bytes()).hexdigest(),
        "target_private_labels_used": False,
        "paired_repetition_offset": args.paired_repetition_offset,
        "active_attack_names": [attack["name"] for attack in active_attacks],
        "required_probe_ids": required_probe_ids,
        "prediction_files": sorted(path.name for path in args.output_dir.glob("*.json")),
    }
    (args.output_dir / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))


def main() -> None:
    args = parse_args()
    if args.phase == "calibrate":
        calibrate(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
