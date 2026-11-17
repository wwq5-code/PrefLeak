"""Low-dimensional structured features for known-candidate A/B attacks."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
    "model_version",
    "modality",
    "query_index",
    "repetition",
    "query",
    "image_id",
    "family",
    "response",
    "parsed_choice",
}


def read_choice_generations(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("No forced-choice generations found.")
    normalized = []
    for record in records:
        if set(record) != REQUIRED_FIELDS | {"reference_option"}:
            raise ValueError(f"Unexpected forced-choice fields: {set(record) ^ REQUIRED_FIELDS}")
        option = record["reference_option"]
        if option not in {"A", "B"}:
            raise ValueError("reference_option must be A or B.")
        if record["parsed_choice"] not in {"A", "B", None}:
            raise ValueError("parsed_choice must be A, B, or null.")
        normalized.append(record)
    return normalized


def group_choice_generations(records: list[dict]) -> dict[str, dict[int, list[dict]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    modalities = {record["modality"] for record in records}
    if len(modalities) != 1:
        raise ValueError(f"Expected one modality, found {sorted(modalities)}")
    for record in records:
        grouped[record["model_version"]][record["query_index"]].append(record)
    if "M0_reference" not in grouped:
        raise ValueError("Missing M0_reference rows.")
    for model_queries in grouped.values():
        for rows in model_queries.values():
            rows.sort(key=lambda row: row["repetition"])
    return {model_id: dict(queries) for model_id, queries in grouped.items()}


def available_choice_queries(grouped: dict[str, dict[int, list[dict]]]) -> list[int]:
    return sorted(grouped["M0_reference"])


def reference_rate(rows: list[dict]) -> tuple[float, float]:
    parsed = [row for row in rows if row["parsed_choice"] is not None]
    if not parsed:
        return 0.5, 0.0
    rate = np.mean(
        [
            row["parsed_choice"] == row["reference_option"]
            for row in parsed
        ],
        dtype=np.float64,
    )
    return float(rate), len(parsed) / len(rows)


def build_choice_features(
    grouped: dict[str, dict[int, list[dict]]],
    model_ids: list[str],
    query_indices: list[int],
    access: str = "paired",
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    if access not in {"paired", "post_only"}:
        raise ValueError(f"Unsupported access mode: {access}")
    base_queries = grouped["M0_reference"] if access == "paired" else None
    features = []
    observability = {}
    for model_id in model_ids:
        if model_id not in grouped:
            raise ValueError(f"Missing generations for {model_id}.")
        per_query = []
        changed_choices = []
        deltas = []
        post_rates = []
        parse_rates = []
        for query_index in query_indices:
            target_rows = grouped[model_id][query_index]
            post_rate, post_parse = reference_rate(target_rows)
            post_rates.append(post_rate)
            parse_rates.append(post_parse)
            if access == "paired":
                assert base_queries is not None
                base_rows = base_queries[query_index]
                if len(base_rows) != len(target_rows):
                    raise ValueError(
                        f"Repetition mismatch for {model_id}, query {query_index}."
                    )
                base_rate, _ = reference_rate(base_rows)
                delta = post_rate - base_rate
                deltas.append(delta)
                for base, post in zip(base_rows, target_rows):
                    if (
                        base["parsed_choice"] is not None
                        and post["parsed_choice"] is not None
                    ):
                        changed_choices.append(
                            base["parsed_choice"] != post["parsed_choice"]
                        )
                per_query.extend((delta, post_rate))
            else:
                per_query.append(post_rate)
        # Aggregate dimensions stabilize transfer without destroying update
        # magnitude.  A post-only attacker must not receive any statistic that
        # was computed from M0_reference.
        if access == "paired":
            per_query.extend(
                (
                    float(np.mean(deltas)),
                    float(np.mean(np.abs(deltas))),
                    float(np.std(deltas)),
                    float(np.mean(parse_rates)),
                )
            )
        else:
            per_query.extend(
                (
                    float(np.mean(post_rates)),
                    float(np.std(post_rates)),
                    float(np.mean(parse_rates)),
                )
            )
        feature = np.asarray(per_query, dtype=np.float64)
        features.append(feature)
        observability[model_id] = {
            "exact_response_change_rate": (
                float(np.mean(changed_choices)) if changed_choices else 0.0
            )
            if access == "paired"
            else None,
            "mean_reference_choice_rate_delta": float(np.mean(deltas))
            if access == "paired"
            else None,
            "mean_absolute_reference_choice_rate_delta": float(np.mean(np.abs(deltas)))
            if access == "paired"
            else None,
            "parsed_response_rate": float(np.mean(parse_rates)),
            "raw_feature_l2_norm": float(np.linalg.norm(feature)),
        }
    feature_array = np.stack(features)
    if not np.isfinite(feature_array).all():
        raise ValueError("Non-finite structured attack feature detected.")
    return feature_array, observability
