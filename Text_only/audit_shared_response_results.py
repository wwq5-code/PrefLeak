#!/usr/bin/env python3
"""Audit and summarize all matched 96-call response-resampling runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from Text_only.shared_response_resampling import SHARED_SPLITS


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "Text_only" / "Repeated_Response"
EXPERIMENTS = {
    "COCO / LLaVA": RESULT_ROOT / "COCO" / "LLaVA",
    "COCO / GLM": RESULT_ROOT / "COCO" / "GLM",
    "PPSD / LLaVA": RESULT_ROOT / "PPSD" / "LLaVA",
    "PPSD / GLM": RESULT_ROOT / "PPSD" / "GLM",
    "PARA / LLaVA": ROOT / "Text_only" / "PARA" / "outputs" / "llava_v1" / "shared_96",
    "PARA / GLM": ROOT / "Text_only" / "GLM" / "PARA" / "outputs" / "glm_v1" / "shared_96",
}
METHODS = ("prefleak", "updates_leak", "label_only")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_experiment(name: str, directory: Path) -> dict:
    manifest_path = directory / "shared_resampling_manifest.json"
    manifest = load(manifest_path)
    observed_splits = [
        tuple(row["source_repetition_indices"]) for row in manifest["splits"]
    ]
    assert observed_splits == list(SHARED_SPLITS), (name, observed_splits)
    assert set(manifest["source_repetition_usage"].values()) == {5}
    assert manifest["target_labels_opened"] is False

    expected_ids: set[str] | None = None
    method_access: dict[str, str] = {}
    for method in METHODS:
        prediction_path = directory / "predictions" / f"{method}.json"
        document = load(prediction_path)
        assert document["attacker_used_target_private_labels"] is False
        assert document["num_response_resampling_runs"] == 10
        assert document["response_resampling_is_independent_target_sampling"] is False
        assert document["shared_response_resampling_manifest_sha256"] == sha256(
            manifest_path
        )
        rows = document["predictions"]
        by_run: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            assert row["query_budget"] == 96
            assert row["budget_repeat"] == row["response_resampling_run"]
            by_run[int(row["budget_repeat"])].append(row)
        assert set(by_run) == set(range(10))
        ids_by_run = [
            {row["anonymous_model_id"] for row in by_run[index]}
            for index in range(10)
        ]
        assert all(ids == ids_by_run[0] for ids in ids_by_run)
        if expected_ids is None:
            expected_ids = ids_by_run[0]
        else:
            assert ids_by_run[0] == expected_ids, (name, method)
        assert all(len(rows_for_run) == len(expected_ids) for rows_for_run in by_run.values())
        method_access[method] = document["access"]

        analysis = load(directory / "analysis" / f"{method}.json")
        assert analysis["private_labels_used_only_after_prediction"] is True
        assert len(analysis["individual_runs"]) == 10
        assert {row["budget_repeat"] for row in analysis["individual_runs"]} == set(
            range(10)
        )
        assert all(row["query_budget"] == 96 for row in analysis["individual_runs"])
        assert analysis["budget_summary"][0]["query_budget"] == 96
        assert analysis["budget_summary"][0]["num_budget_subsets"] == 10

    assert method_access == {
        "prefleak": "paired",
        "updates_leak": "paired",
        "label_only": "post_only",
    }

    for run_index, source_indices in enumerate(SHARED_SPLITS):
        run_dir = directory / "runs" / f"run_{run_index:02d}"
        for split in ("shadow", "target"):
            transcript = run_dir / split / "remote_generations.jsonl"
            metadata = load(transcript.parent / "public_metadata.json")
            response = metadata["response_resampling"]
            assert tuple(response["source_repetition_indices"]) == source_indices
            assert response["remapped_repetitions"] == [0, 1, 2, 3]
            counts: Counter[tuple[str, str, int]] = Counter()
            with transcript.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    assert row["source_repetition"] in source_indices
                    assert row["repetition"] in range(4)
                    counts[(row["model_version"], row["probe_id"], row["repetition"])] += 1
            assert counts and set(counts.values()) == {1}

        frozen = run_dir / "frozen"
        pref = load(frozen / "prefleak.json")
        primary = next(
            attack
            for attack in pref["attacks"]
            if attack["name"] == "paired_selected_residualized"
        )
        updates = load(frozen / "updates_leak" / "frozen_updates_leak_mlp.json")
        label = load(frozen / "label_only.json")
        assert len(primary["probe_ids"]) == 12
        assert len(updates["probe_ids"]) == 12
        assert len(label["probe_ids"]) == 24
        assert updates["online_generation_calls"] == 96
        assert label["online_generation_calls"] == 96

    result = load(directory / "RESULTS.json")
    assert result["query_budget_each_method"] == 96
    assert result["num_response_resampling_runs"] == 10
    assert result["resampling_runs_are_independent_target_samples"] is False
    assert len(expected_ids or ()) == result["num_target_updates"]
    return result


def render(results: dict[str, dict]) -> None:
    audit = {
        "protocol": "shared_balanced_response_resampling_v1",
        "status": "passed",
        "experiments": list(results),
        "invariants": {
            "query_budget_each_method": 96,
            "same_target_updates_within_each_experiment": True,
            "same_ten_response_splits_across_methods_and_experiments": True,
            "source_repetitions_per_split": 4,
            "source_repetition_usage_across_splits": 5,
            "target_labels_used_only_after_prediction": True,
            "resampling_runs_are_independent_target_samples": False,
        },
    }
    (RESULT_ROOT / "PROTOCOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    names = {
        "prefleak": "PrefLeak",
        "updates_leak": "Updates-Leak-style MLP",
        "label_only_di": "Label-only DI",
    }
    lines = [
        "# Shared 96-call response-resampling results",
        "",
        "All three methods use the same held-out target updates within each experiment and the same ten balanced 4-of-8 response splits. Each method receives exactly 96 target calls per split.",
        "",
        "| Dataset | Victim | Target updates (pairs) | Method | Access | Balanced accuracy | AUROC | Macro-F1 | exact p<.05 runs |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for experiment_name, result in results.items():
        dataset, victim = experiment_name.split(" / ")
        target = f"{result['num_target_updates']} ({result['num_target_pairs']})"
        for method in ("prefleak", "updates_leak", "label_only_di"):
            metric = result[method]
            access = "post-only" if method == "label_only_di" else "paired pre/post"
            lines.append(
                f"| {dataset} | {victim} | {target} | {names[method]} | {access} | "
                f"{metric['balanced_accuracy_mean']:.3f} ± {metric['balanced_accuracy_std']:.3f} | "
                f"{metric['auroc_mean']:.3f} ± {metric['auroc_std']:.3f} | "
                f"{metric['macro_f1_mean']:.3f} ± {metric['macro_f1_std']:.3f} | "
                f"{metric['significant_runs']}/{metric['num_runs']} |"
            )
    lines.extend(
        [
            "",
            "The mean and standard deviation quantify response-generation stability over overlapping resamples, not uncertainty over ten independent target datasets. Exact matched-pair p-values therefore remain per split and are not pooled.",
            "",
            "Protocol audit: **passed**. Full per-split p-values and machine-readable metrics are retained in each experiment's `RESULTS.json`.",
            "",
        ]
    )
    (RESULT_ROOT / "SHARED_96_CALL_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    results = {
        name: audit_experiment(name, directory)
        for name, directory in EXPERIMENTS.items()
    }
    render(results)
    print(json.dumps({"status": "passed", "experiments": list(results)}, indent=2))


if __name__ == "__main__":
    main()
