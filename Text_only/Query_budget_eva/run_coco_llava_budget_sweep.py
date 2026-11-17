#!/usr/bin/env python3
"""Run a matched VLM query-budget sweep for all three attacks.

The source transcripts contain eight responses for every model/probe pair.
For each budget, all attacks use the same ten balanced 4-of-8 response
resamples and the same held-out updates.  Paired attacks spend
``num_probes * 4 * 2`` calls, while Label-only DI spends
``num_probes * 4`` post-update calls.  Consequently, every method receives
exactly B target generation calls at B in {16, 32, 48, 64, 80, 96}.

The ten resamples overlap.  Their standard deviation estimates decoding
stability; it is not uncertainty over ten independently trained target sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEXT_ROOT = ROOT / "Text_only"
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Text_only.shared_response_resampling import (  # noqa: E402
    LABEL_ONLY,
    PREFLEAK,
    PROBE_BANK,
    SHARED_SPLITS,
    UPDATES,
    EVALUATOR,
    experiment_config,
    filter_transcript,
    mapping_arguments,
    sha256,
    validate_source,
)


BUDGETS = (16, 32, 48, 64, 80, 96)
METHODS = ("prefleak", "updates_leak", "label_only")
DISPLAY_NAMES = {
    "prefleak": "PrefLeak",
    "updates_leak": "Updates-Leak-style MLP",
    "label_only": "Label-only DI",
}
COLORS = {
    "prefleak": "#0072B2",
    "updates_leak": "#D55E00",
    "label_only": "#009E73",
}


def parse_args(default_output: Path | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "infer", "evaluate", "report", "audit", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output or HERE / "outputs" / "coco_llava",
    )
    parser.add_argument("--shadow-generations", type=Path)
    parser.add_argument("--target-generations", type=Path)
    return parser.parse_args()


def budget_layout(budget: int) -> dict[str, int]:
    if budget not in BUDGETS or budget % 16:
        raise ValueError(f"Unsupported budget: {budget}")
    layout = {
        "paired_probes": budget // 8,
        "label_only_probes": budget // 4,
        "repetitions": 4,
    }
    if layout["paired_probes"] * 4 * 2 != budget:
        raise AssertionError("Paired budget arithmetic failed.")
    if layout["label_only_probes"] * 4 != budget:
        raise AssertionError("Post-only budget arithmetic failed.")
    return layout


def environment(output: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'trl'}:{ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # Matplotlib otherwise tries the read-only user configuration directory.
    env["MPLCONFIGDIR"] = str(output / ".matplotlib")
    return env


def run_logged(command: list[str], env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            handle.write("\n")
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"Command failed with status {completed.returncode}: {' '.join(command)}\n{tail}"
        )


def manifest(experiment, shadow_source: Path, target_source: Path) -> dict:
    usage = {str(index): 0 for index in range(8)}
    split_rows = []
    for run_index, indices in enumerate(SHARED_SPLITS):
        for index in indices:
            usage[str(index)] += 1
        split_rows.append({
            "run_index": run_index,
            "source_repetition_indices": list(indices),
            "selected_repetitions_per_probe": 4,
        })
    if set(usage.values()) != {5}:
        raise AssertionError(f"Response split usage is not balanced: {usage}")
    return {
        "protocol": "matched_query_budget_response_resampling_v1",
        "experiment": experiment.key,
        "dataset": experiment.dataset,
        "victim": experiment.victim,
        "query_modality": "text_only",
        "budgets": list(BUDGETS),
        "budget_definition": "target model generation calls",
        "shadow_calibration_calls_counted_in_budget": False,
        "repetitions_per_probe": 4,
        "budget_allocation": {
            str(budget): {
                "prefleak": (
                    f"{budget_layout(budget)['paired_probes']} probes * 4 responses "
                    f"* pre/post = {budget}"
                ),
                "updates_leak": (
                    f"{budget_layout(budget)['paired_probes']} probes * 4 responses "
                    f"* pre/post = {budget}"
                ),
                "label_only": (
                    f"{budget_layout(budget)['label_only_probes']} probes * 4 responses "
                    f"* post only = {budget}"
                ),
            }
            for budget in BUDGETS
        },
        "num_response_resamples": len(SHARED_SPLITS),
        "response_resamples_are_independent_target_samples": False,
        "independence_boundary": (
            "The ten overlapping 4-of-8 resamples reuse the same held-out target "
            "updates and measure response-generation stability only."
        ),
        "source_repetition_usage": usage,
        "splits": split_rows,
        "num_shadow_updates": len(experiment.shadows),
        "num_target_updates": len(experiment.targets),
        "num_target_pairs": len({spec.pair_id for spec in experiment.targets}),
        "shadow_source": str(shadow_source.resolve()),
        "shadow_source_sha256": sha256(shadow_source),
        "target_source": str(target_source.resolve()),
        "target_source_sha256": sha256(target_source),
        "probe_bank_sha256": sha256(PROBE_BANK),
        "selection_boundary": "all attacks calibrated on shadow outputs only",
        "target_private_labels_used_for_calibration_or_inference": False,
    }


def run_directory(output: Path, run_index: int) -> Path:
    return output / "runs" / f"run_{run_index:02d}"


def budget_directory(output: Path, run_index: int, budget: int) -> Path:
    return run_directory(output, run_index) / f"budget_{budget:03d}"


def calibrate_budget(experiment, output: Path, run_index: int, budget: int, env: dict[str, str]) -> None:
    layout = budget_layout(budget)
    directory = budget_directory(output, run_index, budget)
    frozen = directory / "frozen"
    shadow = run_directory(output, run_index) / "shadow" / "remote_generations.jsonl"
    labels = mapping_arguments(experiment.shadows, "--shadow-label", "preference")
    pairs = mapping_arguments(experiment.shadows, "--shadow-pair", "pair_id")
    log = directory / "commands.log"
    frozen_outputs = (
        frozen / "prefleak.json",
        frozen / "updates_leak" / "frozen_updates_leak_mlp.json",
        frozen / "updates_leak" / "frozen_updates_leak_mlp.pt",
        frozen / "label_only.json",
    )
    if all(path.is_file() for path in frozen_outputs):
        print(f"[reuse-cal] split={run_index:02d} budget={budget}", flush=True)
        return
    print(f"[calibrate] split={run_index:02d} budget={budget}", flush=True)
    run_logged([
        sys.executable, str(PREFLEAK), "calibrate",
        "--shadow-generations", str(shadow),
        "--probe-bank", str(PROBE_BANK),
        "--output", str(frozen / "prefleak.json"),
        "--paired-repetitions", "4", "--post-only-repetitions", "4",
        "--nuisance-rank", "2",
        "--k-grid", str(layout["paired_probes"]),
        "--random-repeats", "0", "--seed", "38401",
        *labels, *pairs,
    ], env, log)
    run_logged([
        sys.executable, str(UPDATES), "calibrate",
        "--shadow-generations", str(shadow),
        "--probe-bank", str(PROBE_BANK),
        "--output-dir", str(frozen / "updates_leak"),
        "--num-probes", str(layout["paired_probes"]),
        "--repetitions", "4", "--attack-runs", "1",
        "--epochs", "50", "--seed", "2020",
        *labels, *pairs,
    ], env, log)
    run_logged([
        sys.executable, str(LABEL_ONLY), "calibrate",
        "--shadow-generations", str(shadow),
        "--probe-bank", str(PROBE_BANK),
        "--output", str(frozen / "label_only.json"),
        "--num-probes", str(layout["label_only_probes"]),
        "--repetitions", "4", "--attack-runs", "1",
        "--kl-frac", "0.8", "--seed", "2020",
        *labels, *pairs,
    ], env, log)


def prepare(experiment, shadow_source: Path, target_source: Path, output: Path, env: dict[str, str]) -> None:
    # This phase deliberately validates/opens only the shadow transcript.
    validate_source(shadow_source, experiment.shadows)
    output.mkdir(parents=True, exist_ok=True)
    protocol = manifest(experiment, shadow_source, target_source)
    (output / "PROTOCOL_MANIFEST.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    for run_index, indices in enumerate(SHARED_SPLITS):
        shadow = run_directory(output, run_index) / "shadow" / "remote_generations.jsonl"
        filter_transcript(shadow_source, shadow, indices)
        for budget in BUDGETS:
            calibrate_budget(experiment, output, run_index, budget, env)
    frozen_files = sorted(output.glob("runs/run_*/budget_*/frozen/**/*.json"))
    boundary = {
        "boundary": "all_budgeted_attacks_frozen_before_target_transcript_inference",
        "target_private_labels_opened": False,
        "target_transcript_used_for_calibration": False,
        "num_frozen_json_files": len(frozen_files),
        "protocol_manifest_sha256": sha256(output / "PROTOCOL_MANIFEST.json"),
        "frozen_files_sha256": {
            str(path.relative_to(output)): sha256(path) for path in frozen_files
        },
    }
    (output / "ATTACK_FREEZE_BOUNDARY.json").write_text(
        json.dumps(boundary, indent=2) + "\n", encoding="utf-8"
    )


def infer_budget(experiment, output: Path, run_index: int, budget: int, env: dict[str, str]) -> dict[str, dict]:
    directory = budget_directory(output, run_index, budget)
    frozen = directory / "frozen"
    predictions = directory / "predictions"
    target = run_directory(output, run_index) / "target" / "remote_generations.jsonl"
    target_ids = [
        value
        for spec in experiment.targets
        for value in ("--target-id", spec.model_id)
    ]
    log = directory / "commands.log"
    prediction_paths = {
        "prefleak": predictions / "prefleak" / "paired_selected_residualized.json",
        "updates_leak": predictions / "updates_leak.json",
        "label_only": predictions / "label_only.json",
    }
    if all(path.is_file() for path in prediction_paths.values()):
        print(f"[reuse]     split={run_index:02d} budget={budget}", flush=True)
        return {
            method: json.loads(path.read_text(encoding="utf-8"))
            for method, path in prediction_paths.items()
        }
    print(f"[infer]     split={run_index:02d} budget={budget}", flush=True)
    run_logged([
        sys.executable, str(PREFLEAK), "infer",
        "--target-generations", str(target), "--probe-bank", str(PROBE_BANK),
        "--frozen-attack", str(frozen / "prefleak.json"),
        "--attack-name", "paired_selected_residualized",
        "--output-dir", str(predictions / "prefleak"), *target_ids,
    ], env, log)
    run_logged([
        sys.executable, str(UPDATES), "infer",
        "--target-generations", str(target), "--probe-bank", str(PROBE_BANK),
        "--frozen-attack", str(frozen / "updates_leak" / "frozen_updates_leak_mlp.json"),
        "--weights", str(frozen / "updates_leak" / "frozen_updates_leak_mlp.pt"),
        "--output", str(predictions / "updates_leak.json"), *target_ids,
    ], env, log)
    run_logged([
        sys.executable, str(LABEL_ONLY), "infer",
        "--target-generations", str(target), "--probe-bank", str(PROBE_BANK),
        "--frozen-attack", str(frozen / "label_only.json"),
        "--output", str(predictions / "label_only.json"), *target_ids,
    ], env, log)
    return {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in prediction_paths.items()
    }


def combine_predictions(output: Path, documents: dict[str, list[tuple[int, int, dict]]]) -> None:
    destination = output / "predictions"
    destination.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256(output / "PROTOCOL_MANIFEST.json")
    for method, entries in documents.items():
        combined = dict(entries[0][2])
        rows = []
        for budget, run_index, document in entries:
            if document.get("attacker_used_target_private_labels") is not False:
                raise ValueError(f"{method} used target labels at B={budget}, split={run_index}")
            for row in document["predictions"]:
                if row["query_budget"] != budget:
                    raise ValueError(
                        f"{method} call count mismatch at B={budget}, split={run_index}: {row}"
                    )
                copied = dict(row)
                copied["budget_repeat"] = run_index
                copied["response_resampling_run"] = run_index
                rows.append(copied)
        combined["predictions"] = rows
        combined["query_budgets"] = list(BUDGETS)
        combined["protocol_manifest_sha256"] = protocol_hash
        combined["num_response_resampling_runs_per_budget"] = len(SHARED_SPLITS)
        combined["response_resampling_is_independent_target_sampling"] = False
        (destination / f"{method}.json").write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8"
        )


def infer(experiment, target_source: Path, output: Path, env: dict[str, str]) -> None:
    if not (output / "ATTACK_FREEZE_BOUNDARY.json").is_file():
        raise FileNotFoundError("Run prepare first; the attack-freeze boundary is missing.")
    validate_source(target_source, experiment.targets)
    documents: dict[str, list[tuple[int, int, dict]]] = {method: [] for method in METHODS}
    for run_index, indices in enumerate(SHARED_SPLITS):
        target = run_directory(output, run_index) / "target" / "remote_generations.jsonl"
        filter_transcript(target_source, target, indices)
        for budget in BUDGETS:
            result = infer_budget(experiment, output, run_index, budget, env)
            for method, document in result.items():
                documents[method].append((budget, run_index, document))
    combine_predictions(output, documents)
    prediction_files = sorted((output / "predictions").glob("*.json"))
    boundary = {
        "boundary": "all_target_predictions_serialized_before_private_label_evaluation",
        "target_private_labels_opened": False,
        "prediction_files_sha256": {
            path.name: sha256(path) for path in prediction_files
        },
    }
    (output / "PREDICTION_BOUNDARY.json").write_text(
        json.dumps(boundary, indent=2) + "\n", encoding="utf-8"
    )


def evaluate(experiment, output: Path, env: dict[str, str]) -> None:
    if not (output / "PREDICTION_BOUNDARY.json").is_file():
        raise FileNotFoundError("All predictions must be serialized before evaluation.")
    analysis = output / "analysis"
    labels = mapping_arguments(experiment.targets, "--target-label", "preference")
    pairs = mapping_arguments(experiment.targets, "--target-pair", "pair_id")
    for method in METHODS:
        print(f"[evaluate]  {method}", flush=True)
        run_logged([
            sys.executable, str(EVALUATOR),
            "--predictions", str(output / "predictions" / f"{method}.json"),
            "--output", str(analysis / f"{method}.json"),
            *labels, *pairs,
        ], env, output / "evaluation.log")
    label_boundary = {
        "target_private_labels_opened": True,
        "opened_by": "Text_only/evaluate_known_preference.py",
        "opened_after_prediction_boundary": True,
        "prediction_boundary_sha256": sha256(output / "PREDICTION_BOUNDARY.json"),
        "analysis_files_sha256": {
            path.name: sha256(path) for path in sorted(analysis.glob("*.json"))
        },
    }
    (output / "EVALUATION_BOUNDARY.json").write_text(
        json.dumps(label_boundary, indent=2) + "\n", encoding="utf-8"
    )


def result_rows(output: Path) -> list[dict]:
    rows = []
    for method in METHODS:
        analysis = json.loads(
            (output / "analysis" / f"{method}.json").read_text(encoding="utf-8")
        )
        individual = defaultdict(list)
        for run in analysis["individual_runs"]:
            individual[int(run["query_budget"])].append(run)
        for summary in analysis["budget_summary"]:
            budget = int(summary["query_budget"])
            runs = individual[budget]
            rows.append({
                "method": method,
                "method_display": DISPLAY_NAMES[method],
                "access": "post-only" if method == "label_only" else "paired pre/post",
                "query_budget": budget,
                "num_response_resamples": int(summary["num_budget_subsets"]),
                "balanced_accuracy_mean": summary["balanced_accuracy_mean"],
                "balanced_accuracy_std": summary["balanced_accuracy_std"],
                "balanced_accuracy_variance": summary["balanced_accuracy_std"] ** 2,
                "auroc_mean": summary["auroc_mean"],
                "auroc_std": summary["auroc_std"],
                "auroc_variance": summary["auroc_std"] ** 2,
                "macro_f1_mean": summary["macro_f1_mean"],
                "macro_f1_std": summary["macro_f1_std"],
                "macro_f1_variance": summary["macro_f1_std"] ** 2,
                "significant_resamples_p_lt_0_05": sum(
                    run["matched_pair_randomization_pvalue_one_sided"] is not None
                    and run["matched_pair_randomization_pvalue_one_sided"] < 0.05
                    for run in runs
                ),
                "per_resample_exact_p": [
                    run["matched_pair_randomization_pvalue_one_sided"] for run in runs
                ],
            })
    return rows


def artifact_prefix(experiment) -> str:
    return {
        "coco_llava": "COCO_LLAVA",
        "coco_glm": "COCO_GLM",
        "ppsd_llava": "PPSD_LLAVA",
        "ppsd_glm": "PPSD_GLM",
        "para_llava": "PARA_LLAVA",
        "para_glm": "PARA_GLM",
    }[experiment.key]


def short_model_name(experiment) -> str:
    return {
        "coco_llava": "LLaVA",
        "coco_glm": "GLM-4.1V",
        "ppsd_llava": "LLaVA",
        "ppsd_glm": "GLM-4.1V",
        "para_llava": "LLaVA",
        "para_glm": "GLM-4.1V",
    }[experiment.key]


def short_dataset_name(experiment) -> str:
    return {
        "coco_llava": "COCO",
        "coco_glm": "COCO",
        "ppsd_llava": "PPSD",
        "ppsd_glm": "PPSD",
        "para_llava": "PARA",
        "para_glm": "PARA",
    }[experiment.key]


def write_plot(rows: list[dict], experiment, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True)
    panels = (
        ("balanced_accuracy", "Balanced accuracy"),
        ("auroc", "AUROC"),
    )
    for axis, (metric, label) in zip(axes, panels):
        for method in METHODS:
            selected = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: row["query_budget"],
            )
            x = [row["query_budget"] for row in selected]
            mean = [row[f"{metric}_mean"] for row in selected]
            std = [row[f"{metric}_std"] for row in selected]
            lower = [max(0.0, value - spread) for value, spread in zip(mean, std)]
            upper = [min(1.0, value + spread) for value, spread in zip(mean, std)]
            axis.plot(
                x, mean, marker="o", linewidth=2, markersize=4.5,
                color=COLORS[method], label=DISPLAY_NAMES[method],
            )
            axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.14)
        axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="Chance")
        axis.set_xticks(BUDGETS)
        axis.set_ylim(0.35, 1.02)
        axis.set_xlabel("Target generation-call budget")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(), loc="upper center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, 1.03),
    )
    model_name = short_model_name(experiment)
    dataset_name = short_dataset_name(experiment)
    fig.suptitle(
        f"{dataset_name} + {model_name}: text-only attack performance vs. query budget",
        y=1.10,
    )
    fig.text(
        0.5, -0.015,
        "Band: mean ± SD across 10 overlapping 4-of-8 response resamples (not independent target sets)",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout()
    prefix = artifact_prefix(experiment)
    fig.savefig(output / f"{prefix}_QUERY_BUDGET.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{prefix}_QUERY_BUDGET.pdf", bbox_inches="tight")
    plt.close(fig)


def report(experiment, output: Path) -> None:
    rows = result_rows(output)
    prefix = artifact_prefix(experiment)
    model_name = short_model_name(experiment)
    dataset_name = short_dataset_name(experiment)
    result = {
        "protocol": "matched_query_budget_response_resampling_v1",
        "dataset": experiment.dataset,
        "victim": experiment.victim,
        "budgets": list(BUDGETS),
        "budget_unit": "target generation calls",
        "num_shadow_updates": len(experiment.shadows),
        "num_target_updates": len(experiment.targets),
        "num_target_pairs": len({spec.pair_id for spec in experiment.targets}),
        "num_response_resamples_per_budget": len(SHARED_SPLITS),
        "response_resamples_are_independent_target_samples": False,
        "results": rows,
    }
    (output / f"{prefix}_QUERY_BUDGET_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    csv_fields = [
        "method", "method_display", "access", "query_budget",
        "num_response_resamples", "balanced_accuracy_mean",
        "balanced_accuracy_std", "balanced_accuracy_variance", "auroc_mean",
        "auroc_std", "auroc_variance", "macro_f1_mean", "macro_f1_std",
        "macro_f1_variance", "significant_resamples_p_lt_0_05",
    ]
    with (output / f"{prefix}_QUERY_BUDGET_RESULTS.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {dataset_name} + {model_name} query-budget evaluation",
        "",
        "All three methods use the same held-out updates and the same ten balanced "
        "4-of-8 response splits at every budget. Paired methods query both model "
        "versions; Label-only DI uses post-update outputs only. The budget is the "
        "exact number of target generation calls, excluding shadow calibration.",
        "",
        "> The ten splits overlap. Mean ± SD measures decoding/resampling stability; "
        "it must not be described as uncertainty over ten independently trained target sets.",
        "",
        "| Method | Access | Calls | Balanced accuracy | AUROC | Macro-F1 | p<.05 resamples |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: (METHODS.index(value["method"]), value["query_budget"])):
        lines.append(
            f"| {row['method_display']} | {row['access']} | {row['query_budget']} | "
            f"{row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f} | "
            f"{row['auroc_mean']:.3f} ± {row['auroc_std']:.3f} | "
            f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} | "
            f"{row['significant_resamples_p_lt_0_05']}/10 |"
        )
    lines.extend([
        "",
        "Exact matched-pair randomization p-values remain available per response "
        "resample in `analysis/*.json` and the machine-readable result JSON. Target "
        "private labels were supplied only to the evaluator after every prediction "
        "file had been serialized.",
        "",
    ])
    (output / f"{prefix}_QUERY_BUDGET_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    write_plot(rows, experiment, output)
    print(json.dumps(result, indent=2))


def audit(experiment, output: Path) -> None:
    expected_target_ids = {spec.model_id for spec in experiment.targets}
    checks = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    record("protocol_manifest_exists", (output / "PROTOCOL_MANIFEST.json").is_file(), "manifest serialized")
    record("attack_freeze_boundary_exists", (output / "ATTACK_FREEZE_BOUNDARY.json").is_file(), "freeze boundary serialized")
    record("prediction_boundary_exists", (output / "PREDICTION_BOUNDARY.json").is_file(), "prediction boundary serialized")

    for run_index, indices in enumerate(SHARED_SPLITS):
        for transcript_kind in ("shadow", "target"):
            transcript = run_directory(output, run_index) / transcript_kind / "remote_generations.jsonl"
            remapped = set()
            sources = set()
            with transcript.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    remapped.add(int(row["repetition"]))
                    sources.add(int(row["source_repetition"]))
            record(
                f"split_{run_index:02d}_{transcript_kind}_repetitions",
                remapped == set(range(4)) and sources == set(indices),
                f"remapped={sorted(remapped)}, source={sorted(sources)}",
            )
        for budget in BUDGETS:
            layout = budget_layout(budget)
            directory = budget_directory(output, run_index, budget)
            pref = json.loads((directory / "frozen" / "prefleak.json").read_text())
            primary = next(
                attack for attack in pref["attacks"]
                if attack["name"] == "paired_selected_residualized"
            )
            updates = json.loads(
                (directory / "frozen" / "updates_leak" / "frozen_updates_leak_mlp.json").read_text()
            )
            label = json.loads((directory / "frozen" / "label_only.json").read_text())
            frozen_ok = (
                primary["num_probes"] == layout["paired_probes"]
                and primary["online_generation_calls"] == budget
                and updates["num_probes"] == layout["paired_probes"]
                and updates["online_generation_calls"] == budget
                and label["num_probes"] == layout["label_only_probes"]
                and label["online_generation_calls"] == budget
            )
            record(
                f"split_{run_index:02d}_budget_{budget}_matched_calls",
                frozen_ok,
                f"paired_probes={layout['paired_probes']}, label_probes={layout['label_only_probes']}",
            )
            prediction_paths = {
                "prefleak": directory / "predictions" / "prefleak" / "paired_selected_residualized.json",
                "updates_leak": directory / "predictions" / "updates_leak.json",
                "label_only": directory / "predictions" / "label_only.json",
            }
            for method, path in prediction_paths.items():
                document = json.loads(path.read_text())
                ids = {row["anonymous_model_id"] for row in document["predictions"]}
                calls = {row["query_budget"] for row in document["predictions"]}
                record(
                    f"split_{run_index:02d}_budget_{budget}_{method}_target_blind",
                    document["attacker_used_target_private_labels"] is False
                    and ids == expected_target_ids and calls == {budget},
                    f"target_labels=False, targets={len(ids)}, calls={sorted(calls)}",
                )

    for method in METHODS:
        combined = json.loads((output / "predictions" / f"{method}.json").read_text())
        expected_rows = len(BUDGETS) * len(SHARED_SPLITS) * len(expected_target_ids)
        budget_counts = defaultdict(int)
        for row in combined["predictions"]:
            budget_counts[int(row["query_budget"])] += 1
        record(
            f"combined_{method}_complete",
            len(combined["predictions"]) == expected_rows
            and set(budget_counts) == set(BUDGETS),
            f"rows={len(combined['predictions'])}, per_budget={dict(budget_counts)}",
        )
        analysis = json.loads((output / "analysis" / f"{method}.json").read_text())
        record(
            f"analysis_{method}_complete",
            len(analysis["budget_summary"]) == len(BUDGETS)
            and len(analysis["individual_runs"]) == len(BUDGETS) * len(SHARED_SPLITS),
            f"budget_rows={len(analysis['budget_summary'])}, run_rows={len(analysis['individual_runs'])}",
        )

    document = {
        "audit": f"{experiment.key}_matched_query_budget_v1",
        "passed": all(check["passed"] for check in checks),
        "num_checks": len(checks),
        "checks": checks,
    }
    (output / "PROTOCOL_AUDIT.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": document["passed"], "num_checks": len(checks)}, indent=2))


def run_main(
    experiment_key: str = "coco_llava",
    default_output: Path | None = None,
) -> None:
    args = parse_args(default_output)
    experiment = experiment_config(experiment_key)
    output = args.output_dir.resolve()
    shadow = (args.shadow_generations or experiment.default_shadow).resolve()
    target = (args.target_generations or experiment.default_target).resolve()
    env = environment(output)
    if args.phase in {"prepare", "all"}:
        prepare(experiment, shadow, target, output, env)
    if args.phase in {"infer", "all"}:
        infer(experiment, target, output, env)
    if args.phase in {"evaluate", "all"}:
        evaluate(experiment, output, env)
    if args.phase in {"report", "all"}:
        report(experiment, output)
    if args.phase in {"audit", "all"}:
        audit(experiment, output)


def main() -> None:
    run_main()


if __name__ == "__main__":
    main()

