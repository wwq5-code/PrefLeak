#!/usr/bin/env python3
"""Run matched 96-call response-resampling evaluations for all three attacks.

Each of ten runs uses the same four response repetition indices for PrefLeak,
Updates-Leak-style MLP, and Label-only DI.  Paired attacks use 12 probes from
both pre- and post-update models (12 * 4 * 2 = 96 calls); the post-only attack
uses 24 probes (24 * 4 = 96 calls).  Shadow calibration and target inference
use the same split, while target labels are supplied only to the evaluator
after every prediction file has been serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_ROOT = ROOT / "Text_only"
# Running this file directly places ``Text_only/`` rather than the repository
# root on sys.path.  Add the root so the existing experiment modules can be
# imported exactly as they are when invoked with ``python -m``.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROBE_BANK = TEXT_ROOT / "prompts" / "remote_text_candidate_bank.json"
PREFLEAK = TEXT_ROOT / "remote_attack.py"
UPDATES = TEXT_ROOT / "Updates-Leak" / "prefleak_mlp.py"
LABEL_ONLY = TEXT_ROOT / "dissecting_dist_inf" / "prefleak_label_kl.py"
EVALUATOR = TEXT_ROOT / "evaluate_known_preference.py"

# Ten balanced 4-of-8 subsets.  Every source repetition occurs exactly five
# times, and adjacent entries are not treated as statistically independent.
SHARED_SPLITS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 2, 4, 6),
    (1, 3, 5, 7),
    (0, 1, 4, 5),
    (2, 3, 6, 7),
    (0, 3, 5, 6),
    (1, 2, 4, 7),
    (0, 2, 5, 7),
    (1, 3, 4, 6),
)


@dataclass(frozen=True)
class Experiment:
    key: str
    dataset: str
    victim: str
    shadows: list
    targets: list
    default_shadow: Path
    default_target: Path
    default_output: Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_config(key: str) -> Experiment:
    if key in {"coco_llava", "coco_glm"}:
        from Text_only.run_joint_vlm_experiment import build_specs

        shadows, targets = build_specs()
        if key == "coco_llava":
            base = TEXT_ROOT / "outputs" / "remote_probe_llava_v1" / "main"
            victim = "LLaVA-OneVision-Qwen2-0.5B"
            family = "LLaVA"
        else:
            base = TEXT_ROOT / "GLM" / "outputs" / "remote_probe_glm_v1" / "main"
            victim = "GLM-4.1V-9B-Base"
            family = "GLM"
        return Experiment(
            key, "COCO 2017 magenta/cyan", victim, shadows, targets,
            base / "probes" / "shadow_candidates" / "attacker_view" / "remote_generations.jsonl",
            base / "probes" / "target_candidates" / "attacker_view" / "remote_generations.jsonl",
            TEXT_ROOT / "Repeated_Response" / "COCO" / family,
        )
    if key == "ppsd_llava":
        from Text_only.PPSD.protocol import build_specs

        shadows, targets = build_specs()
        base = TEXT_ROOT / "PPSD" / "outputs" / "d2_style13_shadow_validation_llava7b"
        return Experiment(
            key, "PPSD D2 style-1/style-3", "LLaVA-OneVision-Qwen2-7B",
            shadows, targets,
            base / "probes" / "shadow_candidates" / "attacker_view" / "remote_generations.jsonl",
            TEXT_ROOT / "Repeated_Response" / "PPSD" / "LLaVA" / "source_target" / "attacker_view" / "remote_generations.jsonl",
            TEXT_ROOT / "Repeated_Response" / "PPSD" / "LLaVA",
        )
    if key == "ppsd_glm":
        from Text_only.GLM.PPSD.protocol import build_specs

        shadows, targets = build_specs()
        base = TEXT_ROOT / "GLM" / "PPSD" / "outputs" / "d1_style01_glm_v2"
        return Experiment(
            key, "PPSD D1 style-0/style-1", "GLM-4.1V-9B-Base",
            shadows, targets,
            base / "probes" / "shadow_candidates" / "attacker_view" / "remote_generations.jsonl",
            TEXT_ROOT / "Repeated_Response" / "PPSD" / "GLM" / "source_target" / "attacker_view" / "remote_generations.jsonl",
            TEXT_ROOT / "Repeated_Response" / "PPSD" / "GLM",
        )
    if key == "para_llava":
        from Text_only.PARA.protocol import build_specs

        shadows, targets = build_specs()
        base = TEXT_ROOT / "PARA" / "outputs" / "llava_v1"
        return Experiment(
            key, "PARA portrait/scene", "LLaVA-OneVision-Qwen2-7B",
            shadows, targets,
            base / "probes" / "shadow_candidates" / "attacker_view" / "remote_generations.jsonl",
            base / "shared_96" / "source_target" / "attacker_view" / "remote_generations.jsonl",
            base / "shared_96",
        )
    if key == "para_glm":
        from Text_only.GLM.PARA.protocol import build_specs

        shadows, targets = build_specs()
        base = TEXT_ROOT / "GLM" / "PARA" / "outputs" / "glm_v1"
        return Experiment(
            key, "PARA portrait/scene", "GLM-4.1V-9B-Base",
            shadows, targets,
            base / "probes" / "shadow_candidates" / "attacker_view" / "remote_generations.jsonl",
            base / "probes" / "target_union" / "attacker_view" / "remote_generations.jsonl",
            base / "shared_96",
        )
    raise ValueError(f"Unknown experiment: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "infer", "evaluate", "report", "all"),
        nargs="?", default="all",
    )
    parser.add_argument(
        "--experiment",
        choices=(
            "coco_llava", "coco_glm", "ppsd_llava", "ppsd_glm",
            "para_llava", "para_glm",
        ),
        required=True,
    )
    parser.add_argument("--shadow-generations", type=Path)
    parser.add_argument("--target-generations", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'trl'}:{ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def split_manifest() -> dict:
    counts = {str(index): 0 for index in range(8)}
    rows = []
    for run_index, indices in enumerate(SHARED_SPLITS):
        for index in indices:
            counts[str(index)] += 1
        rows.append({
            "run_index": run_index,
            "source_repetition_indices": list(indices),
            "repetitions_per_probe": 4,
        })
    if set(counts.values()) != {5}:
        raise AssertionError(f"Unbalanced response splits: {counts}")
    return {
        "protocol": "shared_balanced_response_resampling_v1",
        "num_runs": 10,
        "source_repetitions": 8,
        "selected_repetitions_per_run": 4,
        "source_repetition_usage": counts,
        "query_budgets": {
            "prefleak": "12 probes * 4 repetitions * 2 versions = 96",
            "updates_leak": "12 probes * 4 repetitions * 2 versions = 96",
            "label_only_di": "24 probes * 4 repetitions * 1 version = 96",
        },
        "independence_boundary": (
            "Runs reuse the same held-out update pairs and estimate response-generation "
            "stability; they are not independent target samples."
        ),
        "splits": rows,
    }


def validate_source(path: Path, specs: list, require_reference: bool = True) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_ids: set[str] = set()
    repetitions: dict[tuple[str, str], set[int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            model_id = row["model_version"]
            observed_ids.add(model_id)
            repetitions.setdefault((model_id, row["probe_id"]), set()).add(
                int(row["repetition"])
            )
    expected = {spec.model_id for spec in specs}
    if require_reference:
        expected.add("M0_reference")
    if observed_ids != expected:
        raise ValueError(
            f"Source transcript model IDs differ: expected={sorted(expected)}, "
            f"observed={sorted(observed_ids)}"
        )
    bad = [key for key, value in repetitions.items() if value != set(range(8))]
    if bad:
        raise ValueError(f"Source transcript lacks exactly repetitions 0..7: {bad[:5]}")


def filter_transcript(source: Path, destination: Path, indices: tuple[int, ...]) -> None:
    mapping = {source_index: new_index for new_index, source_index in enumerate(indices)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open(encoding="utf-8") as reader, temporary.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            source_index = int(row["repetition"])
            if source_index not in mapping:
                continue
            row["source_repetition"] = source_index
            row["repetition"] = mapping[source_index]
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    source_metadata = source.parent / "public_metadata.json"
    metadata = (
        json.loads(source_metadata.read_text(encoding="utf-8"))
        if source_metadata.is_file() else {}
    )
    metadata.update({
        "response_resampling": {
            "source_sha256": sha256(source),
            "source_repetition_indices": list(indices),
            "remapped_repetitions": list(range(len(indices))),
        },
        "repetitions": len(indices),
    })
    (destination.parent / "public_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def mapping_arguments(specs: list, option: str, attribute: str) -> list[str]:
    values: list[str] = []
    for spec in specs:
        values.extend([option, f"{spec.model_id}={getattr(spec, attribute)}"])
    return values


def calibrate_run(
    experiment: Experiment, output: Path, run_index: int, shadow: Path,
    env: dict[str, str],
) -> None:
    directory = output / "runs" / f"run_{run_index:02d}"
    frozen = directory / "frozen"
    labels = mapping_arguments(experiment.shadows, "--shadow-label", "preference")
    pairs = mapping_arguments(experiment.shadows, "--shadow-pair", "pair_id")
    run([
        sys.executable, str(PREFLEAK), "calibrate",
        "--shadow-generations", str(shadow),
        "--probe-bank", str(PROBE_BANK),
        "--output", str(frozen / "prefleak.json"),
        "--paired-repetitions", "4", "--post-only-repetitions", "4",
        "--nuisance-rank", "2", "--k-grid", "12",
        "--random-repeats", "0", "--seed", "38401",
        *labels, *pairs,
    ], env)
    run([
        sys.executable, str(UPDATES), "calibrate",
        "--shadow-generations", str(shadow),
        "--probe-bank", str(PROBE_BANK),
        "--output-dir", str(frozen / "updates_leak"),
        "--num-probes", "12", "--repetitions", "4",
        "--attack-runs", "1", "--epochs", "50", "--seed", "2020",
        *labels, *pairs,
    ], env)
    run([
        sys.executable, str(LABEL_ONLY), "calibrate",
        "--shadow-generations", str(shadow),
        "--probe-bank", str(PROBE_BANK),
        "--output", str(frozen / "label_only.json"),
        "--num-probes", "24", "--repetitions", "4",
        "--attack-runs", "1", "--kl-frac", "0.8", "--seed", "2020",
        *labels, *pairs,
    ], env)


def frozen_probe_ids(output: Path) -> set[str]:
    result: set[str] = set()
    for run_index in range(len(SHARED_SPLITS)):
        frozen = output / "runs" / f"run_{run_index:02d}" / "frozen"
        pref = json.loads((frozen / "prefleak.json").read_text(encoding="utf-8"))
        primary = next(
            row for row in pref["attacks"]
            if row["name"] == "paired_selected_residualized"
        )
        result.update(primary["probe_ids"])
        updates = json.loads(
            (frozen / "updates_leak" / "frozen_updates_leak_mlp.json").read_text(
                encoding="utf-8"
            )
        )
        result.update(updates["probe_ids"])
        label = json.loads((frozen / "label_only.json").read_text(encoding="utf-8"))
        result.update(label["probe_ids"])
    return result


def write_union_bank(output: Path) -> Path:
    rows = json.loads(PROBE_BANK.read_text(encoding="utf-8"))
    required = frozen_probe_ids(output)
    selected = [row for row in rows if row["probe_id"] in required]
    if {row["probe_id"] for row in selected} != required:
        raise ValueError("A frozen probe is absent from the public bank.")
    path = output / "target_union_probe_bank.json"
    path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    return path


def prepare(experiment: Experiment, shadow_source: Path, output: Path, env: dict[str, str]) -> None:
    validate_source(shadow_source, experiment.shadows)
    output.mkdir(parents=True, exist_ok=True)
    manifest = split_manifest()
    manifest.update({
        "experiment": experiment.key,
        "dataset": experiment.dataset,
        "victim": experiment.victim,
        "num_shadow_updates": len(experiment.shadows),
        "num_target_updates": len(experiment.targets),
        "shadow_source": str(shadow_source.resolve()),
        "shadow_source_sha256": sha256(shadow_source),
        "target_labels_opened": False,
    })
    (output / "shared_resampling_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for run_index, indices in enumerate(SHARED_SPLITS):
        shadow = output / "runs" / f"run_{run_index:02d}" / "shadow" / "remote_generations.jsonl"
        filter_transcript(shadow_source, shadow, indices)
        calibrate_run(experiment, output, run_index, shadow, env)
    union = write_union_bank(output)
    boundary = {
        "boundary": "all_resampled_attacks_frozen_before_new_target_collection",
        "target_labels_opened": False,
        "shared_manifest_sha256": sha256(output / "shared_resampling_manifest.json"),
        "target_union_probe_bank_sha256": sha256(union),
        "num_union_probes": len(json.loads(union.read_text(encoding="utf-8"))),
    }
    (output / "target_query_boundary.json").write_text(
        json.dumps(boundary, indent=2) + "\n", encoding="utf-8"
    )


def infer_run(
    experiment: Experiment, output: Path, run_index: int, target: Path,
    env: dict[str, str],
) -> dict[str, dict]:
    directory = output / "runs" / f"run_{run_index:02d}"
    frozen = directory / "frozen"
    predictions = directory / "predictions"
    target_ids = [value for spec in experiment.targets for value in ("--target-id", spec.model_id)]
    run([
        sys.executable, str(PREFLEAK), "infer",
        "--target-generations", str(target), "--probe-bank", str(PROBE_BANK),
        "--frozen-attack", str(frozen / "prefleak.json"),
        "--attack-name", "paired_selected_residualized",
        "--output-dir", str(predictions / "prefleak"), *target_ids,
    ], env)
    run([
        sys.executable, str(UPDATES), "infer",
        "--target-generations", str(target), "--probe-bank", str(PROBE_BANK),
        "--frozen-attack", str(frozen / "updates_leak" / "frozen_updates_leak_mlp.json"),
        "--weights", str(frozen / "updates_leak" / "frozen_updates_leak_mlp.pt"),
        "--output", str(predictions / "updates_leak.json"), *target_ids,
    ], env)
    run([
        sys.executable, str(LABEL_ONLY), "infer",
        "--target-generations", str(target), "--probe-bank", str(PROBE_BANK),
        "--frozen-attack", str(frozen / "label_only.json"),
        "--output", str(predictions / "label_only.json"), *target_ids,
    ], env)
    return {
        "prefleak": json.loads(
            (predictions / "prefleak" / "paired_selected_residualized.json").read_text(
                encoding="utf-8"
            )
        ),
        "updates_leak": json.loads((predictions / "updates_leak.json").read_text(encoding="utf-8")),
        "label_only": json.loads((predictions / "label_only.json").read_text(encoding="utf-8")),
    }


def combine_predictions(output: Path, documents: dict[str, list[dict]]) -> None:
    destination = output / "predictions"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(output / "shared_resampling_manifest.json")
    for method, per_run in documents.items():
        combined = dict(per_run[0])
        combined_rows = []
        for run_index, document in enumerate(per_run):
            if document["attacker_used_target_private_labels"] is not False:
                raise ValueError(f"{method} used target labels in run {run_index}")
            for row in document["predictions"]:
                if row["query_budget"] != 96:
                    raise ValueError(f"{method} did not use 96 calls: {row}")
                copied = dict(row)
                copied["budget_repeat"] = run_index
                copied["response_resampling_run"] = run_index
                combined_rows.append(copied)
        combined["predictions"] = combined_rows
        combined["shared_response_resampling_manifest_sha256"] = manifest_hash
        combined["num_response_resampling_runs"] = len(per_run)
        combined["response_resampling_is_independent_target_sampling"] = False
        (destination / f"{method}.json").write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8"
        )


def infer(experiment: Experiment, target_source: Path, output: Path, env: dict[str, str]) -> None:
    validate_source(target_source, experiment.targets)
    required = frozen_probe_ids(output)
    observed: set[str] = set()
    with target_source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                observed.add(json.loads(line)["probe_id"])
    if not required <= observed:
        raise ValueError(f"Target source lacks frozen probes: {sorted(required - observed)}")
    documents: dict[str, list[dict]] = {
        "prefleak": [], "updates_leak": [], "label_only": [],
    }
    for run_index, indices in enumerate(SHARED_SPLITS):
        target = output / "runs" / f"run_{run_index:02d}" / "target" / "remote_generations.jsonl"
        filter_transcript(target_source, target, indices)
        result = infer_run(experiment, output, run_index, target, env)
        for method, document in result.items():
            documents[method].append(document)
    combine_predictions(output, documents)


def evaluate_one(
    experiment: Experiment, prediction: Path, destination: Path, env: dict[str, str]
) -> None:
    run([
        sys.executable, str(EVALUATOR),
        "--predictions", str(prediction), "--output", str(destination),
        *mapping_arguments(experiment.targets, "--target-label", "preference"),
        *mapping_arguments(experiment.targets, "--target-pair", "pair_id"),
    ], env)


def evaluate(experiment: Experiment, output: Path, env: dict[str, str]) -> None:
    analysis = output / "analysis"
    evaluate_one(experiment, output / "predictions" / "prefleak.json", analysis / "prefleak.json", env)
    evaluate_one(experiment, output / "predictions" / "updates_leak.json", analysis / "updates_leak.json", env)
    evaluate_one(experiment, output / "predictions" / "label_only.json", analysis / "label_only.json", env)


def metric_summary(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    row = document["budget_summary"][0]
    runs = document["individual_runs"]
    balanced_accuracy_std = row["balanced_accuracy_std"]
    auroc_std = row["auroc_std"]
    macro_f1_std = row["macro_f1_std"]
    return {
        "calls": row["query_budget"],
        "balanced_accuracy_mean": row["balanced_accuracy_mean"],
        "balanced_accuracy_std": balanced_accuracy_std,
        "balanced_accuracy_variance": balanced_accuracy_std ** 2,
        "auroc_mean": row["auroc_mean"],
        "auroc_std": auroc_std,
        "auroc_variance": auroc_std ** 2,
        "macro_f1_mean": row["macro_f1_mean"],
        "macro_f1_std": macro_f1_std,
        "macro_f1_variance": macro_f1_std ** 2,
        "significant_runs": sum(
            value["matched_pair_randomization_pvalue_one_sided"] is not None
            and value["matched_pair_randomization_pvalue_one_sided"] < 0.05
            for value in runs
        ),
        "num_runs": len(runs),
        "per_run_exact_p": [
            value["matched_pair_randomization_pvalue_one_sided"] for value in runs
        ],
    }


def report(experiment: Experiment, output: Path) -> None:
    metrics = {
        "prefleak": metric_summary(output / "analysis" / "prefleak.json"),
        "updates_leak": metric_summary(output / "analysis" / "updates_leak.json"),
        "label_only_di": metric_summary(output / "analysis" / "label_only.json"),
    }
    result = {
        "protocol": "shared_balanced_response_resampling_v1",
        "dataset": experiment.dataset,
        "victim": experiment.victim,
        "num_shadow_updates": len(experiment.shadows),
        "num_target_updates": len(experiment.targets),
        "num_target_pairs": len({spec.pair_id for spec in experiment.targets}),
        "query_budget_each_method": 96,
        "num_response_resampling_runs": 10,
        "resampling_runs_are_independent_target_samples": False,
        **metrics,
    }
    (output / "RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    names = {
        "prefleak": "PrefLeak",
        "updates_leak": "Updates-Leak-style MLP",
        "label_only_di": "Label-only DI",
    }
    lines = [
        f"# {experiment.dataset} + {experiment.victim}: shared response resampling",
        "",
        "All methods use the same ten balanced 4-of-8 response splits, the same held-out target updates, and 96 target calls per run.",
        "These runs measure response-generation stability and are not ten independent target samples.",
        "",
        "| Method | Access | Calls | Balanced accuracy | AUROC | Macro-F1 | Runs with exact p<.05 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("prefleak", "updates_leak", "label_only_di"):
        value = metrics[key]
        access = "post-only" if key == "label_only_di" else "paired pre/post"
        lines.append(
            f"| {names[key]} | {access} | 96 | "
            f"{value['balanced_accuracy_mean']:.3f} ± {value['balanced_accuracy_std']:.3f} | "
            f"{value['auroc_mean']:.3f} ± {value['auroc_std']:.3f} | "
            f"{value['macro_f1_mean']:.3f} ± {value['macro_f1_std']:.3f} | "
            f"{value['significant_runs']}/{value['num_runs']} |"
        )
    lines.extend([
        "",
        "The exact paired p-values are reported per resampling run in `RESULTS.json`; they are not pooled across runs.",
        "Target labels were opened only by the evaluator after all prediction files were serialized.",
        "",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    experiment = experiment_config(args.experiment)
    output = args.output_dir or experiment.default_output
    shadow = args.shadow_generations or experiment.default_shadow
    target = args.target_generations or experiment.default_target
    env = environment()
    if args.phase in {"prepare", "all"}:
        prepare(experiment, shadow, output, env)
    if args.phase in {"infer", "all"}:
        infer(experiment, target, output, env)
    if args.phase in {"evaluate", "all"}:
        evaluate(experiment, output, env)
    if args.phase in {"report", "all"}:
        report(experiment, output)


if __name__ == "__main__":
    main()
