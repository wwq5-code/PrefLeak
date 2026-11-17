#!/usr/bin/env python3
"""Run PPSD joint-VLM updates and the three text-only leakage attacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from Text_only.PPSD.protocol import (
    DATA_ROOT,
    PREFERENCES,
    PROJECT_ROOT,
    REFERENCE_PREFERENCE,
    UpdateSpec,
    build_specs,
)


PPSD_DIR = Path(__file__).resolve().parent
TEXT_ROOT = PROJECT_ROOT / "Text_only"
GLM_ROOT = TEXT_ROOT / "GLM"
PROBE_BANK = TEXT_ROOT / "prompts" / "remote_text_candidate_bank.json"
UPDATES_ATTACK = TEXT_ROOT / "Updates-Leak" / "prefleak_mlp.py"
LABEL_ATTACK = TEXT_ROOT / "dissecting_dist_inf" / "prefleak_label_kl.py"


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    model: Path
    train_script: Path
    fidelity_script: Path
    remote_script: Path
    source_config: Path
    default_output: Path
    report: Path
    default_jobs: int
    fidelity_max_tokens: int
    fidelity_min_improvement: float
    fidelity_min_parsed: float


def family_config(key: str) -> Family:
    if key == "llava":
        from Text_only.train_llava_dpo import DEFAULT_MODEL

        return Family(
            key="llava",
            label="LLaVA-OneVision-Qwen2-0.5B",
            model=DEFAULT_MODEL,
            train_script=TEXT_ROOT / "train_llava_dpo.py",
            fidelity_script=TEXT_ROOT / "probe_forced_choice.py",
            remote_script=TEXT_ROOT / "probe_remote_text.py",
            source_config=TEXT_ROOT / "outputs" / "joint_vlm_paper_v1" / "main" / "frozen_config.json",
            default_output=PPSD_DIR / "outputs" / "main",
            report=PPSD_DIR / "PPSD_RESULTS.md",
            default_jobs=4,
            fidelity_max_tokens=12,
            # D2 has 21 disjoint fidelity scenes (4.76-point resolution).
            # A 15-point floor plus 75% post accuracy avoids an arbitrary
            # five-scene jump while still rejecting unchanged updates.
            fidelity_min_improvement=0.15,
            fidelity_min_parsed=0.95,
        )
    if key == "glm":
        from Text_only.GLM.train_glm_dpo import DEFAULT_MODEL

        directory = GLM_ROOT / "PPSD"
        return Family(
            key="glm",
            label="GLM-4.1V-9B-Base",
            model=DEFAULT_MODEL,
            train_script=GLM_ROOT / "train_glm_dpo.py",
            fidelity_script=GLM_ROOT / "probe_glm.py",
            remote_script=GLM_ROOT / "probe_remote_glm.py",
            source_config=GLM_ROOT / "outputs" / "main_v1" / "frozen_config.json",
            default_output=directory / "outputs" / "main",
            report=directory / "PPSD_RESULTS.md",
            default_jobs=3,
            fidelity_max_tokens=1,
            fidelity_min_improvement=0.15,
            fidelity_min_parsed=0.90,
        )
    raise ValueError(f"Unknown family: {key}")


def parse_args(default_family: str = "llava") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare", "train", "train-shadow", "train-target", "audit",
            "fidelity", "fidelity-shadow", "fidelity-target", "shadow",
            "calibrate", "target", "infer", "evaluate", "report", "all",
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--family", choices=("llava", "glm"), default=default_family)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int)
    parser.add_argument(
        "--frozen-config-source",
        type=Path,
        help="Optional shadow-only selected update configuration used by prepare.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'trl'}:{PROJECT_ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run(
    command: list[str], env: dict[str, str], check: bool = True
) -> subprocess.CompletedProcess:
    print("\n$ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=check)


def adapter_path(output: Path, spec: UpdateSpec) -> Path:
    return output / "adapters" / spec.split / spec.model_id


def adapter_arguments(output: Path, specs: list[UpdateSpec]) -> list[str]:
    result = []
    for spec in specs:
        result.extend(["--adapter", f"{spec.model_id}={adapter_path(output, spec)}"])
    return result


def label_arguments(specs: list[UpdateSpec], name: str) -> list[str]:
    result = []
    for spec in specs:
        result.extend([name, f"{spec.model_id}={spec.preference}"])
    return result


def pair_arguments(specs: list[UpdateSpec], name: str) -> list[str]:
    result = []
    for spec in specs:
        result.extend([name, f"{spec.model_id}={spec.pair_id}"])
    return result


def prepare(args: argparse.Namespace, family: Family, env: dict[str, str]) -> dict:
    metadata = DATA_ROOT / "private_dataset_metadata.json"
    if not metadata.is_file():
        run([sys.executable, str(PPSD_DIR / "build_ppsd_preferences.py")], env)
    source_path = (args.frozen_config_source or family.source_config).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    selection_boundary = str(source.get("selection_boundary", ""))
    if "shadow" not in selection_boundary.lower():
        raise ValueError(
            f"Frozen configuration was not selected at a shadow-only boundary: {source_path}"
        )
    frozen = {
        **source,
        "ppsd_transfer": {
            "source_frozen_config": str(source_path),
            "source_sha256": sha256(source_path),
            "rule": (
                "PPSD_shadow_only_selection_without_PPSD_target_tuning"
                if args.frozen_config_source
                else "transfer_COCO_frozen_update_hyperparameters_without_PPSD_target_tuning"
            ),
            "ppsd_target_data_opened": False,
        },
    }
    destination = args.output_dir / "frozen_config.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != frozen:
            raise RuntimeError(f"Existing transferred configuration changed: {destination}")
    else:
        destination.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


def read_config(output: Path) -> dict:
    path = output / "frozen_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare first: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    transfer = config.get("ppsd_transfer", {})
    if transfer.get("ppsd_target_data_opened") is not False:
        raise ValueError("PPSD transfer boundary is not target-blind.")
    return config


def train_one(
    args: argparse.Namespace,
    family: Family,
    env: dict[str, str],
    config: dict,
    spec: UpdateSpec,
) -> str:
    destination = adapter_path(args.output_dir, spec)
    if (destination / "adapter_config.json").is_file():
        return f"reuse {spec.model_id}"
    log = args.output_dir / "logs" / spec.split / f"{spec.model_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    seed_base = 27400 if spec.split == "shadow" else 28400
    command = [
        sys.executable,
        str(family.train_script),
        "--model", str(args.model),
        "--train-manifest", str(spec.manifest),
        "--target-id", spec.model_id,
        "--output-dir", str(destination),
        "--seed", str(seed_base + (spec.serial + 1) // 2),
        "--max-steps", str(config["max_steps"]),
        "--per-device-train-batch-size", str(config["per_device_train_batch_size"]),
        "--gradient-accumulation-steps", str(config["gradient_accumulation_steps"]),
        "--learning-rate", str(config["learning_rate"]),
        "--beta", str(config["beta"]),
        "--label-smoothing", str(config["label_smoothing"]),
        "--lora-rank", str(config["lora_rank"]),
        "--lora-alpha", str(config["lora_alpha"]),
        "--lora-dropout", str(config["lora_dropout"]),
    ]
    for key, option in (
        ("vision_lora_rank", "--vision-lora-rank"),
        ("vision_lora_alpha", "--vision-lora-alpha"),
        ("projector_lora_rank", "--projector-lora-rank"),
        ("projector_lora_alpha", "--projector-lora-alpha"),
        ("language_lora_rank", "--language-lora-rank"),
        ("language_lora_alpha", "--language-lora-alpha"),
    ):
        if key in config:
            command.extend([option, str(config[key])])
    if family.key == "glm" and "max_image_pixels" in config:
        command.extend(["--max-image-pixels", str(config["max_image_pixels"])])
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return f"complete {spec.model_id}"


def train_all(
    args: argparse.Namespace,
    family: Family,
    env: dict[str, str],
    specs: list[UpdateSpec],
) -> None:
    config = read_config(args.output_dir)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(train_one, args, family, env, config, spec): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                log = args.output_dir / "logs" / spec.split / f"{spec.model_id}.log"
                raise RuntimeError(f"Training failed for {spec.model_id}; inspect {log}") from error


def audit(args: argparse.Namespace, specs: list[UpdateSpec]) -> dict:
    config = read_config(args.output_dir)
    keys = [
        "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "beta", "label_smoothing", "lora_rank", "lora_alpha",
        "lora_dropout",
    ]
    keys.extend(
        key
        for key in (
            "vision_lora_rank", "vision_lora_alpha",
            "projector_lora_rank", "projector_lora_alpha",
            "language_lora_rank", "language_lora_alpha",
        )
        if key in config
    )
    rows = []
    pair_seeds: dict[str, set[int]] = {}
    for spec in specs:
        path = adapter_path(args.output_dir, spec) / "private_training_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        pair_seeds.setdefault(spec.pair_id, set()).add(metadata["seed"])
        counts = metadata["matched_module_counts"]
        checks = {
            "manifest": Path(metadata["train_manifest"]) == spec.manifest.resolve(),
            "pure_dpo": metadata["objective"] == "dpo"
            and metadata["loss_type"] == ["sigmoid"]
            and metadata["loss_weights"] is None,
            "joint_vlm": metadata["update_scope"] == "joint_vlm"
            and set(counts) == {"vision", "projector", "language"}
            and all(value > 0 for value in counts.values()),
            "frozen_hyperparameters": all(metadata[key] == config[key] for key in keys),
        }
        rows.append({"model_id": spec.model_id, **checks, "passed": all(checks.values())})
    same_pair_seed = all(len(values) == 1 for values in pair_seeds.values())
    result = {
        "audit": "ppsd_joint_vlm_pure_dpo_transferred_configuration",
        "num_updates": len(rows),
        "num_passed": sum(row["passed"] for row in rows),
        "same_seed_within_each_counterfactual_pair": same_pair_seed,
        "all_passed": all(row["passed"] for row in rows) and same_pair_seed,
        "per_update": rows,
    }
    path = args.output_dir / "audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_passed"]:
        raise RuntimeError(f"Adapter audit failed: {path}")
    return result


def probe_fidelity(
    args: argparse.Namespace,
    family: Family,
    env: dict[str, str],
    split: str,
    specs: list[UpdateSpec],
) -> None:
    directory = args.output_dir / "probes" / f"visual_fidelity_{split}"
    generations = directory / "attacker_view" / "forced_choice_generations.jsonl"
    expected_versions = {"M0_reference", *(spec.model_id for spec in specs)}
    observed_versions = set()
    if generations.is_file():
        with generations.open(encoding="utf-8") as handle:
            observed_versions = {
                json.loads(line)["model_version"] for line in handle if line.strip()
            }
    if observed_versions != expected_versions:
        if generations.is_file():
            print(
                f"regenerate incomplete fidelity transcript: observed "
                f"{len(observed_versions)} versions, expected {len(expected_versions)}",
                flush=True,
            )
        split_bank = DATA_ROOT / f"private_fidelity_{split}_probe_bank.json"
        probe_bank = (
            split_bank
            if split_bank.is_file()
            else DATA_ROOT / "private_fidelity_probe_bank.json"
        )
        command = [
            sys.executable,
            str(family.fidelity_script),
            "--model", str(args.model),
            "--modality", "vision_conditioned",
            "--probe-bank", str(probe_bank),
            "--output-dir", str(directory),
            "--max-new-tokens", str(family.fidelity_max_tokens),
            *adapter_arguments(args.output_dir, specs),
        ]
        if family.key == "llava":
            command.extend(["--visual-prompt-style", "training"])
        if family.key == "glm":
            command.append("--constrained-choice")
            config_path = args.output_dir / "frozen_config.json"
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if "max_image_pixels" in config:
                    command.extend(
                        ["--max-image-pixels", str(config["max_image_pixels"])]
                    )
        run(command, env)
    output = args.output_dir / "analysis" / f"visual_fidelity_{split}.json"
    command = [
        sys.executable,
        "-m", "Text_only.evaluate_update_fidelity",
        "--generations", str(generations),
        "--output", str(output),
        "--reference-preference", REFERENCE_PREFERENCE,
        "--min-improvement", str(family.fidelity_min_improvement),
        "--min-parsed-rate", str(family.fidelity_min_parsed),
        *label_arguments(specs, "--target-label"),
    ]
    run(command, env, check=False)
    if not output.is_file():
        raise RuntimeError(f"Fidelity evaluator did not write {output}")


def remote_generation_path(output: Path, split: str) -> Path:
    return output / "probes" / split / "attacker_view" / "remote_generations.jsonl"


def fidelity_valid_complete_pairs(
    output: Path,
    split: str,
    specs: list[UpdateSpec],
    *,
    minimum_pairs: int,
) -> list[UpdateSpec]:
    """Keep only counterfactual pairs for which both private updates are valid."""
    path = output / "analysis" / f"visual_fidelity_{split}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    passed = {
        row["anonymous_model_id"] for row in result["per_model"] if row["passed"]
    }
    grouped: dict[str, list[UpdateSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.pair_id, []).append(spec)
    complete = [
        spec
        for pair in grouped.values()
        if len(pair) == 2 and all(spec.model_id in passed for spec in pair)
        for spec in pair
    ]
    num_pairs = len(complete) // 2
    if num_pairs < minimum_pairs:
        raise RuntimeError(
            f"Only {num_pairs} complete {split} fidelity-valid pairs; "
            f"need at least {minimum_pairs}. Inspect {path}."
        )
    return complete


def probe_remote(
    args: argparse.Namespace,
    family: Family,
    env: dict[str, str],
    split: str,
    specs: list[UpdateSpec],
    bank: Path,
) -> Path:
    output = remote_generation_path(args.output_dir, split)
    if output.is_file():
        print(f"reuse {output}", flush=True)
        return output
    command = [
        sys.executable,
        str(family.remote_script),
        "--model", str(args.model),
        "--probe-bank", str(bank),
        "--output-dir", str(args.output_dir / "probes" / split),
        "--repetitions", "8",
        "--max-new-tokens", "12",
        "--temperature", "0.8",
        "--top-p", "0.95",
        *adapter_arguments(args.output_dir, specs),
    ]
    if bank != PROBE_BANK:
        command.extend(["--seed-reference-bank", str(PROBE_BANK)])
    if family.key == "glm":
        command.extend(["--batch-size", "8"])
    run(command, env)
    return output


def calibrate_attacks(
    args: argparse.Namespace,
    env: dict[str, str],
    shadows: list[UpdateSpec],
) -> None:
    # Calibrate the primary, unconditional estimand on every sampled shadow
    # update. Filtering shadow examples by a label-dependent private fidelity
    # outcome would both shrink the attack-training distribution and introduce
    # selection bias. Fidelity remains a separately reported victim diagnostic.
    shadow_generations = remote_generation_path(args.output_dir, "shadow_candidates")
    frozen_dir = args.output_dir / "frozen_attacks"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    prefleak = frozen_dir / "prefleak.json"
    updates_frozen = frozen_dir / "updates_leak" / "frozen_updates_leak_mlp.json"
    updates_weights = frozen_dir / "updates_leak" / "frozen_updates_leak_mlp.pt"
    label_path = frozen_dir / "label_only.json"
    complete = all(
        path.is_file()
        for path in (prefleak, updates_frozen, updates_weights, label_path)
    )
    if complete:
        print(f"reuse three frozen attacks in {frozen_dir}", flush=True)
        materialize_target_bank(args.output_dir)
        return
    if remote_generation_path(args.output_dir, "target_union").exists():
        raise RuntimeError(
            "Cannot recalibrate an incomplete attack set after target queries exist."
        )
    command = [
        sys.executable, str(TEXT_ROOT / "remote_attack.py"), "calibrate",
        "--shadow-generations", str(shadow_generations),
        "--probe-bank", str(PROBE_BANK),
        "--output", str(prefleak),
        "--paired-repetitions", "4",
        "--post-only-repetitions", "8",
        "--nuisance-rank", "2",
        "--k-grid", "4,8,12,16,24",
        "--random-repeats", "10",
        *label_arguments(shadows, "--shadow-label"),
        *pair_arguments(shadows, "--shadow-pair"),
    ]
    run(command, env)
    selected_k = json.loads(prefleak.read_text(encoding="utf-8"))["selection"][
        "chosen_num_probes"
    ]

    updates_dir = frozen_dir / "updates_leak"
    run(
        [
            sys.executable, str(UPDATES_ATTACK), "calibrate",
            "--shadow-generations", str(shadow_generations),
            "--probe-bank", str(PROBE_BANK),
            "--output-dir", str(updates_dir),
            "--num-probes", str(selected_k),
            "--repetitions", "4",
            "--attack-runs", "10",
            "--epochs", "50",
            "--seed", "2020",
            *label_arguments(shadows, "--shadow-label"),
            *pair_arguments(shadows, "--shadow-pair"),
        ],
        env,
    )
    run(
        [
            sys.executable, str(LABEL_ATTACK), "calibrate",
            "--shadow-generations", str(shadow_generations),
            "--probe-bank", str(PROBE_BANK),
            "--output", str(label_path),
            "--num-probes", str(selected_k),
            "--repetitions", "8",
            "--attack-runs", "10",
            "--kl-frac", "0.8",
            "--seed", "2020",
            *label_arguments(shadows, "--shadow-label"),
            *pair_arguments(shadows, "--shadow-pair"),
        ],
        env,
    )
    materialize_target_bank(args.output_dir)


def materialize_target_bank(output: Path) -> Path:
    frozen = output / "frozen_attacks"
    prefleak = json.loads((frozen / "prefleak.json").read_text(encoding="utf-8"))
    updates = json.loads(
        (frozen / "updates_leak" / "frozen_updates_leak_mlp.json").read_text(
            encoding="utf-8"
        )
    )
    label = json.loads((frozen / "label_only.json").read_text(encoding="utf-8"))
    selected = set(prefleak["selection"]["selected_probe_ids"])
    selected.update(updates["probe_ids"])
    selected.update(label["probe_ids"])
    full_bank = json.loads(PROBE_BANK.read_text(encoding="utf-8"))
    rows = [row for row in full_bank if row["probe_id"] in selected]
    if len(rows) != len(selected):
        raise ValueError("A frozen attack references a missing target probe.")
    path = output / "target_probe_bank.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    boundary = {
        "boundary": "all_three_attacks_frozen_before_target_queries",
        "target_labels_opened": False,
        "full_probe_bank_sha256": sha256(PROBE_BANK),
        "target_probe_bank_sha256": sha256(path),
        "num_union_probes": len(rows),
        "frozen_attack_sha256": {
            "prefleak": sha256(frozen / "prefleak.json"),
            "updates_leak": sha256(frozen / "updates_leak" / "frozen_updates_leak_mlp.json"),
            "label_only": sha256(frozen / "label_only.json"),
        },
        "target_generations_existed": remote_generation_path(output, "target_union").exists(),
    }
    boundary_path = output / "target_query_boundary.json"
    if boundary["target_generations_existed"] and not boundary_path.is_file():
        raise RuntimeError("Target outputs predate the frozen three-attack boundary.")
    if not boundary_path.is_file():
        boundary_path.write_text(json.dumps(boundary, indent=2) + "\n", encoding="utf-8")
    return path


def infer_attacks(
    args: argparse.Namespace,
    env: dict[str, str],
    targets: list[UpdateSpec],
) -> None:
    generations = remote_generation_path(args.output_dir, "target_union")
    frozen = args.output_dir / "frozen_attacks"
    target_ids = [value for spec in targets for value in ("--target-id", spec.model_id)]
    prediction_root = args.output_dir / "predictions"
    run(
        [
            sys.executable, str(TEXT_ROOT / "remote_attack.py"), "infer",
            "--target-generations", str(generations),
            "--probe-bank", str(PROBE_BANK),
            "--frozen-attack", str(frozen / "prefleak.json"),
            "--attack-name", "paired_selected_residualized",
            "--output-dir", str(prediction_root / "prefleak"),
            *target_ids,
        ],
        env,
    )
    run(
        [
            sys.executable, str(UPDATES_ATTACK), "infer",
            "--target-generations", str(generations),
            "--probe-bank", str(PROBE_BANK),
            "--frozen-attack", str(frozen / "updates_leak" / "frozen_updates_leak_mlp.json"),
            "--weights", str(frozen / "updates_leak" / "frozen_updates_leak_mlp.pt"),
            "--output", str(prediction_root / "updates_leak.json"),
            *target_ids,
        ],
        env,
    )
    run(
        [
            sys.executable, str(LABEL_ATTACK), "infer",
            "--target-generations", str(generations),
            "--probe-bank", str(PROBE_BANK),
            "--frozen-attack", str(frozen / "label_only.json"),
            "--output", str(prediction_root / "label_only.json"),
            *target_ids,
        ],
        env,
    )


def evaluate_one(
    args: argparse.Namespace,
    env: dict[str, str],
    prediction: Path,
    output: Path,
    targets: list[UpdateSpec],
    valid_ids: set[str],
) -> None:
    command = [
        sys.executable,
        "-m", "Text_only.evaluate_known_preference",
        "--predictions", str(prediction),
        "--output", str(output),
        *label_arguments(targets, "--target-label"),
        *pair_arguments(targets, "--target-pair"),
    ]
    for model_id in sorted(valid_ids):
        command.extend(["--valid-target-id", model_id])
    run(command, env)


def evaluate_all(
    args: argparse.Namespace,
    env: dict[str, str],
    targets: list[UpdateSpec],
) -> None:
    valid_targets = fidelity_valid_complete_pairs(
        args.output_dir, "target", targets, minimum_pairs=0
    )
    valid_ids = {spec.model_id for spec in valid_targets}
    prediction_root = args.output_dir / "predictions"
    analysis = args.output_dir / "analysis"
    evaluate_one(
        args,
        env,
        prediction_root / "prefleak" / "paired_selected_residualized.json",
        analysis / "prefleak.json",
        targets,
        valid_ids,
    )
    evaluate_one(
        args, env, prediction_root / "updates_leak.json",
        analysis / "updates_leak.json", targets, valid_ids,
    )
    evaluate_one(
        args, env, prediction_root / "label_only.json",
        analysis / "label_only.json", targets, valid_ids,
    )


def first_run(metrics: dict) -> dict:
    return metrics["individual_runs"][0]


def summarize_repeated(metrics: dict) -> dict:
    row = metrics["budget_summary"][0]
    runs = metrics["individual_runs"]
    return {
        "calls": row["query_budget"],
        "balanced_accuracy_mean": row["balanced_accuracy_mean"],
        "balanced_accuracy_std": row["balanced_accuracy_std"],
        "auroc_mean": row["auroc_mean"],
        "auroc_std": row["auroc_std"],
        "significant_runs": sum(
            value.get("matched_pair_randomization_pvalue_one_sided") is not None
            and value["matched_pair_randomization_pvalue_one_sided"] < 0.05
            for value in runs
        ),
        "num_runs": len(runs),
    }


def write_report(args: argparse.Namespace, family: Family) -> Path:
    metadata = json.loads(
        (DATA_ROOT / "private_dataset_metadata.json").read_text(encoding="utf-8")
    )
    audit_result = json.loads((args.output_dir / "audit.json").read_text(encoding="utf-8"))
    shadow_fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_shadow.json").read_text(encoding="utf-8")
    )
    target_fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_target.json").read_text(encoding="utf-8")
    )
    pref_metrics = json.loads(
        (args.output_dir / "analysis" / "prefleak.json").read_text(encoding="utf-8")
    )
    updates_metrics = json.loads(
        (args.output_dir / "analysis" / "updates_leak.json").read_text(encoding="utf-8")
    )
    label_metrics = json.loads(
        (args.output_dir / "analysis" / "label_only.json").read_text(encoding="utf-8")
    )
    frozen_pref = json.loads(
        (args.output_dir / "frozen_attacks" / "prefleak.json").read_text(encoding="utf-8")
    )
    label_predictions = json.loads(
        (args.output_dir / "predictions" / "label_only.json").read_text(encoding="utf-8")
    )
    pref = first_run(pref_metrics)
    updates = summarize_repeated(updates_metrics)
    label = summarize_repeated(label_metrics)
    pref_supported = (
        pref["balanced_accuracy"] > 0.5
        and pref["matched_pair_randomization_pvalue_one_sided"] is not None
        and pref["matched_pair_randomization_pvalue_one_sided"] < 0.05
    )
    style_ids = metadata["style_ids"]
    style_labels = metadata["candidate_preferences"]
    result = {
        "dataset": f"PPSD {metadata['collection']} style-{style_ids[0]}/style-{style_ids[1]}",
        "model": family.label,
        "num_shadow_updates": metadata["num_shadow_updates"],
        "num_target_updates": metadata["num_target_updates"],
        "prefleak": {
            "calls": pref["query_budget"],
            "balanced_accuracy": pref["balanced_accuracy"],
            "auroc": pref["auroc"],
            "exact_p": pref["matched_pair_randomization_pvalue_one_sided"],
            "supported_at_0.05": pref_supported,
        },
        "updates_leak": updates,
        "label_only_di": label,
        "shadow_fidelity_passed": shadow_fidelity["num_passed"],
        "target_fidelity_passed": target_fidelity["num_passed"],
        "target_hard_label_unique_vectors": label_predictions[
            "target_hard_label_diagnostics"
        ]["num_unique_hard_label_vectors"],
    }
    (args.output_dir / "PPSD_RESULTS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    human = metadata["human_preference_support"]
    first_style, second_style = style_ids
    conclusion = (
        "PrefLeak 在 PPSD held-out updates 上显著高于随机水平。"
        if pref_supported
        else "PrefLeak 在本次 PPSD held-out updates 上未通过预设的显著性检验。"
    )
    lines = [
        f"# PPSD + {family.label}: three-method results",
        "",
        "## Core result",
        "",
        conclusion,
        "",
        "| Method | Access | Effective calls | Balanced accuracy | AUROC | Exact paired significance |",
        "|---|---|---:|---:|---:|---:|",
        f"| PrefLeak | paired pre/post | {pref['query_budget']} | {pref['balanced_accuracy']:.3f} | {pref['auroc']:.3f} | {pref['matched_pair_randomization_pvalue_one_sided']:.5f} |",
        f"| Updates-Leak-style MLP | paired pre/post | {updates['calls']} | {updates['balanced_accuracy_mean']:.3f} ± {updates['balanced_accuracy_std']:.3f} | {updates['auroc_mean']:.3f} ± {updates['auroc_std']:.3f} | {updates['significant_runs']}/{updates['num_runs']} runs with p<.05 |",
        f"| Label-only DI | post-only hard labels | {label['calls']} | {label['balanced_accuracy_mean']:.3f} ± {label['balanced_accuracy_std']:.3f} | {label['auroc_mean']:.3f} ± {label['auroc_std']:.3f} | {label['significant_runs']}/{label['num_runs']} runs with p<.05 |",
        "",
        "## Protocol",
        "",
        f"- Victim: {family.label}; genuine joint vision/projector/language LoRA with pure sigmoid DPO.",
        f"- Dataset: PPSD collection {metadata['collection']}, style IDs {first_style} versus {second_style}. The two styles have direct human support rather than being generated color filters.",
        f"- Human direct votes: style {first_style} = {human[f'style_{first_style}_votes']}, style {second_style} = {human[f'style_{second_style}_votes']} (total {human['num_direct_votes']}).",
        (
            f"- Scene split: {metadata['num_shadow_scenes']} shadow, "
            f"{metadata['num_target_scenes']} target, "
            f"{metadata.get('num_shadow_fidelity_scenes', metadata.get('num_fidelity_scenes', 0))} shadow-fidelity, and "
            f"{metadata.get('num_target_fidelity_scenes', 0)} target-fidelity scenes; all disjoint."
            if "num_shadow_fidelity_scenes" in metadata
            else f"- Scene split: {metadata['num_shadow_scenes']} shadow, {metadata['num_target_scenes']} target, {metadata['num_fidelity_scenes']} private fidelity; all disjoint."
        ),
        f"- Updates: {metadata['num_shadow_updates']} shadow updates in {metadata['num_shadow_updates'] // 2} counterfactual pairs and {metadata['num_target_updates']} held-out targets in {metadata['num_target_updates'] // 2} pairs. A matched pair shares images, seed, and hyperparameters; only chosen/rejected answers reverse.",
        f"- Update hyperparameters were transferred from the frozen {read_config(args.output_dir)['selection_boundary']} configuration; PPSD target data were not used for selection.",
        f"- Remote bank: 132 preference-remote text tasks; nested shadow validation selected K={frozen_pref['selection']['chosen_num_probes']} probes.",
        "- Target labels were supplied only to the shared evaluator after all three prediction files were serialized.",
        "",
        "## Update fidelity",
        "",
        f"- Adapter audit: {audit_result['num_passed']}/{audit_result['num_updates']} passed.",
        f"- Shadow visual fidelity: {shadow_fidelity['num_passed']}/{shadow_fidelity['num_models']} passed; mean post accuracy {shadow_fidelity['mean_post_preference_accuracy']:.3f}.",
        f"- Target visual fidelity: {target_fidelity['num_passed']}/{target_fidelity['num_models']} passed; mean post accuracy {target_fidelity['mean_post_preference_accuracy']:.3f}.",
        "",
        "## Claim boundary",
        "",
        "This is an external-dataset replication of closed-set photographic-style direction inference. It does not claim recovery of an arbitrary individual user's identity or unconstrained preference profile.",
        f"Label-only target representation contains {result['target_hard_label_unique_vectors']} unique hard-label vectors; collapse to one vector means that baseline has no ranking signal.",
        "",
        f"Machine-readable metrics, raw predictions, frozen attacks, transcripts, fidelity results, and adapter metadata are stored under `{args.output_dir}`.",
        "",
    ]
    family.report.parent.mkdir(parents=True, exist_ok=True)
    family.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(family.report), **result}, indent=2))
    return family.report


def main(default_family: str = "llava") -> None:
    args = parse_args(default_family)
    family = family_config(args.family)
    args.model = args.model or family.model
    model_text = str(args.model).lower()
    if family.key == "llava" and "7b" in model_text:
        family = replace(family, label="LLaVA-OneVision-Qwen2-7B")
    args.output_dir = args.output_dir or family.default_output
    args.jobs = args.jobs or family.default_jobs
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    env = environment()
    shadows, targets = build_specs()
    specs = [*shadows, *targets]

    if args.phase in {"prepare", "all"}:
        prepare(args, family, env)
    if args.phase in {"train", "all"}:
        train_all(args, family, env, specs)
    if args.phase == "train-shadow":
        train_all(args, family, env, shadows)
    if args.phase == "train-target":
        train_all(args, family, env, targets)
    if args.phase in {"audit", "all"}:
        audit(args, specs)
    if args.phase in {"fidelity", "all"}:
        probe_fidelity(args, family, env, "shadow", shadows)
        probe_fidelity(args, family, env, "target", targets)
    if args.phase == "fidelity-shadow":
        probe_fidelity(args, family, env, "shadow", shadows)
    if args.phase == "fidelity-target":
        probe_fidelity(args, family, env, "target", targets)
    if args.phase in {"shadow", "all"}:
        probe_remote(args, family, env, "shadow_candidates", shadows, PROBE_BANK)
    if args.phase in {"calibrate", "all"}:
        calibrate_attacks(args, env, shadows)
    if args.phase in {"target", "all"}:
        bank = materialize_target_bank(args.output_dir)
        probe_remote(args, family, env, "target_union", targets, bank)
    if args.phase in {"infer", "all"}:
        infer_attacks(args, env, targets)
    if args.phase in {"evaluate", "all"}:
        evaluate_all(args, env, targets)
    if args.phase in {"report", "all"}:
        write_report(args, family)


if __name__ == "__main__":
    main()

