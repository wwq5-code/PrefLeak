#!/usr/bin/env python3
"""Run PARA + LLaVA joint-DPO updates and the matched three-attack protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.PARA.protocol import (
    DATA_ROOT,
    PROJECT_ROOT,
    REFERENCE_PREFERENCE,
    UpdateSpec,
    build_specs,
)
from Text_only.PPSD.run_ppsd_experiment import (
    adapter_arguments,
    adapter_path,
    environment,
    family_config,
    probe_remote,
    run,
    train_one,
)


PARA_DIR = Path(__file__).resolve().parent
TEXT_ROOT = PROJECT_ROOT / "Text_only"
PROBE_BANK = TEXT_ROOT / "prompts" / "remote_text_candidate_bank.json"
DEFAULT_OUTPUT = PARA_DIR / "outputs" / "llava_v1"
DEFAULT_MODEL = Path("llava-hf/llava-onevision-qwen2-7b-ov-hf")
SOURCE_CONFIG = (
    TEXT_ROOT
    / "PPSD"
    / "outputs"
    / "d2_style13_shadow_validation_llava7b"
    / "frozen_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "build",
            "prepare",
            "train",
            "audit",
            "fidelity",
            "shadow",
            "shared-prepare",
            "target",
            "shared-infer",
            "report",
            "all",
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(SOURCE_CONFIG)
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config["selection_boundary"] = (
        "transfer_PPSD_D2_shadow_configuration_without_PARA_target_tuning"
    )
    config["para_transfer"] = {
        "source_frozen_config": str(SOURCE_CONFIG.resolve()),
        "source_sha256": sha256(SOURCE_CONFIG),
        "rule": "external_dataset_transfer_without_PARA_target_tuning",
        "para_target_data_opened": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "frozen_config.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Frozen PARA configuration changed: {destination}")
    else:
        destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def read_config(output: Path) -> dict:
    config = json.loads((output / "frozen_config.json").read_text(encoding="utf-8"))
    transfer = config.get("para_transfer", {})
    if transfer.get("para_target_data_opened") is not False:
        raise ValueError("PARA configuration is not target-blind.")
    return config


def train_all(
    args: argparse.Namespace,
    family,
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
                raise RuntimeError(
                    f"Training failed for {spec.model_id}; inspect {log}"
                ) from error


def audit(args: argparse.Namespace, specs: list[UpdateSpec]) -> dict:
    config = read_config(args.output_dir)
    keys = (
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "beta",
        "label_smoothing",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
    )
    rows = []
    seeds: dict[str, set[int]] = {}
    for spec in specs:
        metadata_path = (
            adapter_path(args.output_dir, spec) / "private_training_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        seeds.setdefault(spec.pair_id, set()).add(metadata["seed"])
        counts = metadata["matched_module_counts"]
        checks = {
            "manifest": Path(metadata["train_manifest"]) == spec.manifest.resolve(),
            "pure_dpo": metadata["objective"] == "dpo"
            and metadata["loss_type"] == ["sigmoid"]
            and metadata["loss_weights"] is None,
            "joint_vlm": metadata["update_scope"] == "joint_vlm"
            and set(counts) == {"vision", "projector", "language"}
            and all(value > 0 for value in counts.values()),
            "frozen_hyperparameters": all(
                metadata[key] == config[key] for key in keys
            ),
        }
        rows.append(
            {"model_id": spec.model_id, **checks, "passed": all(checks.values())}
        )
    same_pair_seed = all(len(values) == 1 for values in seeds.values())
    result = {
        "audit": "para_llava_joint_vlm_pure_dpo_transferred_configuration",
        "num_updates": len(rows),
        "num_passed": sum(row["passed"] for row in rows),
        "same_seed_within_each_counterfactual_pair": same_pair_seed,
        "all_passed": all(row["passed"] for row in rows) and same_pair_seed,
        "per_update": rows,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not result["all_passed"]:
        raise RuntimeError("PARA adapter audit failed.")
    return result


def label_arguments(specs: list[UpdateSpec], option: str) -> list[str]:
    result = []
    for spec in specs:
        result.extend([option, f"{spec.model_id}={spec.preference}"])
    return result


def probe_fidelity(
    args: argparse.Namespace,
    family,
    env: dict[str, str],
    split: str,
    specs: list[UpdateSpec],
) -> None:
    directory = args.output_dir / "probes" / f"visual_fidelity_{split}"
    transcript = directory / "attacker_view" / "forced_choice_generations.jsonl"
    expected = {"M0_reference", *(spec.model_id for spec in specs)}
    observed = set()
    if transcript.is_file():
        with transcript.open(encoding="utf-8") as handle:
            observed = {
                json.loads(line)["model_version"] for line in handle if line.strip()
            }
    if observed != expected:
        run(
            [
                sys.executable,
                str(family.fidelity_script),
                "--model",
                str(args.model),
                "--modality",
                "vision_conditioned",
                "--probe-bank",
                str(DATA_ROOT / "private_fidelity_probe_bank.json"),
                "--output-dir",
                str(directory),
                "--max-new-tokens",
                str(family.fidelity_max_tokens),
                "--visual-prompt-style",
                "training",
                *adapter_arguments(args.output_dir, specs),
            ],
            env,
        )
    output = args.output_dir / "analysis" / f"visual_fidelity_{split}.json"
    run(
        [
            sys.executable,
            "-m",
            "Text_only.evaluate_update_fidelity",
            "--generations",
            str(transcript),
            "--output",
            str(output),
            "--reference-preference",
            REFERENCE_PREFERENCE,
            "--min-improvement",
            str(family.fidelity_min_improvement),
            "--min-parsed-rate",
            str(family.fidelity_min_parsed),
            *label_arguments(specs, "--target-label"),
        ],
        env,
        check=False,
    )
    if not output.is_file():
        raise RuntimeError(f"Fidelity evaluator did not write {output}")


def shared_command(
    phase: str, args: argparse.Namespace, env: dict[str, str]
) -> None:
    run(
        [
            sys.executable,
            str(TEXT_ROOT / "shared_response_resampling.py"),
            phase,
            "--experiment",
            "para_llava",
        ],
        env,
    )


def probe_target(
    args: argparse.Namespace,
    env: dict[str, str],
    targets: list[UpdateSpec],
) -> None:
    shared = args.output_dir / "shared_96"
    output = shared / "source_target"
    transcript = output / "attacker_view" / "remote_generations.jsonl"
    if transcript.is_file():
        print(f"reuse {transcript}", flush=True)
        return
    run(
        [
            sys.executable,
            str(TEXT_ROOT / "probe_remote_text.py"),
            "--model",
            str(args.model),
            "--probe-bank",
            str(shared / "target_union_probe_bank.json"),
            "--seed-reference-bank",
            str(PROBE_BANK),
            "--output-dir",
            str(output),
            "--repetitions",
            "8",
            "--max-new-tokens",
            "12",
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            *adapter_arguments(args.output_dir, targets),
        ],
        env,
    )


def write_report(args: argparse.Namespace) -> None:
    shared = args.output_dir / "shared_96"
    result = json.loads((shared / "RESULTS.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (DATA_ROOT / "private_dataset_metadata.json").read_text(encoding="utf-8")
    )
    audit_result = json.loads((args.output_dir / "audit.json").read_text(encoding="utf-8"))
    shadow_fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_shadow.json").read_text(
            encoding="utf-8"
        )
    )
    target_fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_target.json").read_text(
            encoding="utf-8"
        )
    )
    names = {
        "prefleak": "PrefLeak",
        "updates_leak": "Updates-Leak-style MLP",
        "label_only_di": "Label-only DI",
    }
    lines = [
        "# PARA + LLaVA shared 96-call results",
        "",
        "All methods use the same 10 held-out target updates, the same ten balanced 4-of-8 response splits, and exactly 96 calls per split.",
        "",
        "| Method | Access | Balanced accuracy | AUROC | Macro-F1 | exact p<.05 splits |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in ("prefleak", "updates_leak", "label_only_di"):
        row = result[key]
        access = "post-only" if key == "label_only_di" else "paired pre/post"
        lines.append(
            f"| {names[key]} | {access} | "
            f"{row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f} | "
            f"{row['auroc_mean']:.3f} ± {row['auroc_std']:.3f} | "
            f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} | "
            f"{row['significant_runs']}/{row['num_runs']} |"
        )
    lines.extend(
        [
            "",
            "## Response-resampling variance",
            "",
            "| Method | Balanced accuracy variance | AUROC variance | Macro-F1 variance |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("prefleak", "updates_leak", "label_only_di"):
        row = result[key]
        lines.append(
            f"| {names[key]} | {row['balanced_accuracy_variance']:.6f} | "
            f"{row['auroc_variance']:.6f} | {row['macro_f1_variance']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol and fidelity",
            "",
            f"- Preference task: {metadata['preference_definition']}.",
            f"- Session split: {metadata['num_shadow_sessions']} shadow, {metadata['num_target_sessions']} target, and {metadata['num_fidelity_sessions']} fidelity; sessions and images are disjoint.",
            f"- Counterfactual updates: {metadata['num_shadow_updates']} shadow and {metadata['num_target_updates']} target; paired updates differ only in chosen/rejected answers.",
            f"- Joint-DPO adapter audit: {audit_result['num_passed']}/{audit_result['num_updates']} passed.",
            f"- Shadow visual fidelity: {shadow_fidelity['num_passed']}/{shadow_fidelity['num_models']} passed.",
            f"- Target visual fidelity: {target_fidelity['num_passed']}/{target_fidelity['num_models']} passed.",
            "- Target labels were opened only after every prediction file was serialized.",
            "",
            "The ten overlapping response resamples estimate decoding stability, not uncertainty over ten independent target datasets; exact p-values are not pooled.",
            "",
            f"Machine-readable artifacts: `{shared}`.",
            "",
        ]
    )
    (PARA_DIR / "PARA_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    family = replace(
        family_config("llava"),
        label="LLaVA-OneVision-Qwen2-7B",
        model=args.model,
        default_jobs=args.jobs,
    )
    env = environment()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    shadows, targets = build_specs()
    specs = [*shadows, *targets]

    if args.phase in {"build", "all"}:
        run(
            [sys.executable, str(PARA_DIR / "build_para_preferences.py")], env
        )
    if args.phase in {"prepare", "all"}:
        prepare(args)
    if args.phase in {"train", "all"}:
        train_all(args, family, env, specs)
    if args.phase in {"audit", "all"}:
        audit(args, specs)
    if args.phase in {"fidelity", "all"}:
        probe_fidelity(args, family, env, "shadow", shadows)
        probe_fidelity(args, family, env, "target", targets)
    if args.phase in {"shadow", "all"}:
        probe_remote(args, family, env, "shadow_candidates", shadows, PROBE_BANK)
    if args.phase in {"shared-prepare", "all"}:
        shared_command("prepare", args, env)
    if args.phase in {"target", "all"}:
        probe_target(args, env, targets)
    if args.phase in {"shared-infer", "all"}:
        for phase in ("infer", "evaluate", "report"):
            shared_command(phase, args, env)
    if args.phase in {"report", "all"}:
        write_report(args)


if __name__ == "__main__":
    main()
