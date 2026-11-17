#!/usr/bin/env python3
"""Run PARA + GLM-4.1V joint-DPO updates and three matched attacks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.GLM.PARA.protocol import (  # noqa: E402
    DATA_ROOT,
    PROJECT_ROOT,
    REFERENCE_PREFERENCE,
    UpdateSpec,
    build_specs,
)
from Text_only.PPSD import run_ppsd_experiment as shared  # noqa: E402


PARA_DIR = Path(__file__).resolve().parent
TEXT_ROOT = PROJECT_ROOT / "Text_only"
PROBE_BANK = TEXT_ROOT / "prompts" / "remote_text_candidate_bank.json"
DEFAULT_OUTPUT = PARA_DIR / "outputs" / "glm_v1"
SOURCE_CONFIG = (
    TEXT_ROOT
    / "GLM"
    / "PPSD"
    / "outputs"
    / "d1_style01_glm_v2"
    / "frozen_config.json"
)
MAX_IMAGE_PIXELS = 1_003_520


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
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    # One GLM joint-DPO worker peaks near 40 GiB on the local 96-GiB GPU.
    # Three workers OOM; two are the validated maximum concurrency.
    parser.add_argument("--jobs", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict[str, str]:
    env = shared.environment()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    shared.run(command, env)


def prepare(args: argparse.Namespace) -> dict:
    metadata = DATA_ROOT / "private_dataset_metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"Build the shared PARA protocol before preparing GLM: {metadata}"
        )
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(SOURCE_CONFIG)
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if "shadow" not in str(source.get("selection_boundary", "")).lower():
        raise ValueError("The transferred GLM configuration is not shadow-only.")
    source_target_opened = source.get("selection_evidence", {}).get(
        "target_data_opened"
    )
    if source_target_opened is not False:
        raise ValueError("The source GLM configuration opened target data.")
    config = {
        **source,
        "selection_boundary": (
            "transfer_GLM_PPSD_D1_shadow_configuration_without_PARA_target_tuning"
        ),
        "para_transfer": {
            "source_frozen_config": str(SOURCE_CONFIG.resolve()),
            "source_sha256": sha256(SOURCE_CONFIG),
            "rule": "external_dataset_transfer_without_PARA_target_tuning",
            "para_target_data_opened": False,
        },
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "image_preprocessing": (
            "aspect_ratio_preserving_resize_without_crop; pixel cap chosen "
            "before PARA target training to match GLM-PPSD image-token scale"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "frozen_config.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Frozen GLM PARA configuration changed: {destination}")
    else:
        destination.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    return config


def read_config(output: Path) -> dict:
    path = output / "frozen_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare first: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    transfer = config.get("para_transfer", {})
    if transfer.get("para_target_data_opened") is not False:
        raise ValueError("PARA transfer boundary is not target-blind.")
    return config


def train_all(
    args: argparse.Namespace,
    family: shared.Family,
    env: dict[str, str],
    specs: list[UpdateSpec],
) -> None:
    config = read_config(args.output_dir)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(shared.train_one, args, family, env, config, spec): spec
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
    keys = [
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "beta",
        "label_smoothing",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "max_image_pixels",
    ]
    keys.extend(
        key
        for key in (
            "vision_lora_rank",
            "vision_lora_alpha",
            "projector_lora_rank",
            "projector_lora_alpha",
            "language_lora_rank",
            "language_lora_alpha",
        )
        if key in config
    )
    rows: list[dict] = []
    pair_seeds: dict[str, set[int]] = {}
    for spec in specs:
        metadata_path = (
            shared.adapter_path(args.output_dir, spec)
            / "private_training_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
            "frozen_hyperparameters": all(
                metadata[key] == config[key] for key in keys
            ),
        }
        rows.append(
            {"model_id": spec.model_id, **checks, "passed": all(checks.values())}
        )
    same_pair_seed = all(len(values) == 1 for values in pair_seeds.values())
    result = {
        "audit": "para_glm_joint_vlm_pure_dpo_transferred_configuration",
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
        raise RuntimeError("GLM PARA adapter audit failed.")
    return result


def configure_shared_para_protocol() -> None:
    """Point reused PPSD orchestration helpers at the immutable PARA data."""

    shared.DATA_ROOT = DATA_ROOT
    shared.REFERENCE_PREFERENCE = REFERENCE_PREFERENCE


def shared_command(phase: str, env: dict[str, str]) -> None:
    run(
        [
            sys.executable,
            str(TEXT_ROOT / "shared_response_resampling.py"),
            phase,
            "--experiment",
            "para_glm",
        ],
        env,
    )


def write_report(args: argparse.Namespace) -> None:
    shared_dir = args.output_dir / "shared_96"
    result = json.loads((shared_dir / "RESULTS.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (DATA_ROOT / "private_dataset_metadata.json").read_text(encoding="utf-8")
    )
    audit_result = json.loads(
        (args.output_dir / "audit.json").read_text(encoding="utf-8")
    )
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
        "# PARA + GLM-4.1V shared 96-call results",
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
            "## Protocol and fidelity",
            "",
            f"- Preference task: {metadata['preference_definition']}.",
            f"- Session split: {metadata['num_shadow_sessions']} shadow, {metadata['num_target_sessions']} target, and {metadata['num_fidelity_sessions']} fidelity; sessions and images are disjoint.",
            f"- Counterfactual updates: {metadata['num_shadow_updates']} shadow and {metadata['num_target_updates']} target; paired updates differ only in chosen/rejected answers.",
            f"- GLM image preprocessing: aspect-ratio-preserving resize without crop, capped at {MAX_IMAGE_PIXELS} pixels for both DPO and fidelity.",
            f"- Joint-DPO adapter audit: {audit_result['num_passed']}/{audit_result['num_updates']} passed.",
            f"- Shadow visual fidelity: {shadow_fidelity['num_passed']}/{shadow_fidelity['num_models']} passed.",
            f"- Target visual fidelity: {target_fidelity['num_passed']}/{target_fidelity['num_models']} passed.",
            "- Target labels were opened only after every prediction file was serialized.",
            "",
            "The ten overlapping response resamples estimate decoding stability, not uncertainty over ten independent target datasets; exact p-values are not pooled.",
            "",
            f"Machine-readable artifacts: `{shared_dir}`.",
            "",
        ]
    )
    (PARA_DIR / "PARA_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    base_family = shared.family_config("glm")
    model = args.model or base_family.model
    args.model = model
    family = replace(
        base_family,
        label="GLM-4.1V-9B-Base",
        model=model,
        default_output=args.output_dir,
        report=PARA_DIR / "PARA_RESULTS.md",
        default_jobs=args.jobs,
    )
    env = environment()
    shadows, targets = build_specs()
    specs = [*shadows, *targets]
    configure_shared_para_protocol()

    if args.phase in {"build", "all"}:
        run(
            [sys.executable, str(TEXT_ROOT / "PARA" / "build_para_preferences.py")],
            env,
        )
    if args.phase in {"prepare", "all"}:
        prepare(args)
    if args.phase in {"train", "all"}:
        train_all(args, family, env, specs)
    if args.phase in {"audit", "all"}:
        audit(args, specs)
    if args.phase in {"fidelity", "all"}:
        shared.probe_fidelity(args, family, env, "shadow", shadows)
        shared.probe_fidelity(args, family, env, "target", targets)
    if args.phase in {"shadow", "all"}:
        shared.probe_remote(
            args, family, env, "shadow_candidates", shadows, PROBE_BANK
        )
    if args.phase in {"shared-prepare", "all"}:
        shared_command("prepare", env)
    if args.phase in {"target", "all"}:
        shared.probe_remote(
            args,
            family,
            env,
            "target_union",
            targets,
            args.output_dir / "shared_96" / "target_union_probe_bank.json",
        )
    if args.phase in {"shared-infer", "all"}:
        for phase in ("infer", "evaluate", "report"):
            shared_command(phase, env)
    if args.phase in {"report", "all"}:
        write_report(args)


if __name__ == "__main__":
    main()

