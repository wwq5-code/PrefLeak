#!/usr/bin/env python3
"""Train and evaluate neutral and label-shuffled VLM update controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data" / "joint_vlm_paper" / "val2017" / "magenta_cyan"
LLAVA_SOURCE_UPDATE_ROOT = SCRIPT_DIR / "outputs" / "joint_vlm_paper_v1" / "main"
LLAVA_REMOTE_ROOT = SCRIPT_DIR / "outputs" / "remote_probe_llava_v1" / "main"
LLAVA_DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "update_controls_llava_v1" / "main"
GLM_SOURCE_UPDATE_ROOT = SCRIPT_DIR / "GLM" / "outputs" / "main_v1"
GLM_REMOTE_ROOT = SCRIPT_DIR / "GLM" / "outputs" / "remote_probe_glm_v1" / "main"
GLM_DEFAULT_OUTPUT = SCRIPT_DIR / "GLM" / "outputs" / "update_controls_glm_v1" / "main"
FULL_BANK = SCRIPT_DIR / "prompts" / "remote_text_candidate_bank.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Text_only.run_joint_vlm_experiment import MODEL as LLAVA_MODEL  # noqa: E402
from Text_only.GLM.train_glm_dpo import DEFAULT_MODEL as GLM_MODEL  # noqa: E402


@dataclass(frozen=True)
class ControlSpec:
    model_id: str
    kind: str
    pair_index: int
    manifest: Path


def build_specs() -> list[ControlSpec]:
    specs = []
    for pair_index in range(1, 11):
        serial = 100 + pair_index
        specs.extend(
            [
                ControlSpec(
                    f"target_n{serial:03d}", "neutral", pair_index,
                    DATA_ROOT / "private_controls" / f"target_n{serial:03d}.jsonl",
                ),
                ControlSpec(
                    f"target_s{serial:03d}", "label_shuffled", pair_index,
                    DATA_ROOT / "private_controls" / f"target_s{serial:03d}.jsonl",
                ),
            ]
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "train", "audit", "probe", "probe-visual", "infer", "analyze", "all"
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--model-family", choices=("llava", "glm"), default="llava")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-update-root", type=Path)
    parser.add_argument("--remote-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    defaults = {
        "llava": (
            LLAVA_MODEL, LLAVA_SOURCE_UPDATE_ROOT, LLAVA_REMOTE_ROOT,
            LLAVA_DEFAULT_OUTPUT, 4,
        ),
        "glm": (
            GLM_MODEL, GLM_SOURCE_UPDATE_ROOT, GLM_REMOTE_ROOT,
            GLM_DEFAULT_OUTPUT, 3,
        ),
    }[args.model_family]
    for field, value in zip(
        ("model", "source_update_root", "remote_root", "output_dir", "jobs"),
        defaults,
    ):
        if getattr(args, field) is None:
            setattr(args, field, value)
    return args


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'trl'}:{ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def frozen_config(args: argparse.Namespace) -> dict:
    config = json.loads(
        (args.source_update_root / "frozen_config.json").read_text(encoding="utf-8")
    )
    if config["selection_boundary"] != "shadow_only_train2017":
        raise ValueError("Control updates must reuse a shadow-only frozen configuration.")
    return config


def adapter_path(args: argparse.Namespace, spec: ControlSpec) -> Path:
    return args.output_dir / "adapters" / spec.model_id


def train_one(
    args: argparse.Namespace,
    env: dict[str, str],
    config: dict,
    spec: ControlSpec,
) -> str:
    destination = adapter_path(args, spec)
    if (destination / "adapter_config.json").is_file():
        return f"reuse {spec.model_id}"
    log = args.output_dir / "logs" / f"{spec.model_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(
            SCRIPT_DIR / "train_llava_dpo.py"
            if args.model_family == "llava"
            else SCRIPT_DIR / "GLM" / "train_glm_dpo.py"
        ),
        "--model", str(args.model),
        "--train-manifest", str(spec.manifest),
        "--output-dir", str(destination),
        "--target-id", spec.model_id,
        "--seed", str(10400 + spec.pair_index),
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
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return f"complete {spec.model_id}"


def train_all(
    args: argparse.Namespace,
    env: dict[str, str],
    config: dict,
    specs: list[ControlSpec],
) -> None:
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(train_one, args, env, config, spec): spec for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(f"Control training failed for {spec.model_id}") from error


def audit(
    args: argparse.Namespace, config: dict, specs: list[ControlSpec]
) -> Path:
    expected = {
        key: config[key]
        for key in (
            "max_steps", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "beta",
            "label_smoothing", "lora_rank", "lora_alpha", "lora_dropout",
        )
    }
    rows = []
    for spec in specs:
        metadata = json.loads(
            (adapter_path(args, spec) / "private_training_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        checks = {
            "manifest": Path(metadata["train_manifest"]) == spec.manifest.resolve(),
            "pure_dpo": metadata["objective"] == "dpo"
            and metadata["loss_type"] == ["sigmoid"]
            and metadata["loss_weights"] is None,
            "joint_vlm": metadata["update_scope"] == "joint_vlm"
            and set(metadata["matched_module_counts"]) == {"vision", "projector", "language"}
            and all(value > 0 for value in metadata["matched_module_counts"].values()),
            "frozen_hyperparameters": all(
                metadata[key] == value for key, value in expected.items()
            ),
        }
        rows.append(
            {
                "model_id": spec.model_id,
                "control_kind": spec.kind,
                "pair_index": spec.pair_index,
                **checks,
                "passed": all(checks.values()),
            }
        )
    result = {
        "audit": f"{args.model_family}_matched_update_controls",
        "num_updates": len(rows),
        "num_passed": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "per_update": rows,
    }
    output = args.output_dir / "audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_passed"]:
        raise RuntimeError(f"Control audit failed: {output}")
    return output


def selected_bank(args: argparse.Namespace, env: dict[str, str]) -> Path:
    output = args.remote_root / "frozen_probe_banks" / "primary_selected_probe_bank.json"
    if output.is_file():
        return output
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "materialize_frozen_probe_banks.py"),
            "--probe-bank", str(FULL_BANK),
            "--frozen-attack", str(args.remote_root / "frozen_remote_attack.json"),
            "--output-dir", str(args.remote_root / "frozen_probe_banks"),
        ],
        env,
    )
    return output


def probe(
    args: argparse.Namespace, env: dict[str, str], specs: list[ControlSpec]
) -> Path:
    output_dir = args.output_dir / "probes" / "paired_selected"
    output = output_dir / "attacker_view" / "remote_generations.jsonl"
    if output.is_file():
        return output
    command = [
        sys.executable,
        str(
            SCRIPT_DIR / "probe_remote_text.py"
            if args.model_family == "llava"
            else SCRIPT_DIR / "GLM" / "probe_remote_glm.py"
        ),
        "--model", str(args.model),
        "--probe-bank", str(selected_bank(args, env)),
        "--seed-reference-bank", str(FULL_BANK),
        "--output-dir", str(output_dir),
        "--repetitions", "4",
        "--max-new-tokens", "12",
        "--temperature", "0.8",
        "--top-p", "0.95",
    ]
    if args.model_family == "glm":
        command.extend(["--batch-size", "4"])
    for spec in specs:
        command.extend(["--adapter", f"{spec.model_id}={adapter_path(args, spec)}"])
    run(command, env)
    return output


def infer(
    args: argparse.Namespace,
    env: dict[str, str],
    specs: list[ControlSpec],
) -> Path:
    output_dir = args.output_dir / "predictions"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "remote_attack.py"),
        "infer",
        "--target-generations", str(
            args.output_dir
            / "probes"
            / "paired_selected"
            / "attacker_view"
            / "remote_generations.jsonl"
        ),
        "--probe-bank", str(FULL_BANK),
        "--frozen-attack", str(args.remote_root / "frozen_remote_attack.json"),
        "--attack-name", "paired_selected_residualized",
        "--output-dir", str(output_dir),
    ]
    for spec in specs:
        command.extend(["--target-id", spec.model_id])
    run(command, env)
    return output_dir / "paired_selected_residualized.json"


def probe_visual(
    args: argparse.Namespace,
    env: dict[str, str],
    specs: list[ControlSpec],
) -> Path:
    output_dir = args.output_dir / "probes" / "visual_control"
    generations = output_dir / "attacker_view" / "forced_choice_generations.jsonl"
    if not generations.is_file():
        script = (
            SCRIPT_DIR / "probe_forced_choice.py"
            if args.model_family == "llava"
            else SCRIPT_DIR / "GLM" / "probe_glm.py"
        )
        command = [
            sys.executable,
            str(script),
            "--model", str(args.model),
            "--modality", "vision_conditioned",
            "--probe-bank", str(DATA_ROOT / "private_fidelity_probe_bank.json"),
            "--output-dir", str(output_dir),
            "--max-new-tokens", "8",
            "--max-queries", "32",
        ]
        if args.model_family == "llava":
            command.extend(["--visual-prompt-style", "training"])
        for spec in specs:
            command.extend(["--adapter", f"{spec.model_id}={adapter_path(args, spec)}"])
        run(command, env)
    output = args.output_dir / "analysis" / "control_visual.json"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_control_visual.py"),
        "--generations", str(generations),
        "--output", str(output),
    ]
    for spec in specs:
        command.extend(["--control-kind", f"{spec.model_id}={spec.kind}"])
    run(command, env)
    return output


def logit(value: float) -> float:
    clipped = min(max(value, 1e-9), 1.0 - 1e-9)
    return math.log(clipped / (1.0 - clipped))


def score_summary(scores: list[float]) -> dict:
    return {
        "num_updates": len(scores),
        "mean_positive_class_score": statistics.fmean(scores),
        "score_std": statistics.stdev(scores),
        "score_min": min(scores),
        "score_max": max(scores),
    }


def analyze(args: argparse.Namespace, specs: list[ControlSpec]) -> Path:
    prediction_path = args.output_dir / "predictions" / "paired_selected_residualized.json"
    document = json.loads(prediction_path.read_text(encoding="utf-8"))
    by_id = {row["anonymous_model_id"]: row for row in document["predictions"]}
    frozen = json.loads(
        (args.remote_root / "frozen_remote_attack.json").read_text(encoding="utf-8")
    )
    attack = next(
        row for row in frozen["attacks"]
        if row["name"] == "paired_selected_residualized"
    )
    threshold = float(attack["threshold"])
    threshold_logit = logit(threshold)
    summaries = {}
    per_update = []
    for kind in ("neutral", "label_shuffled"):
        kind_specs = [spec for spec in specs if spec.kind == kind]
        scores = [float(by_id[spec.model_id]["positive_class_score"]) for spec in kind_specs]
        margins = [logit(score) - threshold_logit for score in scores]
        summaries[kind] = {
            **score_summary(scores),
            "median_absolute_logit_margin_from_threshold": statistics.median(
                abs(value) for value in margins
            ),
            "predicted_magenta_fraction": sum(value >= 0 for value in margins)
            / len(margins),
        }
        for spec, score, margin in zip(kind_specs, scores, margins):
            per_update.append(
                {
                    "anonymous_model_id": spec.model_id,
                    "control_kind": kind,
                    "pair_index": spec.pair_index,
                    "positive_class_score": score,
                    "logit_margin_from_frozen_threshold": margin,
                    "predicted_preference": by_id[spec.model_id]["predicted_preference"],
                }
            )
    preference_predictions = json.loads(
        (
            args.remote_root
            / "predictions"
            / "paired_selected_residualized.json"
        ).read_text(encoding="utf-8")
    )["predictions"]
    preference_scores = {"cyan": [], "magenta": []}
    for row in preference_predictions:
        serial = int(row["anonymous_model_id"].removeprefix("target_u"))
        preference = "magenta" if serial % 2 else "cyan"
        preference_scores[preference].append(float(row["positive_class_score"]))
    preference_summary = {
        key: score_summary(scores) for key, scores in preference_scores.items()
    }
    result = {
        "analysis": "preference_absent_update_controls",
        "model_family": args.model_family,
        "query_modality": "text_only_remote",
        "target_private_labels_used": False,
        "frozen_attack": "paired_selected_residualized",
        "num_selected_probes": attack["num_probes"],
        "online_generation_calls_per_update": attack["online_generation_calls"],
        "threshold": threshold,
        "interpretation_rule": (
            "Controls have no preference ground truth; report score and decision-margin "
            "distributions, not attack accuracy."
        ),
        "summary": summaries,
        "preference_bearing_target_context": preference_summary,
        "preference_target_labels_used_only_for_posthoc_context": True,
        "specificity_warning": (
            "A binary preference classifier has no absent-preference class; "
            "systematic high-confidence assignment of controls indicates an "
            "OOD/rejection limitation, not control accuracy."
        ),
        "per_update": sorted(per_update, key=lambda row: row["anonymous_model_id"]),
    }
    visual_path = args.output_dir / "analysis" / "control_visual.json"
    if visual_path.is_file():
        result["visual_control_diagnostic"] = json.loads(
            visual_path.read_text(encoding="utf-8")
        )
    output = args.output_dir / "analysis" / "control_scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    missing = [str(spec.manifest) for spec in specs if not spec.manifest.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing control manifests: {missing}")
    env = environment()
    config = frozen_config(args)
    if args.phase in {"train", "all"}:
        train_all(args, env, config, specs)
    if args.phase in {"audit", "all"}:
        print(json.dumps({"audit": str(audit(args, config, specs))}, indent=2))
    if args.phase in {"probe", "all"}:
        probe(args, env, specs)
    if args.phase in {"probe-visual", "all"}:
        probe_visual(args, env, specs)
    if args.phase in {"infer", "all"}:
        infer(args, env, specs)
    if args.phase in {"analyze", "all"}:
        print(json.dumps({"analysis": str(analyze(args, specs))}, indent=2))


if __name__ == "__main__":
    main()
