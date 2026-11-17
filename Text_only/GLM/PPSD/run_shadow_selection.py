#!/usr/bin/env python3
"""Train and score one target-blind GLM PPSD shadow candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN = PROJECT_ROOT / "Text_only" / "GLM" / "train_glm_dpo.py"
PROBE = PROJECT_ROOT / "Text_only" / "GLM" / "probe_glm.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--serials", type=int, nargs=2, default=(1, 2), metavar=("LEFT", "RIGHT")
    )
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'trl'}:{PROJECT_ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def component_arguments(config: dict) -> list[str]:
    result = []
    for key, option in (
        ("vision_lora_rank", "--vision-lora-rank"),
        ("vision_lora_alpha", "--vision-lora-alpha"),
        ("projector_lora_rank", "--projector-lora-rank"),
        ("projector_lora_alpha", "--projector-lora-alpha"),
        ("language_lora_rank", "--language-lora-rank"),
        ("language_lora_alpha", "--language-lora-alpha"),
    ):
        if key in config:
            result.extend([option, str(config[key])])
    return result


def train_one(args: argparse.Namespace, config: dict, serial: int, env: dict[str, str]) -> None:
    model_id = f"target_b{serial:03d}"
    destination = args.output_dir / "adapters" / model_id
    if (destination / "adapter_config.json").is_file():
        print(f"reuse {model_id}", flush=True)
        return
    log = args.output_dir / "logs" / f"{model_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAIN),
        "--train-manifest", str(args.candidate_dir / "private_updates" / f"{model_id}.jsonl"),
        "--target-id", model_id,
        "--output-dir", str(destination),
        "--seed", str(27400 + (serial + 1) // 2),
        "--max-steps", str(config["max_steps"]),
        "--per-device-train-batch-size", str(config["per_device_train_batch_size"]),
        "--gradient-accumulation-steps", str(config["gradient_accumulation_steps"]),
        "--learning-rate", str(config["learning_rate"]),
        "--beta", str(config["beta"]),
        "--label-smoothing", str(config["label_smoothing"]),
        "--lora-rank", str(config["lora_rank"]),
        "--lora-alpha", str(config["lora_alpha"]),
        "--lora-dropout", str(config["lora_dropout"]),
        *component_arguments(config),
    ]
    if args.model:
        command[2:2] = ["--model", str(args.model)]
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f"complete {model_id}", flush=True)


def main() -> None:
    args = parse_args()
    left_serial, right_serial = args.serials
    if left_serial % 2 != 1 or right_serial != left_serial + 1:
        raise ValueError("--serials must be an odd/even counterfactual pair")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metadata = json.loads(
        (args.candidate_dir / "private_dataset_metadata.json").read_text(encoding="utf-8")
    )
    preferences = metadata["candidate_preferences"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(train_one, args, config, serial, env)
            for serial in args.serials
        ]
        for future in futures:
            future.result()

    probe_dir = args.output_dir / "fidelity"
    generations = probe_dir / "attacker_view" / "forced_choice_generations.jsonl"
    if not generations.is_file():
        left_id = f"target_b{left_serial:03d}"
        right_id = f"target_b{right_serial:03d}"
        command = [
            sys.executable, str(PROBE),
            "--modality", "vision_conditioned",
            "--probe-bank", str(args.candidate_dir / "private_fidelity_shadow_probe_bank.json"),
            "--output-dir", str(probe_dir),
            "--max-new-tokens", "1",
            "--constrained-choice",
            "--adapter", f"{left_id}={args.output_dir / 'adapters' / left_id}",
            "--adapter", f"{right_id}={args.output_dir / 'adapters' / right_id}",
        ]
        if args.model:
            command[2:2] = ["--model", str(args.model)]
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    analysis = args.output_dir / "visual_fidelity_shadow.json"
    left_id = f"target_b{left_serial:03d}"
    right_id = f"target_b{right_serial:03d}"
    subprocess.run(
        [
            sys.executable, "-m", "Text_only.evaluate_update_fidelity",
            "--generations", str(generations),
            "--output", str(analysis),
            "--reference-preference", preferences[0],
            "--min-improvement", "0.15",
            "--min-parsed-rate", "0.90",
            "--target-label", f"{left_id}={preferences[0]}",
            "--target-label", f"{right_id}={preferences[1]}",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )
    fidelity = json.loads(analysis.read_text(encoding="utf-8"))
    rows = fidelity["per_model"]
    symmetry_gap = abs(rows[0]["post_preference_accuracy"] - rows[1]["post_preference_accuracy"])
    selected = fidelity["all_passed"] and symmetry_gap <= 0.15
    result = {
        "selection_boundary": "ppsd_d1_shadow_only",
        "candidate_data": str(args.candidate_dir.resolve()),
        "candidate_data_sha256": hashlib.sha256(
            (args.candidate_dir / "private_dataset_metadata.json").read_bytes()
        ).hexdigest(),
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "target_data_queried": False,
        "fidelity": fidelity,
        "symmetry_gap": symmetry_gap,
        "selected": selected,
    }
    (args.output_dir / "selection_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
