#!/usr/bin/env python3
"""Shadow-only hyperparameter tuning for GLM-4.1V joint visual DPO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.choice_attack_features import (
    available_choice_queries,
    build_choice_features,
    group_choice_generations,
    read_choice_generations,
)
from Text_only.GLM.train_glm_dpo import DEFAULT_MODEL


HERE = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "Text_only" / "data" / "joint_vlm_paper" / "train2017" / "magenta_cyan"
DEFAULT_OUTPUT = HERE / "outputs" / "tuning_v1"


@dataclass(frozen=True)
class Candidate:
    name: str
    learning_rate: float
    max_steps: int = 50


CANDIDATES = (
    Candidate("lr1e5", 1e-5),
    Candidate("lr3e5", 3e-5),
    Candidate("lr5e5", 5e-5),
    Candidate("lr7e5", 7e-5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=3)
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'trl'}:{PROJECT_ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def adapter_path(output: Path, candidate: Candidate, serial: int) -> Path:
    return output / candidate.name / "adapters" / f"target_p{serial:02d}"


def train_one(args: argparse.Namespace, env: dict[str, str], candidate: Candidate, serial: int) -> str:
    adapter = adapter_path(args.output_dir, candidate, serial)
    if (adapter / "adapter_config.json").is_file():
        return f"skip {candidate.name}/target_p{serial:02d}"
    log = args.output_dir / candidate.name / "logs" / f"target_p{serial:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(HERE / "train_glm_dpo.py"),
        "--model", str(args.model),
        "--train-manifest", str(DATA / "private_updates" / f"target_b{serial:03d}.jsonl"),
        "--output-dir", str(adapter),
        "--target-id", f"target_p{serial:02d}",
        "--seed", "12401",
        "--max-steps", str(candidate.max_steps),
        "--per-device-train-batch-size", "1",
        "--gradient-accumulation-steps", "2",
        "--learning-rate", str(candidate.learning_rate),
        "--beta", "0.2",
        "--label-smoothing", "0.05",
        "--lora-rank", "4",
        "--lora-alpha", "8",
    ]
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return f"complete {candidate.name}/target_p{serial:02d}"


def train(args: argparse.Namespace, env: dict[str, str]) -> None:
    tasks = [(candidate, serial) for candidate in CANDIDATES for serial in (1, 2)]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(train_one, args, env, candidate, serial): (candidate, serial)
            for candidate, serial in tasks
        }
        for future in as_completed(futures):
            candidate, serial = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(f"Training failed: {candidate.name}/{serial}") from error


def run(command: list[str], env: dict[str, str], check: bool = True) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=check)


def evaluate(args: argparse.Namespace, env: dict[str, str]) -> dict:
    specs = [
        (candidate, serial, "magenta" if serial == 1 else "cyan")
        for candidate in CANDIDATES for serial in (1, 2)
    ]
    probe_root = args.output_dir / "probes"
    visual_dir = probe_root / "visual_fidelity"
    visual_generations = visual_dir / "attacker_view" / "forced_choice_generations.jsonl"
    if not visual_generations.is_file():
        command = [
            sys.executable,
            str(HERE / "probe_glm.py"),
            "--model", str(args.model),
            "--modality", "vision_conditioned",
            "--probe-bank", str(DATA / "private_fidelity_probe_bank.json"),
            "--output-dir", str(visual_dir),
            "--max-new-tokens", "8",
        ]
        for candidate, serial, _ in specs:
            command.extend([
                "--adapter",
                f"target_{candidate.name}{serial}={adapter_path(args.output_dir, candidate, serial)}",
            ])
        run(command, env)

    fidelity_output = args.output_dir / "visual_fidelity.json"
    command = [
        sys.executable,
        "-m", "Text_only.evaluate_update_fidelity",
        "--generations", str(visual_generations),
        "--output", str(fidelity_output),
        "--reference-preference", "magenta",
        "--min-improvement", "0.0",
        "--min-parsed-rate", "0.90",
        "--min-parsed-rate", "0.90",
    ]
    for candidate, serial, preference in specs:
        command.extend(["--target-label", f"target_{candidate.name}{serial}={preference}"])
    run(command, env, check=False)
    fidelity = json.loads(fidelity_output.read_text(encoding="utf-8"))

    text_dir = probe_root / "text_only"
    text_generations = text_dir / "attacker_view" / "forced_choice_generations.jsonl"
    if not text_generations.is_file():
        command = [
            sys.executable,
            str(HERE / "probe_glm.py"),
            "--model", str(args.model),
            "--modality", "text_only",
            "--concepts", str(HERE / "prompts" / "magenta_cyan_known_preference_concepts.json"),
            "--output-dir", str(text_dir),
            "--max-new-tokens", "8",
        ]
        for candidate, serial, _ in specs:
            command.extend([
                "--adapter",
                f"target_{candidate.name}{serial}={adapter_path(args.output_dir, candidate, serial)}",
            ])
        run(command, env)

    grouped = group_choice_generations(read_choice_generations(text_generations))
    queries = available_choice_queries(grouped)
    fidelity_by_id = {row["anonymous_model_id"]: row for row in fidelity["per_model"]}
    rows = []
    for candidate in CANDIDATES:
        ids = [f"target_{candidate.name}1", f"target_{candidate.name}2"]
        _, observability = build_choice_features(grouped, ids, queries, access="paired")
        magenta_delta = observability[ids[0]]["mean_reference_choice_rate_delta"]
        cyan_delta = observability[ids[1]]["mean_reference_choice_rate_delta"]
        rows.append(
            {
                "candidate": candidate.name,
                "learning_rate": candidate.learning_rate,
                "max_steps": candidate.max_steps,
                "visual_updates_passed": sum(fidelity_by_id[value]["passed"] for value in ids),
                "visual_mean_accuracy": float(np.mean([fidelity_by_id[value]["post_preference_accuracy"] for value in ids])),
                "visual_mean_parse_rate": float(np.mean([fidelity_by_id[value]["parsed_response_rate"] for value in ids])),
                "text_magenta_delta": magenta_delta,
                "text_cyan_delta": cyan_delta,
                "text_pair_separation": magenta_delta - cyan_delta,
                "text_directionally_correct": magenta_delta > 0 and cyan_delta < 0,
                "text_mean_response_change": float(np.mean([observability[value]["exact_response_change_rate"] for value in ids])),
                "trainable_parameters": json.loads(
                    (adapter_path(args.output_dir, candidate, 1) / "private_training_metadata.json").read_text(encoding="utf-8")
                )["trainable_parameters"],
            }
        )
    result = {
        "model": "GLM-4.1V-9B-Base",
        "selection_boundary": "shadow_only_train2017",
        "target_data_opened": False,
        "objective": "pure_sigmoid_dpo_joint_vlm",
        "candidates": rows,
    }
    report = args.output_dir / "candidate_report.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    if args.phase in {"train", "all"}:
        train(args, env)
    if args.phase in {"evaluate", "all"}:
        evaluate(args, env)


if __name__ == "__main__":
    main()
