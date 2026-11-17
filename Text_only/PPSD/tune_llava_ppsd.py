#!/usr/bin/env python3
"""Shadow-only update-strength tuning after the transferred PPSD config failed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.PPSD.protocol import DATA_ROOT, PREFERENCES
from Text_only.train_llava_dpo import DEFAULT_MODEL


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "shadow_tuning_v2"


@dataclass(frozen=True)
class Candidate:
    name: str
    learning_rate: float
    max_steps: int
    beta: float = 0.2
    label_smoothing: float = 0.0


CANDIDATES = (
    Candidate("lr2e4_s200", 2e-4, 200),
    Candidate("lr5e4_s200", 5e-4, 200),
    Candidate("lr1e3_s200", 1e-3, 200),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=tuple(candidate.name for candidate in CANDIDATES),
        help="Run only the named candidate; repeat to select multiple.",
    )
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


def probe_model_id(candidate: Candidate, serial: int) -> str:
    """Anonymous IDs must match the shared target_[a-z0-9]+ validator."""
    return f"target_p{CANDIDATES.index(candidate) + 1}{serial}"


def selected_candidates(args: argparse.Namespace) -> tuple[Candidate, ...]:
    if not args.candidate:
        return CANDIDATES
    requested = set(args.candidate)
    return tuple(candidate for candidate in CANDIDATES if candidate.name in requested)


def train_one(args: argparse.Namespace, env: dict[str, str], candidate: Candidate, serial: int) -> str:
    destination = adapter_path(args.output_dir, candidate, serial)
    if (destination / "adapter_config.json").is_file():
        return f"reuse {candidate.name}/target_p{serial:02d}"
    log = args.output_dir / candidate.name / "logs" / f"target_p{serial:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(PROJECT_ROOT / "Text_only" / "train_llava_dpo.py"),
        "--model", str(args.model),
        "--train-manifest", str(DATA_ROOT / "private_updates" / f"target_b{serial:03d}.jsonl"),
        "--target-id", f"target_p{serial:02d}",
        "--output-dir", str(destination),
        "--seed", "30401",
        "--max-steps", str(candidate.max_steps),
        "--per-device-train-batch-size", "4",
        "--gradient-accumulation-steps", "1",
        "--learning-rate", str(candidate.learning_rate),
        "--beta", str(candidate.beta),
        "--label-smoothing", str(candidate.label_smoothing),
        "--lora-rank", "8",
        "--lora-alpha", "16",
        "--lora-dropout", "0.05",
    ]
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=PROJECT_ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=True,
        )
    return f"complete {candidate.name}/target_p{serial:02d}"


def train(args: argparse.Namespace, env: dict[str, str]) -> None:
    tasks = [
        (candidate, serial)
        for candidate in selected_candidates(args)
        for serial in (1, 2)
    ]
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
                raise RuntimeError(f"Tuning failed: {candidate.name}/{serial}") from error


def run(command: list[str], env: dict[str, str], check: bool = True) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=check)


def evaluate(args: argparse.Namespace, env: dict[str, str]) -> dict:
    specs = [
        (candidate, serial, PREFERENCES[serial - 1])
        for candidate in selected_candidates(args)
        for serial in (1, 2)
    ]
    directory = args.output_dir / "probes" / "visual_fidelity"
    generations = directory / "attacker_view" / "forced_choice_generations.jsonl"
    if not generations.is_file():
        command = [
            sys.executable, str(PROJECT_ROOT / "Text_only" / "probe_forced_choice.py"),
            "--model", str(args.model),
            "--modality", "vision_conditioned",
            "--probe-bank", str(DATA_ROOT / "private_fidelity_probe_bank.json"),
            "--visual-prompt-style", "training",
            "--output-dir", str(directory),
            "--max-new-tokens", "12",
        ]
        for candidate, serial, _ in specs:
            model_id = probe_model_id(candidate, serial)
            command.extend(["--adapter", f"{model_id}={adapter_path(args.output_dir, candidate, serial)}"])
        run(command, env)

    fidelity_path = args.output_dir / "visual_fidelity.json"
    command = [
        sys.executable, "-m", "Text_only.evaluate_update_fidelity",
        "--generations", str(generations),
        "--output", str(fidelity_path),
        "--reference-preference", PREFERENCES[0],
        "--min-improvement", "0.2",
        "--min-parsed-rate", "0.95",
    ]
    for candidate, serial, preference in specs:
        command.extend(["--target-label", f"{probe_model_id(candidate, serial)}={preference}"])
    run(command, env, check=False)
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    by_id = {row["anonymous_model_id"]: row for row in fidelity["per_model"]}
    rows = []
    for candidate in selected_candidates(args):
        ids = [probe_model_id(candidate, serial) for serial in (1, 2)]
        logs = []
        for serial in (1, 2):
            state = json.loads(
                (adapter_path(args.output_dir, candidate, serial) / "trainer_state.json").read_text(encoding="utf-8")
            )
            logs.append(state["log_history"][-1])
        rows.append(
            {
                **asdict(candidate),
                "visual_updates_passed": sum(by_id[model_id]["passed"] for model_id in ids),
                "visual_mean_accuracy": float(np.mean([by_id[model_id]["post_preference_accuracy"] for model_id in ids])),
                "visual_mean_improvement": float(np.mean([by_id[model_id]["preference_accuracy_improvement"] for model_id in ids])),
                "visual_min_accuracy": float(min(by_id[model_id]["post_preference_accuracy"] for model_id in ids)),
                "train_loss_mean": float(np.mean([row["train_loss"] for row in logs])),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["visual_updates_passed"],
            row["visual_min_accuracy"],
            row["visual_mean_accuracy"],
            -row["learning_rate"],
        ),
    )
    result = {
        "selection_boundary": "A1_shadow_pair_and_disjoint_fidelity_only",
        "selection_reason": "transferred_COCO_configuration_failed_0_of_12_shadow_fidelity",
        "target_metrics_used_for_selection": False,
        "objective": "pure_sigmoid_dpo_joint_vlm",
        "candidates": rows,
        "selected_candidate": selected["name"],
    }
    (args.output_dir / "candidate_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    selected_config = {
        "selection_boundary": result["selection_boundary"],
        "selected_candidate": selected["name"],
        "selection_rule": "maximize passed updates, minimum fidelity, mean fidelity, then lower learning rate",
        "selection_evidence": {
            "visual_updates_passed": selected["visual_updates_passed"],
            "visual_updates_total": 2,
            "visual_mean_accuracy": selected["visual_mean_accuracy"],
            "visual_mean_improvement": selected["visual_mean_improvement"],
            "target_data_opened_for_selection": False,
        },
        "learning_rate": selected["learning_rate"],
        "max_steps": selected["max_steps"],
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "beta": selected["beta"],
        "label_smoothing": selected["label_smoothing"],
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "ppsd_transfer": {
            "source_frozen_config": str((args.output_dir / "candidate_report.json").resolve()),
            "rule": "PPSD_shadow_only_update_strength_selection",
            "ppsd_target_data_opened": False,
        },
    }
    (args.output_dir / "selected_frozen_config.json").write_text(
        json.dumps(selected_config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    if args.phase in {"train", "all"}:
        train(args, env)
    if args.phase in {"evaluate", "all"}:
        evaluate(args, env)


if __name__ == "__main__":
    main()

