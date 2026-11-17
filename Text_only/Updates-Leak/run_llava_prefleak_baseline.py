#!/usr/bin/env python3
"""Run the Updates-Leak-style MLP baseline on frozen VLM transcripts.

LLaVA remains the default for backward compatibility.  The GLM entry point
imports this module with a different default, so both model families share the
same calibration, inference, evaluation, and reporting implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
ROOT = REPO_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBE_BANK = ROOT / "Text_only" / "prompts" / "remote_text_candidate_bank.json"
DEFAULT_OUTPUTS = {
    "llava": REPO_DIR / "outputs" / "llava_v1",
    "glm": REPO_DIR / "outputs" / "glm_v1",
}


def experiment_config(model_family: str) -> dict:
    if model_family == "llava":
        from Text_only.run_joint_vlm_experiment import build_specs

        return {
            "build_specs": build_specs,
            "remote_root": ROOT
            / "Text_only"
            / "outputs"
            / "remote_probe_llava_v1"
            / "main",
            "adapter_root": ROOT
            / "Text_only"
            / "outputs"
            / "joint_vlm_paper_v1"
            / "main",
            "model_label": "LLaVA-OneVision-Qwen2-0.5B",
            "report_name": "UPDATES_LEAK_STYLE_LLaVA_RESULTS.md",
            "num_probes": 24,
        }
    if model_family == "glm":
        from Text_only.GLM.run_glm_experiment import build_specs

        return {
            "build_specs": build_specs,
            "remote_root": ROOT
            / "Text_only"
            / "GLM"
            / "outputs"
            / "remote_probe_glm_v1"
            / "main",
            "adapter_root": ROOT
            / "Text_only"
            / "GLM"
            / "outputs"
            / "main_v1",
            "model_label": "GLM-4.1V-9B-Base",
            "report_name": "UPDATES_LEAK_STYLE_GLM_RESULTS.md",
            "num_probes": 12,
        }
    raise ValueError(f"Unsupported model family: {model_family}")


def parse_args(default_model_family: str = "llava") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("calibrate", "infer", "evaluate", "report", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--model-family",
        choices=("llava", "glm"),
        default=default_model_family,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attack-runs", type=int, default=10)
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def calibrate(args: argparse.Namespace, env: dict[str, str], config: dict) -> None:
    shadows, _ = config["build_specs"]()
    remote_root = config["remote_root"]
    command = [
        sys.executable,
        str(REPO_DIR / "prefleak_mlp.py"),
        "calibrate",
        "--shadow-generations",
        str(
            remote_root
            / "probes"
            / "shadow_candidates"
            / "attacker_view"
            / "remote_generations.jsonl"
        ),
        "--probe-bank",
        str(PROBE_BANK),
        "--output-dir",
        str(args.output_dir),
        "--num-probes",
        str(config["num_probes"]),
        "--repetitions",
        "4",
        "--attack-runs",
        str(args.attack_runs),
        "--epochs",
        "50",
        "--seed",
        "2020",
    ]
    for spec in shadows:
        command.extend(["--shadow-label", f"{spec.model_id}={spec.preference}"])
        command.extend(["--shadow-pair", f"{spec.model_id}={spec.pair_id}"])
    run(command, env)


def infer(args: argparse.Namespace, env: dict[str, str], config: dict) -> None:
    _, targets = config["build_specs"]()
    remote_root = config["remote_root"]
    command = [
        sys.executable,
        str(REPO_DIR / "prefleak_mlp.py"),
        "infer",
        "--target-generations",
        str(
            remote_root
            / "probes"
            / "target_candidates"
            / "attacker_view"
            / "remote_generations.jsonl"
        ),
        "--probe-bank",
        str(PROBE_BANK),
        "--frozen-attack",
        str(args.output_dir / "frozen_updates_leak_mlp.json"),
        "--weights",
        str(args.output_dir / "frozen_updates_leak_mlp.pt"),
        "--output",
        str(args.output_dir / "predictions" / "updates_leak_style_mlp.json"),
    ]
    for spec in targets:
        command.extend(["--target-id", spec.model_id])
    run(command, env)


def evaluate(args: argparse.Namespace, env: dict[str, str], config: dict) -> None:
    _, targets = config["build_specs"]()
    adapter_root = config["adapter_root"]
    fidelity = json.loads(
        (adapter_root / "analysis" / "visual_fidelity_target.json").read_text(
            encoding="utf-8"
        )
    )
    valid_ids = {
        row["anonymous_model_id"] for row in fidelity["per_model"] if row["passed"]
    }
    command = [
        sys.executable,
        "-m",
        "Text_only.evaluate_known_preference",
        "--predictions",
        str(args.output_dir / "predictions" / "updates_leak_style_mlp.json"),
        "--output",
        str(args.output_dir / "analysis" / "updates_leak_style_mlp.json"),
    ]
    for spec in targets:
        command.extend(["--target-label", f"{spec.model_id}={spec.preference}"])
        command.extend(["--target-pair", f"{spec.model_id}={spec.pair_id}"])
    if {spec.preference for spec in targets if spec.model_id in valid_ids} == {
        "cyan",
        "magenta",
    }:
        for model_id in sorted(valid_ids):
            command.extend(["--valid-target-id", model_id])
    run(command, env)


def write_report(args: argparse.Namespace, config: dict) -> Path:
    remote_root = config["remote_root"]
    frozen = json.loads(
        (args.output_dir / "frozen_updates_leak_mlp.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (args.output_dir / "analysis" / "updates_leak_style_mlp.json").read_text(
            encoding="utf-8"
        )
    )
    prefleak = json.loads(
        (remote_root / "analysis" / "paired_selected_residualized.json").read_text(
            encoding="utf-8"
        )
    )
    summary = metrics["budget_summary"][0]
    prefleak_run = prefleak["individual_runs"][0]
    pvalues = [
        run["matched_pair_randomization_pvalue_one_sided"]
        for run in metrics["individual_runs"]
    ]
    shadow_balanced = [
        run["shadow_oof_balanced_accuracy"] for run in frozen["runs"]
    ]
    shadow_auc = [run["shadow_oof_auroc"] for run in frozen["runs"]]
    conditional = metrics.get("fidelity_conditioned", {}).get("budget_summary", [])
    lines = [
        f"# Updates-Leak-style MLP on {config['model_label']}",
        "",
        "## Protocol",
        "",
        f"- Victim family: {config['model_label']}.",
        "- Access: paired pre/post, text-only generated strings.",
        f"- The run reuses the existing frozen {args.model_family.upper()} transcripts; the effective attack view is restricted to the declared probes and repetitions, so no VLM re-query is needed.",
        f"- Fixed public probing set: first {frozen['num_probes']} probes; no shadow-label probe selection.",
        f"- Repetitions: {frozen['repetitions_per_version']} per version and probe; effective budget B={frozen['online_generation_calls']}.",
        "- Attack vector: flattened bounded string-feature differences, without pair-centering, stable selection, or nuisance residualization.",
        "- Classifier: official Updates-Leak `labelPredMNIST` architecture with a two-class output.",
        f"- Evaluation repeats: {len(frozen['runs'])} deterministic MLP seeds; thresholds use leave-one-shadow-pair-out predictions only.",
        "- Target labels are opened only by the shared evaluator after prediction serialization.",
        "",
        "## Results",
        "",
        "| Method | Calls | Balanced accuracy | AUROC | Seeds with paired p<0.05 |",
        "|---|---:|---:|---:|---:|",
        f"| Updates-Leak-style MLP | {summary['query_budget']} | {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_std']:.3f} | {summary['auroc_mean']:.3f} ± {summary['auroc_std']:.3f} | {sum(value is not None and value < 0.05 for value in pvalues)}/{len(pvalues)} |",
        f"| Full PrefLeak | {prefleak_run['query_budget']} | {prefleak_run['balanced_accuracy']:.3f} | {prefleak_run['auroc']:.3f} | 1/1 |",
        "",
        f"Shadow leave-one-pair-out balanced accuracy: {sum(shadow_balanced) / len(shadow_balanced):.3f} mean; AUROC: {sum(shadow_auc) / len(shadow_auc):.3f} mean.",
    ]
    if conditional:
        row = conditional[0]
        lines.extend(
            [
                "",
                f"Fidelity-conditioned sensitivity analysis: balanced accuracy {row['balanced_accuracy_mean']:.3f} ± {row['balanced_accuracy_std']:.3f}; AUROC {row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a modified baseline, not an unmodified reproduction: MNIST image probes and posterior vectors are replaced by PrefLeak's text-only bounded string features, while the fixed-probe output-difference MLP principle and official attack network are retained.",
            "",
        ]
    )
    output = args.output_dir / config["report_name"]
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(output)}, indent=2))
    return output


def main(default_model_family: str = "llava") -> None:
    args = parse_args(default_model_family)
    config = experiment_config(args.model_family)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUTS[args.model_family]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    if args.phase in {"calibrate", "all"}:
        calibrate(args, env, config)
    if args.phase in {"infer", "all"}:
        infer(args, env, config)
    if args.phase in {"evaluate", "all"}:
        evaluate(args, env, config)
    if args.phase in {"report", "all"}:
        write_report(args, config)


if __name__ == "__main__":
    main()
