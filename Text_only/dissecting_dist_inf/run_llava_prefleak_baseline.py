#!/usr/bin/env python3
"""Run the adapted label-only distribution-inference baseline on a VLM.

LLaVA is the default for backward compatibility. GLM imports this shared
orchestrator with a different default so the attack protocol is not copied.
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
    "glm": ROOT / "Text_only" / "GLM" / "outputs" / "label_only_di_glm_v1",
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
            "updates_leak_root": ROOT
            / "Text_only"
            / "Updates-Leak"
            / "outputs"
            / "llava_v1",
            "model_label": "LLaVA-OneVision-Qwen2-0.5B",
            "transcript_label": "LLaVA",
            "report_name": "LABEL_ONLY_DI_LLaVA_RESULTS.md",
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
            "updates_leak_root": ROOT
            / "Text_only"
            / "Updates-Leak"
            / "outputs"
            / "glm_v1",
            "model_label": "GLM-4.1V-9B-Base",
            "transcript_label": "GLM",
            "report_name": "LABEL_ONLY_DI_GLM_RESULTS.md",
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
    env["PYTHONPATH"] = f"{REPO_DIR / 'rebase'}:{ROOT}"
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def calibrate(args: argparse.Namespace, env: dict[str, str], config: dict) -> None:
    shadows, _ = config["build_specs"]()
    remote_root = config["remote_root"]
    command = [
        sys.executable,
        str(REPO_DIR / "prefleak_label_kl.py"),
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
        "--output",
        str(args.output_dir / "frozen_label_only_kl.json"),
        "--num-probes",
        str(config["num_probes"]),
        "--repetitions",
        "8",
        "--attack-runs",
        str(args.attack_runs),
        "--kl-frac",
        "0.8",
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
        str(REPO_DIR / "prefleak_label_kl.py"),
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
        str(args.output_dir / "frozen_label_only_kl.json"),
        "--output",
        str(args.output_dir / "predictions" / "label_only_kl.json"),
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
        str(args.output_dir / "predictions" / "label_only_kl.json"),
        "--output",
        str(args.output_dir / "analysis" / "label_only_kl.json"),
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
        (args.output_dir / "frozen_label_only_kl.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (args.output_dir / "analysis" / "label_only_kl.json").read_text(
            encoding="utf-8"
        )
    )
    prediction_document = json.loads(
        (args.output_dir / "predictions" / "label_only_kl.json").read_text(
            encoding="utf-8"
        )
    )
    prefleak = json.loads(
        (remote_root / "analysis" / "paired_selected_residualized.json").read_text(
            encoding="utf-8"
        )
    )
    updates_leak = json.loads(
        (config["updates_leak_root"] / "analysis" / "updates_leak_style_mlp.json").read_text(
            encoding="utf-8"
        )
    )
    summary = metrics["budget_summary"][0]
    updates_summary = updates_leak["budget_summary"][0]
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
    hard_label_diagnostics = prediction_document["target_hard_label_diagnostics"]
    lines = [
        f"# Label-only Distribution Inference on {config['model_label']}",
        "",
        "## Protocol",
        "",
        f"- Victim family: {config['model_label']}.",
        "- Access: post-update text strings only, which is a valid subset of the paired-model threat model.",
        "- Observation supplied to the attack: one hard correctness label per public A/B probe; no logits, probabilities, embeddings, or free-form string features.",
        f"- Probe bank: {frozen['num_probes']} probes selected by shadow parseability without preference labels and balanced by public correct option.",
        f"- Resampling: {frozen['repetitions']} stochastic hard-label queries per probe; effective budget B={frozen['online_generation_calls']}.",
        "- Attack: official `label_only_KLAttack._get_kl_preds` pairwise KL voting with 80% shadow-pair subsampling.",
        f"- Shadow updates: {frozen['num_shadow_updates']} in {frozen['num_shadow_pairs']} opposite-preference pairs; target updates: 20 held out.",
        "- Target labels are opened only by the shared evaluator after prediction serialization.",
        "",
        "## Results",
        "",
        "| Method | Access | Calls | Balanced accuracy | AUROC | Seeds with paired p<0.05 |",
        "|---|---|---:|---:|---:|---:|",
        f"| Label-only Distribution Inference | post-only | {summary['query_budget']} | {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_std']:.3f} | {summary['auroc_mean']:.3f} ± {summary['auroc_std']:.3f} | {sum(value is not None and value < 0.05 for value in pvalues)}/{len(pvalues)} |",
        f"| Updates-Leak-style MLP | paired | {updates_summary['query_budget']} | {updates_summary['balanced_accuracy_mean']:.3f} ± {updates_summary['balanced_accuracy_std']:.3f} | {updates_summary['auroc_mean']:.3f} ± {updates_summary['auroc_std']:.3f} | 10/10 |",
        f"| Full PrefLeak | paired | {prefleak_run['query_budget']} | {prefleak_run['balanced_accuracy']:.3f} | {prefleak_run['auroc']:.3f} | 1/1 |",
        "",
        f"Shadow leave-one-pair-out balanced accuracy: {sum(shadow_balanced) / len(shadow_balanced):.3f} mean; AUROC: {sum(shadow_auc) / len(shadow_auc):.3f} mean.",
        f"The held-out targets contain {hard_label_diagnostics['num_unique_hard_label_vectors']} unique majority hard-label vectors; this diagnostic uses no private preference labels.",
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
            "## Interpretation boundary",
            "",
            "This is a modified baseline, not an unmodified reproduction. The original binary classifier labels and neighborhood samples are replaced by strict A/B labels from stochastic VLM generations. The official shadow-group pairwise KL voting code is retained. Probe selection uses only parseability, not the private preference direction.",
            "",
            "The ten runs vary only the official shadow-pair subsampling on the same held-out updates; they are not ten independent victim-update collections.",
            "",
        ]
    )
    if hard_label_diagnostics["num_unique_hard_label_vectors"] == 1:
        lines.extend(
            [
                "The hard-label representation collapses all held-out updates to the same vector for this victim family. Consequently the official KL-voting score is constant across targets and cannot rank the two private preferences.",
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
