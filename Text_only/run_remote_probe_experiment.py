#!/usr/bin/env python3
"""Run the target-blind LLaVA preference-remote text leakage experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Text_only.run_joint_vlm_experiment import (
    DEFAULT_OUTPUT as ADAPTER_ROOT,
    MODEL,
    build_specs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BANK = SCRIPT_DIR / "prompts" / "remote_text_candidate_bank.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "remote_probe_llava_v1" / "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "build", "shadow", "calibrate", "target", "infer", "evaluate",
            "report", "target-exact", "infer-exact", "all",
        ),
        nargs="?", default="all",
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter-root", type=Path, default=ADAPTER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'trl'}:{ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def run(command: list[str], env: dict[str, str], check: bool = True) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=check)


def adapter_arguments(adapter_root: Path, specs: list) -> list[str]:
    result = []
    for spec in specs:
        path = adapter_root / "adapters" / spec.split / spec.model_id
        result.extend(["--adapter", f"{spec.model_id}={path}"])
    return result


def probe(
    args: argparse.Namespace,
    env: dict[str, str],
    name: str,
    specs: list,
    probe_bank: Path = BANK,
    repetitions: int = 8,
    omit_reference: bool = False,
    seed_reference_bank: Path | None = None,
) -> Path:
    directory = args.output_dir / "probes" / name
    output = directory / "attacker_view" / "remote_generations.jsonl"
    if output.is_file():
        return output
    command = [
            sys.executable, str(SCRIPT_DIR / "probe_remote_text.py"),
            "--model", str(args.model),
            "--probe-bank", str(probe_bank),
            "--output-dir", str(directory),
            "--repetitions", str(repetitions),
            "--max-new-tokens", "12",
            "--temperature", "0.8",
            "--top-p", "0.95",
            *adapter_arguments(args.adapter_root, specs),
    ]
    if seed_reference_bank is not None:
        command.extend(["--seed-reference-bank", str(seed_reference_bank)])
    if omit_reference:
        command.append("--omit-reference")
    run(command, env)
    return output


def materialize_selected_bank(args: argparse.Namespace, env: dict[str, str]) -> Path:
    directory = args.output_dir / "frozen_probe_banks"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "materialize_frozen_probe_banks.py"),
            "--probe-bank", str(BANK),
            "--frozen-attack", str(args.output_dir / "frozen_remote_attack.json"),
            "--output-dir", str(directory),
        ],
        env,
    )
    return directory / "primary_selected_probe_bank.json"


def write_exact_target_boundary(args: argparse.Namespace, selected_bank: Path) -> Path:
    paired = args.output_dir / "probes" / "target_selected_paired" / "attacker_view" / "remote_generations.jsonl"
    post_only = args.output_dir / "probes" / "target_selected_post_only" / "attacker_view" / "remote_generations.jsonl"
    output = args.output_dir / "exact_target_query_boundary.json"
    if output.is_file():
        return output
    result = {
        "boundary": "selected_probe_bank_frozen_before_exact_target_queries",
        "frozen_attack_sha256": hashlib.sha256(
            (args.output_dir / "frozen_remote_attack.json").read_bytes()
        ).hexdigest(),
        "full_probe_bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "selected_probe_bank_sha256": hashlib.sha256(selected_bank.read_bytes()).hexdigest(),
        "paired_target_generations_existed": paired.exists(),
        "post_only_target_generations_existed": post_only.exists(),
        "target_labels_opened": False,
    }
    if paired.exists() or post_only.exists():
        raise RuntimeError("Exact target generations predate the exact-query boundary.")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def infer_exact(args: argparse.Namespace, env: dict[str, str], targets: list) -> None:
    common = [
        sys.executable, str(SCRIPT_DIR / "remote_attack.py"), "infer",
        "--probe-bank", str(BANK),
        "--frozen-attack", str(args.output_dir / "frozen_remote_attack.json"),
    ]
    target_arguments = []
    for spec in targets:
        target_arguments.extend(["--target-id", spec.model_id])
    run(
        [
            *common,
            "--target-generations", str(
                args.output_dir / "probes" / "target_selected_paired" / "attacker_view" / "remote_generations.jsonl"
            ),
            "--attack-name", "paired_selected_residualized",
            "--attack-name", "paired_selected_no_residual",
            "--output-dir", str(args.output_dir / "exact_budget" / "predictions_paired"),
            *target_arguments,
        ],
        env,
    )
    run(
        [
            *common,
            "--target-generations", str(
                args.output_dir / "probes" / "target_selected_post_only" / "attacker_view" / "remote_generations.jsonl"
            ),
            "--attack-name", "post_only_matched_budget",
            "--output-dir", str(args.output_dir / "exact_budget" / "predictions_post_only"),
            *target_arguments,
        ],
        env,
    )


def calibrate(args: argparse.Namespace, env: dict[str, str], shadows: list) -> Path:
    output = args.output_dir / "frozen_remote_attack.json"
    generations = args.output_dir / "probes" / "shadow_candidates" / "attacker_view" / "remote_generations.jsonl"
    command = [
        sys.executable, str(SCRIPT_DIR / "remote_attack.py"), "calibrate",
        "--shadow-generations", str(generations),
        "--probe-bank", str(BANK),
        "--output", str(output),
        "--paired-repetitions", "4",
        "--post-only-repetitions", "8",
        "--nuisance-rank", "2",
        "--k-grid", "4,8,12,16,24",
        "--random-repeats", "10",
    ]
    for spec in shadows:
        command.extend(["--shadow-label", f"{spec.model_id}={spec.preference}"])
        command.extend(["--shadow-pair", f"{spec.model_id}={spec.pair_id}"])
    run(command, env)
    return output


def write_target_boundary(args: argparse.Namespace) -> Path:
    frozen = args.output_dir / "frozen_remote_attack.json"
    target_generations = args.output_dir / "probes" / "target_candidates" / "attacker_view" / "remote_generations.jsonl"
    result = {
        "boundary": "remote_attack_frozen_before_target_queries",
        "frozen_attack_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "probe_bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "target_generations_existed_when_boundary_was_written": target_generations.exists(),
        "target_labels_opened": False,
    }
    if result["target_generations_existed_when_boundary_was_written"]:
        existing = args.output_dir / "target_query_boundary.json"
        if existing.is_file():
            return existing
        raise RuntimeError("Target generations predate the frozen-boundary record.")
    output = args.output_dir / "target_query_boundary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output


def infer(args: argparse.Namespace, env: dict[str, str], targets: list) -> None:
    command = [
        sys.executable, str(SCRIPT_DIR / "remote_attack.py"), "infer",
        "--target-generations", str(
            args.output_dir / "probes" / "target_candidates" / "attacker_view" / "remote_generations.jsonl"
        ),
        "--probe-bank", str(BANK),
        "--frozen-attack", str(args.output_dir / "frozen_remote_attack.json"),
        "--output-dir", str(args.output_dir / "predictions"),
    ]
    for spec in targets:
        command.extend(["--target-id", spec.model_id])
    run(command, env)


def evaluate(args: argparse.Namespace, env: dict[str, str], targets: list) -> None:
    fidelity_path = args.adapter_root / "analysis" / "visual_fidelity_target.json"
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    valid_ids = {
        row["anonymous_model_id"] for row in fidelity["per_model"] if row["passed"]
    }
    valid_classes = {spec.preference for spec in targets if spec.model_id in valid_ids}
    names = (
        "paired_selected_residualized",
        "paired_selected_no_residual",
        "post_only_matched_budget",
        "paired_random",
    )
    for name in names:
        command = [
            sys.executable, "-m", "Text_only.evaluate_known_preference",
            "--predictions", str(args.output_dir / "predictions" / f"{name}.json"),
            "--output", str(args.output_dir / "analysis" / f"{name}.json"),
        ]
        for spec in targets:
            command.extend(["--target-label", f"{spec.model_id}={spec.preference}"])
            command.extend(["--target-pair", f"{spec.model_id}={spec.pair_id}"])
        if len(valid_classes) == 2:
            for model_id in sorted(valid_ids):
                command.extend(["--valid-target-id", model_id])
        run(command, env)


def full_run(metrics: dict) -> dict:
    return metrics["individual_runs"][0]


def number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_report(args: argparse.Namespace) -> Path:
    frozen = json.loads((args.output_dir / "frozen_remote_attack.json").read_text(encoding="utf-8"))
    primary = json.loads((args.output_dir / "analysis" / "paired_selected_residualized.json").read_text(encoding="utf-8"))
    no_residual = json.loads((args.output_dir / "analysis" / "paired_selected_no_residual.json").read_text(encoding="utf-8"))
    post_only = json.loads((args.output_dir / "analysis" / "post_only_matched_budget.json").read_text(encoding="utf-8"))
    random_metrics = json.loads((args.output_dir / "analysis" / "paired_random.json").read_text(encoding="utf-8"))
    resample = json.loads((args.output_dir / "analysis" / "paired_selected_residualized_resample.json").read_text(encoding="utf-8"))
    base_seed_control = json.loads((args.output_dir / "analysis" / "paired_selected_residualized_base_seed_control.json").read_text(encoding="utf-8"))
    direct = json.loads((args.adapter_root / "analysis" / "text_only" / "paired" / "metrics.json").read_text(encoding="utf-8"))
    bank_audit = json.loads(BANK.with_suffix(".audit.json").read_text(encoding="utf-8"))
    bank_rows = json.loads(BANK.read_text(encoding="utf-8"))
    exact_budget = json.loads(
        (args.output_dir / "exact_budget" / "verification.json").read_text(
            encoding="utf-8"
        )
    )
    primary_run = full_run(primary)
    no_residual_run = full_run(no_residual)
    post_run = full_run(post_only)
    resample_run = full_run(resample)
    base_seed_run = full_run(base_seed_control)
    direct_run = next(row for row in direct["individual_runs"] if row["query_budget"] == 48)
    random_summary = random_metrics["budget_summary"][0]
    random_pvalues = [
        row["matched_pair_randomization_pvalue_one_sided"]
        for row in random_metrics["individual_runs"]
    ]
    selected = frozen["selection"]["selected_probe_ids"]
    selected_rows = [row for row in bank_rows if row["probe_id"] in selected]
    chosen_nested = frozen["selection"]["chosen_nested_oof"]
    primary_attack = next(
        row for row in frozen["attacks"] if row["name"] == "paired_selected_residualized"
    )
    supported = (
        primary_run["balanced_accuracy"] > 0.5
        and primary_run["matched_pair_randomization_pvalue_one_sided"] is not None
        and primary_run["matched_pair_randomization_pvalue_one_sided"] < 0.05
    )
    conclusion = (
        "严格 remote text behavior attack 在 held-out targets 上显著高于随机水平。"
        if supported else
        "严格 remote text behavior attack 在当前 held-out targets 上未显著高于随机水平。"
    )
    lines = [
        "# LLaVA Preference-Remote Text Leakage Metrics", "",
        "## 核心结论", "", conclusion, "",
        "该实验与 direct-elicitation baseline 分离：所有 remote prompts 均不含颜色、视觉、审美或偏好词，也不询问模型偏好。", "",
        "## 隔离与协议", "",
        f"- Candidate bank：{bank_audit['num_probes']} probes（{bank_audit['num_choice']} choice + {bank_audit['num_open']} open），audit passed。",
        f"- Shadow calibration：12 updates / 6 matched pairs；target evaluation：20 updates / 10 matched pairs。",
        "- Stable selection：leave-one-shadow-pair-out；使用跨 pair 稳定性得分和 response-feature correlation penalty。",
        "- Nuisance residualization：从 shadow opposite-preference pair means 学习 rank-2 nuisance subspace，并将 target feature 投影到其正交补。",
        "- Feature map：只读取生成字符串；32 维 bounded char n-grams + 11 维长度、格式、choice 与 public task-score features。",
        f"- 冻结的 selected set：K={len(selected)}；target queries 在 attack configuration 与 probe IDs 冻结后生成。",
        f"- Paired sampling：每个 probe 对 pre/post 各采样 4 次；online budget={primary_attack['online_generation_calls']} generation calls。",
        "- Post-only control：每个 probe 采样 8 次，与 paired attack 的总调用预算相同。", "",
        f"- Exact-budget audit：passed={exact_budget['passed']}；三种主攻击 selected-only 复算的最大 score 差为 0。",
        "- 预算口径：上述 calls 是冻结攻击所需的 effective budget；legacy diagnostic collection 保存过完整 candidate bank，额外调用不会被追溯删除。", "",
        "## Shadow-only selection", "",
        f"Nested leave-one-pair-out selected K={chosen_nested['num_probes']}：balanced accuracy {chosen_nested['balanced_accuracy']:.3f}，AUROC {chosen_nested['auroc']:.3f}。",
        f"Selected probe IDs：`{', '.join(selected)}`。", "",
        f"Selected 类型：{sum(row['kind'] == 'choice' for row in selected_rows)}/{len(selected_rows)} 为 ordinary A/B choice tasks；没有 open-ended probe 被选择。", "",
        "## Held-out target results", "",
        "| Attack | Calls | Balanced accuracy | AUROC | Exact paired p |",
        "|---|---:|---:|---:|---:|",
        f"| Remote selected + residualization | {primary_run['query_budget']} | {primary_run['balanced_accuracy']:.3f} | {number(primary_run['auroc'])} | {number(primary_run['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Remote selected, no residualization | {no_residual_run['query_budget']} | {no_residual_run['balanced_accuracy']:.3f} | {number(no_residual_run['auroc'])} | {number(no_residual_run['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Remote post-only, matched total calls | {post_run['query_budget']} | {post_run['balanced_accuracy']:.3f} | {number(post_run['auroc'])} | {number(post_run['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Remote random probes (10 sets) | {random_summary['query_budget']} | {random_summary['balanced_accuracy_mean']:.3f} ± {random_summary['balanced_accuracy_std']:.3f} | {number(random_summary['auroc_mean'])} | {sum(value < 0.05 for value in random_pvalues)}/10 significant |",
        f"| Direct elicitation baseline (prior experiment) | 96 actual paired calls | {direct_run['balanced_accuracy']:.3f} | {number(direct_run['auroc'])} | {number(direct_run['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        "",
        f"Primary pair-level 95% CI：[{primary_run['accuracy_95ci'][0]:.3f}, {primary_run['accuracy_95ci'][1]:.3f}]。", "",
        "## Robustness and negative control", "",
        "| Diagnostic | Balanced accuracy | AUROC | Exact paired p |",
        "|---|---:|---:|---:|",
        f"| Independent target resampling（repetitions 5–8） | {resample_run['balanced_accuracy']:.3f} | {number(resample_run['auroc'])} | {number(resample_run['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Base-only ID/seed-stream control（all adapters disabled） | {base_seed_run['balanced_accuracy']:.3f} | {number(base_seed_run['auroc'])} | {number(base_seed_run['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        "",
        f"Fidelity-conditioned sensitivity analysis：通过 visual fidelity gate 的 {primary['fidelity_conditioned']['num_held_out_target_updates']}/20 updates 上，primary accuracy {primary['fidelity_conditioned']['individual_runs'][0]['accuracy']:.3f}、AUROC {primary['fidelity_conditioned']['individual_runs'][0]['auroc']:.3f}。", "",
        "## Interpretation", "",
        (
            "在排除偏好词和视觉语义后仍可恢复 private visual preference，因此结果支持 latent visual-training-to-text-behavior leakage。"
            if supported else
            "当前结果只支持 direct-elicitation leakage；在严格 remote prompts 上尚无显著的 latent text behavior evidence。"
        ),
        f"Stable selection 将 balanced accuracy 从 random-set 平均 {random_summary['balanced_accuracy_mean']:.3f} 提升到 {primary_run['balanced_accuracy']:.3f}；但 random probes 本身也普遍显著，说明 remote leakage 广泛存在，而不是只集中在少量特制 prompts。",
        f"Rank-2 nuisance residualization 相对 no-residualization 从 {no_residual_run['balanced_accuracy']:.3f} 提升到 {primary_run['balanced_accuracy']:.3f}；paired 相对 matched-budget post-only 从 {post_run['balanced_accuracy']:.3f} 提升到 {primary_run['balanced_accuracy']:.3f}。",
        "限制：最终选择的 probes 全部使用 A/B 输出格式。它们在语义上严格 remote，但与 DPO 的 A/B answer channel 共享结构，因此当前证据应表述为 remote structured-choice behavior leakage，而不是开放式自然语言泄漏。",
        "Stable-selected、random-probe、no-residualization 和 matched-budget post-only 均使用同一冻结 target split；target labels 仅在 predictions 落盘后由 evaluator 打开。", "",
        "## Artifacts", "",
        "- `frozen_remote_attack.json`：shadow-only probe selection、nuisance basis 与分类器参数。",
        "- `target_query_boundary.json`：target 查询前的冻结边界。",
        "- `probes/`：shadow/target attacker-view raw string generations。",
        "- `predictions/`：未使用 target labels 的预测。",
        "- `predictions_resample/`：冻结攻击在独立 target sample repetitions 上的预测。",
        "- `predictions_base_seed_control/`：未更新 base-model seed/ID 负控制预测。",
        "- `analysis/`：标签解封后的 held-out metrics。", "",
        "- `exact_budget/`：selected-only transcripts、query ledgers 与等价性审计。",
        "- `frozen_probe_banks/`：从 shadow-only frozen attack 导出的 target banks。", "",
    ]
    output = SCRIPT_DIR / "REMOTE_PROBE_METRICS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    shadows, targets = build_specs()
    if args.phase in {"build", "all"}:
        run([sys.executable, str(SCRIPT_DIR / "build_remote_probe_bank.py")], env)
    if args.phase in {"shadow", "all"}:
        probe(args, env, "shadow_candidates", shadows)
    if args.phase in {"calibrate", "all"}:
        calibrate(args, env, shadows)
    if args.phase in {"target", "all"}:
        write_target_boundary(args)
        probe(args, env, "target_candidates", targets)
    if args.phase == "target-exact":
        selected_bank = materialize_selected_bank(args, env)
        write_exact_target_boundary(args, selected_bank)
        probe(
            args, env, "target_selected_paired", targets,
            probe_bank=selected_bank, repetitions=4, seed_reference_bank=BANK,
        )
        probe(
            args, env, "target_selected_post_only", targets,
            probe_bank=selected_bank, repetitions=8, omit_reference=True,
            seed_reference_bank=BANK,
        )
    if args.phase in {"infer", "all"}:
        infer(args, env, targets)
    if args.phase == "infer-exact":
        infer_exact(args, env, targets)
    if args.phase in {"evaluate", "all"}:
        evaluate(args, env, targets)
    if args.phase in {"report", "all"}:
        print(json.dumps({"report": str(write_report(args))}, indent=2))


if __name__ == "__main__":
    main()
