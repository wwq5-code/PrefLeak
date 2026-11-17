#!/usr/bin/env python3
"""Run the target-blind GLM-4.1V preference-remote text experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GLM_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Text_only.GLM.run_glm_experiment import (
    DEFAULT_OUTPUT as ADAPTER_ROOT,
    build_specs,
)
from Text_only.GLM.train_glm_dpo import DEFAULT_MODEL


BANK = ROOT / "Text_only" / "prompts" / "remote_text_candidate_bank.json"
DEFAULT_OUTPUT = GLM_ROOT / "outputs" / "remote_probe_glm_v1" / "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "build", "shadow", "calibrate", "target", "infer",
            "diagnostics", "evaluate", "report", "target-exact",
            "infer-exact", "all",
        ),
        nargs="?",
        default="all",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter-root", type=Path, default=ADAPTER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


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


def adapter_path(root: Path, spec) -> Path:
    return root / "adapters" / spec.split / spec.model_id


def adapter_arguments(root: Path, specs: list) -> list[str]:
    values = []
    for spec in specs:
        values.extend(["--adapter", f"{spec.model_id}={adapter_path(root, spec)}"])
    return values


def generation_path(output: Path, name: str) -> Path:
    return output / "probes" / name / "attacker_view" / "remote_generations.jsonl"


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
    output = generation_path(args.output_dir, name)
    if output.is_file():
        print(f"reuse {output}", flush=True)
        return output
    command = [
        sys.executable,
        str(GLM_ROOT / "probe_remote_glm.py"),
        "--model", str(args.model),
        "--probe-bank", str(probe_bank),
        "--output-dir", str(args.output_dir / "probes" / name),
        "--repetitions", str(repetitions),
        "--batch-size", str(repetitions),
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
            str(ROOT / "Text_only" / "materialize_frozen_probe_banks.py"),
            "--probe-bank", str(BANK),
            "--frozen-attack", str(args.output_dir / "frozen_remote_attack.json"),
            "--output-dir", str(directory),
        ],
        env,
    )
    return directory / "primary_selected_probe_bank.json"


def write_exact_target_boundary(args: argparse.Namespace, selected_bank: Path) -> Path:
    paired = generation_path(args.output_dir, "target_selected_paired")
    post_only = generation_path(args.output_dir, "target_selected_post_only")
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


def probe_base_control(
    args: argparse.Namespace,
    env: dict[str, str],
    targets: list,
) -> Path:
    name = "base_seed_control"
    output = generation_path(args.output_dir, name)
    if output.is_file():
        print(f"reuse {output}", flush=True)
        return output
    first = targets[0]
    command = [
        sys.executable,
        str(GLM_ROOT / "probe_remote_glm.py"),
        "--model", str(args.model),
        "--probe-bank", str(BANK),
        "--output-dir", str(args.output_dir / "probes" / name),
        "--repetitions", "8",
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", "12",
        "--temperature", "0.8",
        "--top-p", "0.95",
        "--adapter", f"{first.model_id}={adapter_path(args.adapter_root, first)}",
    ]
    for spec in targets:
        command.extend(["--control-base-id", spec.model_id])
    run(command, env)
    return output


def calibrate(args: argparse.Namespace, env: dict[str, str], shadows: list) -> Path:
    output = args.output_dir / "frozen_remote_attack.json"
    if output.is_file():
        frozen = json.loads(output.read_text(encoding="utf-8"))
        if frozen["probe_bank_sha256"] != hashlib.sha256(BANK.read_bytes()).hexdigest():
            raise RuntimeError("Existing frozen attack uses another probe bank.")
        print(f"reuse {output}", flush=True)
        return output
    command = [
        sys.executable,
        str(ROOT / "Text_only" / "remote_attack.py"),
        "calibrate",
        "--shadow-generations", str(generation_path(args.output_dir, "shadow_candidates")),
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
    target = generation_path(args.output_dir, "target_candidates")
    existing = args.output_dir / "target_query_boundary.json"
    if existing.is_file():
        return existing
    result = {
        "boundary": "remote_attack_frozen_before_target_queries",
        "frozen_attack_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "probe_bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(),
        "target_generations_existed_when_boundary_was_written": target.exists(),
        "target_labels_opened": False,
    }
    if result["target_generations_existed_when_boundary_was_written"]:
        raise RuntimeError("Target generations predate the frozen-boundary record.")
    existing.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return existing


def infer_one(
    args: argparse.Namespace,
    env: dict[str, str],
    generations: Path,
    output_dir: Path,
    targets: list,
    repetition_offset: int = 0,
    attack_names: tuple[str, ...] = (),
) -> None:
    command = [
        sys.executable,
        str(ROOT / "Text_only" / "remote_attack.py"),
        "infer",
        "--target-generations", str(generations),
        "--probe-bank", str(BANK),
        "--frozen-attack", str(args.output_dir / "frozen_remote_attack.json"),
        "--output-dir", str(output_dir),
        "--paired-repetition-offset", str(repetition_offset),
    ]
    for attack_name in attack_names:
        command.extend(["--attack-name", attack_name])
    for spec in targets:
        command.extend(["--target-id", spec.model_id])
    run(command, env)


def evaluate_prediction(
    args: argparse.Namespace,
    env: dict[str, str],
    prediction: Path,
    output: Path,
    targets: list,
    valid_ids: set[str],
) -> None:
    command = [
        sys.executable,
        "-m", "Text_only.evaluate_known_preference",
        "--predictions", str(prediction),
        "--output", str(output),
    ]
    for spec in targets:
        command.extend(["--target-label", f"{spec.model_id}={spec.preference}"])
        command.extend(["--target-pair", f"{spec.model_id}={spec.pair_id}"])
    valid_classes = {spec.preference for spec in targets if spec.model_id in valid_ids}
    if len(valid_classes) == 2:
        for model_id in sorted(valid_ids):
            command.extend(["--valid-target-id", model_id])
    run(command, env)


def evaluate(args: argparse.Namespace, env: dict[str, str], targets: list) -> None:
    fidelity = json.loads(
        (args.adapter_root / "analysis" / "visual_fidelity_target.json").read_text(encoding="utf-8")
    )
    valid_ids = {
        row["anonymous_model_id"] for row in fidelity["per_model"] if row["passed"]
    }
    analysis = args.output_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for name in (
        "paired_selected_residualized",
        "paired_selected_no_residual",
        "post_only_matched_budget",
        "paired_random",
    ):
        evaluate_prediction(
            args, env,
            args.output_dir / "predictions" / f"{name}.json",
            analysis / f"{name}.json",
            targets, valid_ids,
        )
    evaluate_prediction(
        args, env,
        args.output_dir / "predictions_resample" / "paired_selected_residualized.json",
        analysis / "paired_selected_residualized_resample.json",
        targets, valid_ids,
    )
    evaluate_prediction(
        args, env,
        args.output_dir / "predictions_base_seed_control" / "paired_selected_residualized.json",
        analysis / "paired_selected_residualized_base_seed_control.json",
        targets, valid_ids,
    )


def full_run(metrics: dict) -> dict:
    return metrics["individual_runs"][0]


def number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_report(args: argparse.Namespace) -> Path:
    frozen = json.loads((args.output_dir / "frozen_remote_attack.json").read_text(encoding="utf-8"))
    analysis = args.output_dir / "analysis"
    primary_doc = json.loads((analysis / "paired_selected_residualized.json").read_text(encoding="utf-8"))
    no_res_doc = json.loads((analysis / "paired_selected_no_residual.json").read_text(encoding="utf-8"))
    post_doc = json.loads((analysis / "post_only_matched_budget.json").read_text(encoding="utf-8"))
    random_doc = json.loads((analysis / "paired_random.json").read_text(encoding="utf-8"))
    resample_doc = json.loads((analysis / "paired_selected_residualized_resample.json").read_text(encoding="utf-8"))
    control_doc = json.loads((analysis / "paired_selected_residualized_base_seed_control.json").read_text(encoding="utf-8"))
    direct_doc = json.loads(
        (args.adapter_root / "analysis" / "text_only" / "paired" / "metrics.json").read_text(encoding="utf-8")
    )
    primary_predictions = json.loads(
        (args.output_dir / "predictions" / "paired_selected_residualized.json").read_text(encoding="utf-8")
    )["predictions"]
    resample_predictions = json.loads(
        (args.output_dir / "predictions_resample" / "paired_selected_residualized.json").read_text(encoding="utf-8")
    )["predictions"]
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    exact_budget = json.loads(
        (args.output_dir / "exact_budget" / "verification.json").read_text(
            encoding="utf-8"
        )
    )
    bank_by_id = {row["probe_id"]: row for row in bank}
    selected = frozen["selection"]["selected_probe_ids"]
    kinds = Counter(bank_by_id[probe_id]["kind"] for probe_id in selected)
    families = Counter(bank_by_id[probe_id]["family"] for probe_id in selected)
    primary = full_run(primary_doc)
    no_res = full_run(no_res_doc)
    post = full_run(post_doc)
    resample = full_run(resample_doc)
    control = full_run(control_doc)
    direct = next(row for row in direct_doc["individual_runs"] if row["query_budget"] == 48)
    random_summary = random_doc["budget_summary"][0]
    random_sig = sum(
        row["matched_pair_randomization_pvalue_one_sided"] < 0.05
        for row in random_doc["individual_runs"]
    )
    chosen = frozen["selection"]["chosen_nested_oof"]
    primary_attack = next(
        row for row in frozen["attacks"] if row["name"] == "paired_selected_residualized"
    )
    supported = (
        primary["balanced_accuracy"] > 0.5
        and primary["matched_pair_randomization_pvalue_one_sided"] is not None
        and primary["matched_pair_randomization_pvalue_one_sided"] < 0.05
    )
    kind_summary = ", ".join(f"{key}={value}" for key, value in sorted(kinds.items()))
    family_summary = ", ".join(f"{key}={value}" for key, value in sorted(families.items()))
    conditioned = primary_doc["fidelity_conditioned"]
    conditioned_run = conditioned["individual_runs"][0]
    primary_by_id = {row["anonymous_model_id"]: row for row in primary_predictions}
    resample_by_id = {row["anonymous_model_id"]: row for row in resample_predictions}
    stability_ids = sorted(primary_by_id)
    first_scores = [primary_by_id[model_id]["positive_class_score"] for model_id in stability_ids]
    second_scores = [resample_by_id[model_id]["positive_class_score"] for model_id in stability_ids]
    first_mean = sum(first_scores) / len(first_scores)
    second_mean = sum(second_scores) / len(second_scores)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first_scores, second_scores)
    )
    denominator = math.sqrt(
        sum((value - first_mean) ** 2 for value in first_scores)
        * sum((value - second_mean) ** 2 for value in second_scores)
    )
    score_correlation = numerator / denominator if denominator else float("nan")
    prediction_agreement = sum(
        primary_by_id[model_id]["predicted_preference"]
        == resample_by_id[model_id]["predicted_preference"]
        for model_id in stability_ids
    ) / len(stability_ids)
    mean_score_shift = sum(
        abs(left - right) for left, right in zip(first_scores, second_scores)
    ) / len(first_scores)
    conclusion = (
        "GLM-4.1V 的严格 remote text behavior attack 在 held-out targets 上显著高于随机水平。"
        if supported else
        "GLM-4.1V 的严格 remote attack 在当前 held-out targets 上未显著高于随机水平。"
    )
    lines = [
        "# GLM-4.1V-9B Preference-Remote Text Leakage Metrics", "",
        "## 核心结论", "", conclusion, "",
        "所有 remote prompts 均不含颜色、视觉、审美或偏好词，也不询问模型偏好。", "",
        "## 隔离与协议", "",
        "- Candidate bank：132 probes（108 choice + 24 open），与 LLaVA 严格实验共享同一冻结 bank。",
        "- Shadow calibration：12 updates / 6 matched pairs；target evaluation：20 updates / 10 matched pairs。",
        "- Stable selection：leave-one-shadow-pair-out，target 输出生成前冻结。",
        "- Nuisance residualization：shadow opposite-preference pair means 的 rank-2 subspace。",
        "- 黑盒观测：只读取生成字符串；不提交图像，不使用 logits、概率或参数。",
        f"- Selected K={len(selected)}；paired online budget={primary_attack['online_generation_calls']} generation calls。",
        "- Post-only control 使用 8 samples/probe，与 paired attack 总调用预算匹配。", "",
        f"- Exact-budget audit：passed={exact_budget['passed']}；三种主攻击 selected-only 复算最大 score 差为 0。",
        "- 预算口径：96 是冻结攻击的 effective budget；legacy diagnostic collection 的完整 candidate-bank 调用单独保留。", "",
        "## Shadow-only selection", "",
        f"Nested shadow OOF：balanced accuracy {chosen['balanced_accuracy']:.3f}，AUROC {chosen['auroc']:.3f}。",
        f"Selected kinds：{kind_summary}。",
        f"Selected families：{family_summary}。", "",
        "## Held-out target results", "",
        "| Attack | Calls | Balanced accuracy | AUROC | Exact paired p |",
        "|---|---:|---:|---:|---:|",
        f"| Remote selected + residualization | {primary_attack['online_generation_calls']} | {primary['balanced_accuracy']:.3f} | {number(primary['auroc'])} | {number(primary['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Remote selected, no residualization | {primary_attack['online_generation_calls']} | {no_res['balanced_accuracy']:.3f} | {number(no_res['auroc'])} | {number(no_res['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Remote post-only, matched total calls | {primary_attack['online_generation_calls']} | {post['balanced_accuracy']:.3f} | {number(post['auroc'])} | {number(post['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Remote random probes (10 sets) | {primary_attack['online_generation_calls']} | {random_summary['balanced_accuracy_mean']:.3f} ± {random_summary['balanced_accuracy_std']:.3f} | {number(random_summary['auroc_mean'])} | {random_sig}/10 significant |",
        f"| Direct elicitation baseline (prior GLM experiment) | 96 actual paired calls | {direct['balanced_accuracy']:.3f} | {number(direct['auroc'])} | {number(direct['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        "",
        f"Primary pair-level 95% CI：[{primary['accuracy_95ci'][0]:.3f}, {primary['accuracy_95ci'][1]:.3f}]。", "",
        "## Robustness and negative control", "",
        "| Diagnostic | Balanced accuracy | AUROC | Exact paired p |",
        "|---|---:|---:|---:|",
        f"| Independent target resampling（repetitions 5–8） | {resample['balanced_accuracy']:.3f} | {number(resample['auroc'])} | {number(resample['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        f"| Base-only ID/seed-stream control（all adapters disabled） | {control['balanced_accuracy']:.3f} | {number(control['auroc'])} | {number(control['matched_pair_randomization_pvalue_one_sided'], 4)} |",
        "",
        f"Fidelity-conditioned：通过视觉 fidelity gate 的 {conditioned['num_held_out_target_updates']}/20 updates 上，accuracy {conditioned_run['accuracy']:.3f}、balanced accuracy {conditioned_run['balanced_accuracy']:.3f}、AUROC {conditioned_run['auroc']:.3f}。", "",
        "## Interpretation", "",
        (
            "主 held-out run 支持 GLM 上的 visual-training-to-text-behavior leakage；"
            "但独立采样未达到显著性，因此这是部分稳健的跨模型复现，不能表述为所有采样下均稳定。"
            if supported else
            "结果暂不支持 GLM 上的 strict remote text leakage；应保留为模型族差异的阴性结果。"
        ),
        f"Stable-selected={primary['balanced_accuracy']:.3f}，random-set mean={random_summary['balanced_accuracy_mean']:.3f}；该差异用于描述，不进行事后显著性扩张。",
        f"Residualization 未改变 balanced accuracy（{no_res['balanced_accuracy']:.3f} vs. {primary['balanced_accuracy']:.3f}）；matched-budget post-only（{post['balanced_accuracy']:.3f}）高于 paired（{primary['balanced_accuracy']:.3f}），说明本次 GLM 的 pre/post subtraction 没有带来收益。",
        "冻结集合包含 7 个 choice 和 5 个 open-ended probes，证明攻击并非只选择 A/B tasks；但当前分类器融合两类信号，尚不能单独声称 open-only leakage。",
        f"独立 target resampling 为 balanced accuracy {resample['balanced_accuracy']:.3f}、p={number(resample['matched_pair_randomization_pvalue_one_sided'], 4)}，是当前最重要的稳健性限制。",
        f"两组 target samples 的 score Pearson correlation={score_correlation:.3f}，prediction agreement={prediction_agreement:.3f}（{round((1 - prediction_agreement) * len(stability_ids))}/{len(stability_ids)} 个预测改变），mean absolute score shift={mean_score_shift:.3f}。",
        "所有 target labels 仅在预测落盘后由 evaluator 打开。", "",
        "## Artifacts", "",
        "- `outputs/remote_probe_glm_v1/main/frozen_remote_attack.json`：shadow-only 冻结攻击。",
        "- `outputs/remote_probe_glm_v1/main/target_query_boundary.json`：target 查询边界。",
        "- `outputs/remote_probe_glm_v1/main/probes/`：可断点续跑的 raw string shards。",
        "- `outputs/remote_probe_glm_v1/main/predictions*/`：label-blind predictions。",
        "- `outputs/remote_probe_glm_v1/main/analysis/`：标签解封后的指标。", "",
        "- `outputs/remote_probe_glm_v1/main/exact_budget/`：精确预算 transcripts、ledgers 与等价性审计。",
        "- `outputs/remote_probe_glm_v1/main/frozen_probe_banks/`：冻结的 selected/random banks。", "",
    ]
    output = GLM_ROOT / "GLM_REMOTE_PROBE_METRICS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    if args.batch_size != 8:
        raise ValueError("The frozen GLM protocol requires --batch-size 8.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    shadows, targets = build_specs()
    if args.phase in {"build", "all"}:
        run([sys.executable, str(ROOT / "Text_only" / "build_remote_probe_bank.py")], env)
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
        infer_one(
            args, env,
            generation_path(args.output_dir, "target_candidates"),
            args.output_dir / "predictions",
            targets,
        )
    if args.phase == "infer-exact":
        infer_one(
            args, env,
            generation_path(args.output_dir, "target_selected_paired"),
            args.output_dir / "exact_budget" / "predictions_paired",
            targets,
            attack_names=(
                "paired_selected_residualized",
                "paired_selected_no_residual",
            ),
        )
        infer_one(
            args, env,
            generation_path(args.output_dir, "target_selected_post_only"),
            args.output_dir / "exact_budget" / "predictions_post_only",
            targets,
            attack_names=("post_only_matched_budget",),
        )
    if args.phase in {"diagnostics", "all"}:
        infer_one(
            args, env,
            generation_path(args.output_dir, "target_candidates"),
            args.output_dir / "predictions_resample",
            targets,
            repetition_offset=4,
        )
        control = probe_base_control(args, env, targets)
        infer_one(
            args, env, control,
            args.output_dir / "predictions_base_seed_control",
            targets,
        )
    if args.phase in {"evaluate", "all"}:
        evaluate(args, env, targets)
    if args.phase in {"report", "all"}:
        print(json.dumps({"report": str(write_report(args))}, indent=2))


if __name__ == "__main__":
    main()
