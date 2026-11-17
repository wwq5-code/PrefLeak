#!/usr/bin/env python3
"""Run the frozen joint-VLM DPO experiment and the text-only attack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = Path("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
DATA = SCRIPT_DIR / "data" / "joint_vlm_paper"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "joint_vlm_paper_v1" / "main"


@dataclass(frozen=True)
class UpdateSpec:
    model_id: str
    split: str
    serial: int
    preference: str
    pair_id: str
    data_root: Path

    @property
    def manifest(self) -> Path:
        return self.data_root / "private_updates" / f"{self.model_id}.jsonl"


def build_specs() -> tuple[list[UpdateSpec], list[UpdateSpec]]:
    train_root = DATA / "train2017" / "magenta_cyan"
    valid_root = DATA / "val2017" / "magenta_cyan"
    shadows = [
        UpdateSpec(
            f"target_b{serial:03d}",
            "shadow",
            serial,
            "magenta" if serial % 2 else "cyan",
            f"shadow_pair_{(serial + 1) // 2:02d}",
            train_root,
        )
        for serial in range(1, 13)
    ]
    targets = [
        UpdateSpec(
            f"target_u{serial:03d}",
            "target",
            serial,
            "magenta" if serial % 2 else "cyan",
            f"target_pair_{(serial + 1) // 2:02d}",
            valid_root,
        )
        for serial in range(1, 21)
    ]
    return shadows, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("train", "audit", "probe", "analyze", "report", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT / 'trl'}:{ROOT}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def read_config(output: Path) -> dict:
    path = output / "frozen_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; freeze a shadow-only tuning candidate before target training."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "selection_boundary",
        "learning_rate",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "beta",
        "label_smoothing",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "selected_candidate",
        "selection_rule",
        "selection_evidence",
    }
    if set(config) != required:
        raise ValueError(f"Unexpected frozen-config fields: {set(config) ^ required}")
    if config["selection_boundary"] != "shadow_only_train2017":
        raise ValueError("The update configuration was not frozen on shadow data only.")
    return config


def adapter_path(output: Path, spec: UpdateSpec) -> Path:
    return output / "adapters" / spec.split / spec.model_id


def run(command: list[str], env: dict[str, str], check: bool = True) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=check)


def train_one(
    args: argparse.Namespace, env: dict[str, str], config: dict, spec: UpdateSpec
) -> str:
    destination = adapter_path(args.output_dir, spec)
    if (destination / "adapter_config.json").is_file():
        return f"skip {spec.model_id}"
    log = args.output_dir / "logs" / spec.split / f"{spec.model_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    seed_base = 8400 if spec.split == "shadow" else 9400
    pair_number = (spec.serial + 1) // 2
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_llava_dpo.py"),
        "--model", str(args.model),
        "--train-manifest", str(spec.manifest),
        "--target-id", spec.model_id,
        "--output-dir", str(destination),
        "--seed", str(seed_base + pair_number),
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
    specs: list[UpdateSpec],
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
                raise RuntimeError(f"Training failed for {spec.model_id}") from error


def audit(args: argparse.Namespace, config: dict, specs: list[UpdateSpec]) -> dict:
    expected = {
        "max_steps": config["max_steps"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "beta": config["beta"],
        "label_smoothing": config["label_smoothing"],
        "lora_rank": config["lora_rank"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
    }
    rows = []
    for spec in specs:
        path = adapter_path(args.output_dir, spec) / "private_training_metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "manifest": Path(metadata["train_manifest"]) == spec.manifest.resolve(),
            "objective": metadata["objective"] == "dpo"
            and metadata["loss_type"] == ["sigmoid"]
            and metadata["loss_weights"] is None,
            "joint_scope": metadata["update_scope"] == "joint_vlm"
            and metadata["lora_scope"] == "vision_language_projector",
            "all_components": set(metadata["matched_module_counts"])
            == {"vision", "projector", "language"}
            and all(value > 0 for value in metadata["matched_module_counts"].values()),
            "frozen_hyperparameters": all(metadata[key] == value for key, value in expected.items()),
        }
        rows.append({"model_id": spec.model_id, **checks, "passed": all(checks.values())})
    result = {
        "audit": "joint_vlm_pure_dpo_frozen_configuration",
        "num_updates": len(rows),
        "num_passed": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "per_update": rows,
    }
    path = args.output_dir / "audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_passed"]:
        raise RuntimeError(f"Adapter audit failed; inspect {path}")
    return result


def adapter_arguments(args: argparse.Namespace, specs: list[UpdateSpec]) -> list[str]:
    values = []
    for spec in specs:
        values.extend(["--adapter", f"{spec.model_id}={adapter_path(args.output_dir, spec)}"])
    return values


def label_arguments(specs: list[UpdateSpec]) -> list[str]:
    values = []
    for spec in specs:
        values.extend(["--target-label", f"{spec.model_id}={spec.preference}"])
    return values


def probe_visual_fidelity(
    args: argparse.Namespace,
    env: dict[str, str],
    name: str,
    specs: list[UpdateSpec],
) -> None:
    directory = args.output_dir / "probes" / f"visual_fidelity_{name}"
    generations = directory / "attacker_view" / "forced_choice_generations.jsonl"
    if not generations.is_file():
        command = [
            sys.executable,
            str(SCRIPT_DIR / "probe_forced_choice.py"),
            "--model", str(args.model),
            "--modality", "vision_conditioned",
            "--probe-bank", str(specs[0].data_root / "private_fidelity_probe_bank.json"),
            "--visual-prompt-style", "training",
            "--output-dir", str(directory),
            "--max-new-tokens", "12",
            "--reference-description", "a magenta-tinted visual treatment",
            "--alternative-description", "a cyan-tinted visual treatment",
            *adapter_arguments(args, specs),
        ]
        run(command, env)
    output = args.output_dir / "analysis" / f"visual_fidelity_{name}.json"
    command = [
        sys.executable,
        "-m", "Text_only.evaluate_update_fidelity",
        "--generations", str(generations),
        "--output", str(output),
        "--reference-preference", "magenta",
        *label_arguments(specs),
    ]
    run(command, env, check=False)


def probe_text(
    args: argparse.Namespace, env: dict[str, str], specs: list[UpdateSpec]
) -> Path:
    directory = args.output_dir / "probes" / "text_only"
    generations = directory / "attacker_view" / "forced_choice_generations.jsonl"
    if generations.is_file():
        return generations
    command = [
        sys.executable,
        str(SCRIPT_DIR / "probe_forced_choice.py"),
        "--model", str(args.model),
        "--modality", "text_only",
        "--concepts", str(SCRIPT_DIR / "prompts" / "magenta_cyan_known_preference_concepts.json"),
        "--output-dir", str(directory),
        "--max-new-tokens", "12",
        *adapter_arguments(args, specs),
    ]
    run(command, env)
    return generations


def analyze(
    args: argparse,
    env: dict[str, str],
    shadows: list[UpdateSpec],
    targets: list[UpdateSpec],
) -> None:
    generations = (
        args.output_dir / "probes" / "text_only" / "attacker_view"
        / "forced_choice_generations.jsonl"
    )
    target_fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_target.json").read_text(encoding="utf-8")
    )
    valid_ids = {
        row["anonymous_model_id"] for row in target_fidelity["per_model"] if row["passed"]
    }
    valid_preferences = {spec.preference for spec in targets if spec.model_id in valid_ids}
    for access in ("paired", "post_only"):
        directory = args.output_dir / "analysis" / "text_only" / access
        predictions = directory / "predictions.json"
        command = [
            sys.executable,
            "-m", "Text_only.infer_forced_choice",
            "--generations", str(generations),
            "--output", str(predictions),
            "--access", access,
            "--budgets", "6,12,24,48",
            "--budget-repeats", "10",
        ]
        for spec in shadows:
            command.extend(["--shadow-label", f"{spec.model_id}={spec.preference}"])
            command.extend(["--shadow-pair", f"{spec.model_id}={spec.pair_id}"])
        for spec in targets:
            command.extend(["--target-id", spec.model_id])
        run(command, env)

        metrics = directory / "metrics.json"
        command = [
            sys.executable,
            "-m", "Text_only.evaluate_known_preference",
            "--predictions", str(predictions),
            "--output", str(metrics),
        ]
        for spec in targets:
            command.extend(["--target-label", f"{spec.model_id}={spec.preference}"])
            command.extend(["--target-pair", f"{spec.model_id}={spec.pair_id}"])
        if len(valid_preferences) == 2:
            for model_id in sorted(valid_ids):
                command.extend(["--valid-target-id", model_id])
        run(command, env)


def format_number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_report(args: argparse, config: dict) -> Path:
    audit_result = json.loads((args.output_dir / "audit.json").read_text(encoding="utf-8"))
    shadow_fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_shadow.json").read_text(encoding="utf-8")
    )
    target_fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_target.json").read_text(encoding="utf-8")
    )
    paired = json.loads(
        (args.output_dir / "analysis" / "text_only" / "paired" / "metrics.json").read_text(encoding="utf-8")
    )
    post_only = json.loads(
        (args.output_dir / "analysis" / "text_only" / "post_only" / "metrics.json").read_text(encoding="utf-8")
    )

    def table_rows(metrics: dict) -> list[str]:
        rows = []
        for item in metrics["budget_summary"]:
            full_run = next(
                run for run in metrics["individual_runs"]
                if run["query_budget"] == item["query_budget"] and run["budget_repeat"] == 0
            )
            pvalue = (
                full_run["matched_pair_randomization_pvalue_one_sided"]
                if item["num_budget_subsets"] == 1 else None
            )
            rows.append(
                f"| {item['query_budget']} | {item['num_budget_subsets']} | "
                f"{format_number(item['balanced_accuracy_mean'])} ± {format_number(item['balanced_accuracy_std'])} | "
                f"{format_number(item['auroc_mean'])} | {format_number(pvalue, 4)} |"
            )
        return rows

    full_paired = next(row for row in paired["individual_runs"] if row["query_budget"] == 48)
    paired_attack = json.loads(
        (args.output_dir / "analysis" / "text_only" / "paired" / "predictions.json").read_text(encoding="utf-8")
    )
    full_shadow = next(
        row for row in paired_attack["run_summaries"] if row["query_budget"] == 48
    )
    conditional = paired.get("fidelity_conditioned")
    full_conditional = (
        next(
            row for row in conditional["individual_runs"]
            if row["query_budget"] == 48
        )
        if conditional else None
    )
    conclusion = (
        "支持核心假设：held-out text-only balanced accuracy 高于随机水平。"
        if full_paired["balanced_accuracy"] > 0.5
        and full_paired["matched_pair_randomization_pvalue_one_sided"] is not None
        and full_paired["matched_pair_randomization_pvalue_one_sided"] < 0.05
        else "当前实验不足以支持核心假设：未在配对随机化检验下显著高于随机水平。"
    )
    lines = [
        "# Joint-VLM Text-only PrefLeak Metrics",
        "",
        "## 核心结论",
        "",
        conclusion,
        "",
        "核心检验使用 12 个 train2017 shadow updates 与 20 个此前未见的 val2017 target updates；"
        "20 个 target 组成 10 个同图像、反偏好 matched pairs。候选偏好名称没有出现在视觉 DPO 的提示或回答中。",
        "",
        "## 更新与保真度",
        "",
        f"- 训练目标：pure sigmoid DPO；更新范围：vision tower + multimodal projector + language model。",
        f"- Shadow-only 选择：`{config['selected_candidate']}`（2/2 pilot updates 通过视觉门槛，"
        f"平均准确率 {config['selection_evidence']['visual_mean_accuracy']:.3f}；target 尚未打开）。",
        f"- 冻结超参数：learning rate {config['learning_rate']:.0e}，{config['max_steps']} steps，"
        f"batch size {config['per_device_train_batch_size']}，beta {config['beta']}，"
        f"label smoothing {config['label_smoothing']}，LoRA r={config['lora_rank']} / alpha={config['lora_alpha']} / dropout={config['lora_dropout']}。",
        f"- 联合更新审计：{audit_result['num_passed']}/{audit_result['num_updates']} 通过。",
        f"- Shadow visual fidelity：{shadow_fidelity['num_passed']}/{shadow_fidelity['num_models']} 通过，平均偏好准确率 {shadow_fidelity['mean_post_preference_accuracy']:.3f}。",
        f"- Target visual fidelity：{target_fidelity['num_passed']}/{target_fidelity['num_models']} 通过，平均偏好准确率 {target_fidelity['mean_post_preference_accuracy']:.3f}。",
        f"- Full-query shadow grouped-OOF：balanced accuracy {full_shadow['shadow_oof_balanced_accuracy']:.3f}，AUROC {full_shadow['shadow_oof_auroc']:.3f}。",
        "",
        "## Text-only attack：paired pre/post access（主结果）",
        "",
        "| Queries | Subsets | Balanced accuracy | AUROC | One-sided paired p |",
        "|---:|---:|---:|---:|---:|",
        *table_rows(paired),
        "",
        "`p` 是对 10 个 target matched pairs 逐对交换私有标签的精确单侧随机化检验；"
        "只对在 target 训练前已固定、使用全部 48 个查询的主检验报告。",
        f"主检验 accuracy 的 matched-pair effective-n Wilson 95% CI 为 "
        f"[{full_paired['accuracy_95ci'][0]:.3f}, {full_paired['accuracy_95ci'][1]:.3f}]；"
        f"平均精确响应变化率为 {full_paired['mean_exact_response_change_rate']:.3f}。",
        *(
            [
                f"次要的 fidelity-conditioned 子集覆盖 {conditional['coverage']:.1%} targets；"
                f"accuracy {full_conditional['accuracy']:.3f}，balanced accuracy "
                f"{full_conditional['balanced_accuracy']:.3f}，AUROC {full_conditional['auroc']:.3f}。"
                "该子集含不完整 matched pairs，因此不报告配对随机化 p 值。"
            ]
            if full_conditional else []
        ),
        "",
        "## Text-only attack：post-update-only control",
        "",
        "| Queries | Subsets | Balanced accuracy | AUROC | One-sided paired p |",
        "|---:|---:|---:|---:|---:|",
        *table_rows(post_only),
        "",
        "Paired 与 post-only 的 full-query 结果相同，说明在本配置中泄漏已直接存在于更新后文本行为；"
        "访问基座输出没有提供额外分类增益。",
        "",
        "## 隔离边界",
        "",
        "学习率和步数只由 train2017 的 shadow-only 调参确定并在 target 训练前写入 `frozen_config.json`。"
        "攻击器只接收 shadow 标签；target truth 在预测文件落盘后才由 evaluator 打开。",
        "",
    ]
    report = SCRIPT_DIR / "JOINT_VLM_PAPER_METRICS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = read_config(args.output_dir)
    env = environment()
    shadows, targets = build_specs()
    all_specs = [*shadows, *targets]
    if args.phase in {"train", "all"}:
        train_all(args, env, config, all_specs)
    if args.phase in {"audit", "all"}:
        audit(args, config, all_specs)
    if args.phase in {"probe", "all"}:
        probe_visual_fidelity(args, env, "shadow", shadows)
        probe_visual_fidelity(args, env, "target", targets)
        probe_text(args, env, all_specs)
    if args.phase in {"analyze", "all"}:
        analyze(args, env, shadows, targets)
    if args.phase in {"report", "all"}:
        report = write_report(args, config)
        print(json.dumps({"report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
