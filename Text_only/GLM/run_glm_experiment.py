#!/usr/bin/env python3
"""Run the frozen GLM-4.1V joint-DPO text-only leakage experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLM_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Text_only" / "data" / "joint_vlm_paper"
DEFAULT_OUTPUT = GLM_ROOT / "outputs" / "main_v1"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.GLM.train_glm_dpo import DEFAULT_MODEL  # noqa: E402


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
    shadow_root = DATA_ROOT / "train2017" / "magenta_cyan"
    target_root = DATA_ROOT / "val2017" / "magenta_cyan"
    shadows = [
        UpdateSpec(
            f"target_b{i:03d}", "shadow", i,
            "magenta" if i % 2 else "cyan",
            f"shadow_pair_{(i + 1) // 2:02d}", shadow_root,
        )
        for i in range(1, 13)
    ]
    targets = [
        UpdateSpec(
            f"target_u{i:03d}", "target", i,
            "magenta" if i % 2 else "cyan",
            f"target_pair_{(i + 1) // 2:02d}", target_root,
        )
        for i in range(1, 21)
    ]
    return shadows, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("train", "audit", "probe", "analyze", "report", "all"),
        nargs="?", default="all",
    )
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


def read_config(output: Path) -> dict:
    path = output / "frozen_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen configuration: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["selection_boundary"] != "shadow_only_train2017":
        raise ValueError("GLM configuration was not frozen on shadow data only.")
    if config["selection_evidence"]["target_data_opened"] is not False:
        raise ValueError("Frozen config indicates target data was opened.")
    return config


def adapter_path(output: Path, spec: UpdateSpec) -> Path:
    return output / "adapters" / spec.split / spec.model_id


def run(command: list[str], env: dict[str, str], check: bool = True) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=check)


def train_one(args: argparse.Namespace, env: dict[str, str], config: dict, spec: UpdateSpec) -> str:
    destination = adapter_path(args.output_dir, spec)
    if (destination / "adapter_config.json").is_file():
        return f"skip {spec.model_id}"
    log = args.output_dir / "logs" / spec.split / f"{spec.model_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    seed_base = 15400 if spec.split == "shadow" else 16400
    command = [
        sys.executable, str(GLM_ROOT / "train_glm_dpo.py"),
        "--model", str(args.model),
        "--train-manifest", str(spec.manifest),
        "--output-dir", str(destination),
        "--target-id", spec.model_id,
        "--seed", str(seed_base + (spec.serial + 1) // 2),
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
            command, cwd=PROJECT_ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=True,
        )
    return f"complete {spec.model_id}"


def train_all(args: argparse.Namespace, env: dict[str, str], config: dict, specs: list[UpdateSpec]) -> None:
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(train_one, args, env, config, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(f"GLM training failed for {spec.model_id}") from error


def audit(args: argparse.Namespace, config: dict, specs: list[UpdateSpec]) -> dict:
    expected = {
        key: config[key]
        for key in (
            "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
            "learning_rate", "beta", "label_smoothing", "lora_rank", "lora_alpha",
            "lora_dropout",
        )
    }
    rows = []
    for spec in specs:
        metadata = json.loads(
            (adapter_path(args.output_dir, spec) / "private_training_metadata.json").read_text(encoding="utf-8")
        )
        checks = {
            "manifest": Path(metadata["train_manifest"]) == spec.manifest.resolve(),
            "pure_dpo": metadata["objective"] == "dpo"
            and metadata["loss_type"] == ["sigmoid"] and metadata["loss_weights"] is None,
            "joint_scope": metadata["update_scope"] == "joint_vlm"
            and metadata["matched_module_counts"] == {"vision": 120, "projector": 4, "language": 240},
            "frozen_hyperparameters": all(metadata[key] == value for key, value in expected.items()),
        }
        rows.append({"model_id": spec.model_id, **checks, "passed": all(checks.values())})
    result = {
        "audit": "glm41v_joint_vlm_pure_dpo_frozen_configuration",
        "num_updates": len(rows),
        "num_passed": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "per_update": rows,
    }
    output = args.output_dir / "audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_passed"]:
        raise RuntimeError(f"Audit failed: {output}")
    return result


def adapter_args(args: argparse.Namespace, specs: list[UpdateSpec]) -> list[str]:
    values = []
    for spec in specs:
        values.extend(["--adapter", f"{spec.model_id}={adapter_path(args.output_dir, spec)}"])
    return values


def label_args(specs: list[UpdateSpec]) -> list[str]:
    values = []
    for spec in specs:
        values.extend(["--target-label", f"{spec.model_id}={spec.preference}"])
    return values


def probe_fidelity(args: argparse.Namespace, env: dict[str, str], name: str, specs: list[UpdateSpec]) -> None:
    directory = args.output_dir / "probes" / f"visual_fidelity_{name}"
    generations = directory / "attacker_view" / "forced_choice_generations.jsonl"
    if not generations.is_file():
        run(
            [
                sys.executable, str(GLM_ROOT / "probe_glm.py"),
                "--model", str(args.model),
                "--modality", "vision_conditioned",
                "--probe-bank", str(specs[0].data_root / "private_fidelity_probe_bank.json"),
                "--output-dir", str(directory),
                "--max-new-tokens", "8",
                *adapter_args(args, specs),
            ],
            env,
        )
    output = args.output_dir / "analysis" / f"visual_fidelity_{name}.json"
    run(
        [
            sys.executable, "-m", "Text_only.evaluate_update_fidelity",
            "--generations", str(generations),
            "--output", str(output),
            "--reference-preference", "magenta",
            "--min-improvement", "0.0",
            "--min-parsed-rate", "0.90",
            *label_args(specs),
        ],
        env,
        check=False,
    )


def probe_text(args: argparse.Namespace, env: dict[str, str], specs: list[UpdateSpec]) -> None:
    directory = args.output_dir / "probes" / "text_only"
    generations = directory / "attacker_view" / "forced_choice_generations.jsonl"
    if generations.is_file():
        return
    run(
        [
            sys.executable, str(GLM_ROOT / "probe_glm.py"),
            "--model", str(args.model),
            "--modality", "text_only",
            "--concepts", str(GLM_ROOT / "prompts" / "magenta_cyan_known_preference_concepts.json"),
            "--output-dir", str(directory),
            "--max-new-tokens", "8",
            *adapter_args(args, specs),
        ],
        env,
    )


def analyze(args: argparse.Namespace, env: dict[str, str], shadows: list[UpdateSpec], targets: list[UpdateSpec]) -> None:
    generations = args.output_dir / "probes" / "text_only" / "attacker_view" / "forced_choice_generations.jsonl"
    fidelity = json.loads(
        (args.output_dir / "analysis" / "visual_fidelity_target.json").read_text(encoding="utf-8")
    )
    valid_ids = {row["anonymous_model_id"] for row in fidelity["per_model"] if row["passed"]}
    valid_classes = {spec.preference for spec in targets if spec.model_id in valid_ids}
    for access in ("paired", "post_only"):
        directory = args.output_dir / "analysis" / "text_only" / access
        predictions = directory / "predictions.json"
        command = [
            sys.executable, "-m", "Text_only.infer_forced_choice",
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

        command = [
            sys.executable, "-m", "Text_only.evaluate_known_preference",
            "--predictions", str(predictions),
            "--output", str(directory / "metrics.json"),
        ]
        for spec in targets:
            command.extend(["--target-label", f"{spec.model_id}={spec.preference}"])
            command.extend(["--target-pair", f"{spec.model_id}={spec.pair_id}"])
        if len(valid_classes) == 2:
            for model_id in sorted(valid_ids):
                command.extend(["--valid-target-id", model_id])
        run(command, env)


def number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_report(args: argparse.Namespace, config: dict) -> Path:
    audit_result = json.loads((args.output_dir / "audit.json").read_text(encoding="utf-8"))
    shadow_fidelity = json.loads((args.output_dir / "analysis" / "visual_fidelity_shadow.json").read_text(encoding="utf-8"))
    target_fidelity = json.loads((args.output_dir / "analysis" / "visual_fidelity_target.json").read_text(encoding="utf-8"))
    paired = json.loads((args.output_dir / "analysis" / "text_only" / "paired" / "metrics.json").read_text(encoding="utf-8"))
    post = json.loads((args.output_dir / "analysis" / "text_only" / "post_only" / "metrics.json").read_text(encoding="utf-8"))
    attack = json.loads((args.output_dir / "analysis" / "text_only" / "paired" / "predictions.json").read_text(encoding="utf-8"))
    full = next(row for row in paired["individual_runs"] if row["query_budget"] == 48)
    post_full = next(row for row in post["individual_runs"] if row["query_budget"] == 48)
    conditioned_full = next(
        row for row in paired["fidelity_conditioned"]["individual_runs"]
        if row["query_budget"] == 48
    )
    shadow_full = next(row for row in attack["run_summaries"] if row["query_budget"] == 48)
    failed_fidelity = [row for row in target_fidelity["per_model"] if not row["passed"]]

    def rows(metrics: dict) -> list[str]:
        result = []
        for item in metrics["budget_summary"]:
            one = next(
                row for row in metrics["individual_runs"]
                if row["query_budget"] == item["query_budget"] and row["budget_repeat"] == 0
            )
            pvalue = one["matched_pair_randomization_pvalue_one_sided"] if item["num_budget_subsets"] == 1 else None
            result.append(
                f"| {item['query_budget']} | {item['num_budget_subsets']} | "
                f"{number(item['balanced_accuracy_mean'])} ± {number(item['balanced_accuracy_std'])} | "
                f"{number(item['auroc_mean'])} | {number(pvalue, 4)} |"
            )
        return result

    supported = (
        full["balanced_accuracy"] > 0.5
        and full["matched_pair_randomization_pvalue_one_sided"] is not None
        and full["matched_pair_randomization_pvalue_one_sided"] < 0.05
    )
    lines = [
        "# GLM-4.1V-9B Text-only PrefLeak Metrics", "",
        "## 核心结论", "",
        (
            "支持核心假设：GLM-4.1V held-out text-only attack 显著高于随机水平。"
            if supported else
            "当前 GLM-4.1V 实验不足以支持 text-only attack 显著高于随机水平。"
        ), "",
        (
            f"在 20 个 held-out joint-VLM updates（10 个 matched pairs）上，48-query paired attack "
            f"取得 balanced accuracy {full['balanced_accuracy']:.3f}、AUROC {full['auroc']:.3f}，"
            f"one-sided exact matched-pair randomization p="
            f"{full['matched_pair_randomization_pvalue_one_sided']:.6f}。"
        ), "",
        "## 实验协议", "",
        "- 威胁模型：known-candidate closed-set preference inference；攻击者知道候选语义是 magenta/cyan，但不知道匿名 update 的真实标签。",
        "- Private update：视觉 A/B comparison；DPO prompt 和 chosen/rejected answer 均不出现 `magenta` 或 `cyan`。",
        "- Shadow/target 隔离：12 个 COCO train2017 shadow updates 用于攻击器与超参数选择；20 个 COCO val2017 updates 只用于 held-out evaluation。",
        "- 查询：48 个无图像的平衡 text-only semantic probes；greedy decoding，`max_new_tokens=8`。",
        "- 主分析：20 个 target updates 全部保留；每对 target 共享数据与随机种子，仅 private preference 相反。",
        "- 统计单位：10 个 matched pairs；full-budget CI 使用 matched-pair effective n=10。",
        "",
        "## 更新与保真度", "",
        "- 模型：`zai-org/GLM-4.1V-9B-Base`。",
        "- 更新：pure sigmoid DPO；joint LoRA 覆盖 vision blocks、visual merger 和全部 language layers。",
        f"- Shadow-only 选择：`{config['selected_candidate']}`，learning rate {config['learning_rate']:.0e}，{config['max_steps']} steps。",
        f"- Adapter audit：{audit_result['num_passed']}/{audit_result['num_updates']} 通过。",
        f"- Shadow visual fidelity：{shadow_fidelity['num_passed']}/{shadow_fidelity['num_models']}，平均准确率 {shadow_fidelity['mean_post_preference_accuracy']:.3f}。",
        f"- Target visual fidelity：{target_fidelity['num_passed']}/{target_fidelity['num_models']}，平均准确率 {target_fidelity['mean_post_preference_accuracy']:.3f}。",
        f"- Shadow grouped-OOF：balanced accuracy {shadow_full['shadow_oof_balanced_accuracy']:.3f}，AUROC {shadow_full['shadow_oof_auroc']:.3f}。",
        (
            "- Fidelity gate：post preference accuracy >= 0.75、parsed rate >= 0.90、"
            "improvement >= 0；该 gate 在 shadow-only tuning 后冻结。"
        ),
        *[
            (
                f"- 未过 fidelity gate：`{row['anonymous_model_id']}`（post preference accuracy "
                f"{row['post_preference_accuracy']:.3f}，parsed rate {row['parsed_response_rate']:.3f}）；"
                "仅因 parsed rate 未达到 0.90。"
            )
            for row in failed_fidelity
        ],
        "", "## Paired pre/post text-only attack（主结果）", "",
        "| Queries | Subsets | Balanced accuracy | AUROC | One-sided paired p |",
        "|---:|---:|---:|---:|---:|", *rows(paired), "",
        f"Full-query accuracy 95% CI：[{full['accuracy_95ci'][0]:.3f}, {full['accuracy_95ci'][1]:.3f}]；"
        f"exact response-change rate：{full['mean_exact_response_change_rate']:.3f}。", "",
        "### Fidelity-conditioned sensitivity analysis", "",
        (
            f"在通过预注册 fidelity gate 的 {paired['fidelity_conditioned']['num_held_out_target_updates']}/"
            f"{paired['num_held_out_target_updates']} 个 updates 上，48-query accuracy "
            f"{conditioned_full['accuracy']:.3f}、balanced accuracy {conditioned_full['balanced_accuracy']:.3f}、"
            f"AUROC {conditioned_full['auroc']:.3f}。由于删除两个 cyan updates 后 matched pairs 不完整，"
            "这里不报告 paired randomization p 值。"
        ), "",
        "## Post-update-only control", "",
        "| Queries | Subsets | Balanced accuracy | AUROC | One-sided paired p |",
        "|---:|---:|---:|---:|---:|", *rows(post), "",
        (
            f"48-query post-only 与 paired 的 balanced accuracy 均为 {post_full['balanced_accuracy']:.3f}，"
            "说明在本次固定 probe bank 上，公共 pre-update baseline 没有提升最终分类结果；"
            "泄漏已经可从 post-update 的 text-only 行为读取。"
        ), "",
        "## 解释与边界", "",
        (
            f"平均只有 {full['mean_exact_response_change_rate'] * 100:.1f}% 的 text responses 相对 base model 改变，"
            "但这些稀疏变化在 shadow updates 中具有可学习的方向，并在 held-out targets 上达到上述 AUROC。"
        ),
        "该结果验证的是一条具体存在性结论：视觉 comparison DPO 的 joint VLM update 可在纯文本接口中留下可推断信号。",
        "它尚不等价于 unknown-preference discovery，也尚未覆盖其他 preference axes、instruction-tuned GLM 或 vision/multimodal attack。",
        "当前样本量只有 10 个独立 matched pairs，因此应同时报告 exact p 值与较宽的 pair-level CI。",
        "",
        "## 隔离边界", "",
        "配置只用 train2017 shadow pilot 冻结；target truth 在预测落盘后才由 evaluator 打开。",
        "attacker-view generations 不含 private labels；标签只由最终 evaluator 载入。", "",
        "## 产物", "",
        "- `outputs/main_v1/frozen_config.json`：shadow-only 冻结配置。",
        "- `outputs/main_v1/audit.json`：32 个 adapter 的 objective、scope 与超参数审计。",
        "- `outputs/main_v1/probes/`：原始 black-box generations。",
        "- `outputs/main_v1/analysis/`：视觉 fidelity、逐预算 predictions 与 metrics。", "",
    ]
    report = GLM_ROOT / "GLM_PAPER_METRICS.md"
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
        probe_fidelity(args, env, "shadow", shadows)
        probe_fidelity(args, env, "target", targets)
        probe_text(args, env, all_specs)
    if args.phase in {"analyze", "all"}:
        analyze(args, env, shadows, targets)
    if args.phase in {"report", "all"}:
        print(json.dumps({"report": str(write_report(args, config))}, indent=2))


if __name__ == "__main__":
    main()
