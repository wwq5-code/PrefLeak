#!/usr/bin/env python3
"""Train a pure-DPO joint GLM-4.1V LoRA update on visual comparisons."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset, Image, Sequence
from peft import LoraConfig, TaskType
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TRL = PROJECT_ROOT / "trl"
if LOCAL_TRL.is_dir():
    sys.path.insert(0, str(LOCAL_TRL))

from trl import DPOConfig, DPOTrainer  # noqa: E402


DEFAULT_MODEL = Path("zai-org/GLM-4.1V-9B-Base")
VISION_LORA_PATTERN = (
    r"model\.visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
)
PROJECTOR_LORA_PATTERN = (
    r"model\.visual\.merger\.(?:proj|gate_proj|up_proj|down_proj)"
)
LANGUAGE_LORA_PATTERN = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_up_proj|down_proj))"
)
JOINT_VLM_LORA_PATTERN = (
    f"(?:{VISION_LORA_PATTERN}|{PROJECTOR_LORA_PATTERN}|{LANGUAGE_LORA_PATTERN})"
)
PURE_DPO_LOSS_TYPES = ("sigmoid",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--seed", type=int, default=12401)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--vision-lora-rank", type=int)
    parser.add_argument("--vision-lora-alpha", type=int)
    parser.add_argument("--projector-lora-rank", type=int)
    parser.add_argument("--projector-lora-alpha", type=int)
    parser.add_argument("--language-lora-rank", type=int)
    parser.add_argument("--language-lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        help="Optional longest-edge pixel-area cap applied without cropping.",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dataset(path: Path) -> Dataset:
    root = path.parent.parent
    rows = [
        {
            "images": [str((root / image_path).resolve()) for image_path in row["images"]],
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in read_jsonl(path)
    ]
    dataset = Dataset.from_list(rows)
    features = dataset.features.copy()
    features["images"] = Sequence(Image(decode=True))
    return dataset.cast(features)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLM-4.1V DPO training.")
    if not 0.0 <= args.label_smoothing < 0.5:
        raise ValueError("--label-smoothing must be in [0, 0.5).")
    component_values = (
        args.lora_rank,
        args.lora_alpha,
        args.vision_lora_rank,
        args.vision_lora_alpha,
        args.projector_lora_rank,
        args.projector_lora_alpha,
        args.language_lora_rank,
        args.language_lora_alpha,
    )
    if any(value is not None and value <= 0 for value in component_values):
        raise ValueError("All LoRA ranks and alpha values must be positive.")
    if args.max_image_pixels is not None and args.max_image_pixels < 12544:
        raise ValueError("--max-image-pixels must be at least 12544.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    dataset = load_dataset(args.train_manifest)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if args.max_image_pixels is not None:
        processor.image_processor.size["longest_edge"] = args.max_image_pixels
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = False

    matched_modules = {
        "vision": sorted(
            name for name, _ in model.named_modules()
            if re.fullmatch(VISION_LORA_PATTERN, name)
        ),
        "projector": sorted(
            name for name, _ in model.named_modules()
            if re.fullmatch(PROJECTOR_LORA_PATTERN, name)
        ),
        "language": sorted(
            name for name, _ in model.named_modules()
            if re.fullmatch(LANGUAGE_LORA_PATTERN, name)
        ),
    }
    missing = [component for component, modules in matched_modules.items() if not modules]
    if missing:
        raise RuntimeError(f"Joint GLM LoRA matched no modules for: {missing}")

    component_lora = {
        "vision": {
            "rank": args.vision_lora_rank or args.lora_rank,
            "alpha": args.vision_lora_alpha or args.lora_alpha,
        },
        "projector": {
            "rank": args.projector_lora_rank or args.lora_rank,
            "alpha": args.projector_lora_alpha or args.lora_alpha,
        },
        "language": {
            "rank": args.language_lora_rank or args.lora_rank,
            "alpha": args.language_lora_alpha or args.lora_alpha,
        },
    }
    rank_pattern = {
        VISION_LORA_PATTERN: component_lora["vision"]["rank"],
        PROJECTOR_LORA_PATTERN: component_lora["projector"]["rank"],
        LANGUAGE_LORA_PATTERN: component_lora["language"]["rank"],
    }
    alpha_pattern = {
        VISION_LORA_PATTERN: component_lora["vision"]["alpha"],
        PROJECTOR_LORA_PATTERN: component_lora["projector"]["alpha"],
        LANGUAGE_LORA_PATTERN: component_lora["language"]["alpha"],
    }
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=args.lora_dropout,
        target_modules=JOINT_VLM_LORA_PATTERN,
        bias="none",
    )
    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        run_name=args.target_id,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        loss_type=list(PURE_DPO_LOSS_TYPES),
        bf16=True,
        tf32=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        max_length=None,
        pad_to_multiple_of=None,
        optim="adamw_torch_fused",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)
    trainer.save_state()

    trainable = sum(
        parameter.numel() for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_by_component = {"vision": 0, "projector": 0, "language": 0}
    for name, parameter in trainer.model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".visual.blocks." in name:
            trainable_by_component["vision"] += parameter.numel()
        elif ".visual.merger." in name:
            trainable_by_component["projector"] += parameter.numel()
        elif ".language_model.layers." in name:
            trainable_by_component["language"] += parameter.numel()
    metadata = {
        "target_id": args.target_id,
        "base_model": str(args.model.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "vision_lora_rank": component_lora["vision"]["rank"],
        "vision_lora_alpha": component_lora["vision"]["alpha"],
        "projector_lora_rank": component_lora["projector"]["rank"],
        "projector_lora_alpha": component_lora["projector"]["alpha"],
        "language_lora_rank": component_lora["language"]["rank"],
        "language_lora_alpha": component_lora["language"]["alpha"],
        "lora_dropout": args.lora_dropout,
        "max_image_pixels": args.max_image_pixels,
        "loss_type": list(PURE_DPO_LOSS_TYPES),
        "loss_weights": None,
        "objective": "dpo",
        "update_scope": "joint_vlm",
        "lora_scope": "vision_language_projector",
        "lora_target_pattern": JOINT_VLM_LORA_PATTERN,
        "matched_module_counts": {
            component: len(modules) for component, modules in matched_modules.items()
        },
        "trainable_parameters": trainable,
        "trainable_parameters_by_component": trainable_by_component,
        "total_parameters_with_adapter": total,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "private_training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
