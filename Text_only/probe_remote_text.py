#!/usr/bin/env python3
"""Collect text-only black-box outputs for a frozen remote probe bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.train_llava_dpo import DEFAULT_MODEL


TARGET_ID_PATTERN = re.compile(r"target_[a-z0-9]+$")
DIRECT_CHOICE_PATTERN = re.compile(r"^\s*([AB])(?:[.):,-]|\s*$)", re.I)
EXPLICIT_CHOICE_PATTERN = re.compile(
    r"\b(?:answer|choice|option)\s*(?:is|:)?\s*([AB])(?:\s|[.):,;-]|$)",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--probe-bank", type=Path, required=True)
    parser.add_argument(
        "--seed-reference-bank",
        type=Path,
        help=(
            "Optional full bank that fixes each probe's RNG index when querying "
            "a frozen subset."
        ),
    )
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument(
        "--omit-reference",
        action="store_true",
        help="Do not query M0; required for an exact-budget post-update-only run.",
    )
    parser.add_argument(
        "--control-base-id",
        action="append",
        default=[],
        help="Emit an anonymous version using the disabled-adapter base model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=27191)
    parser.add_argument("--max-probes", type=int)
    return parser.parse_args()


def parse_adapters(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Adapter must be MODEL_ID=PATH: {value}")
        model_id, raw_path = value.split("=", 1)
        if not TARGET_ID_PATTERN.fullmatch(model_id):
            raise ValueError(f"Invalid anonymous model ID: {model_id}")
        path = Path(raw_path).resolve()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(path / "adapter_config.json")
        parsed.append((model_id, path))
    if not parsed or len({value[0] for value in parsed}) != len(parsed):
        raise ValueError("At least one unique adapter is required.")
    return parsed


def load_bank(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    required = {"probe_id", "family", "kind", "prompt", "expected_answer"}
    if not rows or any(set(row) != required for row in rows):
        raise ValueError("Remote probe bank has an unexpected schema.")
    if len({row["probe_id"] for row in rows}) != len(rows):
        raise ValueError("Remote probe IDs must be unique.")
    return rows


def parse_choice(response: str) -> str | None:
    normalized = response.strip()
    match = (
        DIRECT_CHOICE_PATTERN.search(normalized)
        or EXPLICIT_CHOICE_PATTERN.search(normalized)
    )
    return match.group(1).upper() if match else None


def stable_model_offset(model_id: str) -> int:
    return int.from_bytes(hashlib.sha256(model_id.encode()).digest()[:4], "big")


def generate(model, processor, prompt: str, args: argparse.Namespace, seed: int) -> str:
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=formatted, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    completion = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(completion, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaVA probing.")
    adapters = parse_adapters(args.adapter)
    control_ids = list(args.control_base_id)
    if control_ids:
        if any(not TARGET_ID_PATTERN.fullmatch(value) for value in control_ids):
            raise ValueError("Control IDs must use anonymous target_* identifiers.")
        if len(control_ids) != len(set(control_ids)):
            raise ValueError("Control IDs must be unique.")
    bank = load_bank(args.probe_bank)
    seed_bank = load_bank(args.seed_reference_bank) if args.seed_reference_bank else bank
    seed_index = {probe["probe_id"]: index for index, probe in enumerate(seed_bank)}
    missing_seed_ids = sorted({probe["probe_id"] for probe in bank} - set(seed_index))
    if missing_seed_ids:
        raise ValueError(f"Seed reference bank lacks probes: {missing_seed_ids}")
    if args.max_probes is not None:
        bank = bank[: args.max_probes]
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive.")

    attacker_dir = args.output_dir / "attacker_view"
    attacker_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    first_id, first_path = adapters[0]
    model = PeftModel.from_pretrained(
        base, first_path, adapter_name=first_id, is_trainable=False
    )
    for model_id, path in adapters[1:]:
        model.load_adapter(path, adapter_name=model_id, is_trainable=False)
    model.eval()

    output = attacker_dir / "remote_generations.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    reference = [] if args.omit_reference else [("M0_reference", None)]
    versions = (
        [*reference, *[(model_id, None) for model_id in control_ids]]
        if control_ids else [*reference, *adapters]
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for model_index, (model_id, adapter_path) in enumerate(versions):
            context = model.disable_adapter() if adapter_path is None else nullcontext()
            if adapter_path is not None:
                model.set_adapter(model_id)
            with context:
                offset = stable_model_offset(model_id)
                for probe in bank:
                    probe_index = seed_index[probe["probe_id"]]
                    for repetition in range(args.repetitions):
                        # Each model version receives an independent reproducible
                        # sample stream, matching an ordinary black-box API.
                        seed = (
                            args.seed + offset + probe_index * 1009 + repetition * 9176
                        ) % (2**31 - 1)
                        response = generate(model, processor, probe["prompt"], args, seed)
                        record = {
                            "model_version": model_id,
                            "modality": "text_only",
                            "probe_id": probe["probe_id"],
                            "repetition": repetition,
                            "family": probe["family"],
                            "kind": probe["kind"],
                            "prompt": probe["prompt"],
                            "response": response,
                            "parsed_choice": parse_choice(response)
                            if probe["kind"] == "choice" else None,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"completed {model_id} ({model_index + 1}/{len(versions)})", flush=True)
    temporary.replace(output)

    model_path_lower = str(args.model).lower()
    if "qwen2-7b" in model_path_lower:
        base_model_family = "llava-onevision-qwen2-7b-ov-hf"
    elif "qwen2-0.5b" in model_path_lower:
        base_model_family = "llava-onevision-qwen2-0.5b-ov-hf"
    else:
        base_model_family = "llava-onevision-qwen2-unknown"

    metadata = {
        "base_model_family": base_model_family,
        "base_model_path": str(args.model.resolve()),
        "query_modality": "text_only",
        "attack_setting": "strict_preference_remote_behavior",
        "num_anonymous_updates": len(control_ids) if control_ids else len(adapters),
        "num_probes": len(bank),
        "repetitions": args.repetitions,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "randomness_across_models": "independent_reproducible",
        },
        "probe_bank_sha256": hashlib.sha256(args.probe_bank.read_bytes()).hexdigest(),
        "seed_reference_bank_sha256": hashlib.sha256(
            (args.seed_reference_bank or args.probe_bank).read_bytes()
        ).hexdigest(),
        "contains_images": False,
        "contains_preference_or_visual_terms": False,
        "control_base_only": bool(control_ids),
        "includes_pre_update_reference": not args.omit_reference,
    }
    (attacker_dir / "public_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
