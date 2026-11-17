#!/usr/bin/env python3
"""Collect strict preference-remote text outputs from GLM-4.1V.

The collector exposes only generated strings, never images, logits, or model
parameters.  Each anonymous model is written to an atomic shard so a long 9B
run can be resumed without repeating completed model versions.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.GLM.train_glm_dpo import DEFAULT_MODEL


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
            "Optional full bank that fixes probe RNG indices for frozen subsets."
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
        help="Emit this anonymous version with every adapter disabled.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=27191)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Eight keeps all repetitions of one probe in the same RNG batch.",
    )
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
    if not parsed or len({row[0] for row in parsed}) != len(parsed):
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


def formatted_prompt(processor, prompt: str) -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    model,
    processor,
    prompts: list[str],
    args: argparse.Namespace,
    seed: int,
) -> list[str]:
    formatted = [formatted_prompt(processor, prompt) for prompt in prompts]
    inputs = processor(text=formatted, return_tensors="pt", padding=True).to(model.device)
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
    return [row.strip() for row in processor.batch_decode(completion, skip_special_tokens=True)]


def shard_is_complete(path: Path, expected_rows: int, model_id: str) -> bool:
    if not path.is_file():
        return False
    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["model_version"] != model_id:
                    return False
                count += 1
    except (OSError, KeyError, json.JSONDecodeError):
        return False
    return count == expected_rows


def collect_version(
    model,
    processor,
    model_id: str,
    bank: list[dict],
    seed_index: dict[str, int],
    args: argparse.Namespace,
    destination: Path,
) -> None:
    expected = len(bank) * args.repetitions
    if shard_is_complete(destination, expected, model_id):
        print(f"reuse {model_id} ({expected} rows)", flush=True)
        return
    tasks = [
        (seed_index[probe["probe_id"]], probe, repetition)
        for probe in bank
        for repetition in range(args.repetitions)
    ]
    temporary = destination.with_suffix(".jsonl.tmp")
    offset = stable_model_offset(model_id)
    with temporary.open("w", encoding="utf-8") as handle:
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            # With the frozen batch size of eight, this seed names one
            # anonymous model/probe stream containing eight independent draws.
            seed = (args.seed + offset + batch[0][0] * 1009) % (2**31 - 1)
            responses = generate_batch(
                model,
                processor,
                [probe["prompt"] for _, probe, _ in batch],
                args,
                seed,
            )
            for (probe_index, probe, repetition), response in zip(batch, responses):
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
    temporary.replace(destination)
    print(f"completed {model_id} ({expected} rows)", flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLM probing.")
    if args.repetitions < 1 or args.batch_size < 1:
        raise ValueError("Repetitions and batch size must be positive.")
    if args.batch_size != args.repetitions:
        raise ValueError(
            "For the frozen protocol, batch size must equal repetitions so each "
            "probe has a separate deterministic RNG stream."
        )
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

    attacker_dir = args.output_dir / "attacker_view"
    shard_dir = attacker_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
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

    reference = [] if args.omit_reference else [("M0_reference", None)]
    versions = (
        [*reference, *[(model_id, None) for model_id in control_ids]]
        if control_ids else [*reference, *adapters]
    )
    for index, (model_id, adapter_path) in enumerate(versions, start=1):
        if adapter_path is None:
            context = model.disable_adapter()
        else:
            model.set_adapter(model_id)
            context = nullcontext()
        with context:
            collect_version(
                model,
                processor,
                model_id,
                bank,
                seed_index,
                args,
                shard_dir / f"{model_id}.jsonl",
            )
        print(f"version {index}/{len(versions)}", flush=True)

    output = attacker_dir / "remote_generations.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for model_id, _ in versions:
            shard = shard_dir / f"{model_id}.jsonl"
            if not shard_is_complete(shard, len(bank) * args.repetitions, model_id):
                raise RuntimeError(f"Incomplete generation shard: {shard}")
            with shard.open(encoding="utf-8") as source:
                for line in source:
                    destination.write(line)
    temporary.replace(output)

    metadata = {
        "base_model_family": "GLM-4.1V-9B-Base",
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
            "batch_size": args.batch_size,
            "seed_scope": "anonymous_model_and_probe",
        },
        "probe_bank_sha256": hashlib.sha256(args.probe_bank.read_bytes()).hexdigest(),
        "seed_reference_bank_sha256": hashlib.sha256(
            (args.seed_reference_bank or args.probe_bank).read_bytes()
        ).hexdigest(),
        "contains_images": False,
        "contains_preference_or_visual_terms": False,
        "control_base_only": bool(control_ids),
        "includes_pre_update_reference": not args.omit_reference,
        "resumable_atomic_model_shards": True,
    }
    (attacker_dir / "public_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
