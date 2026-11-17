#!/usr/bin/env python3
"""Run GLM-4.1V forced-choice probes with common randomness."""

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
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.GLM.train_glm_dpo import DEFAULT_MODEL


TRAINING_PROMPT = (
    "Image A is the first image and Image B is the second image. "
    "Which image has the more appealing visual treatment? Answer with exactly A or B."
)
TARGET_ID_PATTERN = re.compile(r"target_[a-z0-9]+$")
DIRECT_CHOICE_PATTERN = re.compile(r"^\s*([AB])(?:\s|[.):,-]|$)", re.I)
LEADING_NAMED_CHOICE_PATTERN = re.compile(
    r"^\s*(?:option|image)\s*([AB])(?:\s|[.):,-]|$)", re.I
)
FINAL_CHOICE_PATTERN = re.compile(
    r"(?:answer|choice|choose|select|prefer)\s*(?:is|:)?\s*"
    r"(?:option|image)?\s*([AB])\s*[.!]*\s*$",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--modality", choices=("text_only", "vision_conditioned"), required=True
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path(__file__).resolve().parent
        / "prompts" / "magenta_cyan_known_preference_concepts.json",
    )
    parser.add_argument("--probe-bank", type=Path)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Allow a reference-only visual probe without loading an adapter.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        help="Optional longest-edge pixel-area cap applied without cropping.",
    )
    parser.add_argument(
        "--constrained-choice",
        action="store_true",
        help="Constrain visual-fidelity decoding to a single A/B token.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--seed", type=int, default=14901)
    return parser.parse_args()


def parse_adapters(values: list[str]) -> list[tuple[str, Path]]:
    adapters = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Adapter must be TARGET_ID=PATH, got: {value}")
        model_id, raw_path = value.split("=", 1)
        if not TARGET_ID_PATTERN.fullmatch(model_id):
            raise ValueError(f"Invalid anonymous target ID: {model_id}")
        path = Path(raw_path).resolve()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(path / "adapter_config.json")
        adapters.append((model_id, path))
    if len({model_id for model_id, _ in adapters}) != len(adapters):
        raise ValueError("Anonymous target IDs must be unique.")
    return adapters


def parse_choice(response: str) -> str | None:
    normalized = response.strip()
    match = (
        DIRECT_CHOICE_PATTERN.search(normalized)
        or LEADING_NAMED_CHOICE_PATTERN.search(normalized)
        or FINAL_CHOICE_PATTERN.search(normalized)
    )
    if match:
        return match.group(1).upper()
    lowered = normalized.lower()
    if re.fullmatch(r"(?:i\s+)?(?:choose|select|prefer)\s+(?:image\s+)?1[.!]?", lowered):
        return "A"
    if re.fullmatch(r"(?:i\s+)?(?:choose|select|prefer)\s+(?:image\s+)?2[.!]?", lowered):
        return "B"
    if re.fullmatch(r"(?:i\s+)?(?:choose|select|prefer)\s+the\s+first(?:\s+image)?[.!]?", lowered):
        return "A"
    if re.fullmatch(r"(?:i\s+)?(?:choose|select|prefer)\s+the\s+second(?:\s+image)?[.!]?", lowered):
        return "B"
    return None


def load_text_queries(path: Path) -> list[dict]:
    concepts = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for index, concept in enumerate(concepts):
        if set(concept) != {"family", "context", "reference", "alternative"}:
            raise ValueError(f"Unexpected concept fields: {set(concept)}")
        for reference_first in (True, False):
            first = concept["reference"] if reference_first else concept["alternative"]
            second = concept["alternative"] if reference_first else concept["reference"]
            queries.append(
                {
                    "query_index": index * 2 + (0 if reference_first else 1),
                    "query": (
                        f"For {concept['context']}, which visual treatment better matches "
                        f"the aesthetic you generally prefer?\nA. {first}\nB. {second}\n"
                        "Answer with exactly A or B."
                    ),
                    "image_id": None,
                    "image_paths": [],
                    "reference_option": "A" if reference_first else "B",
                    "family": concept["family"],
                }
            )
    return queries


def load_visual_queries(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "query_index", "image_id", "image_paths", "source_image_id", "reference_option"
    }
    queries = []
    for record in records:
        if set(record) != required:
            raise ValueError(f"Unexpected visual-probe fields: {set(record) ^ required}")
        resolved = [(path.parent / value).resolve() for value in record["image_paths"]]
        if len(resolved) != 2 or not all(value.is_file() for value in resolved):
            raise FileNotFoundError(f"Expected two visual alternatives: {resolved}")
        queries.append(
            {
                "query_index": record["query_index"],
                "query": TRAINING_PROMPT,
                "image_id": record["image_id"],
                "image_paths": resolved,
                "reference_option": record["reference_option"],
                "family": "visual_pair",
            }
        )
    return sorted(queries, key=lambda row: row["query_index"])


def generate_response(
    model,
    processor,
    query: dict,
    max_new_tokens: int,
    seed: int,
    constrained_choice_ids: list[int] | None = None,
) -> str:
    images = []
    for path in query["image_paths"]:
        with Image.open(path) as loaded:
            images.append(loaded.convert("RGB").copy())
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": query["query"]})
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    kwargs = {"text": formatted, "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs).to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generation = {
            "max_new_tokens": 1 if constrained_choice_ids else max_new_tokens,
            "do_sample": False,
        }
        if constrained_choice_ids:
            generation["prefix_allowed_tokens_fn"] = (
                lambda _batch_id, _input_ids: constrained_choice_ids
            )
        generated = model.generate(**inputs, **generation)
    completion = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(completion, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GLM probing.")
    adapters = parse_adapters(args.adapter)
    if not adapters and not args.base_only:
        raise ValueError("Provide at least one adapter or pass --base-only.")
    if adapters and args.base_only:
        raise ValueError("--base-only cannot be combined with --adapter.")
    queries = (
        load_text_queries(args.concepts)
        if args.modality == "text_only"
        else load_visual_queries(args.probe_bank)
    )
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    attacker_dir = args.output_dir / "attacker_view"
    attacker_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if args.max_image_pixels is not None:
        if args.max_image_pixels < 12544:
            raise ValueError("--max-image-pixels must be at least 12544.")
        processor.image_processor.size["longest_edge"] = args.max_image_pixels
    base = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    if adapters:
        first_id, first_path = adapters[0]
        model = PeftModel.from_pretrained(
            base, first_path, adapter_name=first_id, is_trainable=False
        )
        for model_id, path in adapters[1:]:
            model.load_adapter(path, adapter_name=model_id, is_trainable=False)
        model.eval()
    else:
        model = base

    output = attacker_dir / "forced_choice_generations.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    versions = [("M0_reference", None), *adapters]
    constrained_choice_ids = None
    if args.constrained_choice:
        constrained_choice_ids = sorted(
            {
                token_id
                for text in ("A", "B", " A", " B")
                for token_id in processor.tokenizer.encode(
                    text, add_special_tokens=False
                )
                if len(processor.tokenizer.encode(text, add_special_tokens=False)) == 1
            }
        )
        if not constrained_choice_ids:
            raise RuntimeError("Tokenizer exposes no single-token A/B choices.")
    with temporary.open("w", encoding="utf-8") as handle:
        for model_index, (model_id, adapter_path) in enumerate(versions):
            if adapter_path is None and adapters:
                context = model.disable_adapter()
            else:
                if adapter_path is not None:
                    model.set_adapter(model_id)
                context = nullcontext()
            with context:
                for query in queries:
                    for repetition in range(args.repetitions):
                        response = generate_response(
                            model,
                            processor,
                            query,
                            args.max_new_tokens,
                            args.seed + query["query_index"] * 1009 + repetition,
                            constrained_choice_ids,
                        )
                        record = {
                            "model_version": model_id,
                            "modality": args.modality,
                            "query_index": query["query_index"],
                            "repetition": repetition,
                            "query": query["query"],
                            "image_id": query["image_id"],
                            "reference_option": query["reference_option"],
                            "family": query["family"],
                            "response": response,
                            "parsed_choice": parse_choice(response),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"completed {model_id} ({model_index + 1}/{len(versions)})", flush=True)
    temporary.replace(output)

    source = args.concepts if args.modality == "text_only" else args.probe_bank
    metadata = {
        "base_model_family": "GLM-4.1V-9B-Base",
        "base_model_path": str(args.model.resolve()),
        "query_modality": args.modality,
        "attack_setting": "known_candidate_forced_choice",
        "num_anonymous_targets": len(adapters),
        "num_queries": len(queries),
        "repetitions": args.repetitions,
        "generation": {
            "max_new_tokens": 1 if args.constrained_choice else args.max_new_tokens,
            "do_sample": False,
            "randomness_across_models": "common",
            "constrained_choice": args.constrained_choice,
        },
        "max_image_pixels": args.max_image_pixels,
        "query_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (attacker_dir / "public_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
