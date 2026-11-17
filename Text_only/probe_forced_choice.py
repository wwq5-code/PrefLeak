#!/usr/bin/env python3
"""Run known-candidate A/B probes with common randomness across model versions."""

import argparse
import hashlib
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.train_llava_dpo import DEFAULT_MODEL
from Text_only.build_visual_preferences import PROMPT as TRAINING_CHOICE_PROMPT


TARGET_ID_PATTERN = re.compile(r"target_[a-z0-9]+$")
DIRECT_CHOICE_PATTERN = re.compile(r"^\s*([AB])(?:\s|[.):,-]|$)", re.I)
NAMED_CHOICE_PATTERN = re.compile(r"\b(?:option|image)\s*([AB])\b", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--modality",
        choices=("text_only", "vision_conditioned", "multimodal"),
        required=True,
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=Path(__file__).resolve().parent
        / "prompts"
        / "magenta_cyan_known_preference_concepts.json",
    )
    parser.add_argument("--probe-bank", type=Path)
    parser.add_argument(
        "--visual-prompt-style", choices=("ab", "training"), default="ab"
    )
    parser.add_argument("--adapter", action="append", default=[], metavar="TARGET_ID=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--reference-description", default="a magenta-tinted treatment")
    parser.add_argument("--alternative-description", default="a cyan-tinted treatment")
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
    match = DIRECT_CHOICE_PATTERN.search(normalized) or NAMED_CHOICE_PATTERN.search(normalized)
    if match:
        return match.group(1).upper()
    lowered = normalized.lower()
    if re.search(r"\bimage\s*1\b", lowered) and not re.search(r"\bimage\s*2\b", lowered):
        return "A"
    if re.search(r"\bimage\s*2\b", lowered) and not re.search(r"\bimage\s*1\b", lowered):
        return "B"
    if "first" in lowered and "second" not in lowered:
        return "A"
    if "second" in lowered and "first" not in lowered:
        return "B"
    return None


def load_text_queries(path: Path) -> list[dict]:
    concepts = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for index, concept in enumerate(concepts):
        if set(concept) == {"family", "context", "reference", "alternative"}:
            reference, alternative = concept["reference"], concept["alternative"]
        else:
            raise ValueError(f"Unexpected concept fields: {set(concept)}")
        for reference_first in (True, False):
            first = reference if reference_first else alternative
            second = alternative if reference_first else reference
            query = (
                f"For {concept['context']}, which visual treatment better matches the "
                "aesthetic you generally prefer?\n"
                f"A. {first}\nB. {second}\nAnswer with exactly A or B."
            )
            queries.append(
                {
                    "query_index": index * 2 + (0 if reference_first else 1),
                    "query": query,
                    "image_id": None,
                    "image_paths": [],
                    "reference_option": "A" if reference_first else "B",
                    "family": concept["family"],
                }
            )
    return queries


def load_visual_queries(
    path: Path,
    modality: str,
    prompt_style: str = "ab",
    reference_description: str = "a warm golden treatment",
    alternative_description: str = "a cool blue treatment",
) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    required = {"query_index", "image_id", "image_paths", "source_image_id"}
    queries = []
    for record in records:
        if set(record) != required | {"reference_option"}:
            raise ValueError(f"Unexpected choice-probe fields: {set(record) - required}")
        reference_option = record["reference_option"]
        resolved = [(path.parent / value).resolve() for value in record["image_paths"]]
        if len(resolved) not in {1, 2} or not all(value.is_file() for value in resolved):
            raise FileNotFoundError(f"Choice probe requires one composite or two images: {resolved}")
        if prompt_style == "training":
            query = TRAINING_CHOICE_PROMPT if len(resolved) == 2 else (
                "A single comparison image is provided. Option A is the left half and "
                "option B is the right half. Which option has the more appealing visual "
                "treatment? Answer with exactly A or B."
            )
        elif modality == "vision_conditioned":
            query = (
                "Two images are provided. Image A is the first image and Image B is the "
                "second image. Which image has the more appealing visual treatment? "
                "Answer with exactly A or B."
                if len(resolved) == 2
                else "A single comparison image is provided. Option A is the left half and "
                "option B is the right half. Which option has the more appealing visual "
                "treatment? Answer with exactly A or B."
            )
        else:
            location = (
                "Image A is the first image and Image B is the second image."
                if len(resolved) == 2
                else "Option A is the left half and option B is the right half."
            )
            query = (
                f"{location} The candidate styles are {reference_description} and "
                f"{alternative_description}. Which option better matches the visual style you generally "
                "prefer? Answer with exactly A or B."
            )
        queries.append(
            {
                "query_index": record["query_index"],
                "query": query,
                "image_id": record["image_id"],
                "image_paths": resolved,
                "reference_option": reference_option,
                "family": "visual_pair",
            }
        )
    return sorted(queries, key=lambda row: row["query_index"])


def build_queries(args: argparse.Namespace) -> list[dict]:
    if args.modality == "text_only":
        return load_text_queries(args.concepts)
    if args.probe_bank is None:
        raise ValueError(f"--probe-bank is required for {args.modality}.")
    return load_visual_queries(
        args.probe_bank,
        args.modality,
        args.visual_prompt_style,
        args.reference_description,
        args.alternative_description,
    )


def generate_response(
    model,
    processor,
    prompt: str,
    images: list[Image.Image],
    args: argparse.Namespace,
    seed: int,
) -> str:
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    processor_kwargs = {"text": formatted, "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs).to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    completion = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(completion, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaVA probing.")
    adapters = parse_adapters(args.adapter)
    if not adapters:
        raise ValueError("Provide at least one --adapter TARGET_ID=PATH.")
    queries = build_queries(args)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attacker_dir = args.output_dir / "attacker_view"
    attacker_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    first_id, first_path = adapters[0]
    model = PeftModel.from_pretrained(
        base_model, first_path, adapter_name=first_id, is_trainable=False
    )
    for model_id, path in adapters[1:]:
        model.load_adapter(path, adapter_name=model_id, is_trainable=False)
    model.eval()

    output_path = attacker_dir / "forced_choice_generations.jsonl"
    temporary = output_path.with_suffix(".jsonl.tmp")
    model_versions = [("M0_reference", None), *adapters]
    with temporary.open("w", encoding="utf-8") as handle:
        for model_index, (model_id, adapter_path) in enumerate(model_versions):
            if adapter_path is None:
                adapter_context = model.disable_adapter()
            else:
                model.set_adapter(model_id)
                adapter_context = nullcontext()
            with adapter_context:
                for query_spec in queries:
                    images = []
                    for image_path in query_spec["image_paths"]:
                        with Image.open(image_path) as loaded:
                            images.append(loaded.convert("RGB").copy())
                    for repetition in range(args.repetitions):
                        # Common random numbers isolate the update effect from sampling noise.
                        seed = args.seed + query_spec["query_index"] * 1009 + repetition
                        response = generate_response(
                            model, processor, query_spec["query"], images, args, seed
                        )
                        choice = parse_choice(response)
                        record = {
                            "model_version": model_id,
                            "modality": args.modality,
                            "query_index": query_spec["query_index"],
                            "repetition": repetition,
                            "query": query_spec["query"],
                            "image_id": query_spec["image_id"],
                            "reference_option": query_spec["reference_option"],
                            "family": query_spec["family"],
                            "response": response,
                            "parsed_choice": choice,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"completed {model_id} ({model_index + 1}/{len(model_versions)})", flush=True)
    temporary.replace(output_path)

    source_path = args.concepts if args.modality == "text_only" else args.probe_bank
    public_metadata = {
        "base_model_family": args.model.name,
        "base_model_path": str(args.model.resolve()),
        "query_modality": args.modality,
        "attack_setting": "known_candidate_forced_choice",
        "visual_prompt_style": args.visual_prompt_style if args.modality != "text_only" else None,
        "num_anonymous_targets": len(adapters),
        "num_queries": len(queries),
        "repetitions": args.repetitions,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "randomness_across_models": "common",
        },
        "query_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "parsed_response_rate": None,
    }
    (attacker_dir / "public_metadata.json").write_text(
        json.dumps(public_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path), **public_metadata}, indent=2))


if __name__ == "__main__":
    main()
