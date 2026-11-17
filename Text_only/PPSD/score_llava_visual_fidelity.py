#!/usr/bin/env python3
"""Score private PPSD visual-update fidelity from A/B next-token logits.

This is an experimenter-side update validation only.  Attack features remain
restricted to generated text in the paired black-box protocol.
"""

from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from Text_only.PPSD.protocol import DATA_ROOT
from Text_only.train_llava_dpo import DEFAULT_MODEL


TARGET_ID_PATTERN = re.compile(r"target_[a-z0-9]+$")
PROMPT = (
    "Image A is the first image and Image B is the second image. "
    "Which image has the more appealing visual treatment? Answer with exactly A or B."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--probe-bank", type=Path, default=DATA_ROOT / "private_fidelity_probe_bank.json"
    )
    parser.add_argument("--adapter", action="append", default=[], metavar="TARGET_ID=PATH")
    parser.add_argument("--target-label", action="append", default=[], metavar="TARGET_ID=STYLE_N")
    parser.add_argument("--reference-preference", default="style_0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_mapping(values: list[str], *, paths: bool) -> dict:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, got {value}")
        key, raw = value.split("=", 1)
        if not TARGET_ID_PATTERN.fullmatch(key) or key in result:
            raise ValueError(f"Invalid or duplicate anonymous ID: {key}")
        result[key] = Path(raw).resolve() if paths else raw
    return result


def score(model, processor, prompt: str, images: list[Image.Image], token_ids: tuple[int, int]) -> tuple[float, float]:
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    formatted = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=formatted, images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
        pair = torch.log_softmax(logits[list(token_ids)], dim=0).cpu().tolist()
    return float(pair[0]), float(pair[1])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    adapters = parse_mapping(args.adapter, paths=True)
    labels = parse_mapping(args.target_label, paths=False)
    if set(adapters) != set(labels):
        raise ValueError("Adapter IDs and target-label IDs must match.")
    if any(not re.fullmatch(r"style_\d+", value) for value in labels.values()):
        raise ValueError("PPSD labels must use the style_N form.")
    if args.reference_preference not in set(labels.values()):
        raise ValueError("The reference preference must occur in the target labels.")
    for path in adapters.values():
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(path / "adapter_config.json")

    queries = json.loads(args.probe_bank.read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    token_a = processor.tokenizer.encode("A", add_special_tokens=False)
    token_b = processor.tokenizer.encode("B", add_special_tokens=False)
    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError("A and B must each be a single token.")
    base = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to("cuda").eval()
    first_id = next(iter(adapters))
    model = PeftModel.from_pretrained(
        base, adapters[first_id], adapter_name=first_id, is_trainable=False
    )
    for model_id, path in list(adapters.items())[1:]:
        model.load_adapter(path, adapter_name=model_id, is_trainable=False)
    model.eval()

    versions = [("M0_reference", None), *adapters.items()]
    raw = []
    for model_id, path in versions:
        if path is None:
            context = model.disable_adapter()
        else:
            model.set_adapter(model_id)
            context = nullcontext()
        with context:
            for query in queries:
                images = []
                for image_path in query["image_paths"]:
                    with Image.open(image_path) as loaded:
                        images.append(loaded.convert("RGB").copy())
                logp_a, logp_b = score(model, processor, PROMPT, images, (token_a[0], token_b[0]))
                raw.append(
                    {
                        "model_version": model_id,
                        "query_index": query["query_index"],
                        "reference_option": query["reference_option"],
                        "logp_a_normalized": logp_a,
                        "logp_b_normalized": logp_b,
                    }
                )
        print(f"completed {model_id}", flush=True)

    base_rows = [row for row in raw if row["model_version"] == "M0_reference"]
    base_by_query = {row["query_index"]: row for row in base_rows}
    per_model = []
    for model_id, preference in labels.items():
        rows = [row for row in raw if row["model_version"] == model_id]
        signed_deltas = []
        correct = []
        base_correct = []
        for row in rows:
            desired = row["reference_option"] if preference == args.reference_preference else (
                "B" if row["reference_option"] == "A" else "A"
            )
            margin = row["logp_a_normalized"] - row["logp_b_normalized"]
            base = base_by_query[row["query_index"]]
            base_margin = base["logp_a_normalized"] - base["logp_b_normalized"]
            sign = 1.0 if desired == "A" else -1.0
            signed_deltas.append(sign * (margin - base_margin))
            correct.append(sign * margin > 0)
            base_correct.append(sign * base_margin > 0)
        per_model.append(
            {
                "anonymous_model_id": model_id,
                "private_preference": preference,
                "mean_preference_log_odds_delta": sum(signed_deltas) / len(signed_deltas),
                "fraction_positive_preference_delta": sum(x > 0 for x in signed_deltas) / len(signed_deltas),
                "post_pairwise_accuracy": sum(correct) / len(correct),
                "base_pairwise_accuracy": sum(base_correct) / len(base_correct),
            }
        )
    result = {
        "evaluation": "experimenter_side_ppsd_visual_next_token_fidelity",
        "attacker_access_to_logits": False,
        "num_queries": len(queries),
        "per_model": per_model,
        "raw_scores": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "raw_scores": f"{len(raw)} rows"}, indent=2))


if __name__ == "__main__":
    main()
