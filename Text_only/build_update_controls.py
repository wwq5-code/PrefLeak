#!/usr/bin/env python3
"""Build matched neutral and stratified label-shuffled DPO controls.

Each input pair must contain identical visual comparisons with exactly reversed
chosen/rejected answers.  Controls keep the row count, images, prompts, and
answer vocabulary fixed.  Neutral controls cancel treatment within every
two-order scene; shuffled controls have exactly 50% agreement with either
preference while preserving the global A/B answer count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--neutral-id", required=True)
    parser.add_argument("--shuffled-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer(row: dict, field: str = "chosen") -> str:
    messages = row[field]
    if len(messages) != 1 or messages[0].get("role") != "assistant":
        raise ValueError(f"Unexpected {field} schema in {row.get('sample_id')}")
    value = messages[0].get("content")
    if value not in {"A", "B"}:
        raise ValueError(f"Controls require opaque A/B answers, got {value!r}")
    return value


def set_choice(row: dict, chosen: str) -> dict:
    result = json.loads(json.dumps(row))
    rejected = "B" if chosen == "A" else "A"
    result["chosen"] = [{"role": "assistant", "content": chosen}]
    result["rejected"] = [{"role": "assistant", "content": rejected}]
    return result


def validate_pair(first: list[dict], second: list[dict]) -> None:
    by_id_first = {row["sample_id"]: row for row in first}
    by_id_second = {row["sample_id"]: row for row in second}
    if len(by_id_first) != len(first) or set(by_id_first) != set(by_id_second):
        raise ValueError("Counterfactual manifests have different or duplicate sample IDs.")
    for sample_id, left in by_id_first.items():
        right = by_id_second[sample_id]
        public_fields = {"source_image_id", "order_index", "images", "prompt"}
        if any(left[field] != right[field] for field in public_fields):
            raise ValueError(f"Counterfactual public fields differ for {sample_id}")
        if answer(left) != answer(right, "rejected") or answer(left, "rejected") != answer(right):
            raise ValueError(f"Answers are not exactly reversed for {sample_id}")


def neutral_rows(rows: list[dict], rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_image_id"]].append(row)
    if not groups or any(len(group) != 2 for group in groups.values()):
        raise ValueError("Neutral control requires two presentation orders per scene.")
    scene_ids = sorted(groups)
    if len(scene_ids) % 2:
        raise ValueError("Neutral control requires an even number of scenes.")
    rng.shuffle(scene_ids)
    choice_by_scene = {
        scene_id: ("A" if index < len(scene_ids) // 2 else "B")
        for index, scene_id in enumerate(scene_ids)
    }
    # The two rows reverse visual order but choose the same position, so each
    # visual treatment wins exactly once within every scene.
    return [set_choice(row, choice_by_scene[row["source_image_id"]]) for row in rows]


def shuffled_rows(rows: list[dict], rng: random.Random) -> list[dict]:
    by_original = {
        choice: [index for index, row in enumerate(rows) if answer(row) == choice]
        for choice in ("A", "B")
    }
    if any(len(indices) % 2 for indices in by_original.values()):
        raise ValueError("Stratified shuffle requires even original A and B counts.")
    keep = set()
    for indices in by_original.values():
        rng.shuffle(indices)
        keep.update(indices[: len(indices) // 2])
    result = []
    for index, row in enumerate(rows):
        original = answer(row)
        chosen = original if index in keep else ("B" if original == "A" else "A")
        result.append(set_choice(row, chosen))
    return result


def audit(rows: list[dict], source: list[dict], kind: str) -> dict:
    choices = Counter(answer(row) for row in rows)
    agreements = sum(answer(row) == answer(original) for row, original in zip(rows, source))
    result = {
        "kind": kind,
        "num_rows": len(rows),
        "chosen_counts": dict(sorted(choices.items())),
        "agreement_with_first_preference": agreements / len(rows),
        "same_public_fields": all(
            all(row[field] == original[field] for field in ("sample_id", "source_image_id", "order_index", "images", "prompt"))
            for row, original in zip(rows, source)
        ),
    }
    if choices["A"] != choices["B"] or result["agreement_with_first_preference"] != 0.5:
        raise ValueError(f"Control is not exactly balanced: {result}")
    if not result["same_public_fields"]:
        raise ValueError("A control changed a public visual-comparison field.")
    if kind == "neutral":
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["source_image_id"]].append(row)
        if any(len({answer(row) for row in group}) != 1 for group in grouped.values()):
            raise ValueError("Neutral within-scene cancellation failed.")
        result["within_scene_visual_treatment_cancels"] = True
    return result


def main() -> None:
    args = parse_args()
    first = read_jsonl(args.first)
    second = read_jsonl(args.second)
    validate_pair(first, second)
    neutral = neutral_rows(first, random.Random(args.seed))
    shuffled = shuffled_rows(first, random.Random(args.seed + 1))
    neutral_audit = audit(neutral, first, "neutral")
    shuffled_audit = audit(shuffled, first, "label_shuffled")

    output_dir = args.data_root / "private_controls"
    neutral_path = output_dir / f"{args.neutral_id}.jsonl"
    shuffled_path = output_dir / f"{args.shuffled_id}.jsonl"
    write_jsonl(neutral_path, neutral)
    write_jsonl(shuffled_path, shuffled)
    manifest = {
        "control_protocol": "matched_preference_direction_controls_v1",
        "seed": args.seed,
        "source_pair": {
            "first": str(args.first.resolve()),
            "first_sha256": sha256(args.first),
            "second": str(args.second.resolve()),
            "second_sha256": sha256(args.second),
            "counterfactual_pair_validated": True,
        },
        "neutral": {
            "model_id": args.neutral_id,
            "path": str(neutral_path.resolve()),
            "sha256": sha256(neutral_path),
            **neutral_audit,
        },
        "label_shuffled": {
            "model_id": args.shuffled_id,
            "path": str(shuffled_path.resolve()),
            "sha256": sha256(shuffled_path),
            **shuffled_audit,
        },
    }
    metadata_path = output_dir / f"{args.neutral_id}_{args.shuffled_id}.control.json"
    metadata_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(metadata_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
