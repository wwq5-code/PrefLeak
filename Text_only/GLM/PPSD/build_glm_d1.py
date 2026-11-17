#!/usr/bin/env python3
"""Build target-blind, pair-diverse PPSD D1 manifests for GLM."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from Text_only.GLM.PPSD.protocol import (
    COLLECTION,
    DATA_ROOT,
    NUM_SHADOW_PAIRS,
    NUM_TARGET_PAIRS,
    SOURCE_ROOT,
)


PROMPT = (
    "Image A is the first image and Image B is the second image. "
    "Which image has the more appealing visual treatment? Answer with exactly A or B."
)
SEED = 53191
NUM_SHADOW_SCENES = 40
NUM_TARGET_SCENES = 50
NUM_SHADOW_FIDELITY_SCENES = 15
NUM_TARGET_FIDELITY_SCENES = 16
SHADOW_SCENES_PER_UPDATE = 32
TARGET_SCENES_PER_UPDATE = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppsd-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--style-ids", type=int, nargs=2, required=True)
    parser.add_argument("--num-shadow-pairs", type=int, default=NUM_SHADOW_PAIRS)
    parser.add_argument("--num-target-pairs", type=int, default=NUM_TARGET_PAIRS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def conversation(role: str, text: str) -> list[dict[str, str]]:
    return [{"role": role, "content": text}]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def available_scenes(root: Path, styles: tuple[int, int]) -> dict[str, dict[int, Path]]:
    image_root = root / "img_src" / "src" / COLLECTION
    result: dict[str, dict[int, Path]] = {}
    first_style, second_style = styles
    for first in sorted(image_root.glob(f"{COLLECTION}-*_{first_style}.jpg")):
        scene_id = first.stem.rsplit("_", 1)[0]
        second = image_root / f"{scene_id}_{second_style}.jpg"
        if second.is_file():
            result[scene_id] = {
                first_style: first.resolve(),
                second_style: second.resolve(),
            }
    return result


def direct_human_votes(root: Path, styles: tuple[int, int]) -> dict:
    path = root / "responses" / "responses" / "raw" / "votes_items.jsonl"
    counts: Counter[int] = Counter()
    voters = {style: set() for style in styles}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("collection") != COLLECTION:
                continue
            if {int(row["left_style"]), int(row["right_style"])} != set(styles):
                continue
            choice = row.get("choice")
            if choice not in {"left", "right"}:
                continue
            selected = int(row[f"{choice}_style"])
            counts[selected] += 1
            voters[selected].add(str(row["user_id"]))
    total = sum(counts.values())
    if not total or any(counts[style] == 0 for style in styles):
        raise ValueError(f"No two-sided human support for D1 styles {styles}")
    result = {"num_direct_votes": total}
    for style in styles:
        result[f"style_{style}_votes"] = counts[style]
        result[f"style_{style}_fraction"] = counts[style] / total
        result[f"style_{style}_unique_voters"] = len(voters[style])
    return result


def split_scenes(scene_ids: list[str], seed: int) -> dict[str, list[str]]:
    expected = (
        NUM_SHADOW_SCENES
        + NUM_TARGET_SCENES
        + NUM_SHADOW_FIDELITY_SCENES
        + NUM_TARGET_FIDELITY_SCENES
    )
    if len(scene_ids) != expected:
        raise ValueError(f"D1 protocol requires {expected} scenes, found {len(scene_ids)}")
    shuffled = sorted(scene_ids)
    random.Random(seed).shuffle(shuffled)
    shadow_end = NUM_SHADOW_SCENES
    target_end = shadow_end + NUM_TARGET_SCENES
    shadow_fidelity_end = target_end + NUM_SHADOW_FIDELITY_SCENES
    return {
        "shadow": shuffled[:shadow_end],
        "target": shuffled[shadow_end:target_end],
        "shadow_fidelity": shuffled[target_end:shadow_fidelity_end],
        "target_fidelity": shuffled[shadow_fidelity_end:],
    }


def pair_scene_subset(
    pool: list[str], count: int, seed: int, split: str, pair_index: int
) -> list[str]:
    offset = 0 if split == "shadow" else 100_000
    return random.Random(seed + offset + pair_index * 1009).sample(pool, count)


def update_rows(
    scenes: list[str],
    paths: dict[str, dict[int, Path]],
    styles: tuple[int, int],
    preferred_style: int,
    seed: int,
    split: str,
    pair_index: int,
) -> list[dict]:
    first_orders = [index % 2 == 0 for index in range(len(scenes))]
    random.Random(seed + pair_index * 9176).shuffle(first_orders)
    rows = []
    for scene_index, (scene_id, first_style_first) in enumerate(zip(scenes, first_orders)):
        for order_index, first_first in enumerate((first_style_first, not first_style_first)):
            order = styles if first_first else tuple(reversed(styles))
            chosen = "A" if order[0] == preferred_style else "B"
            rows.append(
                {
                    "sample_id": f"{split}_p{pair_index:02d}_{scene_index:03d}_o{order_index}",
                    "source_image_id": scene_id,
                    "order_index": order_index,
                    "images": [str(paths[scene_id][style]) for style in order],
                    "prompt": conversation("user", PROMPT),
                    "chosen": conversation("assistant", chosen),
                    "rejected": conversation("assistant", "B" if chosen == "A" else "A"),
                }
            )
    return rows


def fidelity_bank(
    scenes: list[str], paths: dict[str, dict[int, Path]], styles: tuple[int, int], seed: int
) -> list[dict]:
    first_orders = [index % 2 == 0 for index in range(len(scenes))]
    random.Random(seed).shuffle(first_orders)
    rows = []
    for index, (scene_id, first_first) in enumerate(zip(scenes, first_orders)):
        order = styles if first_first else tuple(reversed(styles))
        rows.append(
            {
                "query_index": index,
                "image_id": f"d1_fidelity_{index:03d}",
                "image_paths": [str(paths[scene_id][style]) for style in order],
                "source_image_id": scene_id,
                "reference_option": "A" if first_first else "B",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    styles = tuple(args.style_ids)
    if len(set(styles)) != 2:
        raise ValueError("--style-ids must be distinct")
    if args.num_shadow_pairs < 1 or args.num_target_pairs < 0:
        raise ValueError("Pair counts must be non-negative, with at least one shadow pair")
    paths = available_scenes(args.ppsd_root, styles)
    splits = split_scenes(list(paths), args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    pair_subsets: dict[str, list[str]] = {}
    for split, pair_count, scenes_per_update in (
        ("shadow", args.num_shadow_pairs, SHADOW_SCENES_PER_UPDATE),
        ("target", args.num_target_pairs, TARGET_SCENES_PER_UPDATE),
    ):
        prefix = "b" if split == "shadow" else "u"
        for pair_index in range(1, pair_count + 1):
            selected = pair_scene_subset(
                splits[split], scenes_per_update, args.seed, split, pair_index
            )
            pair_subsets[f"{split}_pair_{pair_index:02d}"] = selected
            for member_index, preferred_style in enumerate(styles):
                serial = (pair_index - 1) * 2 + member_index + 1
                rows = update_rows(
                    selected,
                    paths,
                    styles,
                    preferred_style,
                    args.seed + (0 if split == "shadow" else 10_000),
                    split,
                    pair_index,
                )
                write_jsonl(
                    output / "private_updates" / f"target_{prefix}{serial:03d}.jsonl",
                    rows,
                )

    (output / "private_fidelity_shadow_probe_bank.json").write_text(
        json.dumps(
            fidelity_bank(splits["shadow_fidelity"], paths, styles, args.seed + 20_000),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "private_fidelity_target_probe_bank.json").write_text(
        json.dumps(
            fidelity_bank(splits["target_fidelity"], paths, styles, args.seed + 30_000),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "scene_splits.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )
    (output / "pair_scene_subsets.json").write_text(
        json.dumps(pair_subsets, indent=2) + "\n", encoding="utf-8"
    )
    preferences = [f"style_{style}" for style in styles]
    metadata = {
        "dataset": "PPSD",
        "source_root": str(args.ppsd_root.resolve()),
        "collection": COLLECTION,
        "candidate_preferences": preferences,
        "style_ids": list(styles),
        "preference_definition": "opposing PPSD photographic style identities",
        "human_preference_support": direct_human_votes(args.ppsd_root, styles),
        "seed": args.seed,
        "num_shadow_scenes": NUM_SHADOW_SCENES,
        "num_target_scenes": NUM_TARGET_SCENES,
        "num_shadow_fidelity_scenes": NUM_SHADOW_FIDELITY_SCENES,
        "num_target_fidelity_scenes": NUM_TARGET_FIDELITY_SCENES,
        "shadow_scenes_per_update": SHADOW_SCENES_PER_UPDATE,
        "target_scenes_per_update": TARGET_SCENES_PER_UPDATE,
        "num_shadow_updates": args.num_shadow_pairs * 2,
        "num_target_updates": args.num_target_pairs * 2,
        "scene_disjoint_splits": True,
        "pair_specific_scene_subsets": True,
        "per_scene_swap_augmentation": True,
        "position_balanced_fidelity_banks": True,
        "choice_response_format": "single_token_A_or_B",
        "preference_words_present_in_prompt_or_answers": False,
        "individual_user_identity_inference_claimed": False,
        "target_outputs_opened_during_build": False,
    }
    (output / "private_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()

