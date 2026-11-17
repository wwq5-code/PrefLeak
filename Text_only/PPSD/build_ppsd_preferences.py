#!/usr/bin/env python3
"""Build scene-disjoint PPSD visual-style DPO updates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.PPSD.protocol import (
    COLLECTION,
    DATA_ROOT,
    PREFERENCES,
    SOURCE_ROOT,
    build_specs,
)


PROMPT = (
    "Image A is the first image and Image B is the second image. "
    "Which image has the more appealing visual treatment? Answer with exactly A or B."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppsd-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--num-shadow-scenes", type=int, default=80)
    parser.add_argument("--num-target-scenes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42701)
    parser.add_argument(
        "--style-ids",
        type=int,
        nargs=2,
        default=tuple(int(label.rsplit("_", 1)[1]) for label in PREFERENCES),
        metavar=("STYLE_A", "STYLE_B"),
    )
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


def available_scenes(
    ppsd_root: Path, collection: str, style_ids: tuple[int, int]
) -> dict[str, tuple[Path, Path]]:
    image_root = ppsd_root / "img_src" / "src" / collection
    result = {}
    first_style, second_style = style_ids
    for first in sorted(image_root.glob(f"{collection}-*_{first_style}.jpg")):
        scene_id = first.stem.rsplit("_", 1)[0]
        second = image_root / f"{scene_id}_{second_style}.jpg"
        if second.is_file():
            result[scene_id] = (first.resolve(), second.resolve())
    return result


def direct_human_votes(
    ppsd_root: Path, collection: str, style_ids: tuple[int, int]
) -> dict:
    path = ppsd_root / "responses" / "responses" / "raw" / "votes_items.jsonl"
    counts: Counter[int] = Counter()
    voters: dict[int, set[str]] = {style: set() for style in style_ids}
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            styles = {int(row["left_style"]), int(row["right_style"])}
            if row.get("collection") != collection or styles != set(style_ids):
                continue
            if row.get("choice") not in {"left", "right"}:
                continue
            selected = int(
                row["left_style"] if row["choice"] == "left" else row["right_style"]
            )
            counts[selected] += 1
            voters[selected].add(str(row["user_id"]))
            total += 1
    if not total or any(not counts[style] for style in style_ids):
        raise ValueError(
            f"PPSD has no two-sided direct human support for {collection} styles {style_ids}."
        )
    result = {
        "num_direct_votes": total,
    }
    for style in style_ids:
        result[f"style_{style}_votes"] = counts[style]
        result[f"style_{style}_fraction"] = counts[style] / total
        result[f"style_{style}_unique_voters"] = len(voters[style])
    return result


def update_rows(
    scenes: list[str],
    scene_paths: dict[str, tuple[Path, Path]],
    preference: str,
    style_ids: tuple[int, int],
    seed: int,
    split: str,
) -> list[dict]:
    preference_labels = tuple(f"style_{style}" for style in style_ids)
    preferred_style = style_ids[preference_labels.index(preference)]
    first_orders = [index % 2 == 0 for index in range(len(scenes))]
    random.Random(seed).shuffle(first_orders)
    rows = []
    for scene_index, (scene_id, style_zero_first) in enumerate(
        zip(scenes, first_orders)
    ):
        for order_index, zero_first in enumerate(
            (style_zero_first, not style_zero_first)
        ):
            styles = style_ids if zero_first else tuple(reversed(style_ids))
            by_style = {
                style: path for style, path in zip(style_ids, scene_paths[scene_id])
            }
            chosen = "A" if styles[0] == preferred_style else "B"
            rows.append(
                {
                    "sample_id": f"{split}_{scene_index:03d}_o{order_index}",
                    "source_image_id": scene_id,
                    "order_index": order_index,
                    "images": [str(by_style[style]) for style in styles],
                    "prompt": conversation("user", PROMPT),
                    "chosen": conversation("assistant", chosen),
                    "rejected": conversation("assistant", "B" if chosen == "A" else "A"),
                }
            )
    return rows


def fidelity_bank(
    scenes: list[str], scene_paths: dict[str, tuple[Path, Path]], seed: int,
    style_ids: tuple[int, int],
) -> list[dict]:
    orders = [index % 2 == 0 for index in range(len(scenes))]
    random.Random(seed).shuffle(orders)
    rows = []
    for index, (scene_id, zero_first) in enumerate(zip(scenes, orders)):
        styles = style_ids if zero_first else tuple(reversed(style_ids))
        by_style = {style: path for style, path in zip(style_ids, scene_paths[scene_id])}
        rows.append(
            {
                "query_index": index,
                "image_id": f"ppsd_fidelity_{index:03d}",
                "image_paths": [str(by_style[style]) for style in styles],
                "source_image_id": scene_id,
                "reference_option": "A" if zero_first else "B",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    style_ids = tuple(args.style_ids)
    if len(set(style_ids)) != 2:
        raise ValueError("--style-ids must contain two distinct styles.")
    preferences = tuple(f"style_{style}" for style in style_ids)
    scene_paths = available_scenes(args.ppsd_root, args.collection, style_ids)
    requested = args.num_shadow_scenes + args.num_target_scenes
    if len(scene_paths) <= requested:
        raise ValueError(
            f"Need more than {requested} complete scenes for a disjoint fidelity split; "
            f"found {len(scene_paths)}."
        )
    scenes = sorted(scene_paths)
    random.Random(args.seed).shuffle(scenes)
    shadow_scenes = scenes[: args.num_shadow_scenes]
    target_scenes = scenes[
        args.num_shadow_scenes : args.num_shadow_scenes + args.num_target_scenes
    ]
    fidelity_scenes = scenes[requested:]
    split_scenes = {"shadow": shadow_scenes, "target": target_scenes}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shadows, targets = build_specs()
    specs = [*shadows, *targets]
    if any(spec.data_root != DATA_ROOT for spec in specs) and args.output_dir != DATA_ROOT:
        raise ValueError("A non-default output directory requires protocol regeneration.")
    cached_rows = {}
    for split, selected in split_scenes.items():
        for preference in preferences:
            cached_rows[(split, preference)] = update_rows(
                selected,
                scene_paths,
                preference,
                style_ids,
                args.seed + (0 if split == "shadow" else 100),
                split,
            )
    for spec in specs:
        preference = preferences[(spec.serial + 1) % 2]
        write_jsonl(
            args.output_dir / "private_updates" / f"{spec.model_id}.jsonl",
            cached_rows[(spec.split, preference)],
        )

    bank = fidelity_bank(fidelity_scenes, scene_paths, args.seed + 200, style_ids)
    (args.output_dir / "private_fidelity_probe_bank.json").write_text(
        json.dumps(bank, indent=2) + "\n", encoding="utf-8"
    )
    split_document = {
        "shadow": shadow_scenes,
        "target": target_scenes,
        "fidelity": fidelity_scenes,
    }
    (args.output_dir / "scene_splits.json").write_text(
        json.dumps(split_document, indent=2) + "\n", encoding="utf-8"
    )
    human_support = direct_human_votes(args.ppsd_root, args.collection, style_ids)
    metadata = {
        "dataset": "PPSD",
        "source_root": str(args.ppsd_root.resolve()),
        "collection": args.collection,
        "candidate_preferences": list(preferences),
        "style_ids": list(style_ids),
        "preference_definition": "opposing PPSD photographic style identities",
        "human_preference_support": human_support,
        "seed": args.seed,
        "num_shadow_scenes": len(shadow_scenes),
        "num_target_scenes": len(target_scenes),
        "num_fidelity_scenes": len(fidelity_scenes),
        "comparisons_per_update": {
            "shadow": 2 * len(shadow_scenes),
            "target": 2 * len(target_scenes),
        },
        "num_shadow_updates": len(shadows),
        "num_target_updates": len(targets),
        "scene_disjoint_splits": True,
        "per_scene_swap_augmentation": True,
        "position_balanced_fidelity_bank": True,
        "choice_response_format": "single_token_A_or_B",
        "preference_words_present_in_prompt_or_answers": False,
        "individual_user_identity_inference_claimed": False,
    }
    (args.output_dir / "private_dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()

