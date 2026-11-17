#!/usr/bin/env python3
"""Build session-disjoint PARA portrait/scene visual comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.PARA.protocol import DATA_ROOT, PREFERENCES, SOURCE_ROOT, build_specs


PROMPT = (
    "Image A is the first image and Image B is the second image. "
    "Which image has the more appealing visual treatment? Answer with exactly A or B."
)
MATCH_FEATURES = (
    "aestheticScore_mean",
    "qualityScore_mean",
    "compositionScore_mean",
    "colorScore_mean",
    "dofScore_mean",
    "lightScore_mean",
    "contentScore_mean",
    "contentPreference_mean",
    "willingnessToShare_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--para-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--num-shadow-sessions", type=int, default=50)
    parser.add_argument("--num-target-sessions", type=int, default=50)
    parser.add_argument("--num-fidelity-sessions", type=int, default=50)
    parser.add_argument("--max-matched-score-difference", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=59321)
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


def load_candidates(para_root: Path, max_difference: float) -> list[dict]:
    annotation = para_root / "annotation" / "PARA-GiaaTrain.csv"
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with annotation.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            semantic = row["semantic"]
            if semantic not in PREFERENCES:
                continue
            image = para_root / "imgs" / row["sessionId"] / row["imageName"]
            if image.is_file():
                copied = dict(row)
                copied["resolved_image"] = str(image.resolve())
                grouped[row["sessionId"]][semantic].append(copied)

    candidates = []
    for session_id, by_semantic in grouped.items():
        if any(not by_semantic[label] for label in PREFERENCES):
            continue
        best = None
        for portrait in by_semantic["portrait"]:
            for scene in by_semantic["scene"]:
                differences = {
                    feature: abs(float(portrait[feature]) - float(scene[feature]))
                    for feature in MATCH_FEATURES
                }
                maximum = max(differences.values())
                distance = sum(value * value for value in differences.values())
                candidate = (distance, maximum, portrait["imageName"], scene["imageName"])
                if best is None or candidate < best[0]:
                    best = (candidate, portrait, scene, differences)
        if best is None or best[0][1] > max_difference:
            continue
        _, portrait, scene, differences = best
        candidates.append(
            {
                "session_id": session_id,
                "portrait_image": portrait["resolved_image"],
                "scene_image": scene["resolved_image"],
                "portrait_image_name": portrait["imageName"],
                "scene_image_name": scene["imageName"],
                "squared_match_distance": best[0][0],
                "maximum_score_difference": best[0][1],
                "score_differences": differences,
            }
        )
    return sorted(candidates, key=lambda row: row["session_id"])


def update_rows(
    pairs: list[dict], preference: str, seed: int, split: str
) -> list[dict]:
    base_orders = [index % 2 == 0 for index in range(len(pairs))]
    random.Random(seed).shuffle(base_orders)
    rows = []
    for pair_index, (pair, portrait_first) in enumerate(zip(pairs, base_orders)):
        for order_index, first_is_portrait in enumerate(
            (portrait_first, not portrait_first)
        ):
            ordered_labels = (
                ("portrait", "scene")
                if first_is_portrait
                else ("scene", "portrait")
            )
            by_label = {
                "portrait": pair["portrait_image"],
                "scene": pair["scene_image"],
            }
            chosen = "A" if ordered_labels[0] == preference else "B"
            rows.append(
                {
                    "sample_id": f"{split}_{pair_index:03d}_o{order_index}",
                    "source_image_id": pair["session_id"],
                    "order_index": order_index,
                    "images": [by_label[label] for label in ordered_labels],
                    "prompt": conversation("user", PROMPT),
                    "chosen": conversation("assistant", chosen),
                    "rejected": conversation(
                        "assistant", "B" if chosen == "A" else "A"
                    ),
                }
            )
    return rows


def fidelity_bank(pairs: list[dict], seed: int) -> list[dict]:
    orders = [index % 2 == 0 for index in range(len(pairs))]
    random.Random(seed).shuffle(orders)
    rows = []
    for index, (pair, portrait_first) in enumerate(zip(pairs, orders)):
        paths = [pair["portrait_image"], pair["scene_image"]]
        if not portrait_first:
            paths.reverse()
        rows.append(
            {
                "query_index": index,
                "image_id": f"para_fidelity_{index:03d}",
                "image_paths": paths,
                "source_image_id": pair["session_id"],
                "reference_option": "A" if portrait_first else "B",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.para_root, args.max_matched_score_difference)
    requested = (
        args.num_shadow_sessions
        + args.num_target_sessions
        + args.num_fidelity_sessions
    )
    if len(candidates) < requested:
        raise ValueError(
            f"Need {requested} matched sessions but found only {len(candidates)}."
        )
    random.Random(args.seed).shuffle(candidates)
    shadow = candidates[: args.num_shadow_sessions]
    target_start = args.num_shadow_sessions
    target_end = target_start + args.num_target_sessions
    target = candidates[target_start:target_end]
    fidelity = candidates[target_end:requested]
    split_pairs = {"shadow": shadow, "target": target, "fidelity": fidelity}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shadows, targets = build_specs()
    cached: dict[tuple[str, str], list[dict]] = {}
    for split, pairs in (("shadow", shadow), ("target", target)):
        for preference in PREFERENCES:
            cached[(split, preference)] = update_rows(
                pairs,
                preference,
                args.seed + (0 if split == "shadow" else 100),
                split,
            )
    for spec in [*shadows, *targets]:
        write_jsonl(spec.manifest, cached[(spec.split, spec.preference)])

    (args.output_dir / "private_fidelity_probe_bank.json").write_text(
        json.dumps(fidelity_bank(fidelity, args.seed + 200), indent=2) + "\n",
        encoding="utf-8",
    )
    split_document = {
        split: [row["session_id"] for row in rows]
        for split, rows in split_pairs.items()
    }
    (args.output_dir / "session_splits.json").write_text(
        json.dumps(split_document, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "matched_pairs.json").write_text(
        json.dumps(split_pairs, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "dataset": "PARA",
        "source_root": str(args.para_root.resolve()),
        "source_annotation": "annotation/PARA-GiaaTrain.csv",
        "candidate_preferences": list(PREFERENCES),
        "preference_definition": "portrait-versus-scene content preference",
        "pairing_rule": (
            "one portrait and one scene from the same PARA session, nearest "
            "under nine aggregate aesthetic/quality attributes"
        ),
        "matched_features": list(MATCH_FEATURES),
        "maximum_per_feature_score_difference": args.max_matched_score_difference,
        "num_eligible_sessions": len(candidates),
        "seed": args.seed,
        "num_shadow_sessions": len(shadow),
        "num_target_sessions": len(target),
        "num_fidelity_sessions": len(fidelity),
        "comparisons_per_update": {
            "shadow": 2 * len(shadow),
            "target": 2 * len(target),
        },
        "num_shadow_updates": len(shadows),
        "num_target_updates": len(targets),
        "session_disjoint_splits": True,
        "image_disjoint_splits": True,
        "per_pair_swap_augmentation": True,
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
