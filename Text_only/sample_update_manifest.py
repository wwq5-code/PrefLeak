#!/usr/bin/env python3
"""Create a deterministic per-update subset from a private DPO manifest pool."""

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--balance-choice-position",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample equal numbers of chosen-A and chosen-B rows (default: enabled).",
    )
    parser.add_argument(
        "--preserve-counterfactual-pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When each scene has A/B and B/A rows, sample complete scene pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not 0 < args.num_samples <= len(records):
        raise ValueError(
            f"--num-samples must be in [1, {len(records)}], got {args.num_samples}"
        )
    rng = random.Random(args.seed)
    grouped = {}
    for row in records:
        grouped.setdefault(row.get("source_image_id"), []).append(row)
    paired = bool(grouped) and all(
        len(rows) == 2
        and {row["chosen"][0]["content"] for row in rows} == {"A", "B"}
        for rows in grouped.values()
    )
    if args.preserve_counterfactual_pairs and paired:
        if args.num_samples % 2:
            raise ValueError("Counterfactual-pair sampling requires an even --num-samples.")
        selected_groups = rng.sample(list(grouped.values()), args.num_samples // 2)
        selected = [row for rows in selected_groups for row in rows]
        rng.shuffle(selected)
    elif args.balance_choice_position:
        if args.num_samples % 2:
            raise ValueError("Position-balanced sampling requires an even --num-samples.")
        by_choice = {
            "A": [row for row in records if row["chosen"][0]["content"] in {"A", "I prefer the first image."}],
            "B": [row for row in records if row["chosen"][0]["content"] in {"B", "I prefer the second image."}],
        }
        per_position = args.num_samples // 2
        if any(len(rows) < per_position for rows in by_choice.values()):
            raise ValueError(
                "Source manifest has too few rows for exact choice-position balancing: "
                f"A={len(by_choice['A'])}, B={len(by_choice['B'])}, requested={per_position} each"
            )
        selected = rng.sample(by_choice["A"], per_position) + rng.sample(
            by_choice["B"], per_position
        )
        rng.shuffle(selected)
    else:
        selected = rng.sample(records, args.num_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "num_samples": len(selected),
                "seed": args.seed,
                "choice_position_balanced": args.balance_choice_position,
                "counterfactual_pairs_preserved": bool(
                    args.preserve_counterfactual_pairs and paired
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
