#!/usr/bin/env python3
"""Build paired visual-preference data from COCO without semantic label text."""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps
from torchvision.transforms import functional as vision_f


PROMPT = (
    "Image A is the first image and Image B is the second image. "
    "Which image has the more appealing visual treatment? Answer with exactly A or B."
)
SIDE_BY_SIDE_PROMPT = (
    "A single comparison image is provided. Option A is the left half and option B "
    "is the right half. Which option has the more appealing visual treatment? "
    "Answer with exactly A or B."
)
# Single-token alternatives avoid a shared-response-prefix shortcut and make the
# update objective identical to the black-box forced-choice measurement.
FIRST = "A"
SECOND = "B"
PREFERENCES = ("warm", "cool")
PREFERENCE_AXES = {
    "warm_cool": ("warm", "cool"),
    "vivid_muted": ("vivid", "muted"),
    "bright_dark": ("bright", "dark"),
    "magenta_cyan": ("magenta", "cyan"),
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-images",
        type=Path,
        default=PROJECT_ROOT / "data" / "coco2017" / "train2017",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "coco_warm_cool_validation",
    )
    parser.add_argument("--num-shadow-images", type=int, default=32)
    parser.add_argument("--num-target-images", type=int, default=32)
    parser.add_argument("--num-fidelity-images", type=int, default=0)
    parser.add_argument("--num-probe-images", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--hue-shift", type=float, default=0.06)
    parser.add_argument(
        "--preference-axis", choices=sorted(PREFERENCE_AXES), default="warm_cool"
    )
    parser.add_argument(
        "--variant-mode", choices=("hue", "temperature"), default="hue"
    )
    parser.add_argument(
        "--pair-presentation",
        choices=("separate", "side_by_side"),
        default="separate",
        help="Present each comparison as two images or one left/right composite.",
    )
    parser.add_argument(
        "--swap-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit both A/B and B/A orders for every private update scene.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def make_variant(
    image: Image.Image,
    preference: str,
    hue_shift: float,
    variant_mode: str = "hue",
    preference_axis: str = "warm_cool",
) -> Image.Image:
    """Use torchvision's tested image operators for the two visual treatments."""
    first, second = PREFERENCE_AXES[preference_axis]
    if preference not in {first, second}:
        raise ValueError(f"{preference!r} is not a member of axis {preference_axis!r}")
    is_first = preference == first
    strength = hue_shift
    if preference_axis == "vivid_muted":
        factor = 1.0 + 2.0 * strength if is_first else max(0.05, 1.0 - 2.0 * strength)
        image = vision_f.adjust_saturation(image, factor)
        return vision_f.adjust_contrast(image, 1.05)
    if preference_axis == "bright_dark":
        factor = 1.0 + strength if is_first else max(0.1, 1.0 - strength)
        image = vision_f.adjust_brightness(image, factor)
        return vision_f.adjust_contrast(image, 1.05)
    if preference_axis == "magenta_cyan":
        tensor = vision_f.to_tensor(image)
        if is_first:
            multipliers = tensor.new_tensor((1.0 + strength, 1.0 - strength, 1.0 + strength))
        else:
            multipliers = tensor.new_tensor((1.0 - strength, 1.0 + strength, 1.0 + strength))
        image = vision_f.to_pil_image((tensor * multipliers[:, None, None]).clamp(0.0, 1.0))
        return vision_f.adjust_contrast(image, 1.05)
    if variant_mode == "temperature":
        tensor = vision_f.to_tensor(image)
        if is_first:
            multipliers = tensor.new_tensor((1.0 + strength, 1.0 + 0.2 * strength, 1.0 - strength))
        else:
            multipliers = tensor.new_tensor((1.0 - strength, 1.0 + 0.2 * strength, 1.0 + strength))
        image = vision_f.to_pil_image((tensor * multipliers[:, None, None]).clamp(0.0, 1.0))
    else:
        shift = -hue_shift if is_first else hue_shift
        image = vision_f.adjust_hue(image, shift)
    image = vision_f.adjust_saturation(image, 1.15)
    return vision_f.adjust_contrast(image, 1.05)


def conversational(content: str, role: str) -> list[dict[str, str]]:
    return [{"role": role, "content": content}]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def concatenate_pair(first: Image.Image, second: Image.Image) -> Image.Image:
    """Join equal-sized alternatives without adding semantic labels or metadata."""
    if first.size != second.size:
        raise ValueError(f"Pair images must have equal sizes: {first.size} != {second.size}")
    canvas = Image.new("RGB", (first.width * 2, first.height), color=(0, 0, 0))
    canvas.paste(first, (0, 0))
    canvas.paste(second, (first.width, 0))
    return canvas


def build_rows(
    sources: list[Path],
    split: str,
    output_dir: Path,
    image_size: int,
    hue_shift: float,
    seed: int,
    variant_mode: str = "hue",
    pair_presentation: str = "separate",
    preference_axis: str = "warm_cool",
    swap_augmentation: bool = False,
) -> dict[str, list[dict]]:
    preferences = PREFERENCE_AXES[preference_axis]
    order = [index % 2 == 0 for index in range(len(sources))]
    random.Random(seed).shuffle(order)
    rows = {preference: [] for preference in preferences}

    variant_dir = output_dir / "variants" / split
    variant_dir.mkdir(parents=True, exist_ok=True)
    for index, (source, warm_first) in enumerate(zip(sources, order)):
        with Image.open(source) as raw_image:
            image = ImageOps.pad(
                raw_image.convert("RGB"),
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0),
            )
        variant_paths = {}
        variants = {}
        for preference in preferences:
            variant_path = variant_dir / f"{source.stem}_{preference}.jpg"
            variant = make_variant(
                image, preference, hue_shift, variant_mode, preference_axis
            )
            variant.save(variant_path, format="JPEG", quality=95)
            variants[preference] = variant
            variant_paths[preference] = variant_path.relative_to(output_dir).as_posix()

        base_order = preferences if warm_first else tuple(reversed(preferences))
        orders = [base_order, tuple(reversed(base_order))] if swap_augmentation else [base_order]
        for order_index, ordered_preferences in enumerate(orders):
            if pair_presentation == "side_by_side":
                pair_path = variant_dir / f"{source.stem}_pair_{order_index}.jpg"
                concatenate_pair(*(variants[value] for value in ordered_preferences)).save(
                    pair_path, format="JPEG", quality=95
                )
                images = [pair_path.relative_to(output_dir).as_posix()]
                prompt = SIDE_BY_SIDE_PROMPT
            else:
                images = [variant_paths[value] for value in ordered_preferences]
                prompt = PROMPT
            for preference in preferences:
                chosen_is_first = ordered_preferences[0] == preference
                rows[preference].append(
                    {
                        "sample_id": f"{split}_{index:05d}_o{order_index}",
                        "source_image_id": source.stem,
                        "order_index": order_index,
                        "images": images,
                        "prompt": conversational(prompt, "user"),
                        "chosen": conversational(FIRST if chosen_is_first else SECOND, "assistant"),
                        "rejected": conversational(SECOND if chosen_is_first else FIRST, "assistant"),
                    }
                )
    return rows


def build_public_probe_bank(
    sources: list[Path], output_dir: Path, image_size: int
) -> list[dict[str, str | int]]:
    probe_dir = output_dir / "public_probe_images"
    probe_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(sources):
        destination = probe_dir / f"probe_{index:05d}.jpg"
        with Image.open(source) as raw_image:
            image = ImageOps.pad(
                raw_image.convert("RGB"),
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0),
            )
        image.save(destination, format="JPEG", quality=95)
        records.append(
            {
                "query_index": index,
                "image_id": f"public_image_{index:05d}",
                "image_path": destination.relative_to(output_dir).as_posix(),
                "source_image_id": source.stem,
            }
        )
    return records


def build_choice_probe_bank(
    sources: list[Path],
    output_dir: Path,
    image_size: int,
    hue_shift: float,
    seed: int,
    namespace: str,
    variant_mode: str = "hue",
    pair_presentation: str = "separate",
    preference_axis: str = "warm_cool",
) -> list[dict]:
    """Build opaque A/B visual pairs for known-candidate forced-choice probes."""
    order = [index % 2 == 0 for index in range(len(sources))]
    random.Random(seed).shuffle(order)
    probe_dir = output_dir / f"{namespace}_choice_images"
    probe_dir.mkdir(parents=True, exist_ok=True)
    records = []
    preferences = PREFERENCE_AXES[preference_axis]
    for index, (source, reference_first) in enumerate(zip(sources, order)):
        with Image.open(source) as raw_image:
            image = ImageOps.pad(
                raw_image.convert("RGB"),
                (image_size, image_size),
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0),
            )
        treatments = preferences if reference_first else tuple(reversed(preferences))
        image_paths = []
        option_images = []
        for option_index, treatment in enumerate(treatments):
            option = "A" if option_index == 0 else "B"
            destination = probe_dir / f"pair_{index:05d}_{option}.jpg"
            option_image = make_variant(
                image, treatment, hue_shift, variant_mode, preference_axis
            )
            option_image.save(destination, format="JPEG", quality=95)
            option_images.append(option_image)
            image_paths.append(destination.relative_to(output_dir).as_posix())
        if pair_presentation == "side_by_side":
            destination = probe_dir / f"pair_{index:05d}.jpg"
            concatenate_pair(*option_images).save(destination, format="JPEG", quality=95)
            image_paths = [destination.relative_to(output_dir).as_posix()]
        records.append(
            {
                "query_index": index,
                "image_id": f"{namespace}_pair_{index:05d}",
                "image_paths": image_paths,
                "source_image_id": source.stem,
                "reference_option": "A" if reference_first else "B",
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.hue_shift <= 0.5:
        raise ValueError("--hue-shift/strength must be in [0, 0.5].")
    if min(args.num_shadow_images, args.num_target_images, args.num_probe_images) < 2:
        raise ValueError("Shadow, target, and probe splits each need at least two source images.")
    if args.num_fidelity_images < 0 or args.num_fidelity_images == 1:
        raise ValueError("Fidelity split must be empty or contain at least two images.")

    candidates = sorted(args.coco_images.glob("*.jpg"))
    requested = (
        args.num_shadow_images
        + args.num_target_images
        + args.num_fidelity_images
        + args.num_probe_images
    )
    if len(candidates) < requested:
        raise FileNotFoundError(
            f"Need {requested} COCO JPEGs in {args.coco_images}, found {len(candidates)}. "
            "Wait for COCO download/extraction to complete."
        )
    selected = random.Random(args.seed).sample(candidates, requested)
    split_sources = {
        "shadow": selected[: args.num_shadow_images],
        "target": selected[
            args.num_shadow_images : args.num_shadow_images + args.num_target_images
        ],
    }
    after_updates = args.num_shadow_images + args.num_target_images
    fidelity_sources = selected[after_updates : after_updates + args.num_fidelity_images]
    probe_sources = selected[after_updates + args.num_fidelity_images :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, sources in split_sources.items():
        rows = build_rows(
            sources,
            split,
            args.output_dir,
            args.image_size,
            args.hue_shift,
            args.seed + (0 if split == "shadow" else 1),
            args.variant_mode,
            args.pair_presentation,
            args.preference_axis,
            args.swap_augmentation,
        )
        for preference in PREFERENCE_AXES[args.preference_axis]:
            write_jsonl(args.output_dir / preference / f"{split}.jsonl", rows[preference])
            counts[f"{preference}_{split}"] = len(rows[preference])

    public_probe_bank = build_public_probe_bank(
        probe_sources, args.output_dir, args.image_size
    )
    (args.output_dir / "public_probe_bank.json").write_text(
        json.dumps(public_probe_bank, indent=2) + "\n", encoding="utf-8"
    )
    counts["public_probe"] = len(public_probe_bank)

    public_choice_bank = build_choice_probe_bank(
        probe_sources,
        args.output_dir,
        args.image_size,
        args.hue_shift,
        args.seed + 2,
        "public",
        args.variant_mode,
        args.pair_presentation,
        args.preference_axis,
    )
    (args.output_dir / "public_choice_probe_bank.json").write_text(
        json.dumps(public_choice_bank, indent=2) + "\n", encoding="utf-8"
    )
    counts["public_choice_probe"] = len(public_choice_bank)

    if fidelity_sources:
        fidelity_bank = build_choice_probe_bank(
            fidelity_sources,
            args.output_dir,
            args.image_size,
            args.hue_shift,
            args.seed + 3,
            "fidelity",
            args.variant_mode,
            args.pair_presentation,
            args.preference_axis,
        )
        (args.output_dir / "private_fidelity_probe_bank.json").write_text(
            json.dumps(fidelity_bank, indent=2) + "\n", encoding="utf-8"
        )
        counts["private_fidelity_probe"] = len(fidelity_bank)

    metadata = {
        "source": f"COCO 2017 {args.coco_images.name} directory",
        "coco_images": str(args.coco_images.resolve()),
        "seed": args.seed,
        "image_size": args.image_size,
        "hue_shift_magnitude": args.hue_shift,
        "variant_mode": args.variant_mode,
        "preference_axis": args.preference_axis,
        "candidate_preferences": list(PREFERENCE_AXES[args.preference_axis]),
        "reference_preference": PREFERENCE_AXES[args.preference_axis][0],
        "pair_presentation": args.pair_presentation,
        "counts": counts,
        "shadow_target_fidelity_probe_scene_disjoint": True,
        "position_balanced": True,
        "per_scene_swap_augmentation": args.swap_augmentation,
        "choice_response_format": "single_token_A_or_B",
        "preference_words_present_in_prompt_or_answers": False,
        "public_choice_bank_assumes_known_candidate_set": True,
    }
    metadata_path = args.output_dir / "private_dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
