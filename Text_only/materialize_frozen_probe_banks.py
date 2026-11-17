#!/usr/bin/env python3
"""Materialize target probe banks from a shadow-only frozen attack.

The primary bank contains only the probes used by the selected attacks.  A
separate diagnostic bank contains the union of frozen random-probe baselines;
it is never part of the primary attack's online query ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIMARY_ATTACKS = (
    "paired_selected_residualized",
    "paired_selected_no_residual",
    "post_only_matched_budget",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bank", type=Path, required=True)
    parser.add_argument("--frozen-attack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_bank(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = json.loads(args.probe_bank.read_text(encoding="utf-8"))
    by_id = {row["probe_id"]: row for row in rows}
    frozen = json.loads(args.frozen_attack.read_text(encoding="utf-8"))
    attacks = {attack["name"]: attack for attack in frozen["attacks"]}
    missing = sorted(set(PRIMARY_ATTACKS) - set(attacks))
    if missing:
        raise ValueError(f"Frozen attack lacks primary variants: {missing}")

    primary_ids = sorted(
        {probe_id for name in PRIMARY_ATTACKS for probe_id in attacks[name]["probe_ids"]}
    )
    random_names = sorted(name for name in attacks if name.startswith("paired_random_"))
    random_ids = sorted(
        {probe_id for name in random_names for probe_id in attacks[name]["probe_ids"]}
    )
    unknown = sorted((set(primary_ids) | set(random_ids)) - set(by_id))
    if unknown:
        raise ValueError(f"Frozen attack references unknown probes: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_path = args.output_dir / "primary_selected_probe_bank.json"
    random_path = args.output_dir / "random_union_probe_bank.json"
    write_bank(primary_path, [by_id[probe_id] for probe_id in primary_ids])
    write_bank(random_path, [by_id[probe_id] for probe_id in random_ids])
    per_random = {}
    random_dir = args.output_dir / "random_sets"
    random_dir.mkdir(parents=True, exist_ok=True)
    for name in random_names:
        path = random_dir / f"{name}.json"
        probe_ids = attacks[name]["probe_ids"]
        write_bank(path, [by_id[probe_id] for probe_id in probe_ids])
        per_random[name] = {
            "path": str(path),
            "num_probes": len(probe_ids),
            "sha256": sha256(path),
            "online_generation_calls": attacks[name]["online_generation_calls"],
        }

    manifest = {
        "boundary": "derived_only_from_shadow_frozen_attack",
        "full_probe_bank_sha256": sha256(args.probe_bank),
        "frozen_attack_sha256": sha256(args.frozen_attack),
        "target_data_opened": False,
        "primary_attack_names": list(PRIMARY_ATTACKS),
        "primary_probe_ids": primary_ids,
        "primary_num_probes": len(primary_ids),
        "primary_path": str(primary_path),
        "primary_sha256": sha256(primary_path),
        "random_attack_names": random_names,
        "random_union_num_probes": len(random_ids),
        "random_union_path": str(random_path),
        "random_union_sha256": sha256(random_path),
        "per_random_attack": per_random,
    }
    output = args.output_dir / "frozen_probe_banks_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
