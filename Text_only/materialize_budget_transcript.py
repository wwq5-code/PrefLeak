#!/usr/bin/env python3
"""Materialize and audit one exact-budget target transcript.

This utility is primarily a migration/audit tool for legacy runs that queried
the complete target candidate bank.  It selects exactly the model/probe/sample
rows that a frozen attack would have requested.  New runs should give the
materialized selected probe bank directly to the model collector instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REFERENCE_ID = "M0_reference"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-generations", type=Path, required=True)
    parser.add_argument("--frozen-attack", type=Path, required=True)
    parser.add_argument("--attack-name", action="append", required=True)
    parser.add_argument("--target-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetition-offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.frozen_attack.read_text(encoding="utf-8"))
    attacks_by_name = {row["name"]: row for row in frozen["attacks"]}
    unknown = sorted(set(args.attack_name) - set(attacks_by_name))
    if unknown:
        raise ValueError(f"Unknown frozen attacks: {unknown}")
    attacks = [attacks_by_name[name] for name in args.attack_name]
    access_modes = {row["access"] for row in attacks}
    repetitions = {row["repetitions_per_version"] for row in attacks}
    probe_sets = {tuple(row["probe_ids"]) for row in attacks}
    if len(access_modes) != 1 or len(repetitions) != 1 or len(probe_sets) != 1:
        raise ValueError(
            "One transcript may combine only attacks with identical access, "
            "repetition count, and frozen probe set."
        )
    access = next(iter(access_modes))
    num_repetitions = next(iter(repetitions))
    probe_ids = set(next(iter(probe_sets)))
    target_ids = set(args.target_id)
    if len(target_ids) != len(args.target_id):
        raise ValueError("Target IDs must be unique.")
    if args.repetition_offset < 0:
        raise ValueError("--repetition-offset must be non-negative.")

    model_ids = set(target_ids)
    if access == "paired":
        model_ids.add(REFERENCE_ID)
    selected = []
    source_counts: Counter[tuple[str, str]] = Counter()
    with args.source_generations.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["model_version"] not in model_ids or row["probe_id"] not in probe_ids:
                continue
            repetition = int(row["repetition"])
            if not (
                args.repetition_offset
                <= repetition
                < args.repetition_offset + num_repetitions
            ):
                continue
            selected.append(row)
            source_counts[(row["model_version"], row["probe_id"])] += 1

    expected_keys = {(model_id, probe_id) for model_id in model_ids for probe_id in probe_ids}
    if set(source_counts) != expected_keys:
        missing = sorted(expected_keys - set(source_counts))
        extra = sorted(set(source_counts) - expected_keys)
        raise ValueError(f"Transcript key mismatch; missing={missing}, extra={extra}")
    bad_counts = {
        f"{model_id}/{probe_id}": count
        for (model_id, probe_id), count in source_counts.items()
        if count != num_repetitions
    }
    if bad_counts:
        raise ValueError(f"Unexpected repetition counts: {bad_counts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "remote_generations.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)

    per_target_post_calls = len(probe_ids) * num_repetitions
    reference_calls = per_target_post_calls if access == "paired" else 0
    effective_calls_per_target = per_target_post_calls + reference_calls
    declared = {row["online_generation_calls"] for row in attacks}
    if declared != {effective_calls_per_target}:
        raise ValueError(
            f"Frozen budget {sorted(declared)} != materialized budget "
            f"{effective_calls_per_target}."
        )
    ledger = {
        "artifact_type": "exact_budget_target_transcript",
        "provenance": "post_hoc_projection_from_legacy_full_bank_transcript",
        "validates_predictions_but_does_not_retroactively_erase_legacy_queries": True,
        "source_generations_sha256": sha256(args.source_generations),
        "frozen_attack_sha256": sha256(args.frozen_attack),
        "attack_names": args.attack_name,
        "access": access,
        "num_target_updates": len(target_ids),
        "num_selected_probes": len(probe_ids),
        "repetitions_per_model_version": num_repetitions,
        "repetition_offset": args.repetition_offset,
        "post_calls_per_target": per_target_post_calls,
        "reference_calls": reference_calls,
        "effective_online_calls_per_target": effective_calls_per_target,
        "serialized_rows": len(selected),
        "expected_serialized_rows": len(probe_ids)
        * num_repetitions
        * (len(target_ids) + (1 if access == "paired" else 0)),
        "target_private_labels_used": False,
        "output_sha256": sha256(output),
    }
    ledger_path = args.output_dir / "query_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "ledger": str(ledger_path), **ledger}, indent=2))


if __name__ == "__main__":
    main()
