#!/usr/bin/env python3
"""Validate a frozen update configuration on one counterfactual PPSD shadow pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from Text_only.PPSD.protocol import build_specs
from Text_only.PPSD.run_ppsd_experiment import (
    audit,
    environment,
    family_config,
    prepare,
    probe_fidelity,
    train_all,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("llava", "glm"), required=True)
    parser.add_argument("--frozen-config-source", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    family = family_config(cli.family)
    output = cli.output_dir or (
        HERE / "outputs" / f"d2_shadow_validation_{cli.family}"
    )
    args = SimpleNamespace(
        model=cli.model or family.model,
        output_dir=output,
        jobs=cli.jobs,
        frozen_config_source=cli.frozen_config_source,
    )
    env = environment()
    shadows, _ = build_specs()
    pair = shadows[:2]
    prepare(args, family, env)
    train_all(args, family, env, pair)
    audit_result = audit(args, pair)
    probe_fidelity(args, family, env, "shadow", pair)
    fidelity_path = output / "analysis" / "visual_fidelity_shadow.json"
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    result = {
        "selection_boundary": "PPSD_D2_first_counterfactual_shadow_pair_only",
        "target_data_opened": False,
        "family": cli.family,
        "source_frozen_config": str(
            (cli.frozen_config_source or family.source_config).resolve()
        ),
        "audit_passed": audit_result["all_passed"],
        "visual_updates_passed": fidelity["num_passed"],
        "visual_updates_total": fidelity["num_models"],
        "visual_mean_accuracy": fidelity["mean_post_preference_accuracy"],
        "visual_mean_improvement": fidelity[
            "mean_preference_accuracy_improvement"
        ],
        "accepted": audit_result["all_passed"] and fidelity["num_passed"] == 2,
    }
    (output / "shadow_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
