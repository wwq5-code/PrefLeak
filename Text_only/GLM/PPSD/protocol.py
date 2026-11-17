"""Frozen, target-blind D1 protocol for the GLM PPSD replication."""

from __future__ import annotations

import json
import os
from pathlib import Path

from Text_only.PPSD.protocol import UpdateSpec


PPSD_DIR = Path(__file__).resolve().parent
DATA_ROOT = PPSD_DIR / "data" / "d1_final"
SOURCE_ROOT = Path(
    os.environ.get("PREFLEAK_PPSD_ROOT", PPSD_DIR.parents[2] / "data" / "PPSD")
)
COLLECTION = "D1"
NUM_SHADOW_PAIRS = 8
NUM_TARGET_PAIRS = 12


def _preferences() -> tuple[str, str]:
    metadata = DATA_ROOT / "private_dataset_metadata.json"
    if not metadata.is_file():
        return ("style_0", "style_1")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    values = tuple(payload["candidate_preferences"])
    if len(values) != 2:
        raise ValueError(f"Expected two D1 preferences in {metadata}")
    return values  # type: ignore[return-value]


PREFERENCES = _preferences()
REFERENCE_PREFERENCE = PREFERENCES[0]


def build_specs() -> tuple[list[UpdateSpec], list[UpdateSpec]]:
    shadows = [
        UpdateSpec(
            model_id=f"target_b{serial:03d}",
            split="shadow",
            serial=serial,
            preference=PREFERENCES[(serial + 1) % 2],
            pair_id=f"shadow_pair_{(serial + 1) // 2:02d}",
            data_root=DATA_ROOT,
        )
        for serial in range(1, NUM_SHADOW_PAIRS * 2 + 1)
    ]
    targets = [
        UpdateSpec(
            model_id=f"target_u{serial:03d}",
            split="target",
            serial=serial,
            preference=PREFERENCES[(serial + 1) % 2],
            pair_id=f"target_pair_{(serial + 1) // 2:02d}",
            data_root=DATA_ROOT,
        )
        for serial in range(1, NUM_TARGET_PAIRS * 2 + 1)
    ]
    return shadows, targets

