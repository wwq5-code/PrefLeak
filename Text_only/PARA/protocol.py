"""Frozen PARA portrait-versus-scene preference protocol."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARA_DIR = Path(__file__).resolve().parent
DATA_ROOT = PARA_DIR / "data" / "portrait_scene"
SOURCE_ROOT = Path(
    os.environ.get("PREFLEAK_PARA_ROOT", PROJECT_ROOT / "data" / "PARA")
)
PREFERENCES = ("portrait", "scene")
REFERENCE_PREFERENCE = PREFERENCES[0]


@dataclass(frozen=True)
class UpdateSpec:
    model_id: str
    split: str
    serial: int
    preference: str
    pair_id: str

    @property
    def data_root(self) -> Path:
        return DATA_ROOT

    @property
    def manifest(self) -> Path:
        return DATA_ROOT / "private_updates" / f"{self.model_id}.jsonl"


def build_specs() -> tuple[list[UpdateSpec], list[UpdateSpec]]:
    shadows = [
        UpdateSpec(
            model_id=f"target_b{serial:03d}",
            split="shadow",
            serial=serial,
            preference=PREFERENCES[(serial + 1) % 2],
            pair_id=f"shadow_pair_{(serial + 1) // 2:02d}",
        )
        for serial in range(1, 9)
    ]
    targets = [
        UpdateSpec(
            model_id=f"target_u{serial:03d}",
            split="target",
            serial=serial,
            preference=PREFERENCES[(serial + 1) % 2],
            pair_id=f"target_pair_{(serial + 1) // 2:02d}",
        )
        for serial in range(1, 11)
    ]
    return shadows, targets
