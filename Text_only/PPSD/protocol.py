"""Frozen PPSD protocol shared by the LLaVA and GLM experiment runners."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PPSD_DIR = Path(__file__).resolve().parent
DATA_ROOT = PPSD_DIR / "data" / "d2_style_1_3"
SOURCE_ROOT = Path(
    os.environ.get("PREFLEAK_PPSD_ROOT", PROJECT_ROOT / "data" / "PPSD")
)
COLLECTION = "D2"
PREFERENCES = ("style_1", "style_3")
REFERENCE_PREFERENCE = PREFERENCES[0]


@dataclass(frozen=True)
class UpdateSpec:
    model_id: str
    split: str
    serial: int
    preference: str
    pair_id: str
    data_root: Path = DATA_ROOT

    @property
    def manifest(self) -> Path:
        return self.data_root / "private_updates" / f"{self.model_id}.jsonl"


def build_specs() -> tuple[list[UpdateSpec], list[UpdateSpec]]:
    """Return the frozen 8-shadow/10-target counterfactual updates."""
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

