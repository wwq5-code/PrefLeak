#!/usr/bin/env python3
"""Run the matched PARA LLaVA query-budget evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Text_only.Query_budget_eva.run_coco_llava_budget_sweep import run_main  # noqa: E402


def main() -> None:
    run_main("para_llava", HERE / "outputs" / "para_llava")


if __name__ == "__main__":
    main()

