#!/usr/bin/env python3
"""Run the shared PPSD three-method protocol with GLM-4.1V."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Text_only.GLM.PPSD import protocol
from Text_only.PPSD import run_ppsd_experiment as shared


shared.DATA_ROOT = protocol.DATA_ROOT
shared.PREFERENCES = protocol.PREFERENCES
shared.REFERENCE_PREFERENCE = protocol.REFERENCE_PREFERENCE
shared.build_specs = protocol.build_specs


if __name__ == "__main__":
    shared.main(default_family="glm")

