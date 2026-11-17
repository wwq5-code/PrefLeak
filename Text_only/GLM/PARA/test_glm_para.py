"""Regression tests for the GLM PARA replication wrapper."""

from __future__ import annotations

import json
import unittest

from Text_only.GLM.PARA.protocol import DATA_ROOT, build_specs
from Text_only.GLM.PARA.run_glm_para_experiment import (
    DEFAULT_OUTPUT,
    MAX_IMAGE_PIXELS,
    SOURCE_CONFIG,
)
from Text_only.shared_response_resampling import experiment_config


class GlmParaProtocolTests(unittest.TestCase):
    def test_reuses_exact_para_update_registry(self) -> None:
        shadows, targets = build_specs()
        self.assertEqual(len(shadows), 8)
        self.assertEqual(len(targets), 10)
        self.assertEqual({row.preference for row in shadows}, {"portrait", "scene"})
        self.assertEqual({row.preference for row in targets}, {"portrait", "scene"})

    def test_shared_para_data_exist(self) -> None:
        self.assertTrue((DATA_ROOT / "private_dataset_metadata.json").is_file())
        for spec in sum((list(rows) for rows in build_specs()), []):
            self.assertTrue(spec.manifest.is_file())

    def test_source_configuration_is_shadow_only(self) -> None:
        source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("shadow", source["selection_boundary"].lower())
        self.assertIs(source["selection_evidence"]["target_data_opened"], False)

    def test_shared_resampling_registration(self) -> None:
        experiment = experiment_config("para_glm")
        self.assertEqual(experiment.default_output, DEFAULT_OUTPUT / "shared_96")
        self.assertEqual(experiment.victim, "GLM-4.1V-9B-Base")
        self.assertEqual(len(experiment.shadows), 8)
        self.assertEqual(len(experiment.targets), 10)

    def test_glm_image_cap_matches_ppsd_scale(self) -> None:
        self.assertEqual(MAX_IMAGE_PIXELS, 1_003_520)
        self.assertGreaterEqual(MAX_IMAGE_PIXELS, 1024 * 683)


if __name__ == "__main__":
    unittest.main()

