"""GLM-specific registration tests for the shared query-budget engine."""

import unittest
from pathlib import Path

from Text_only.Query_budget_eva.run_coco_llava_budget_sweep import (
    BUDGETS,
    artifact_prefix,
    budget_layout,
)
from Text_only.shared_response_resampling import experiment_config


class GlmQueryBudgetProtocolTests(unittest.TestCase):
    def test_glm_transcripts_are_registered(self) -> None:
        experiment = experiment_config("coco_glm")
        self.assertEqual(experiment.key, "coco_glm")
        self.assertTrue(experiment.default_shadow.is_file())
        self.assertTrue(experiment.default_target.is_file())
        self.assertEqual(artifact_prefix(experiment), "COCO_GLM")

    def test_glm_uses_the_same_exact_budget_schedule(self) -> None:
        self.assertEqual(BUDGETS, (16, 32, 48, 64, 80, 96))
        for budget in BUDGETS:
            layout = budget_layout(budget)
            self.assertEqual(layout["paired_probes"] * 4 * 2, budget)
            self.assertEqual(layout["label_only_probes"] * 4, budget)

    def test_glm_default_output_stays_inside_glm_directory(self) -> None:
        here = Path(__file__).resolve().parent
        self.assertEqual(here.name, "Query_budget_eva")
        self.assertEqual(here.parent.name, "GLM")


if __name__ == "__main__":
    unittest.main()

