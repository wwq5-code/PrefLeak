"""PPSD-GLM registration tests for the shared query-budget engine."""

import unittest

from Text_only.Query_budget_eva.run_coco_llava_budget_sweep import (
    BUDGETS,
    artifact_prefix,
    budget_layout,
    short_dataset_name,
)
from Text_only.shared_response_resampling import experiment_config


class PpsdGlmQueryBudgetTests(unittest.TestCase):
    def test_ppsd_glm_transcripts_are_registered(self) -> None:
        experiment = experiment_config("ppsd_glm")
        self.assertTrue(experiment.default_shadow.is_file())
        self.assertTrue(experiment.default_target.is_file())
        self.assertEqual(artifact_prefix(experiment), "PPSD_GLM")
        self.assertEqual(short_dataset_name(experiment), "PPSD")

    def test_exact_budget_schedule(self) -> None:
        self.assertEqual(BUDGETS, (16, 32, 48, 64, 80, 96))
        for budget in BUDGETS:
            layout = budget_layout(budget)
            self.assertEqual(layout["paired_probes"] * 4 * 2, budget)
            self.assertEqual(layout["label_only_probes"] * 4, budget)


if __name__ == "__main__":
    unittest.main()

