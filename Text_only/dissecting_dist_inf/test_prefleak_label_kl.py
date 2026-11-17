"""Deterministic checks for the PrefLeak label-only KL adapter."""

from __future__ import annotations

import unittest

import numpy as np

from Text_only.dissecting_dist_inf.prefleak_label_kl import (
    correctness_probabilities,
    score_targets,
)


class LabelOnlyKlAdapterTests(unittest.TestCase):
    def test_projection_exposes_only_public_hard_correctness(self) -> None:
        grouped = {
            "m1": {
                "p_a": [{"response": "A"}, {"response": "explanation"}],
                "p_b": [{"response": "B"}, {"response": "A"}],
            }
        }
        bank = {
            "p_a": {"expected_answer": "A"},
            "p_b": {"expected_answer": "B"},
        }
        matrix, parse_rates = correctness_probabilities(
            grouped, bank, ["m1"], ["p_a", "p_b"], repetitions=2
        )
        np.testing.assert_allclose(matrix, [[0.5, 0.5]])
        np.testing.assert_allclose(parse_rates, [0.75])

    def test_official_kl_voting_separates_opposite_templates(self) -> None:
        probe_ids = ["a1", "a2", "b1", "b2"]
        groups = {"answer_A": probe_ids[:2], "answer_B": probe_ids[2:]}
        cyan = {key: [[0.0] * 2 for _ in range(3)] for key in groups}
        magenta = {key: [[1.0] * 2 for _ in range(3)] for key in groups}
        targets = np.asarray([[0.0] * 4, [1.0] * 4])
        scores = score_targets(
            cyan, magenta, targets, probe_ids, groups, kl_frac=1.0, seed=7
        )
        self.assertLess(scores[0], 0.5)
        self.assertGreaterEqual(scores[1], 0.5)


if __name__ == "__main__":
    unittest.main()
