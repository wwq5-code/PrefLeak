"""Unit tests for the strict preference-remote LLaVA experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Text_only.build_remote_probe_bank import audit_bank, build_bank
from Text_only.remote_attack import (
    FEATURE_DIM,
    load_generations,
    parsed_choice,
    response_feature,
)
from Text_only.probe_remote_text import stable_model_offset


ROOT = Path(__file__).resolve().parent


class RemoteAttackTests(unittest.TestCase):
    def test_generated_bank_is_strictly_remote_and_balanced(self) -> None:
        rows = build_bank()
        audit = audit_bank(rows)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["choice_answer_balance"]["A"], audit["choice_answer_balance"]["B"])
        self.assertGreaterEqual(audit["num_probes"], 100)

    def test_frozen_bank_matches_generator(self) -> None:
        frozen = json.loads(
            (ROOT / "prompts" / "remote_text_candidate_bank.json").read_text(encoding="utf-8")
        )
        self.assertEqual(frozen, build_bank())

    def test_choice_parser_requires_an_explicit_answer(self) -> None:
        self.assertEqual(parsed_choice("A"), "A")
        self.assertEqual(parsed_choice("The answer is B."), "B")
        self.assertIsNone(parsed_choice("A river runs through a valley."))
        self.assertIsNone(parsed_choice("Because the calculation is short."))

    def test_response_features_are_fixed_and_bounded(self) -> None:
        probe = build_bank()[0]
        feature = response_feature("The answer is A.", probe)
        self.assertEqual(feature.shape, (FEATURE_DIM,))
        self.assertTrue(np.isfinite(feature).all())
        self.assertTrue((feature >= 0).all())
        self.assertTrue((feature <= 1).all())

    def test_model_seed_streams_are_hash_based_not_parity_based(self) -> None:
        values = [stable_model_offset(f"target_u{i:03d}") for i in range(1, 21)]
        self.assertEqual(len(values), len(set(values)))
        parity_match = np.mean(
            [(value % 2) == (index % 2) for index, value in enumerate(values, 1)]
        )
        self.assertGreater(parity_match, 0.2)
        self.assertLess(parity_match, 0.8)

    def test_post_only_transcript_does_not_require_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "post_only.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "model_version": "target_u001",
                        "probe_id": "remote_000",
                        "repetition": 0,
                        "response": "A",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            grouped = load_generations(path, require_reference=False)
            self.assertEqual(set(grouped), {"target_u001"})
            with self.assertRaisesRegex(ValueError, "M0_reference"):
                load_generations(path, require_reference=True)


if __name__ == "__main__":
    unittest.main()
