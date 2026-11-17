"""Regression tests for the target-blind GLM PPSD D1 protocol."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Text_only.GLM.PPSD.protocol import DATA_ROOT, PREFERENCES, build_specs


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class GlmPpsdD1ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (DATA_ROOT / "private_dataset_metadata.json").is_file():
            raise unittest.SkipTest("Build the GLM D1 protocol first")

    def test_four_scene_splits_are_disjoint(self) -> None:
        splits = json.loads((DATA_ROOT / "scene_splits.json").read_text())
        values = [set(splits[key]) for key in (
            "shadow", "target", "shadow_fidelity", "target_fidelity"
        )]
        self.assertEqual([len(value) for value in values], [40, 50, 15, 16])
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                self.assertFalse(left & right)

    def test_pair_members_are_exact_counterfactuals(self) -> None:
        shadows, targets = build_specs()
        self.assertEqual((len(shadows), len(targets)), (16, 24))
        for specs in (shadows, targets):
            for left, right in zip(specs[::2], specs[1::2]):
                self.assertEqual(left.pair_id, right.pair_id)
                self.assertEqual({left.preference, right.preference}, set(PREFERENCES))
                a_rows, b_rows = read_jsonl(left.manifest), read_jsonl(right.manifest)
                self.assertEqual(len(a_rows), len(b_rows))
                for a, b in zip(a_rows, b_rows):
                    self.assertEqual(a["images"], b["images"])
                    self.assertEqual(a["prompt"], b["prompt"])
                    self.assertEqual(a["chosen"], b["rejected"])
                    self.assertEqual(a["rejected"], b["chosen"])

    def test_pairs_have_distinct_scene_subsets(self) -> None:
        subsets = json.loads((DATA_ROOT / "pair_scene_subsets.json").read_text())
        shadow = [tuple(value) for key, value in subsets.items() if key.startswith("shadow")]
        target = [tuple(value) for key, value in subsets.items() if key.startswith("target")]
        self.assertEqual(len(set(shadow)), len(shadow))
        self.assertEqual(len(set(target)), len(target))

    def test_training_channel_is_visual_and_private_name_free(self) -> None:
        shadows, _ = build_specs()
        rows = read_jsonl(shadows[0].manifest)
        payload = json.dumps(
            [
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
                for row in rows
            ]
        ).lower()
        for forbidden in (*PREFERENCES, "ppsd", "preference"):
            self.assertNotIn(forbidden, payload)
        self.assertTrue(all(Path(image).is_file() for row in rows for image in row["images"]))


if __name__ == "__main__":
    unittest.main()
