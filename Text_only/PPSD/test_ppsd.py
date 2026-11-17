"""Protocol regression tests for the PPSD external-dataset experiment."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Text_only.PPSD.protocol import DATA_ROOT, PREFERENCES, build_specs


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PPSDProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (DATA_ROOT / "private_dataset_metadata.json").is_file():
            raise unittest.SkipTest("Run build_ppsd_preferences.py first.")

    def test_scene_splits_are_disjoint_and_complete(self) -> None:
        splits = json.loads((DATA_ROOT / "scene_splits.json").read_text())
        shadow, target, fidelity = map(
            set, (splits["shadow"], splits["target"], splits["fidelity"])
        )
        metadata = json.loads(
            (DATA_ROOT / "private_dataset_metadata.json").read_text()
        )
        self.assertFalse(shadow & target)
        self.assertFalse(shadow & fidelity)
        self.assertFalse(target & fidelity)
        self.assertEqual(
            (len(shadow), len(target), len(fidelity)),
            (
                metadata["num_shadow_scenes"],
                metadata["num_target_scenes"],
                metadata["num_fidelity_scenes"],
            ),
        )

    def test_every_matched_pair_is_exactly_counterfactual(self) -> None:
        shadows, targets = build_specs()
        for specs in (shadows, targets):
            for left, right in zip(specs[::2], specs[1::2]):
                self.assertEqual(left.pair_id, right.pair_id)
                self.assertEqual({left.preference, right.preference}, set(PREFERENCES))
                left_rows = read_jsonl(left.manifest)
                right_rows = read_jsonl(right.manifest)
                self.assertEqual(len(left_rows), len(right_rows))
                for a, b in zip(left_rows, right_rows):
                    self.assertEqual(a["images"], b["images"])
                    self.assertEqual(a["prompt"], b["prompt"])
                    self.assertEqual(a["chosen"], b["rejected"])
                    self.assertEqual(a["rejected"], b["chosen"])

    def test_training_channel_contains_no_private_class_names(self) -> None:
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
        for forbidden in (*PREFERENCES, "preference", "ppsd"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual({row["chosen"][0]["content"] for row in rows}, {"A", "B"})

    def test_all_referenced_images_exist(self) -> None:
        shadows, targets = build_specs()
        for spec in [*shadows, *targets]:
            for row in read_jsonl(spec.manifest):
                self.assertTrue(all(Path(path).is_file() for path in row["images"]))


if __name__ == "__main__":
    unittest.main()

