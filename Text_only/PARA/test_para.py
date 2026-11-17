"""Protocol regression tests for the PARA replication."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Text_only.PARA.protocol import DATA_ROOT, PREFERENCES, build_specs


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PARAProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (DATA_ROOT / "private_dataset_metadata.json").is_file():
            raise unittest.SkipTest("Run build_para_preferences.py first.")

    def test_session_splits_are_disjoint(self) -> None:
        splits = json.loads((DATA_ROOT / "session_splits.json").read_text())
        shadow, target, fidelity = map(
            set, (splits["shadow"], splits["target"], splits["fidelity"])
        )
        self.assertFalse(shadow & target)
        self.assertFalse(shadow & fidelity)
        self.assertFalse(target & fidelity)

    def test_images_and_sessions_are_disjoint(self) -> None:
        document = json.loads((DATA_ROOT / "matched_pairs.json").read_text())
        session_sets = []
        image_sets = []
        for split in ("shadow", "target", "fidelity"):
            session_sets.append({row["session_id"] for row in document[split]})
            image_sets.append(
                {
                    image
                    for row in document[split]
                    for image in (row["portrait_image"], row["scene_image"])
                }
            )
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertFalse(session_sets[left] & session_sets[right])
                self.assertFalse(image_sets[left] & image_sets[right])

    def test_every_update_pair_is_exactly_counterfactual(self) -> None:
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

    def test_training_channel_hides_candidate_names(self) -> None:
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
        for forbidden in (*PREFERENCES, "preference", "para"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual({row["chosen"][0]["content"] for row in rows}, {"A", "B"})

    def test_matched_scores_and_images(self) -> None:
        metadata = json.loads((DATA_ROOT / "private_dataset_metadata.json").read_text())
        document = json.loads((DATA_ROOT / "matched_pairs.json").read_text())
        threshold = metadata["maximum_per_feature_score_difference"]
        for split in document.values():
            for row in split:
                self.assertLessEqual(row["maximum_score_difference"], threshold)
                self.assertTrue(Path(row["portrait_image"]).is_file())
                self.assertTrue(Path(row["scene_image"]).is_file())

    def test_remote_probe_bank_hides_para_preferences(self) -> None:
        bank = DATA_ROOT.parents[2] / "prompts" / "remote_text_candidate_bank.json"
        payload = bank.read_text(encoding="utf-8").lower()
        for forbidden in (*PREFERENCES, "preference", "image", "visual"):
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
