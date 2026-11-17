import json
import tempfile
import unittest
from pathlib import Path

from Text_only.shared_response_resampling import (
    SHARED_SPLITS,
    filter_transcript,
    split_manifest,
)


class SharedResponseResamplingTests(unittest.TestCase):
    def test_ten_balanced_unique_splits(self) -> None:
        self.assertEqual(len(SHARED_SPLITS), 10)
        self.assertEqual(len(set(SHARED_SPLITS)), 10)
        manifest = split_manifest()
        self.assertEqual(set(manifest["source_repetition_usage"].values()), {5})
        for indices in SHARED_SPLITS:
            self.assertEqual(len(indices), 4)
            self.assertEqual(len(set(indices)), 4)
            self.assertTrue(set(indices) <= set(range(8)))

    def test_filter_remaps_same_response_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            source = source_dir / "remote_generations.jsonl"
            rows = [
                {
                    "model_version": model,
                    "probe_id": "remote_000",
                    "repetition": repetition,
                    "response": f"{model}-{repetition}",
                }
                for model in ("M0_reference", "target_u001")
                for repetition in range(8)
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            destination = root / "filtered" / "remote_generations.jsonl"
            filter_transcript(source, destination, SHARED_SPLITS[2])
            observed = [
                json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(observed), 8)
            for model in ("M0_reference", "target_u001"):
                selected = [row for row in observed if row["model_version"] == model]
                self.assertEqual([row["source_repetition"] for row in selected], [0, 2, 4, 6])
                self.assertEqual([row["repetition"] for row in selected], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
