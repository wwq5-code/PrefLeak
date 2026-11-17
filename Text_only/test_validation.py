"""Regression tests for the joint-VLM text-only leakage experiment."""

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from Text_only.build_visual_preferences import PROMPT, build_choice_probe_bank, build_rows
from Text_only.choice_attack_features import build_choice_features, group_choice_generations
from Text_only.infer_forced_choice import calibrated_threshold
from Text_only.evaluate_known_preference import one_run_metrics
from Text_only.probe_forced_choice import load_text_queries, parse_choice
from Text_only.train_llava_dpo import (
    JOINT_VLM_LORA_PATTERN,
    LANGUAGE_LORA_PATTERN,
    PROJECTOR_LORA_PATTERN,
    PURE_DPO_LOSS_TYPES,
    VISION_LORA_PATTERN,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "joint_vlm_paper"
TRAIN = DATA / "train2017" / "magenta_cyan"
VALID = DATA / "val2017" / "magenta_cyan"
TEXT_PROBES = ROOT / "prompts" / "magenta_cyan_known_preference_concepts.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def choice_records(base_choices: list[str], target_choices: dict[str, list[str]]) -> list[dict]:
    records = []
    for model_id, choices in {"M0_reference": base_choices, **target_choices}.items():
        for query_index, choice in enumerate(choices):
            records.append(
                {
                    "model_version": model_id,
                    "modality": "text_only",
                    "query_index": query_index,
                    "repetition": 0,
                    "query": "A or B?",
                    "image_id": None,
                    "reference_option": "A",
                    "family": "test",
                    "response": choice,
                    "parsed_choice": choice,
                }
            )
    return records


class JointVLMValidationTests(unittest.TestCase):
    def test_training_is_pure_dpo_and_joint(self):
        self.assertEqual(PURE_DPO_LOSS_TYPES, ("sigmoid",))
        modules = {
            "language": "model.language_model.layers.0.self_attn.q_proj",
            "vision": "model.vision_tower.encoder.layers.0.self_attn.q_proj",
            "projector": "model.multi_modal_projector.linear_1",
        }
        self.assertIsNotNone(re.fullmatch(LANGUAGE_LORA_PATTERN, modules["language"]))
        self.assertIsNotNone(re.fullmatch(VISION_LORA_PATTERN, modules["vision"]))
        self.assertIsNotNone(re.fullmatch(PROJECTOR_LORA_PATTERN, modules["projector"]))
        for module in modules.values():
            self.assertIsNotNone(re.fullmatch(JOINT_VLM_LORA_PATTERN, module))

    def test_real_manifests_are_balanced_inverse_pairs(self):
        train_paths = sorted((TRAIN / "private_updates").glob("target_b*.jsonl"))
        valid_paths = sorted((VALID / "private_updates").glob("target_u*.jsonl"))
        self.assertEqual((len(train_paths), len(valid_paths)), (12, 20))
        for paths in (train_paths, valid_paths):
            for left_path, right_path in zip(paths[::2], paths[1::2]):
                left, right = read_jsonl(left_path), read_jsonl(right_path)
                self.assertEqual(len(left), len(right))
                self.assertGreater(len(left), 0)
                for first, second in zip(left, right):
                    self.assertEqual(first["source_image_id"], second["source_image_id"])
                    self.assertEqual(first["images"], second["images"])
                    self.assertEqual(first["chosen"], second["rejected"])
                    self.assertEqual(first["rejected"], second["chosen"])

    def test_train_and_validation_source_pools_are_disjoint(self):
        def source_ids(root: Path) -> set[str]:
            values = set()
            for manifest in (root / "private_updates").glob("*.jsonl"):
                values.update(row["source_image_id"] for row in read_jsonl(manifest))
            return values

        training, validation = source_ids(TRAIN), source_ids(VALID)
        self.assertTrue(training)
        self.assertTrue(validation)
        self.assertFalse(training & validation)

    def test_visual_training_language_hides_candidate_names(self):
        self.assertNotIn("magenta", PROMPT.lower())
        self.assertNotIn("cyan", PROMPT.lower())
        for root in (TRAIN, VALID):
            manifest = next((root / "private_updates").glob("*.jsonl"))
            for row in read_jsonl(manifest):
                semantic_text = json.dumps(
                    {key: row[key] for key in ("prompt", "chosen", "rejected")}
                ).lower()
                self.assertNotIn("magenta", semantic_text)
                self.assertNotIn("cyan", semantic_text)

    def test_synthetic_visual_pair_is_counterbalanced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = []
            for index in range(4):
                path = root / f"source_{index}.jpg"
                Image.new("RGB", (32, 24), (40 * index, 80, 120)).save(path)
                sources.append(path)
            rows = build_rows(
                sources,
                "train",
                root / "data",
                32,
                0.15,
                7,
                preference_axis="magenta_cyan",
                swap_augmentation=True,
            )
            self.assertEqual(set(rows), {"magenta", "cyan"})
            self.assertEqual(len(rows["magenta"]), 8)
            for magenta, cyan in zip(rows["magenta"], rows["cyan"]):
                self.assertEqual(magenta["images"], cyan["images"])
                self.assertEqual(magenta["chosen"], cyan["rejected"])
                self.assertEqual(magenta["rejected"], cyan["chosen"])

    def test_text_only_probes_are_balanced(self):
        probes = load_text_queries(TEXT_PROBES)
        self.assertEqual(len(probes), 48)
        self.assertEqual(sum(row["reference_option"] == "A" for row in probes), 24)
        self.assertTrue(all("exactly A or B" in row["query"] for row in probes))
        self.assertTrue(all(row.get("image_paths") == [] for row in probes))

    def test_opaque_visual_probe_bank_is_position_balanced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = []
            for index in range(4):
                path = root / f"source_{index}.jpg"
                Image.new("RGB", (32, 24), (50 * index, 90, 130)).save(path)
                sources.append(path)
            bank = build_choice_probe_bank(
                sources,
                root / "data",
                32,
                0.15,
                9,
                "public",
                preference_axis="magenta_cyan",
            )
            self.assertEqual(sum(row["reference_option"] == "A" for row in bank), 2)
            paths = " ".join(value for row in bank for value in row["image_paths"])
            self.assertNotIn("magenta", paths)
            self.assertNotIn("cyan", paths)

    def test_choice_parser(self):
        self.assertEqual(parse_choice("A"), "A")
        self.assertEqual(parse_choice("Option B."), "B")
        self.assertEqual(parse_choice("Image 1"), "A")
        self.assertEqual(parse_choice("the second image"), "B")
        self.assertIsNone(parse_choice("I cannot decide."))

    def test_post_only_features_do_not_use_base_model(self):
        records = choice_records(
            ["A", "B"], {"target_1": ["A", "A"], "target_2": ["B", "B"]}
        )
        grouped = group_choice_generations(records)
        original, observability = build_choice_features(
            grouped, ["target_1", "target_2"], [0, 1], access="post_only"
        )
        changed = choice_records(
            ["B", "A"], {"target_1": ["A", "A"], "target_2": ["B", "B"]}
        )
        modified, _ = build_choice_features(
            group_choice_generations(changed),
            ["target_1", "target_2"],
            [0, 1],
            access="post_only",
        )
        np.testing.assert_array_equal(original, modified)
        self.assertIsNone(observability["target_1"]["exact_response_change_rate"])

    def test_threshold_is_calibrated_on_shadow_scores(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.42, 0.44, 0.58, 0.61])
        threshold = calibrated_threshold(labels, scores)
        self.assertGreater(threshold, 0.44)
        self.assertLessEqual(threshold, 0.58)

    def test_matched_pair_randomization_test_uses_pairs(self):
        rows = []
        labels = {}
        pairs = {}
        for pair_index in range(3):
            for member, label in enumerate(("cyan", "magenta")):
                model_id = f"target_{pair_index}_{member}"
                labels[model_id] = label
                pairs[model_id] = f"pair_{pair_index}"
                rows.append(
                    {
                        "anonymous_model_id": model_id,
                        "predicted_preference": label,
                        "positive_class": "magenta",
                        "positive_class_score": 0.1 if label == "cyan" else 0.9,
                        "raw_feature_l2_norm": 1.0,
                    }
                )
        metrics = one_run_metrics(rows, labels, pairs)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["accuracy_effective_sample_size"], 3)
        self.assertEqual(metrics["matched_pair_randomization_pvalue_one_sided"], 0.125)

    def test_adapter_identifiers_are_anonymous(self):
        expected = [f"target_b{index:03d}.jsonl" for index in range(1, 13)]
        actual = sorted(path.name for path in (TRAIN / "private_updates").glob("*.jsonl"))
        self.assertEqual(actual, expected)

    def test_neutral_and_shuffled_controls_are_exactly_balanced(self):
        expected_counts = ((TRAIN, 6), (VALID, 10))
        for root, expected_pairs in expected_counts:
            metadata_paths = sorted((root / "private_controls").glob("*.control.json"))
            self.assertEqual(len(metadata_paths), expected_pairs)
            for metadata_path in metadata_paths:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertTrue(
                    metadata["source_pair"]["counterfactual_pair_validated"]
                )
                for control_name in ("neutral", "label_shuffled"):
                    control = metadata[control_name]
                    self.assertEqual(control["num_rows"], 256)
                    self.assertEqual(control["chosen_counts"], {"A": 128, "B": 128})
                    self.assertEqual(control["agreement_with_first_preference"], 0.5)
                    self.assertTrue(control["same_public_fields"])
                    rows = read_jsonl(Path(control["path"]))
                    semantic_text = json.dumps(
                        [
                            {key: row[key] for key in ("prompt", "chosen", "rejected")}
                            for row in rows
                        ]
                    ).lower()
                    self.assertNotIn("magenta", semantic_text)
                    self.assertNotIn("cyan", semantic_text)
                self.assertTrue(
                    metadata["neutral"]["within_scene_visual_treatment_cancels"]
                )


if __name__ == "__main__":
    unittest.main()
