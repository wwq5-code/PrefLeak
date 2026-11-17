"""Deterministic checks for the GLM-4.1V replication."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from Text_only.GLM.probe_glm import load_text_queries, parse_choice
from Text_only.GLM.probe_remote_glm import (
    load_bank,
    parse_choice as parse_remote_choice,
    shard_is_complete,
    stable_model_offset,
)
from Text_only.GLM.run_glm_experiment import build_specs
from Text_only.GLM.train_glm_dpo import (
    JOINT_VLM_LORA_PATTERN,
    LANGUAGE_LORA_PATTERN,
    PROJECTOR_LORA_PATTERN,
    PURE_DPO_LOSS_TYPES,
    VISION_LORA_PATTERN,
)
from Text_only.dissecting_dist_inf.run_llava_prefleak_baseline import (
    experiment_config as label_only_di_config,
)


ROOT = Path(__file__).resolve().parent


class GlmReplicationTests(unittest.TestCase):
    def test_label_only_di_uses_glm_artifacts_and_matched_budget(self):
        config = label_only_di_config("glm")
        self.assertEqual(config["num_probes"], 12)
        self.assertEqual(config["model_label"], "GLM-4.1V-9B-Base")
        self.assertIn("Text_only/GLM/outputs", str(config["remote_root"]))
        self.assertIn("Text_only/GLM/outputs", str(config["adapter_root"]))

    def test_joint_module_patterns(self):
        modules = {
            "vision": "model.visual.blocks.0.attn.qkv",
            "projector": "model.visual.merger.proj",
            "language": "model.language_model.layers.0.mlp.gate_up_proj",
        }
        patterns = {
            "vision": VISION_LORA_PATTERN,
            "projector": PROJECTOR_LORA_PATTERN,
            "language": LANGUAGE_LORA_PATTERN,
        }
        for component, module in modules.items():
            self.assertIsNotNone(re.fullmatch(patterns[component], module))
            self.assertIsNotNone(re.fullmatch(JOINT_VLM_LORA_PATTERN, module))
        self.assertEqual(PURE_DPO_LOSS_TYPES, ("sigmoid",))

    def test_shadow_target_registry(self):
        shadows, targets = build_specs()
        self.assertEqual((len(shadows), len(targets)), (12, 20))
        for specs in (shadows, targets):
            for first, second in zip(specs[::2], specs[1::2]):
                self.assertEqual(first.pair_id, second.pair_id)
                self.assertEqual({first.preference, second.preference}, {"magenta", "cyan"})
                self.assertTrue(first.manifest.is_file())
                self.assertTrue(second.manifest.is_file())

    def test_strict_parser_does_not_parse_reasoning_mentions(self):
        self.assertEqual(parse_choice("A"), "A")
        self.assertEqual(parse_choice("Option B."), "B")
        self.assertEqual(parse_choice("My answer is option A."), "A")
        self.assertIsNone(
            parse_choice("<think>Image A is bright. Image B is dark. Comparing the two")
        )

    def test_text_probe_bank(self):
        path = ROOT / "prompts" / "magenta_cyan_known_preference_concepts.json"
        queries = load_text_queries(path)
        self.assertEqual(len(queries), 48)
        self.assertEqual(sum(row["reference_option"] == "A" for row in queries), 24)
        self.assertTrue(all(row["image_paths"] == [] for row in queries))

    def test_config_was_frozen_without_target(self):
        path = ROOT / "outputs" / "main_v1" / "frozen_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["selection_boundary"], "shadow_only_train2017")
        self.assertFalse(config["selection_evidence"]["target_data_opened"])
        self.assertEqual(config["selected_candidate"], "lr5e5")

    def test_remote_bank_is_shared_and_strict(self):
        path = ROOT.parent / "prompts" / "remote_text_candidate_bank.json"
        rows = load_bank(path)
        audit = json.loads(path.with_suffix(".audit.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 132)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["violations"], [])
        self.assertEqual({row["kind"] for row in rows}, {"choice", "open"})

    def test_remote_seed_streams_are_anonymous_id_specific(self):
        ids = [f"target_u{index:03d}" for index in range(1, 21)]
        offsets = [stable_model_offset(model_id) for model_id in ids]
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertGreater(len({value % 2 for value in offsets}), 1)

    def test_remote_parser_and_atomic_shard_validation(self):
        self.assertEqual(parse_remote_choice("A"), "A")
        self.assertEqual(parse_remote_choice("Answer: B."), "B")
        self.assertIsNone(parse_remote_choice("A sentence containing an article."))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target_u001.jsonl"
            rows = [
                {"model_version": "target_u001", "response": "A"},
                {"model_version": "target_u001", "response": "B"},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertTrue(shard_is_complete(path, 2, "target_u001"))
            self.assertFalse(shard_is_complete(path, 3, "target_u001"))
            self.assertFalse(shard_is_complete(path, 2, "target_u002"))


if __name__ == "__main__":
    unittest.main()
