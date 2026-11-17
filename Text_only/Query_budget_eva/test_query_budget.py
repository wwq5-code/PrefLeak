"""Protocol-level tests for the COCO-LLaVA query-budget sweep."""

import unittest

from Text_only.Query_budget_eva.run_coco_llava_budget_sweep import (
    BUDGETS,
    SHARED_SPLITS,
    budget_layout,
)


class QueryBudgetProtocolTests(unittest.TestCase):
    def test_every_budget_is_exactly_matched(self) -> None:
        for budget in BUDGETS:
            layout = budget_layout(budget)
            self.assertEqual(
                layout["paired_probes"] * layout["repetitions"] * 2, budget
            )
            self.assertEqual(
                layout["label_only_probes"] * layout["repetitions"], budget
            )

    def test_probe_counts_grow_with_budget(self) -> None:
        self.assertEqual(
            [budget_layout(value)["paired_probes"] for value in BUDGETS],
            [2, 4, 6, 8, 10, 12],
        )
        self.assertEqual(
            [budget_layout(value)["label_only_probes"] for value in BUDGETS],
            [4, 8, 12, 16, 20, 24],
        )

    def test_response_resamples_are_balanced(self) -> None:
        counts = {index: 0 for index in range(8)}
        self.assertEqual(len(SHARED_SPLITS), 10)
        for split in SHARED_SPLITS:
            self.assertEqual(len(split), 4)
            self.assertEqual(len(set(split)), 4)
            for index in split:
                counts[index] += 1
        self.assertEqual(set(counts.values()), {5})


if __name__ == "__main__":
    unittest.main()

