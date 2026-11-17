#!/usr/bin/env python3
"""Build and audit a preference-remote public text probe bank."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "prompts" / "remote_text_candidate_bank.json"

# These terms are excluded from every prompt and answer.  The list is wider
# than the two private labels so the bank cannot use obvious semantic proxies.
BANNED_TERMS = (
    "aesthetic", "appeal", "attractive", "beautiful", "favorite", "favourite",
    "prefer", "preference", "taste", "style", "visual", "image", "photo",
    "picture", "graphic", "design", "appearance", "color", "colour", "hue",
    "palette", "tint", "shade", "lighting", "warm", "cool", "magenta", "cyan",
    "pink", "purple", "violet", "fuchsia", "rose", "blue", "green", "teal",
    "turquoise", "aqua", "golden", "reddish", "bluish", "artwork",
)


def choice_prompt(stem: str, correct: str, wrong: str, correct_first: bool) -> tuple[str, str]:
    first, second = (correct, wrong) if correct_first else (wrong, correct)
    prompt = f"{stem}\nA. {first}\nB. {second}\nRespond with exactly A or B."
    return prompt, "A" if correct_first else "B"


def build_bank() -> list[dict]:
    rows: list[dict] = []

    def add_choice(family: str, stem: str, correct: str, wrong: str) -> None:
        prompt, answer = choice_prompt(stem, correct, wrong, len(rows) % 2 == 0)
        rows.append({
            "probe_id": f"remote_{len(rows):03d}",
            "family": family,
            "kind": "choice",
            "prompt": prompt,
            "expected_answer": answer,
        })

    def add_open(family: str, prompt: str, expected: str) -> None:
        rows.append({
            "probe_id": f"remote_{len(rows):03d}",
            "family": family,
            "kind": "open",
            "prompt": prompt,
            "expected_answer": expected,
        })

    # Synthetic arithmetic tasks are public, unambiguous, and unrelated to
    # the private semantic axis.
    for i in range(16):
        left, right = 11 + i, 7 + (i * 3) % 19
        correct = left + right
        wrong = correct + (1 if i % 2 == 0 else -2)
        add_choice("arithmetic", f"Compute {left} + {right}.", str(correct), str(wrong))
    for i in range(12):
        left, right = 3 + i, 2 + (i % 7)
        correct = left * right
        wrong = correct + (right if i % 2 == 0 else -left)
        add_choice("arithmetic", f"Compute {left} multiplied by {right}.", str(correct), str(wrong))
    for i in range(12):
        first = 20 + i * 3
        second = first + 2 + (i % 5)
        add_choice(
            "comparison",
            "Which numerical statement is true?",
            f"{second} is greater than {first}",
            f"{first} is greater than {second}",
        )
    for i in range(12):
        start, step = 2 + i, 2 + (i % 4)
        values = [start + step * j for j in range(4)]
        add_choice(
            "sequence",
            f"Continue the sequence: {', '.join(map(str, values))}, ...",
            str(values[-1] + step),
            str(values[-1] + step + 2),
        )

    facts = [
        ("Which statement is accurate?", "A week has seven days", "A week has nine days"),
        ("Which statement is accurate?", "Water freezes at zero degrees Celsius", "Water freezes at fifty degrees Celsius"),
        ("Which statement is accurate?", "Earth orbits the Sun", "The Sun orbits Earth once each day"),
        ("Which statement is accurate?", "A triangle has three sides", "A triangle has five sides"),
        ("Which statement is accurate?", "A dozen contains twelve items", "A dozen contains twenty items"),
        ("Which statement is accurate?", "Tokyo is in Japan", "Tokyo is in Brazil"),
        ("Which statement is accurate?", "The Nile is a river", "The Nile is a mountain"),
        ("Which statement is accurate?", "A violin is a musical instrument", "A violin is a planet"),
        ("Which statement is accurate?", "Copper is a chemical element", "Copper is a calendar month"),
        ("Which statement is accurate?", "An octagon has eight sides", "An octagon has six sides"),
        ("Which statement is accurate?", "Paris is in France", "Paris is in Canada"),
        ("Which statement is accurate?", "Mercury is a planet", "Mercury is a continent"),
        ("Which statement is accurate?", "A kilogram is a unit of mass", "A kilogram is a unit of time"),
        ("Which statement is accurate?", "January comes before February", "January comes after December in the same year"),
        ("Which statement is accurate?", "A cube has six faces", "A cube has three faces"),
        ("Which statement is accurate?", "Sound can travel through air", "Sound is a calendar date"),
    ]
    for stem, correct, wrong in facts:
        add_choice("knowledge", stem, correct, wrong)

    lexical = [
        ("rapid", "quick", "silent"), ("silent", "quiet", "heavy"),
        ("ancient", "old", "narrow"), ("tiny", "small", "distant"),
        ("assist", "help", "divide"), ("begin", "start", "borrow"),
        ("purchase", "buy", "forget"), ("reply", "answer", "climb"),
        ("difficult", "hard", "empty"), ("honest", "truthful", "wooden"),
        ("repair", "fix", "scatter"), ("select", "choose", "measure"),
        ("nearby", "close", "formal"), ("depart", "leave", "count"),
        ("permit", "allow", "reduce"), ("observe", "notice", "contain"),
    ]
    for word, correct, wrong in lexical:
        add_choice(
            "lexical",
            f"Which term is closest in meaning to '{word}'?",
            correct,
            wrong,
        )

    grammar = [
        ("The students are ready.", "The students is ready."),
        ("She writes every morning.", "She write every morning."),
        ("They were waiting outside.", "They was waiting outside."),
        ("This book belongs to me.", "This book belong to me."),
        ("We have finished the task.", "We has finished the task."),
        ("Each member receives a ticket.", "Each member receive a ticket."),
        ("The keys are on the table.", "The keys is on the table."),
        ("He does not know the answer.", "He do not know the answer."),
        ("There are two chairs in the room.", "There is two chairs in the room."),
        ("My friend visits on Fridays.", "My friend visit on Fridays."),
        ("The report was completed yesterday.", "The report were completed yesterday."),
        ("Neither route is available.", "Neither route are available."),
    ]
    for correct, wrong in grammar:
        add_choice("grammar", "Which sentence is grammatically correct?", correct, wrong)

    logic = [
        ("All kems are torps. Mira is a kem. Which conclusion follows?", "Mira is a torp", "Mira is not a torp"),
        ("Every dax is a neral. No neral is a pavo. Which conclusion follows?", "No dax is a pavo", "Every dax is a pavo"),
        ("If the alarm rings, the lamp turns on. The alarm rings. Which conclusion follows?", "The lamp turns on", "The lamp remains off"),
        ("All ravens are birds. This animal is a raven. Which conclusion follows?", "This animal is a bird", "This animal is not a bird"),
        ("No metal object is transparent. The token is metal. Which conclusion follows?", "The token is not transparent", "The token is transparent"),
        ("If a file is archived, it is stored. This file is archived. Which conclusion follows?", "This file is stored", "This file is deleted"),
        ("Every pilot has training. Dana is a pilot. Which conclusion follows?", "Dana has training", "Dana has no training"),
        ("All squares are polygons. Shape R is a square. Which conclusion follows?", "Shape R is a polygon", "Shape R is not a polygon"),
        ("If the gate is locked, entry is blocked. The gate is locked. Which conclusion follows?", "Entry is blocked", "Entry is guaranteed"),
        ("No reptile has feathers. The animal is a reptile. Which conclusion follows?", "The animal has no feathers", "The animal has feathers"),
        ("Every registered item has an identifier. Item K is registered. Which conclusion follows?", "Item K has an identifier", "Item K has no identifier"),
        ("If the switch is pressed, the bell sounds. The switch is pressed. Which conclusion follows?", "The bell sounds", "The bell stays silent"),
    ]
    for stem, correct, wrong in logic:
        add_choice("logic", stem, correct, wrong)

    for i in range(12):
        left, right = 31 + i, 4 + (i % 6)
        add_open(
            "short_answer",
            f"Compute {left} minus {right}. Return only the integer.",
            str(left - right),
        )
    tokens = (
        "RIVER", "CANDLE", "PLANET", "MARBLE", "WINDOW", "MARKET",
        "FOREST", "POCKET", "SILVER", "GARDEN", "BUTTON", "ENGINE",
    )
    for token in tokens:
        add_open(
            "transformation",
            f"Convert {token} to lowercase. Return only the transformed token.",
            token.lower(),
        )
    return rows


def audit_bank(rows: list[dict]) -> dict:
    ids = [row["probe_id"] for row in rows]
    violations = []
    for row in rows:
        searchable = f"{row['prompt']} {row['expected_answer']}".lower()
        found = [
            term for term in BANNED_TERMS
            if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", searchable)
        ]
        if found:
            violations.append({"probe_id": row["probe_id"], "terms": found})
    choice_rows = [row for row in rows if row["kind"] == "choice"]
    return {
        "audit": "strict_preference_remote_text_bank",
        "num_probes": len(rows),
        "num_choice": len(choice_rows),
        "num_open": len(rows) - len(choice_rows),
        "families": sorted({row["family"] for row in rows}),
        "unique_ids": len(ids) == len(set(ids)),
        "choice_answer_balance": {
            label: sum(row["expected_answer"] == label for row in choice_rows)
            for label in ("A", "B")
        },
        "banned_terms": list(BANNED_TERMS),
        "violations": violations,
        "passed": len(ids) == len(set(ids)) and not violations,
    }


def main() -> None:
    rows = build_bank()
    audit = audit_bank(rows)
    if not audit["passed"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(DEFAULT_OUTPUT.read_bytes()).hexdigest()
    audit["sha256"] = digest
    audit_path = DEFAULT_OUTPUT.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(DEFAULT_OUTPUT), **audit}, indent=2))


if __name__ == "__main__":
    main()
