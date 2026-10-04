"""Two ordinary row-selected checkers and two misleading-answer cases."""

import json

from eval.accuracy import grade


def test_math_checker_accepts_bare_anchored_and_boxed_final_answers():
    evaluation = json.dumps({"method": "math_verify", "answer": "16"})
    for text in ("16", "Final answer: 16", r"<think>15</think>\boxed{16}"):
        assert grade(evaluation, text)


def test_letter_checker_accepts_final_answer_and_legacy_formats():
    evaluation = json.dumps({"method": "letter", "answer": "C"})
    for text in (
        "C",
        "**Answer:** (c)",
        r"\boxed{\text{C}}",
        "<think>The answer is A.</think>The correct answer is (C)",
        "Answer: A. Final answer: C.",
    ):
        assert grade(evaluation, text)


def test_math_checker_rejects_wrong_answers_and_incidental_gold_numbers():
    evaluation = json.dumps({"method": "math_verify", "answer": "16"})
    for text in ("15", "The calculation uses 16 numbers.", r"\boxed{15}"):
        assert not grade(evaluation, text)


def test_letter_checker_rejects_ambiguous_options_and_incidental_letters():
    evaluation = json.dumps({"method": "letter", "answer": "B"})
    for text in ("C", "The answer is a prime", "either B or C.", "Answer: B or C."):
        assert not grade(evaluation, text)
