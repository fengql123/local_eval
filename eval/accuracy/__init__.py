"""Resolve the accuracy function named by a prepared row's evaluation column."""

import json

from . import letter, math_verify

CHECKERS = {"letter": letter.check, "math_verify": math_verify.check}


def grade(evaluation: str, prediction: str) -> bool:
    """Ignore reasoning tags and apply the row's trusted accuracy function."""
    rule = json.loads(evaluation)
    answer = prediction.rsplit("</think>", 1)[-1].strip()
    return CHECKERS[rule["method"]](answer, rule["answer"])
