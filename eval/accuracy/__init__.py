"""Resolve the accuracy function named by a prepared row's evaluation column."""

import json
import re

from . import code, exact, letter, math_verify

CHECKERS = {
    "letter": letter.check,
    "math_verify": math_verify.check,
    "code": code.check,
    "exact": exact.check,
}


def grade(evaluation: str, prediction: str) -> bool:
    """Ignore reasoning tags and apply the row's trusted accuracy function."""
    rule = json.loads(evaluation)
    answer = re.split(r"</(?:mm:)?think>", prediction)[-1].strip()
    return CHECKERS[rule["method"]](answer, rule["answer"])
