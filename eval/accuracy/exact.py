"""Check the explicit final value for deterministic long-context tasks."""

import re


def check(prediction: str, answer: str) -> bool:
    prediction = re.sub(r"(?:<\|[^>]+\|>|</s>)+\s*$", "", prediction).strip()
    matches = re.findall(r"(?im)^\s*Final answer:\s*(.*?)\s*$", prediction)
    value = matches[-1] if matches else prediction.strip()
    return " ".join(value.split()) == " ".join(answer.split())
