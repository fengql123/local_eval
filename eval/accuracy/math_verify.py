"""Mathematical equivalence using the dataset's authored gold answer."""

import re

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


def check(prediction: str, answer: str) -> bool:
    """Extract an anchored answer, then compare gold to prediction on the main thread."""
    if re.fullmatch(r"[+\-\d.,/\s]+", prediction):
        prediction = "Final answer: " + prediction
    gold = parse(str(answer), fallback_mode="no_fallback", parsing_timeout=2)
    predicted = parse(
        prediction,
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=False),
            ExprExtractionConfig(try_extract_without_anchor=False),
        ],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=2,
    )
    return bool(predicted and verify(gold, predicted, timeout_seconds=2))
