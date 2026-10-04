"""Extract explicit final answers and legacy multiple-choice formats."""

import re

_LETTER = r"\(?\s*([A-Da-d])\s*\)?"
_ANCHORED = (
    re.compile(r"answer\s+is\s*:?\s*" + _LETTER, re.IGNORECASE),
    re.compile(r"answer\s*:\s*" + _LETTER, re.IGNORECASE),
    re.compile(r"\\boxed\s*\{\s*(?:\\text(?:bf|rm|it)?\s*\{)?\s*\(?\s*([A-Da-d])\s*\)?\s*\}?\s*\}"),
    re.compile(r"\b(?:option|choice)\s+\(?([A-D])\)?(?![A-Za-z0-9])"),
)
_ALTERNATIVE = re.compile(r"(?:or|and|/)\s*\(?[A-D](?![A-Za-z0-9])", re.IGNORECASE)


def _letter(text: str) -> str | None:
    """Prefer the last explicit answer, then the legacy bare-letter formats."""
    text = text.replace("*", "")
    anchored = []
    for pattern in _ANCHORED:
        for match in pattern.finditer(text):
            value = match.group(1)
            rest = text[match.end(1) :].lstrip(" \t)*")
            if _ALTERNATIVE.match(rest):
                continue
            if value.isupper():
                accepted = not text[match.end(1) : match.end(1) + 1].isalnum()
            else:
                accepted = not rest or rest[0] in ".,;:!?\n\r]"
            if accepted:
                anchored.append((match.start(), value.upper()))
    if anchored:
        return max(anchored)[1]
    whole = re.fullmatch(r"\s*\(?\s*([A-Da-d])\s*\)?\s*[.:)]?\s*", text)
    if whole:
        return whole.group(1).upper()
    for line in text.splitlines():
        if line.strip():
            match = re.match(r"\s*(?:\(([A-D])\)|([A-D])[).:])(?=\s|$)", line)
            if match:
                return match.group(1) or match.group(2)
    parens = re.findall(r"\(([A-D])\)", text)
    if parens:
        return parens[-1]
    trailing = re.search(r"(?:^|[\s(])([A-D])\)?[.!]?\s*$", text)
    paired = re.search(r"\b(?:or|and|vs\.?|/)\s*\(?[A-D]\)?[.!]?\s*$", text, re.IGNORECASE)
    return trailing.group(1) if trailing and not paired else None


def check(prediction: str, answer: str) -> bool:
    """Check a final multiple-choice answer against its authored letter."""
    return _letter(prediction) == answer.strip().upper()
