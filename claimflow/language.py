"""
The wording guard. ClaimFlow identifies risk indicators and inconsistencies. It
never says, or implies, that a person committed fraud or acted dishonestly.

Fixed text in the agents is written to respect that. Text produced by a language
model is checked here before it is shown, and dropped if it fails. The check is
deliberately blunt: a false alarm costs a sentence, a miss puts an accusation on a
reviewer's screen.

Policy quotes are shown as written, because they are the policy's words. They are
never copied into risk or compliance text without going through this check.
"""

from __future__ import annotations

import re

_ACCUSATION = re.compile(
    r"\b(fraud\w*|scam\w*|swindl\w*|liars?|lied|dishonest\w*|deceit\w*|decei\w+|deceptive|"
    r"criminal\w*|cheat\w*|forg(?:ed|ery|eries)|fabricat\w+|counterfeit\w*|bogus|phony|perjur\w*)\b",
    re.IGNORECASE,
)


def find_accusation_terms(text: object) -> list[str]:
    seen: list[str] = []
    for match in _ACCUSATION.finditer(str(text or "")):
        word = match.group(0).lower()
        if word not in seen:
            seen.append(word)
    return seen


def mentions_accusation(text: object) -> bool:
    return bool(_ACCUSATION.search(str(text or "")))


def neutral_title(title: str) -> str:
    """
    A policy exclusion can be about conduct (the policy's own words). Risk and
    compliance text must not repeat that wording as if it described this claimant,
    so such a title is replaced with a neutral phrase.
    """
    return "a conduct-related policy condition" if mentions_accusation(title) else title
