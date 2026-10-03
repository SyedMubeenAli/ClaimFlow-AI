"""
Money handling with Decimal, never float.

The rule that matters: a figure from a language model is only trusted if the same
figure appears in the source text. `amount_appears` is that check. It compares
numbers, so "1,250", "1250.00" and "Rs. 1,250/-" all match each other.

Limit: a small number such as 25 can match by coincidence. Callers check a short
quote, not a whole document, wherever they can.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def parse_amount(value: object) -> Decimal | None:
    """First number in `value` as a Decimal, or None. Accepts '1,250.00', 'Rs. 50,000' and plain numbers."""
    if value is None or isinstance(value, bool):
        return None
    match = _NUMBER.search(str(value))
    return _to_decimal(match.group(0)) if match else None


def amounts_in(text: str) -> set[Decimal]:
    found = (_to_decimal(m.group(0)) for m in _NUMBER.finditer(text or ""))
    return {d for d in found if d is not None}


def amount_appears(amount: Decimal, text: str) -> bool:
    return amount in amounts_in(text)


def format_amount(amount: Decimal) -> str:
    return f"{amount:.2f}"
