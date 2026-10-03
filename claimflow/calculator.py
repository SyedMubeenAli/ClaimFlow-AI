"""
Claim calculation with fixed rules. No language model is involved anywhere in this file.

    after_deductible = max(estimate - deductible, 0)
    payable          = min(after_deductible, coverage_limit)

All arithmetic uses Decimal rounded half-up to cents, so there is no floating point
error. Bad input raises ValueError instead of producing a plausible wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
MAX_AMOUNT = Decimal("1000000000000")  # one trillion: far above any real claim, below Decimal's precision limit


@dataclass(frozen=True)
class Calculation:
    estimate: Decimal
    deductible: Decimal
    after_deductible: Decimal
    limit: Decimal
    payable: Decimal
    limit_applied: bool
    steps: tuple[str, ...]


def to_cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _checked(name: str, value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise ValueError(f"{name} must be a Decimal, not {type(value).__name__}.")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be zero or more.")
    if value > MAX_AMOUNT:
        raise ValueError(f"{name} is out of range.")
    return to_cents(value)


def _fmt(value: Decimal) -> str:
    return f"{value:,.2f}"


def calculate_claim(estimate: Decimal, deductible: Decimal, limit: Decimal) -> Calculation:
    estimate = _checked("The repair estimate", estimate)
    deductible = _checked("The deductible", deductible)
    limit = _checked("The coverage limit", limit)

    after = max(estimate - deductible, Decimal("0.00"))
    payable = min(after, limit)
    limit_applied = after > limit

    steps = [f"Repair estimate: {_fmt(estimate)}"]
    if estimate <= deductible:
        steps.append(f"The estimate does not exceed the deductible of {_fmt(deductible)}, so the amount after the deductible is 0.00.")
    else:
        steps.append(f"After the deductible of {_fmt(deductible)}: {_fmt(after)}")
    if limit_applied:
        steps.append(f"This is above the coverage limit of {_fmt(limit)}, so the limit applies.")
    else:
        steps.append(f"This is within the coverage limit of {_fmt(limit)}.")
    steps.append(f"Potential claim amount: {_fmt(payable)}")
    return Calculation(estimate, deductible, after, limit, payable, limit_applied, tuple(steps))
