"""
Says plainly what the AI recommended and what the person decided, as two separate things.
Used by the review page and by the claim package, so both always word it the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ReviewRecord, is_override
from .schemas import RECOMMENDATION_HEADLINES

HUMAN_TONES = {"approve": "ok", "request_info": "warn", "escalate": "bad"}


@dataclass(frozen=True)
class DecisionSummary:
    human: str                 # "Approved for Processing" or "Pending"
    human_detail: str          # who and when, or why it is pending
    human_tone: str            # ok, warn, bad or mute
    ai: str                    # the recommendation, or "None"
    ai_detail: str
    relation: str | None       # whether the person agreed with the AI, once both exist
    note: str | None


def summarize(recommendation: str | None, latest: ReviewRecord | None) -> DecisionSummary:
    ai = RECOMMENDATION_HEADLINES.get(recommendation or "")
    if latest is None:
        human, detail, tone, relation, note = "Pending", "No human decision has been recorded yet.", "mute", None, None
    else:
        human, tone, note = latest.decision.outcome, HUMAN_TONES[latest.decision.value], latest.notes or None
        detail = f"Decided by {latest.reviewer} on {latest.created_at.replace('T', ' ').replace('+00:00', ' UTC')}."
        relation = None
        at_decision = latest.ai_recommendation or recommendation   # what the AI said when the person decided
        if at_decision in RECOMMENDATION_HEADLINES:
            relation = ("The reviewer overrode the AI recommendation." if is_override(latest.decision, at_decision)
                        else "The reviewer agreed with the AI recommendation.")
            if latest.ai_recommendation and recommendation and latest.ai_recommendation != recommendation:
                relation += " The recommendation shown above was updated after that decision."
    return DecisionSummary(human, detail, tone, ai or "None",
                           "Advice only. It has no effect until a person decides." if ai else
                           "The agents have not produced a recommendation for this claim.", relation, note)
