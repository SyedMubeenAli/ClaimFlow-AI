"""
The seven agents, in pipeline order. Each one's `name` matches a step in pipeline.py.
"""

from __future__ import annotations

from .assessment import ClaimAssessmentAgent
from .base import Agent, AgentContext
from .compliance import ComplianceReviewAgent
from .damage import DamageAnalysisAgent
from .evidence import EvidenceVerificationAgent
from .intake import ClaimIntakeAgent
from .policy import PolicyAnalystAgent
from .risk import RiskAnalysisAgent

__all__ = [
    "Agent", "AgentContext", "ClaimAssessmentAgent", "ClaimIntakeAgent", "ComplianceReviewAgent",
    "DamageAnalysisAgent", "EvidenceVerificationAgent", "PolicyAnalystAgent", "RiskAnalysisAgent", "default_agents",
]


def default_agents() -> list[Agent]:
    return [ClaimIntakeAgent(), PolicyAnalystAgent(), DamageAnalysisAgent(), EvidenceVerificationAgent(),
            RiskAnalysisAgent(), ClaimAssessmentAgent(), ComplianceReviewAgent()]
