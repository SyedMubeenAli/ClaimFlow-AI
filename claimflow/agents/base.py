"""
The agent contract.

An agent is a class with a `name` that matches a step in pipeline.py and a
`run(ctx)` method. It reads what it needs from `ctx.state`, optionally asks the
language model through `ctx.ask` / `ctx.ask_json`, and returns one AgentResult.

Three rules keep the system trustworthy:
  1. Agents never call each other. They share information only through the state.
  2. Agents return results; they do not write to the database or change the claim.
     The orchestrator does that.
  3. A model never supplies a number that code then trusts. Money comes from
     code that checks the figure against the source text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

from ..config import Settings
from ..llm import LLMProvider, LLMRequest, LLMResponse, parse_json_object
from ..models import AgentResult, ClaimState, EvidenceRef, StepStatus


@dataclass
class AgentContext:
    state: ClaimState
    llm: LLMProvider
    settings: Settings
    llm_calls: list[LLMResponse] = field(default_factory=list)
    rag: Any = None   # the PolicyIndexService, or None (agents then use a temporary in-memory one)

    def ask(self, request: LLMRequest) -> LLMResponse:
        """Call the model. The call is recorded, so the trace can show which model answered."""
        response = self.llm.complete(request)
        self.llm_calls.append(response)
        return response

    def ask_json(self, request: LLMRequest) -> dict[str, Any]:
        """Call the model in JSON mode and return the parsed object."""
        response = self.ask(replace(request, json_mode=True))
        return parse_json_object(response.text)


class Agent(ABC):
    name: str
    label: str
    description: str = ""

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentResult:
        ...

    # Small helpers so agents read clearly.
    def done(self, summary: str, output: dict[str, Any] | None = None,
             evidence: list[EvidenceRef] | None = None) -> AgentResult:
        return AgentResult(self.name, StepStatus.DONE, summary, output or {}, evidence or [])

    def warning(self, summary: str, warnings: list[str], output: dict[str, Any] | None = None,
                evidence: list[EvidenceRef] | None = None) -> AgentResult:
        return AgentResult(self.name, StepStatus.WARNING, summary, output or {}, evidence or [], list(warnings))

    def skipped(self, reason: str) -> AgentResult:
        return AgentResult(self.name, StepStatus.SKIPPED, reason)
