"""
The workflow orchestrator. Plain code, no model: it decides what runs, in what
order, and what happens when something fails.

What it does for every run
  * walks the pipeline in order and runs each selected step
  * skips a step, with a stated reason, if a step it requires did not finish
  * turns an agent crash into a FAILED result instead of crashing the app
  * saves every result to the database and emits a trace event for the UI
  * sets the claim's status from the outcome

Selective rerun: `rerun(claim_id, changed={"verification"})` runs only the
changed steps and what depends on them. Results of the other steps are kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .agents.base import Agent, AgentContext
from .config import Settings
from .db import Database
from .errors import AgentError, ClaimFlowError, WorkflowError
from .llm import LLMProvider
from .logging_setup import get_logger
from .models import (
    AgentResult, ClaimState, ClaimStatus, StepStatus, TraceEvent, can_transition, utc_now,
)
from .pipeline import PIPELINE, StepSpec, affected_steps, step_label, validate_pipeline

log = get_logger("orchestrator")

EventCallback = Callable[[TraceEvent], None]


@dataclass
class RunSummary:
    claim_id: int
    ran: list[str]
    results: dict[str, AgentResult]
    final_status: ClaimStatus
    trace: list[TraceEvent] = field(default_factory=list)


def final_claim_status(results: dict[str, AgentResult]) -> ClaimStatus:
    """
    Awaiting review only when the last step (compliance review) finished, since it
    produces the recommendation a human reviews. Any failure means the claim needs attention.
    Otherwise nothing useful happened and the claim stays submitted.
    """
    review = results.get("compliance")
    if review is not None and review.succeeded:
        return ClaimStatus.AWAITING_REVIEW
    if any(r.status == StepStatus.FAILED for r in results.values()):
        return ClaimStatus.FAILED
    return ClaimStatus.SUBMITTED


class WorkflowOrchestrator:
    def __init__(self, db: Database, llm: LLMProvider, settings: Settings,
                 agents: Sequence[Agent], pipeline: tuple[StepSpec, ...] = PIPELINE, rag=None):
        validate_pipeline(pipeline)
        self.db = db
        self.llm = llm
        self.settings = settings
        self.pipeline = pipeline
        self.rag = rag
        self.agents = {agent.name: agent for agent in agents}
        missing = [s.name for s in pipeline if s.name not in self.agents]
        extra = [name for name in self.agents if name not in {s.name for s in pipeline}]
        if missing or extra:
            raise WorkflowError(f"Agents do not match the pipeline (missing: {missing}, unknown: {extra}).")

    def run(self, claim_id: int, *, only: Iterable[str] | None = None,
            on_event: EventCallback | None = None) -> RunSummary:
        """Run the whole pipeline, or only the named steps."""
        claim = self.db.get_claim(claim_id)
        if not can_transition(claim.status, ClaimStatus.PROCESSING):
            raise WorkflowError(f"Claim {claim.claim_number} is {claim.status.label.lower()} "
                                "and can no longer be processed by the agents.")

        selected = {s.name for s in self.pipeline} if only is None else set(only)
        unknown = selected - {s.name for s in self.pipeline}
        if unknown:
            raise WorkflowError(f"Unknown workflow step(s): {', '.join(sorted(unknown))}.")

        documents = self.db.list_documents(claim_id)
        shared_files = self.db.shared_documents(claim_id)
        # A partial run keeps the earlier results of the steps it does not touch.
        results: dict[str, AgentResult] = {} if only is None else dict(self.db.latest_agent_results(claim_id))
        for name in selected:
            results.pop(name, None)

        self.db.update_claim_status(claim_id, ClaimStatus.PROCESSING)
        trace: list[TraceEvent] = []
        ran: list[str] = []

        def emit(spec: StepSpec, status: StepStatus, message: str) -> None:
            event = TraceEvent(spec.name, spec.label, status, message, utc_now())
            trace.append(event)
            log.info("claim %s | %s | %s | %s", claim_id, spec.name, status.value, message)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:  # a UI problem must never stop the pipeline
                    log.exception("Trace callback failed")

        try:
            for spec in self.pipeline:
                if spec.name not in selected:
                    continue
                blocking = [d for d in spec.requires if not (d in results and results[d].succeeded)]
                if blocking:
                    names = [self._label(d) for d in blocking]
                    labels = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
                    result = AgentResult(spec.name, StepStatus.SKIPPED,
                                         f"Skipped: it needs {labels} to finish first.")
                else:
                    emit(spec, StepStatus.RUNNING, "Started")
                    state = ClaimState(claim, documents, dict(results), shared_files)
                    result = self._run_agent(self.agents[spec.name], state)
                results[spec.name] = result
                ran.append(spec.name)
                self.db.save_agent_result(claim_id, result)
                emit(spec, result.status, result.error or result.summary or result.status.label)
        finally:
            # Whatever happened, the claim must not be left stuck in "processing".
            final = final_claim_status(results)
            self.db.update_claim_status(claim_id, final)

        return RunSummary(claim_id, ran, results, final, trace)

    def rerun(self, claim_id: int, changed: Iterable[str], *,
              on_event: EventCallback | None = None) -> RunSummary:
        """Rerun the changed steps and everything downstream of them."""
        return self.run(claim_id, only=affected_steps(changed, self.pipeline), on_event=on_event)

    def _label(self, name: str) -> str:
        return step_label(name, self.pipeline)

    def _run_agent(self, agent: Agent, state: ClaimState) -> AgentResult:
        ctx = AgentContext(state=state, llm=self.llm, settings=self.settings, rag=self.rag)
        started = time.perf_counter()
        try:
            result = agent.run(ctx)
            if not isinstance(result, AgentResult) or result.step != agent.name:
                raise AgentError(f"The {agent.label} agent returned an invalid result.")
        except ClaimFlowError as exc:
            log.warning("%s failed: %s", agent.name, exc.message)
            result = AgentResult(agent.name, StepStatus.FAILED, "This step failed.", error=exc.user_message)
        except Exception as exc:  # a bug in one agent must not take the app down
            log.exception("%s crashed", agent.name)
            result = AgentResult(agent.name, StepStatus.FAILED, "This step failed.",
                                 error=f"Unexpected error ({type(exc).__name__}). The log has the details.")
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        if ctx.llm_calls:
            result.model = ctx.llm_calls[-1].model
            result.cached = all(call.cached for call in ctx.llm_calls)
        return result
