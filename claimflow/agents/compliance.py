"""
Compliance review agent: the last step. It combines every earlier result into a
recommended next action for the human reviewer.

Split of responsibility
  * Code decides the recommendation, from fixed rules in `decide()`.
  * A model writes the plain-language explanation. That text is accepted only if it
    contains no figure that is not in the facts it was given and no accusatory
    wording. Otherwise a fixed template explanation is used instead.
  * Code audits every agent's text for accusatory wording before the reviewer sees it.

The recommendation is advice. Nothing happens until a person records a decision.
"""

from __future__ import annotations

from ..errors import ConfigError, LLMError
from ..language import find_accusation_terms, mentions_accusation, neutral_title
from ..llm import LLMRequest
from ..logging_setup import get_logger
from ..models import AgentResult, StepStatus
from ..money import amounts_in
from ..pipeline import step_label
from ..schemas import (
    RECOMMENDATIONS, RISK_INCOMPLETE, AssessmentOutput, ComplianceOutput, DamageOutput, EvidenceOutput, PolicyOutput, RiskFlag,
    RiskOutput,
)
from ..textutil import plural
from .base import Agent, AgentContext
from .common import str_or_none

log = get_logger("agent.compliance")

SYSTEM = (
    "You write a short note for an insurance claims reviewer. Use only the facts you are given. Add no "
    "numbers and no facts of your own. Do not speculate about anyone's intent, honesty or conduct, and never "
    "use words such as fraud or lying. Answer with one JSON object and nothing else."
)


def decide(*, coverage: str, exclusions_that_may_apply: list[str], missing_evidence: list[str],
           flags: list[RiskFlag], calculated: bool, policy_failed: bool) -> tuple[str, list[str]]:
    """
    The recommendation rules. Pure code, so the same inputs always give the same answer.

    escalate_for_investigation     a high-severity indicator, an exclusion that may apply, a policy that
                                   may not cover the incident, or two or more medium indicators
    request_more_information       otherwise, if evidence is missing, the amount could not be calculated,
                                   coverage is unconfirmed, the policy analysis failed, or any medium
                                   indicator needs clarifying
    approve_for_processing         otherwise (low indicators alone do not block)
    """
    escalate: list[str] = []
    ask: list[str] = []
    medium = [f for f in flags if f.severity == "medium"]

    for flag in flags:
        if flag.severity == "high":
            escalate.append(f"A high-severity risk indicator is present: {flag.title}.")
    for title in exclusions_that_may_apply:
        escalate.append(f"A policy exclusion may apply: {neutral_title(title)}.")
    if coverage == "no":
        escalate.append("The policy text indicates this incident may not be covered.")
    if len(medium) >= 2:
        escalate.append("Two or more medium-severity risk indicators are present.")

    if missing_evidence:
        ask.append(f"Required evidence is missing or unusable: {', '.join(missing_evidence)}.")
    if not calculated:
        ask.append("The potential claim amount could not be calculated.")
    if coverage == "unclear":
        ask.append("Coverage could not be confirmed from the policy.")
    if policy_failed:
        ask.append("The policy analysis failed, so the policy terms are unknown.")
    if len(medium) == 1 or (len(medium) >= 2 and not escalate):
        for flag in medium:
            ask.append(f"A medium-severity risk indicator needs clarifying: {flag.title}.")

    if escalate:
        return "escalate_for_investigation", escalate + ask
    if ask:
        return "request_more_information", ask
    return "approve_for_processing", ["The automated checks found no blocking issue."]


def template_explanation(recommendation: str, reasons: list[str]) -> str:
    label = RECOMMENDATIONS[recommendation].lower()
    return f"The recommendation is to {label}. {' '.join(reasons[:4])} A person makes the final decision."


class ComplianceReviewAgent(Agent):
    name = "compliance"
    label = "Compliance review"
    description = "Combines every result into a recommended next action and a plain explanation."

    def run(self, ctx: AgentContext) -> AgentResult:
        state = ctx.state
        policy = state.typed("policy", PolicyOutput)
        evidence = state.typed("evidence", EvidenceOutput)
        risk = state.typed("risk", RiskOutput)
        assessment = state.typed("assessment", AssessmentOutput)
        damage = state.typed("damage", DamageOutput)

        policy_result = state.result("policy")
        policy_failed = policy_result is not None and policy_result.status == StepStatus.FAILED
        possible = [str(e.get("title", "")) for e in policy.exclusions if e.get("applies_to_claim") is True]

        recommendation, reasons = decide(
            coverage=policy.coverage_applies if state.succeeded("policy") else "unclear",
            exclusions_that_may_apply=possible, missing_evidence=evidence.missing, flags=risk.flags,
            calculated=assessment.calculated, policy_failed=policy_failed)

        limitations = []
        damage_result = state.result("damage")
        if damage_result is None or not damage_result.succeeded:
            why = damage_result.error or damage_result.summary if damage_result else "it did not run"
            limitations.append(f"Damage photos were not analysed: {why}")
        if damage_result is not None and damage_result.succeeded and not damage.photos:
            limitations.append("Damage photos were not analysed. A reviewer should look at them directly.")
        if not risk.assessment_complete:
            limitations.append(RISK_INCOMPLETE)
        if risk.model_check == "failed":
            limitations.append("The model-based consistency check did not run.")
        if risk.model_check == "skipped":
            limitations.append("The model-based consistency check was skipped because there were no documents to compare.")
        open_items = list(evidence.missing) + ([f"Policy analysis failed: {policy_result.error}"] if policy_failed else [])

        explanation, source = self._explanation(ctx, recommendation, reasons, assessment)
        audit = self._audit(risk, evidence, assessment, damage, reasons, explanation)

        output = ComplianceOutput(
            recommendation=recommendation, recommendation_label=RECOMMENDATIONS[recommendation], reasons=reasons,
            explanation=explanation, explanation_source=source, open_items=open_items, limitations=limitations,
            language_check_passed=not audit)
        warnings = [f"Wording check: accusatory wording was found in {where}." for where in audit]
        warnings += [f"Analysis incomplete: {item}" for item in open_items if item.startswith("Policy analysis failed")]
        summary = f"Recommendation: {RECOMMENDATIONS[recommendation]}. {plural(len(reasons), 'reason')} recorded."
        log.info("claim %s: recommendation %s (%s)", state.claim.id, recommendation, source)
        if warnings:
            return self.warning(summary, warnings, output.to_dict())
        return self.done(summary, output.to_dict())

    def _explanation(self, ctx: AgentContext, recommendation: str, reasons: list[str],
                     assessment: AssessmentOutput) -> tuple[str, str]:
        facts = [f"Recommendation: {RECOMMENDATIONS[recommendation]}", "Reasons:"] + [f"- {r}" for r in reasons]
        if assessment.calculated:
            facts.append(f"Potential claim amount: {assessment.potential_amount}")
        facts_text = "\n".join(facts)
        fallback = (template_explanation(recommendation, reasons), "template")
        prompt = "\n".join(["TASK: explain recommendation", "", facts_text, "",
                            "Write 2 to 4 plain sentences for the reviewer that explain the recommendation using only "
                            'these facts. Return JSON: {"explanation": "..."}'])
        try:
            data = ctx.ask_json(LLMRequest(system=SYSTEM, prompt=prompt, max_tokens=500))
        except (LLMError, ConfigError) as exc:
            log.warning("claim %s: explanation fell back to the template: %s", ctx.state.claim.id, exc.message)
            return fallback
        text = str_or_none(data.get("explanation"), 800)
        if not text or mentions_accusation(text) or not amounts_in(text) <= amounts_in(facts_text):
            log.warning("claim %s: model explanation rejected; using the template", ctx.state.claim.id)
            return fallback
        return text, "model"

    @staticmethod
    def _audit(risk: RiskOutput, evidence: EvidenceOutput, assessment: AssessmentOutput, damage: DamageOutput,
               reasons: list[str], explanation: str) -> list[str]:
        """Where accusatory wording appears in text that agents produced for the reviewer. Normally empty."""
        sections = {
            "the risk indicators": [f"{f.title} {f.detail}" for f in risk.flags],
            "the evidence check": [f"{i.label} {i.detail}" for i in evidence.items] + [c.detail for c in evidence.checks],
            "the assessment": assessment.steps + assessment.conditions,
            "the damage description": [damage.overall_summary] + [x for p in damage.photos for x in p.visible_damage],
            "the recommendation": reasons + [explanation],
        }
        return [where for where, texts in sections.items() if any(find_accusation_terms(t) for t in texts)]
