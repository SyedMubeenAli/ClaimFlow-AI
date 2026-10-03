"""
Claim assessment agent: the potential claim amount.

All arithmetic happens in claimflow.calculator, in Decimal, with fixed rules. This
agent only gathers the three inputs from earlier agents and calls it. It makes no
model call, so there is no way for a model to put a number into the result.

Every input is already verified upstream: the estimate was found in a document, and
the deductible and limit were found inside quoted policy text. If any input is not
confirmed, nothing is calculated and the step says which input is missing.
"""

from __future__ import annotations

from ..errors import AgentError
from ..language import neutral_title
from ..logging_setup import get_logger
from ..models import AgentResult, EvidenceRef
from ..money import format_amount, parse_amount
from ..schemas import AssessmentOutput, IntakeOutput, PolicyOutput
from ..calculator import calculate_claim
from .base import Agent, AgentContext

log = get_logger("agent.assessment")


class ClaimAssessmentAgent(Agent):
    name = "assessment"
    label = "Claim assessment"
    description = "Works out the potential claim amount with fixed rules, not with the model."

    def run(self, ctx: AgentContext) -> AgentResult:
        intake = ctx.state.typed("intake", IntakeOutput)
        policy = ctx.state.typed("policy", PolicyOutput)
        policy_ran = ctx.state.succeeded("policy")

        estimate = parse_amount(intake.repair_estimate)
        deductible = parse_amount(policy.deductible)
        limit = parse_amount(policy.coverage_limit)

        missing = []
        if estimate is None:
            missing.append("a confirmed repair estimate")
        if deductible is None:
            reason = ("the policy was not analysed" if not policy_ran else
                      "the policy uses a percentage, which is not supported yet" if policy.deductible_kind == "percent"
                      else "it was not found in the policy text")
            missing.append(f"a confirmed deductible ({reason})")
        if limit is None:
            missing.append("a confirmed coverage limit" + ("" if policy_ran else " (the policy was not analysed)"))

        base = AssessmentOutput(
            estimate=format_amount(estimate) if estimate is not None else None,
            deductible=format_amount(deductible) if deductible is not None else None,
            coverage_limit=format_amount(limit) if limit is not None else None,
        )
        if missing:
            base.missing_inputs = missing
            log.info("claim %s: not calculated, missing %d input(s)", ctx.state.claim.id, len(missing))
            return self.warning("The potential claim amount could not be calculated.",
                                [f"Missing: {m}." for m in missing], base.to_dict())

        try:
            calc = calculate_claim(estimate, deductible, limit)
        except ValueError as exc:
            raise AgentError(f"The calculation inputs were not valid. {exc}") from exc

        base.calculated = True
        base.after_deductible = format_amount(calc.after_deductible)
        base.potential_amount = format_amount(calc.payable)
        base.limit_applied = calc.limit_applied
        base.steps = list(calc.steps)

        if policy.coverage_applies == "no":
            base.conditions.append("The policy text indicates this incident may not be covered. "
                                   "The amount is shown for information only.")
        elif policy.coverage_applies != "yes":
            base.conditions.append("Coverage was not confirmed. The amount is shown for information only.")
        for exclusion in policy.exclusions:
            if exclusion.get("applies_to_claim") is True:
                base.conditions.append(f"A policy exclusion may apply ({neutral_title(str(exclusion.get('title', '')))}). "
                                       "The amount depends on a reviewer's decision.")

        # Every figure traces to a source: the estimate to the document it was found in, the others to policy citations.
        evidence = [e for e in (ctx.state.result("intake").evidence if ctx.state.result("intake") else [])
                    if e.detail.startswith("Repair estimate")]
        for wanted in ("the deductible", "the coverage limit"):
            evidence += [e for e in (ctx.state.result("policy").evidence if ctx.state.result("policy") else [])
                         if e.detail.startswith(wanted)]
        if not evidence:
            evidence.append(EvidenceRef(intake.estimate_source or "claim", f"Repair estimate {base.estimate}"))

        summary = f"Potential claim amount: {base.potential_amount}."
        log.info("claim %s: calculated %s (limit applied: %s)", ctx.state.claim.id, base.potential_amount, base.limit_applied)
        if base.conditions:
            return self.warning(summary, base.conditions, base.to_dict(), evidence)
        return self.done(summary, base.to_dict(), evidence)
