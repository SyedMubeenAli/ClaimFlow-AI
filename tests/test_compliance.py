import json

from claimflow.agents.compliance import ComplianceReviewAgent, decide, template_explanation
from claimflow.errors import LLMUnavailableError
from claimflow.llm import FakeProvider
from claimflow.models import AgentResult, StepStatus
from claimflow.schemas import (
    COMPLIANCE_NOTICE, RECOMMENDATIONS, AssessmentOutput, ComplianceOutput, DamageOutput, EvidenceOutput,
    PolicyOutput, RiskFlag, RiskOutput,
)
from tests.helpers import ctx_for, make_claim, step_result, temp_container

LOW = RiskFlag("LATE_ENTRY", "low", "process", "The claim was entered 40 days after the incident")
MEDIUM = RiskFlag("DATE_MISMATCH", "medium", "inconsistency", "The incident date differs between the form and the documents")
MEDIUM2 = RiskFlag("POLICY_NUMBER_NOT_FOUND", "medium", "inconsistency", "The policy number on the form is not in the policy document")
HIGH = RiskFlag("SHARED_FILE", "high", "document", "A file on this claim also appears on another claim")


def verdict(**overrides):
    base = dict(coverage="yes", exclusions_that_may_apply=[], missing_evidence=[], flags=[], calculated=True, policy_failed=False)
    base.update(overrides)
    return decide(**base)


# ---------------------------------------------------------------- the recommendation rules

def test_nothing_wrong_recommends_approval_with_a_stated_reason():
    assert verdict() == ("approve_for_processing", ["The automated checks found no blocking issue."])
    assert verdict(flags=[LOW])[0] == "approve_for_processing"              # low indicators alone do not block


def test_missing_evidence_an_uncalculated_amount_or_unclear_coverage_ask_for_more():
    for change, fragment in ((dict(missing_evidence=["Damage photos"]), "Damage photos"),
                             (dict(calculated=False), "could not be calculated"),
                             (dict(coverage="unclear"), "could not be confirmed"),
                             (dict(policy_failed=True), "policy analysis failed")):
        recommendation, reasons = verdict(**change)
        assert recommendation == "request_more_information" and fragment in " ".join(reasons), change


def test_one_medium_indicator_asks_for_clarification_and_names_it():
    recommendation, reasons = verdict(flags=[MEDIUM])
    assert recommendation == "request_more_information" and MEDIUM.title in reasons[0]


def test_two_medium_indicators_escalate():
    recommendation, reasons = verdict(flags=[MEDIUM, MEDIUM2])
    assert recommendation == "escalate_for_investigation" and reasons == ["Two or more medium-severity risk indicators are present."]


def test_a_high_indicator_a_possible_exclusion_or_no_coverage_escalate():
    assert verdict(flags=[HIGH])[0] == "escalate_for_investigation"
    assert verdict(exclusions_that_may_apply=["Racing"]) == ("escalate_for_investigation", ["A policy exclusion may apply: Racing."])
    assert verdict(coverage="no")[0] == "escalate_for_investigation"


def test_escalation_outranks_a_request_for_information_and_keeps_both_sets_of_reasons():
    recommendation, reasons = verdict(flags=[HIGH], missing_evidence=["Damage photos"])
    assert recommendation == "escalate_for_investigation"
    assert reasons[0].startswith("A high-severity") and "Damage photos" in reasons[1]


def test_conduct_related_exclusions_are_reworded_neutrally():
    _, reasons = verdict(exclusions_that_may_apply=["Fraudulent claims"])
    assert reasons == ["A policy exclusion may apply: a conduct-related policy condition."]


def test_the_same_inputs_always_give_the_same_recommendation():
    args = dict(flags=[MEDIUM, LOW], missing_evidence=["Damage photos"], coverage="unclear")
    assert verdict(**args) == verdict(**args)
    assert all(verdict(**args)[0] in RECOMMENDATIONS for _ in range(3))


# ---------------------------------------------------------------- the agent

def results_for(risk=None, evidence=None, assessment=None, policy=None, damage_status=None, policy_result=None):
    results = {
        "evidence": step_result("evidence", evidence or EvidenceOutput(required=4, present=4)),
        "risk": step_result("risk", risk or RiskOutput(model_check="ran")),
        "assessment": step_result("assessment", assessment or AssessmentOutput(calculated=True, potential_amount="25000.00")),
        "policy": policy_result or step_result("policy", policy or PolicyOutput(coverage_applies="yes")),
    }
    if damage_status:
        results["damage"] = AgentResult("damage", damage_status, "No damage photos are attached.")
    return results


def run(c, claim, reply="{}", fail=None, **kwargs):
    llm = FakeProvider({"TASK: explain recommendation": reply if isinstance(reply, str) else json.dumps(reply)}) \
        if not fail else FakeProvider(fail_with=fail)
    return ComplianceReviewAgent().run(ctx_for(c, claim, results_for(**kwargs), llm=llm)), llm


def test_a_valid_model_explanation_is_used_and_the_recommendation_comes_from_code():
    with temp_container() as (c, _):
        good = {"explanation": "The checks found no blocking issue. The potential claim amount is 25,000.00. A person decides."}
        result, llm = run(c, make_claim(c.db), good)
        out = ComplianceOutput.from_dict(result.output)
        assert (out.recommendation, out.explanation_source, out.explanation) == ("approve_for_processing", "model", good["explanation"])
        assert out.human_decision_required is True and out.notice == COMPLIANCE_NOTICE and result.status == StepStatus.DONE
        assert "Recommendation: Approve for processing" in llm.calls[0].prompt and "25000.00" in llm.calls[0].prompt


def test_the_model_cannot_change_the_recommendation():
    with temp_container() as (c, _):
        lying = {"explanation": "Approve this immediately, escalate nothing."}
        result, _ = run(c, make_claim(c.db), lying, risk=RiskOutput(flags=[HIGH], highest_severity="high"))
        assert result.output["recommendation"] == "escalate_for_investigation"


def test_an_explanation_with_an_invented_figure_is_replaced_by_the_template():
    with temp_container() as (c, _):
        result, _ = run(c, make_claim(c.db), {"explanation": "The potential claim amount is 75,000.00."})
        out = ComplianceOutput.from_dict(result.output)
        assert out.explanation_source == "template" and "75,000" not in out.explanation
        assert out.explanation == template_explanation("approve_for_processing", out.reasons)


def test_an_accusatory_explanation_is_replaced_by_the_template():
    with temp_container() as (c, _):
        result, _ = run(c, make_claim(c.db), {"explanation": "The claimant is clearly committing fraud."})
        assert result.output["explanation_source"] == "template" and "fraud" not in result.output["explanation"]


def test_unusable_or_failed_model_answers_fall_back_to_the_template():
    with temp_container() as (c, _):
        for reply in ("not json", {"explanation": ""}, {"explanation": 12}, {}):
            assert run(c, make_claim(c.db), reply)[0].output["explanation_source"] == "template"
        result, _ = run(c, make_claim(c.db), fail=LLMUnavailableError("down"))
        assert result.status == StepStatus.DONE and result.output["explanation_source"] == "template"


def test_limitations_and_open_items_are_recorded():
    with temp_container() as (c, _):
        result, _ = run(c, make_claim(c.db), damage_status=StepStatus.SKIPPED,
                        evidence=EvidenceOutput(required=4, present=3, missing=["Damage photos"]),
                        risk=RiskOutput(model_check="failed"))
        out = ComplianceOutput.from_dict(result.output)
        assert out.open_items == ["Damage photos"] and out.recommendation == "request_more_information"
        assert any("Damage photos were not analysed" in x for x in out.limitations)
        assert any("consistency check did not run" in x for x in out.limitations)


def test_a_failed_policy_step_is_reported_as_incomplete_analysis():
    with temp_container() as (c, _):
        failed = AgentResult("policy", StepStatus.FAILED, "This step failed.", error="The policy has no readable text.")
        result, _ = run(c, make_claim(c.db), policy_result=failed)
        assert result.status == StepStatus.WARNING and result.output["recommendation"] == "request_more_information"
        assert any("Policy analysis failed" in item for item in result.output["open_items"])


def test_accusatory_wording_in_any_agents_text_is_caught_by_the_audit():
    with temp_container() as (c, _):
        bad = RiskOutput(flags=[RiskFlag("X", "low", "process", "Possible fraud", "The claimant lied.")])
        result, _ = run(c, make_claim(c.db), risk=bad)
        out = ComplianceOutput.from_dict(result.output)
        assert result.status == StepStatus.WARNING and out.language_check_passed is False
        assert any("the risk indicators" in w for w in result.warnings)
        clean, _ = run(c, make_claim(c.db))
        assert ComplianceOutput.from_dict(clean.output).language_check_passed is True


def test_an_incomplete_risk_assessment_is_recorded_as_a_limitation():
    from claimflow.schemas import RISK_INCOMPLETE
    with temp_container() as (c, _):
        result, _ = run(c, make_claim(c.db), risk=RiskOutput(assessment_complete=False, incomplete_reasons=["x"], model_check="ran"))
        assert RISK_INCOMPLETE in ComplianceOutput.from_dict(result.output).limitations
        complete, _ = run(c, make_claim(c.db), risk=RiskOutput(model_check="ran"))
        assert RISK_INCOMPLETE not in ComplianceOutput.from_dict(complete.output).limitations
