"""Claim calculation: the deterministic arithmetic, and the agent that feeds it."""

import json
from decimal import Decimal as D

from claimflow.agents.assessment import ClaimAssessmentAgent
from claimflow.calculator import MAX_AMOUNT, calculate_claim
from claimflow.llm import FakeProvider
from claimflow.models import EvidenceRef, StepStatus
from claimflow.schemas import AssessmentOutput, IntakeOutput, PolicyOutput
from tests.helpers import ctx_for, make_claim, step_result, temp_container


def rejected(*args):
    try:
        calculate_claim(*args)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("ValueError was not raised")


# ---------------------------------------------------------------- the arithmetic

def test_estimate_minus_deductible_within_the_limit():
    calc = calculate_claim(D("50000"), D("25000"), D("500000"))
    assert (calc.after_deductible, calc.payable, calc.limit_applied) == (D("25000.00"), D("25000.00"), False)


def test_an_estimate_at_or_below_the_deductible_pays_nothing():
    for estimate in ("10000", "25000", "0"):
        calc = calculate_claim(D(estimate), D("25000"), D("500000"))
        assert calc.payable == D("0.00") and calc.limit_applied is False
    assert "does not exceed the deductible" in calculate_claim(D("100"), D("500"), D("900")).steps[1]


def test_the_limit_caps_the_amount_and_says_so():
    calc = calculate_claim(D("700000"), D("25000"), D("500000"))
    assert calc.after_deductible == D("675000.00") and calc.payable == D("500000.00") and calc.limit_applied is True
    assert "limit applies" in calc.steps[2]


def test_an_amount_exactly_at_the_limit_is_not_capped():
    calc = calculate_claim(D("525000"), D("25000"), D("500000"))
    assert calc.payable == D("500000.00") and calc.limit_applied is False


def test_a_zero_limit_pays_nothing_and_counts_as_applied():
    calc = calculate_claim(D("1000"), D("0"), D("0"))
    assert calc.payable == D("0.00") and calc.limit_applied is True


def test_decimal_arithmetic_has_no_floating_point_error():
    assert 0.30 - 0.10 != 0.20                                       # why floats are not used
    assert calculate_claim(D("0.30"), D("0.10"), D("1")).payable == D("0.20")
    assert calculate_claim(D("1234567.89"), D("0.01"), D("9999999")).payable == D("1234567.88")


def test_amounts_round_half_up_to_cents():
    assert calculate_claim(D("100.005"), D("0"), D("1000")).estimate == D("100.01")
    assert calculate_claim(D("100.004"), D("0"), D("1000")).estimate == D("100.00")


def test_the_result_matches_an_independent_calculation_over_many_inputs():
    values = [0, 1, 99, 100, 2500, 24999, 25000, 25001, 500000, 999999]
    for estimate in values:
        for deductible in (0, 100, 25000):
            for limit in (0, 50, 500000):
                expected_cents = min(max(estimate - deductible, 0), limit) * 100          # integer reference
                got = calculate_claim(D(estimate), D(deductible), D(limit)).payable
                assert got * 100 == expected_cents, (estimate, deductible, limit)


def test_invalid_inputs_are_refused_not_guessed():
    assert "zero or more" in rejected(D("-1"), D("0"), D("10"))
    assert "zero or more" in rejected(D("10"), D("-5"), D("10"))
    assert "Decimal" in rejected(1000.5, D("0"), D("10"))                  # floats are refused
    assert "Decimal" in rejected(True, D("0"), D("10"))
    assert "zero or more" in rejected(D("NaN"), D("0"), D("10"))
    assert "zero or more" in rejected(D("Infinity"), D("0"), D("10"))
    assert "out of range" in rejected(MAX_AMOUNT + 1, D("0"), D("10"))


def test_the_steps_show_the_working():
    steps = calculate_claim(D("50000"), D("25000"), D("500000")).steps
    assert steps == ("Repair estimate: 50,000.00", "After the deductible of 25,000.00: 25,000.00",
                     "This is within the coverage limit of 500,000.00.", "Potential claim amount: 25,000.00")


# ---------------------------------------------------------------- the agent

def run_assessment(c, claim, intake=None, policy=None, policy_status=StepStatus.DONE, evidence=None, llm=None):
    results = {"intake": step_result("intake", intake or IntakeOutput(repair_estimate="50000.00", estimate_source="quote.pdf"),
                                     evidence=[EvidenceRef("quote.pdf", "Repair estimate 50000.00", 1)])}
    if policy is not False:
        results["policy"] = step_result(
            "policy", policy or PolicyOutput(coverage_applies="yes", deductible="25000.00", deductible_kind="fixed",
                                             coverage_limit="500000.00"),
            status=policy_status, evidence=evidence if evidence is not None else
            [EvidenceRef("policy.pdf", "the deductible: You must pay the first 25,000", 2),
             EvidenceRef("policy.pdf", "the coverage limit: sum insured of 500,000", 1)])
    llm = llm or FakeProvider()
    return ClaimAssessmentAgent().run(ctx_for(c, claim, results, llm=llm)), llm


def test_the_agent_calculates_from_verified_inputs_and_cites_each_one():
    with temp_container() as (c, _):
        result, _ = run_assessment(c, make_claim(c.db))
        out = AssessmentOutput.from_dict(result.output)
        assert result.status == StepStatus.DONE and out.calculated and out.potential_amount == "25000.00"
        assert (out.estimate, out.deductible, out.coverage_limit) == ("50000.00", "25000.00", "500000.00")
        assert [(e.source, e.page) for e in result.evidence] == [("quote.pdf", 1), ("policy.pdf", 2), ("policy.pdf", 1)]


def test_the_agent_never_calls_the_model():
    with temp_container() as (c, _):
        # Even a model that offers a number cannot get it into the result: it is never asked.
        llm = FakeProvider(default=json.dumps({"potential_amount": "999999.00"}))
        result, llm = run_assessment(c, make_claim(c.db), llm=llm)
        assert llm.calls == [] and result.output["potential_amount"] == "25000.00"


def test_each_missing_input_is_named_and_nothing_is_calculated():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        result, _ = run_assessment(c, claim, intake=IntakeOutput())
        out = AssessmentOutput.from_dict(result.output)
        assert result.status == StepStatus.WARNING and out.calculated is False and out.potential_amount is None
        assert out.missing_inputs == ["a confirmed repair estimate"]
        result, _ = run_assessment(c, claim, policy=PolicyOutput(coverage_applies="yes"))
        assert len(result.output["missing_inputs"]) == 2 and result.output["potential_amount"] is None


def test_a_percentage_deductible_is_not_guessed():
    with temp_container() as (c, _):
        result, _ = run_assessment(c, make_claim(c.db), policy=PolicyOutput(
            coverage_applies="yes", deductible_kind="percent", coverage_limit="500000.00"))
        assert result.output["calculated"] is False and "percentage" in result.output["missing_inputs"][0]


def test_a_missing_policy_gives_a_clear_reason():
    with temp_container() as (c, _):
        result, _ = run_assessment(c, make_claim(c.db), policy=False)
        assert result.status == StepStatus.WARNING
        assert any("the policy was not analysed" in m for m in result.output["missing_inputs"])


def test_unconfirmed_coverage_or_a_possible_exclusion_keeps_the_amount_but_adds_conditions():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        unclear = PolicyOutput(coverage_applies="unclear", deductible="25000.00", coverage_limit="500000.00")
        result, _ = run_assessment(c, claim, policy=unclear)
        assert result.status == StepStatus.WARNING and result.output["potential_amount"] == "25000.00"
        assert "information only" in result.output["conditions"][0]
        excluded = PolicyOutput(coverage_applies="yes", deductible="25000.00", coverage_limit="500000.00",
                                exclusions=[{"title": "Racing", "applies_to_claim": True, "page": 2}])
        result, _ = run_assessment(c, claim, policy=excluded)
        assert "Racing" in result.output["conditions"][0] and "reviewer" in result.output["conditions"][0]


def test_the_limit_is_applied_by_the_agent_too():
    with temp_container() as (c, _):
        result, _ = run_assessment(c, make_claim(c.db), intake=IntakeOutput(repair_estimate="900000.00"))
        assert result.output["potential_amount"] == "500000.00" and result.output["limit_applied"] is True
