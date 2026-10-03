"""
How the agents communicate, tested end to end: the shared state, typed hand-offs,
the full seven-agent run, and containment when a step fails.
"""

import json
from datetime import date, timedelta

from claimflow.agents import default_agents
from claimflow.agents.base import Agent
from claimflow.errors import LLMUnavailableError
from claimflow.language import find_accusation_terms, neutral_title
from claimflow.llm import FakeProvider
from claimflow.models import ClaimState, ClaimStatus, DocumentKind, StepStatus
from claimflow.orchestrator import WorkflowOrchestrator
from claimflow.schemas import (
    AssessmentOutput, ComplianceOutput, ConsistencyCheck, DamageOutput, DamagePhoto, EvidenceItem, EvidenceOutput,
    IntakeOutput, PolicyOutput, RiskFlag, RiskOutput,
)
from tests.helpers import (
    POLICY_PAGES, QUOTATION_PAGES, REPORT_PAGES, attach, attach_photo, make_claim, quiet_logs, temp_container,
)
from tests.test_agent_policy import RACING, chunk_id, good_reply

SCHEMAS = [IntakeOutput, PolicyOutput, DamageOutput, EvidenceOutput, RiskOutput, AssessmentOutput, ComplianceOutput]

INTAKE = {"incident_date": "2026-09-01", "location": "Karachi", "vehicle": "Honda City 2019",
          "police_report_number": "KHI-4411", "injuries_mentioned": False, "repair_estimate": "50,000",
          "summary": "The claimant was rear-ended at a light. The bumper was damaged."}
DAMAGE = {"photos": [{"photo": 1, "visible_damage": ["dented bumper"], "affected_areas": ["front bumper"],
                      "severity": "moderate", "image_quality": "good"}],
          "overall_summary": "A dented front bumper is visible.", "consistent_with_description": "yes"}
EXPLAIN = {"explanation": "The checks found no blocking issue. The potential claim amount is 25,000.00. "
                          "A person makes the final decision."}


def script(**overrides):
    base = {
        "TASK: extract claim facts": json.dumps(INTAKE),
        "TASK: write search queries": '{"queries": ["collision cover"]}',
        "TASK: extract policy terms": good_reply(),
        "TASK: describe damage photos": json.dumps(DAMAGE),
        "TASK: find inconsistencies": '{"inconsistencies": []}',
        "TASK: explain recommendation": json.dumps(EXPLAIN),
    }
    base.update(overrides)
    return base


def full_claim(c, with_policy=True, **claim_args):
    claim = make_claim(c.db, **({"incident_date": date(2026, 9, 1)} | claim_args))
    if with_policy:
        attach(c, claim, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
    attach(c, claim, DocumentKind.ACCIDENT_REPORT, "report.pdf", REPORT_PAGES)
    attach(c, claim, DocumentKind.QUOTATION, "quote.pdf", QUOTATION_PAGES)
    attach_photo(c, claim)
    return claim


def orchestrate(c, fake, agents=None):
    return WorkflowOrchestrator(c.db, fake, c.settings, agents or default_agents())


def boom(request):
    raise LLMUnavailableError("The provider is down.")


# ---------------------------------------------------------------- typed schemas

def test_every_schema_survives_a_round_trip_through_json():
    for schema in SCHEMAS:
        empty = schema()
        assert schema.from_dict(json.loads(json.dumps(empty.to_dict()))) == empty


def test_nested_schemas_come_back_as_typed_objects():
    risk = RiskOutput(flags=[RiskFlag("X", "high", "document", "t", "d")], highest_severity="high", counts={"high": 1})
    back = RiskOutput.from_dict(json.loads(json.dumps(risk.to_dict())))
    assert back == risk and isinstance(back.flags[0], RiskFlag)
    evidence = EvidenceOutput(items=[EvidenceItem("policy", "Insurance policy", "present")],
                              checks=[ConsistencyCheck("policy_number", "x", "match")], required=1, present=1)
    assert EvidenceOutput.from_dict(evidence.to_dict()) == evidence
    damage = DamageOutput(photos=[DamagePhoto("a.png", ["dent"], ["bumper"], "minor", "good")])
    assert isinstance(DamageOutput.from_dict(damage.to_dict()).photos[0], DamagePhoto)


def test_reading_is_tolerant_of_missing_garbage_and_unknown_data():
    for schema in SCHEMAS:
        for bad in (None, [], "text", 7, {}, {"unknown_key": 1}):
            assert schema.from_dict(bad) == schema()
    assert RiskOutput.from_dict({"flags": ["not a flag", None, {"code": "X"}]}).flags == [RiskFlag(code="X")]
    assert IntakeOutput.from_dict({"summary": "ok", "future_field": 1}).summary == "ok"


def test_a_skipped_step_reads_as_safe_defaults_not_a_crash():
    with temp_container() as (c, _):
        state = ClaimState(make_claim(c.db))
        assert state.typed("policy", PolicyOutput) == PolicyOutput() and state.succeeded("policy") is False
        assert state.typed("risk", RiskOutput).highest_severity == "none"


# ---------------------------------------------------------------- the wording guard

def test_the_wording_guard_blocks_accusations_and_allows_ordinary_words():
    for text in ("fraud", "This is fraudulent.", "He lied.", "a dishonest account", "a scam", "forged papers", "fabricated quote",
                 "The claimant is a liar", "criminal activity", "deceptive"):
        assert find_accusation_terms(text), text
    for text in ("The car was lying on its side.", "The damage lies on the left.", "He forgot the report.",
                 "The bumper was dented.", "Theft of the vehicle was reported.", ""):
        assert find_accusation_terms(text) == [], text
    assert neutral_title("Fraudulent claims") == "a conduct-related policy condition" and neutral_title("Racing") == "Racing"


# ---------------------------------------------------------------- the full run

def test_a_complete_claim_runs_all_seven_agents_and_reaches_a_person():
    with temp_container() as (c, _):
        claim = full_claim(c)
        fake = FakeProvider(script())
        summary = orchestrate(c, fake).run(claim.id)
        statuses = {n: r.status for n, r in summary.results.items()}
        assert statuses == {n: StepStatus.DONE for n in ("intake", "policy", "damage", "evidence", "risk", "assessment", "compliance")}
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW
        assert len(fake.calls) == 6                                         # intake 1, policy 2, damage 1, risk 1, compliance 1
        assessment = AssessmentOutput.from_dict(summary.results["assessment"].output)
        assert assessment.potential_amount == "25000.00"                    # 50,000 - 25,000, from code
        compliance = ComplianceOutput.from_dict(summary.results["compliance"].output)
        assert compliance.recommendation == "approve_for_processing" and compliance.explanation_source == "model"
        assert compliance.human_decision_required is True


def test_each_agent_sees_only_what_ran_before_it():
    with temp_container() as (c, _):
        seen = {}

        class Spy(Agent):
            def __init__(self, inner):
                self.inner, self.name, self.label = inner, inner.name, inner.label

            def run(self, ctx):
                seen[self.name] = sorted(ctx.state.results)
                return self.inner.run(ctx)

        orchestrate(c, FakeProvider(script()), [Spy(a) for a in default_agents()]).run(full_claim(c).id)
        assert seen == {
            "intake": [],
            "policy": ["intake"],
            "damage": ["intake", "policy"],
            "evidence": ["damage", "intake", "policy"],
            "risk": ["damage", "evidence", "intake", "policy"],
            "assessment": ["damage", "evidence", "intake", "policy", "risk"],
            "compliance": ["assessment", "damage", "evidence", "intake", "policy", "risk"],
        }


def test_an_agent_cannot_corrupt_what_the_others_see():
    with temp_container() as (c, _):
        seen = {}

        class Vandal(Agent):
            """Wipes the state it was given. The orchestrator hands every agent its own copy."""
            def __init__(self, inner):
                self.inner, self.name, self.label = inner, inner.name, inner.label

            def run(self, ctx):
                result = self.inner.run(ctx)
                ctx.state.results.clear()
                return result

        class Spy(Agent):
            def __init__(self, inner):
                self.inner, self.name, self.label = inner, inner.name, inner.label

            def run(self, ctx):
                seen[self.name] = sorted(ctx.state.results)
                return self.inner.run(ctx)

        agents = [Vandal(a) if a.name == "policy" else Spy(a) if a.name == "evidence" else a for a in default_agents()]
        orchestrate(c, FakeProvider(script()), agents).run(full_claim(c).id)
        assert seen["evidence"] == ["damage", "intake", "policy"]


def test_a_fact_travels_from_intake_through_risk_to_the_recommendation():
    with temp_container() as (c, _):
        form_date, document_date = date.today() - timedelta(days=3), date.today() - timedelta(days=5)
        claim = make_claim(c.db, incident_date=form_date)
        attach(c, claim, DocumentKind.ACCIDENT_REPORT, "report.pdf",
               [[f"Date of accident: {document_date.day} {document_date:%B %Y}.", "Vehicle: Honda City."]])
        fake = FakeProvider(script(**{"TASK: extract claim facts": json.dumps({**INTAKE, "incident_date": document_date.isoformat(),
                                                                               "repair_estimate": None})}))
        summary = orchestrate(c, fake).run(claim.id)
        assert summary.results["intake"].output["incident_date_in_documents"] == document_date.isoformat()    # 1. intake finds it
        risk = RiskOutput.from_dict(summary.results["risk"].output)
        assert "DATE_MISMATCH" in [f.code for f in risk.flags]                                               # 2. risk reads it
        compliance = ComplianceOutput.from_dict(summary.results["compliance"].output)
        assert compliance.recommendation == "request_more_information"
        assert any("incident date differs" in r for r in compliance.reasons)                                 # 3. compliance cites it


def test_a_claim_with_no_policy_still_reaches_a_person_with_a_clear_request():
    with temp_container() as (c, _):
        claim = full_claim(c, with_policy=False)
        fake = FakeProvider(script())
        summary = orchestrate(c, fake).run(claim.id)
        assert summary.results["policy"].status == StepStatus.SKIPPED
        assert summary.results["evidence"].status == StepStatus.WARNING and "Insurance policy" in summary.results["evidence"].output["missing"]
        assert summary.results["assessment"].output["calculated"] is False
        compliance = summary.results["compliance"].output
        assert compliance["recommendation"] == "request_more_information"
        assert any("Insurance policy" in r for r in compliance["reasons"]) and any("could not be calculated" in r for r in compliance["reasons"])
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW
        assert not any("extract policy terms" in call.prompt for call in fake.calls)


def test_a_possible_exclusion_escalates():
    with temp_container() as (c, _):
        def reply(request):
            data = json.loads(good_reply()(request))
            data["exclusions"][0]["applies_to_claim"] = True
            return json.dumps(data)
        summary = orchestrate(c, FakeProvider(script(**{"TASK: extract policy terms": reply}))).run(full_claim(c).id)
        compliance = summary.results["compliance"].output
        assert compliance["recommendation"] == "escalate_for_investigation"
        assert "A policy exclusion may apply: Racing." in compliance["reasons"]
        assert any("Racing" in w for w in summary.results["assessment"].warnings)      # the amount carries the condition


def test_the_same_file_on_two_claims_is_a_high_indicator_and_escalates():
    with temp_container() as (c, _):
        first, second = full_claim(c), full_claim(c)
        summary = orchestrate(c, FakeProvider(script())).run(second.id)
        risk = RiskOutput.from_dict(summary.results["risk"].output)
        assert risk.highest_severity == "high" and risk.counts["high"] >= 1
        assert summary.results["compliance"].output["recommendation"] == "escalate_for_investigation"
        assert first.claim_number in " ".join(f.detail for f in risk.flags)


# ---------------------------------------------------------------- failures stay contained

def test_failing_model_calls_degrade_the_result_but_never_stop_the_claim():
    with temp_container() as (c, _):
        claim = full_claim(c)
        fake = FakeProvider(script(**{"TASK: extract policy terms": boom, "TASK: find inconsistencies": boom,
                                      "TASK: explain recommendation": boom}))
        with quiet_logs():
            summary = orchestrate(c, fake).run(claim.id)
        assert summary.results["policy"].status == StepStatus.FAILED and "down" in summary.results["policy"].error
        assert summary.results["risk"].status == StepStatus.WARNING and summary.results["risk"].output["model_check"] == "failed"
        assert summary.results["assessment"].output["calculated"] is False       # no policy terms, so no number
        compliance = summary.results["compliance"]
        assert compliance.status == StepStatus.WARNING and compliance.output["explanation_source"] == "template"
        assert compliance.output["recommendation"] == "request_more_information"
        assert any("Policy analysis failed" in item for item in compliance.output["open_items"])
        assert c.db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW    # a person still gets it, with the gaps listed


def test_a_failed_intake_stops_the_dependent_steps_and_says_so():
    with temp_container() as (c, _):
        claim = full_claim(c)
        with quiet_logs():
            summary = orchestrate(c, FakeProvider(script(**{"TASK: extract claim facts": "no json here"}))).run(claim.id)
        assert summary.results["intake"].status == StepStatus.FAILED
        for step in ("evidence", "risk", "assessment", "compliance"):
            assert summary.results[step].status == StepStatus.SKIPPED
        assert summary.results["policy"].status == StepStatus.DONE and summary.results["damage"].status == StepStatus.DONE
        assert c.db.get_claim(claim.id).status == ClaimStatus.FAILED


def test_a_rerun_after_new_evidence_runs_only_the_affected_steps_and_reuses_the_rest():
    with temp_container() as (c, _):
        claim = full_claim(c, with_policy=False)
        fake = FakeProvider(script())
        orchestrator = orchestrate(c, fake)
        orchestrator.run(claim.id)
        attach(c, claim, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
        before = len(fake.calls)
        from claimflow.pipeline import steps_changed_by_document
        summary = orchestrator.rerun(claim.id, steps_changed_by_document(DocumentKind.POLICY))
        assert summary.ran == ["policy", "evidence", "risk", "assessment", "compliance"]
        assert len(fake.calls) - before == 4                    # policy 2, risk 1, compliance 1; intake and damage did not run again
        assert summary.results["assessment"].output["potential_amount"] == "25000.00"
        assert summary.results["compliance"].output["recommendation"] == "approve_for_processing"


def test_no_text_any_agent_produces_contains_accusatory_wording():
    """The whole pipeline, with a model that tries to accuse at every opportunity."""
    with temp_container() as (c, _):
        claim = full_claim(c, incident_date=date.today() - timedelta(days=90))
        accuse = {"inconsistencies": [{"topic": "events", "description": "The claimant is committing fraud.",
                                       "quote_a": "Date of accident: 1 September 2026.", "quote_b": "Police report number: KHI-4411."}]}
        fake = FakeProvider(script(**{"TASK: find inconsistencies": json.dumps(accuse),
                                      "TASK: explain recommendation": json.dumps({"explanation": "This claim is fraudulent."}),
                                      "TASK: describe damage photos": json.dumps({**DAMAGE, "overall_summary": "A fraudulent dent."})}))
        summary = orchestrate(c, fake).run(claim.id)

        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for v in value.values():
                    yield from strings(v)
            elif isinstance(value, list):
                for v in value:
                    yield from strings(v)

        for step, result in summary.results.items():
            if step == "policy":
                continue                                          # policy quotes are the policy's own words, shown as written
            text = list(strings(result.output)) + [result.summary] + result.warnings
            assert not any(find_accusation_terms(t) for t in text), (step, [t for t in text if find_accusation_terms(t)])
