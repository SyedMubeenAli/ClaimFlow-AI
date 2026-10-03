"""Demo Mode: the whole workflow offline, on the three demo claims, with no network."""

import json

import requests

from claimflow.demo import DEMO_CLAIMS, held_back_quotation, load_demo_claims
from claimflow.demo_model import DemoProvider
from claimflow.errors import LLMResponseError
from claimflow.llm import FakeProvider, LLMProvider, LLMRequest
from claimflow.models import ClaimStatus, DocumentKind, ReviewDecision, StepStatus
from claimflow.pipeline import steps_changed_by_document
from claimflow.rag import HashingEmbedder, answer_question
from claimflow.schemas import AssessmentOutput, ComplianceOutput, DamageOutput, PolicyOutput, RiskOutput
from claimflow.uploads import store_document
from tests.helpers import attach_sample_policy, make_claim, temp_container
from tests.test_rag_index_answer import service


class NoNetwork:
    """While active, any attempt to send an HTTP request fails the test."""
    def __enter__(self):
        self.original = requests.sessions.Session.send
        def forbidden(*args, **kwargs):
            raise AssertionError("Demo Mode tried to use the network")
        requests.sessions.Session.send = forbidden
        return self

    def __exit__(self, *exc):
        requests.sessions.Session.send = self.original


def process(c, claim_id):
    return c.orchestrator.run(claim_id)


def by_name(c):
    return {claim.claimant_name: claim for claim, _ in load_demo_claims(c)}


def recommendation(summary):
    return ComplianceOutput.from_dict(summary.results["compliance"].output).recommendation


# ---------------------------------------------------------------- the three stories

def test_demo_mode_runs_all_three_claims_with_no_network_even_with_a_provider_configured_but_no_key():
    with NoNetwork(), temp_container(DEMO_MODE="true", LLM_PROVIDER="gemini") as (c, _):
        assert isinstance(c.llm, DemoProvider)
        claims = by_name(c)
        for name in claims:
            summary = process(c, claims[name].id)
            assert summary.final_status == ClaimStatus.AWAITING_REVIEW
            assert all(r.status in (StepStatus.DONE, StepStatus.WARNING) for r in summary.results.values()), name
            assert all(r.model in (None, "demo-scripted-model") for r in summary.results.values())


def test_the_clean_claim_is_approved_with_the_right_amount_and_cited_findings():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = by_name(c)["Ayesha Khan"]
        summary = process(c, claim.id)
        assert recommendation(summary) == "approve_for_processing"
        assessment = AssessmentOutput.from_dict(summary.results["assessment"].output)
        assert (assessment.estimate, assessment.deductible, assessment.coverage_limit) == ("185000.00", "25000.00", "1500000.00")
        assert assessment.potential_amount == "160000.00" and assessment.limit_applied is False      # 185,000 - 25,000
        policy = PolicyOutput.from_dict(summary.results["policy"].output)
        by_topic = {f["topic"]: f for f in policy.findings}
        assert by_topic["deductible"]["page"] == 3 and by_topic["deductible"]["section"] == "Section 4. Excess"
        assert by_topic["coverage_limit"]["page"] == 1 and by_topic["coverage"]["page"] == 2
        assert policy.not_found == [] or [n["topic"] for n in policy.not_found] == []
        assert RiskOutput.from_dict(summary.results["risk"].output).flags == []
        damage = DamageOutput.from_dict(summary.results["damage"].output)
        assert damage.photos[0].affected_areas == ["rear bumper", "boot lid"] and damage.photos[0].image_quality == "good"
        assert "cannot see images" in damage.overall_summary and damage.consistent_with_description == "yes"


def test_the_missing_quotation_claim_asks_for_information_then_is_approved_after_a_selective_rerun():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = by_name(c)["Bilal Raza"]
        first = process(c, claim.id)
        assert recommendation(first) == "request_more_information"
        assert "Repair quotation with a confirmed total" in first.results["evidence"].output["missing"]
        assert first.results["assessment"].output["calculated"] is False

        c.db.add_review(claim.id, ReviewDecision.REQUEST_INFO, "Sara Ahmed", "Please send the repair quotation.",
                        "request_more_information")
        assert c.db.get_claim(claim.id).status == ClaimStatus.INFO_REQUESTED

        name, data = held_back_quotation(claim, c.db.list_documents(claim.id))
        store_document(c.db, c.settings.upload_dir, claim.id, DocumentKind.QUOTATION, name, data, c.settings.max_upload_bytes)
        assert held_back_quotation(claim, c.db.list_documents(claim.id)) is None                    # once attached, no longer offered
        second = c.orchestrator.rerun(claim.id, steps_changed_by_document(DocumentKind.QUOTATION))
        assert second.ran == ["intake", "evidence", "risk", "assessment", "compliance"]            # policy and damage were not repeated
        assert recommendation(second) == "approve_for_processing"
        assert second.results["assessment"].output["potential_amount"] == "67500.00"               # 92,500 - 25,000
        assert c.db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW


def test_the_conflict_claim_is_escalated_and_the_injected_instruction_is_reported_not_followed():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = by_name(c)["Sara Ahmed"]
        summary = process(c, claim.id)
        assert recommendation(summary) == "escalate_for_investigation"
        risk = RiskOutput.from_dict(summary.results["risk"].output)
        assert {f.code for f in risk.flags} >= {"DATE_MISMATCH", "INSTRUCTION_TEXT"} and risk.counts["medium"] >= 2
        assert "accident_report_conflict.pdf" in summary.results["intake"].output["instruction_like_text"]
        assert summary.results["compliance"].output["recommendation"] != "approve_for_processing"      # the file's order was ignored
        assert summary.results["assessment"].output["potential_amount"] == "215000.00"


def test_loading_is_repeatable_and_no_evidence_file_is_shared_between_demo_claims():
    with temp_container(DEMO_MODE="true") as (c, _):
        first = load_demo_claims(c)
        again = load_demo_claims(c)
        assert [new for _, new in first] == [True] * 3 and [new for _, new in again] == [False] * 3
        assert len(c.db.list_claims()) == 3
        for claim, _ in first:
            shared = [s for s in c.db.shared_documents(claim.id) if s.kind != DocumentKind.POLICY]
            assert shared == [], claim.claimant_name                       # the policy is shared on purpose, nothing else is
        assert len(DEMO_CLAIMS) == 3


# ---------------------------------------------------------------- the scripted model itself

def test_the_demo_model_answers_only_what_it_has_a_script_for_and_is_deterministic():
    model = DemoProvider()
    assert model.supports_vision and model.name == "demo"
    try:
        model.complete(LLMRequest(prompt="TASK: write a poem"))
    except LLMResponseError as exc:
        assert "no script" in exc.message
    else:
        raise AssertionError("expected LLMResponseError")
    request = LLMRequest(prompt="TASK: write search queries\n\nIncident type: Theft\nWhat happened: car stolen")
    assert model.complete(request).text == model.complete(request).text
    assert json.loads(model.complete(request).text)["queries"][0].startswith("theft cover")


def test_demo_policy_questions_are_answered_from_the_policy_or_reported_as_not_found():
    with temp_container() as (c, tmp):
        index = service(c, tmp, HashingEmbedder()).corpus([attach_sample_policy(c, make_claim(c.db))])
        expected = {"What is the excess for a collision claim?": ("PKR 25,000", 3),
                    "What is the excess for a theft claim?": ("PKR 50,000", 3),
                    "Is racing covered?": ("racing", 4),
                    "How long do I have to report a theft to the police?": ("24 hours", 2),
                    "What is the sum insured?": ("1,500,000", 1),
                    "Does the policy cover wear and tear?": ("wear and tear", 4)}
        for question, (words, page) in expected.items():
            result = answer_question(question, index, DemoProvider())
            assert result.found and words in result.answer and result.citations[0].page == page, (question, result.answer)
        for question in ("Is a rental car provided while my car is repaired?", "What is the roadside assistance phone number?",
                         "Can I add my spouse as a named driver?"):
            result = answer_question(question, index, DemoProvider())
            assert result.status == "not_found" and not result.citations, question


def test_demo_photo_descriptions_come_from_file_names_and_the_answer_says_so():
    with temp_container(DEMO_MODE="true") as (c, _):
        from tests.helpers import attach_photo, png_bytes
        claim = make_claim(c.db, description="Hit on the front side by a bus.")
        attach_photo(c, claim, "front_bumper_damage.png", png_bytes((600, 400)))
        attach_photo(c, claim, "IMG_0042.png", png_bytes((100, 80), (30, 30, 30)))
        out = DamageOutput.from_dict(process(c, claim.id).results["damage"].output)
        front, unknown = out.photos
        assert front.affected_areas == ["front bumper"] and front.severity == "moderate"
        assert unknown.affected_areas == [] and unknown.severity == "unclear" and unknown.image_quality == "poor"   # tiny image
        assert "file names" in out.overall_summary


# ---------------------------------------------------------------- switching modes

def test_the_mode_can_be_switched_while_running_and_a_missing_key_is_explained():
    with temp_container(DEMO_MODE="true", LLM_PROVIDER="gemini") as (c, _):
        assert c.set_demo_mode(False) and "GEMINI_API_KEY" in c.set_demo_mode(False)
        assert c.settings.demo_mode is True and isinstance(c.llm, DemoProvider)           # unchanged
    with temp_container(DEMO_MODE="true", LLM_PROVIDER="gemini", GEMINI_API_KEY="k" * 12) as (c, _):
        assert c.set_demo_mode(False) is None
        assert c.settings.demo_mode is False and not isinstance(c.llm, DemoProvider) and c.orchestrator.llm is c.llm
        assert c.set_demo_mode(True) is None and isinstance(c.orchestrator.llm, DemoProvider) and c.orchestrator.settings.demo_mode
    with temp_container(DEMO_MODE="true") as (c, _):
        assert "No live provider" in c.set_demo_mode(False) and c.settings.demo_mode is True


def test_demo_mode_never_needs_a_model_download():
    from claimflow.rag import AutoEmbedder
    with temp_container(DEMO_MODE="true", EMBEDDING_BACKEND="auto") as (c, _):
        assert isinstance(c.rag.embedder, HashingEmbedder)
    with temp_container(DEMO_MODE="false", EMBEDDING_BACKEND="auto", LLM_PROVIDER="gemini", GEMINI_API_KEY="k" * 12) as (c, _):
        assert isinstance(c.rag.embedder, AutoEmbedder)                                  # live mode may use the real model
        c.set_demo_mode(True)
        assert isinstance(c.rag.embedder, HashingEmbedder)
