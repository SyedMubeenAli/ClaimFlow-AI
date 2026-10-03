from datetime import date, timedelta

from claimflow.errors import DatabaseError, InvalidInputError, WorkflowError
from claimflow.models import (
    AgentResult, ClaimStatus, DocumentKind, EvidenceRef, NewClaim, ReviewDecision, StepStatus,
)
from tests.helpers import make_claim, temp_db


def raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} was not raised")


def move_to_review(db, claim_id):
    db.update_claim_status(claim_id, ClaimStatus.PROCESSING)
    db.update_claim_status(claim_id, ClaimStatus.AWAITING_REVIEW)


# ---- claim input

def test_form_cleans_whitespace_and_converts_types():
    claim = NewClaim.from_form("  Ayesha Khan ", " POL-1 ", date(2026, 9, 1), "theft", "  Car stolen overnight.  ")
    assert claim.claimant_name == "Ayesha Khan" and claim.policy_number == "POL-1"
    assert claim.incident_type.value == "theft" and claim.description == "Car stolen overnight."


def test_form_reports_every_problem_at_once():
    error = raises(InvalidInputError, NewClaim.from_form, "", "", None, "nonsense", "short")
    for fragment in ("claimant", "policy number", "incident date", "incident type", "Describe"):
        assert fragment in error.message


def test_form_rejects_future_dates_and_odd_policy_numbers():
    tomorrow = date.today() + timedelta(days=1)
    assert "future" in raises(InvalidInputError, NewClaim.from_form, "Ali", "P1", tomorrow, "fire", "A fire in the garage.").message
    assert "policy number" in raises(InvalidInputError, NewClaim.from_form, "Ali", "P1; DROP TABLE", date(2026, 1, 1), "fire", "A fire in the garage.").message


# ---- claims

def test_claim_numbers_and_initial_status():
    with temp_db() as (db, _):
        first, second = make_claim(db), make_claim(db)
        assert first.claim_number.startswith("CF-") and first.claim_number.endswith("-0001")
        assert second.claim_number.endswith("-0002")
        assert first.status == ClaimStatus.SUBMITTED
        assert [c.id for c in db.list_claims()] == [second.id, first.id]


def test_missing_claim_is_a_clear_error():
    with temp_db() as (db, _):
        assert "999" in raises(DatabaseError, db.get_claim, 999).message


def test_status_transitions_are_enforced():
    with temp_db() as (db, _):
        claim = make_claim(db)
        raises(WorkflowError, db.update_claim_status, claim.id, ClaimStatus.APPROVED)
        move_to_review(db, claim.id)
        assert db.update_claim_status(claim.id, ClaimStatus.AWAITING_REVIEW).status == ClaimStatus.AWAITING_REVIEW


# ---- documents

def test_documents_round_trip():
    with temp_db() as (db, _):
        claim = make_claim(db)
        doc = db.add_document(claim.id, DocumentKind.POLICY, "policy.pdf", "claim_1/a.pdf", "application/pdf", 1234, "ab" * 32)
        assert [d.id for d in db.list_documents(claim.id)] == [doc.id]
        db.set_extracted_text(doc.id, "Section 4.2 Exclusions")
        assert db.list_documents(claim.id)[0].extracted_text == "Section 4.2 Exclusions"


# ---- agent runs

def test_latest_result_wins_and_history_keeps_everything():
    with temp_db() as (db, _):
        claim = make_claim(db)
        db.save_agent_result(claim.id, AgentResult("intake", StepStatus.FAILED, "first", error="boom"))
        db.save_agent_result(claim.id, AgentResult("intake", StepStatus.DONE, "second"))
        db.save_agent_result(claim.id, AgentResult("policy", StepStatus.WARNING, "p", warnings=["w"]))
        latest = db.latest_agent_results(claim.id)
        assert latest["intake"].summary == "second" and latest["policy"].status == StepStatus.WARNING
        assert [r.summary for r in db.agent_history(claim.id)] == ["first", "second", "p"]


def test_agent_result_survives_storage_unchanged():
    with temp_db() as (db, _):
        claim = make_claim(db)
        original = AgentResult("policy", StepStatus.DONE, "Found 2 clauses", output={"deductible": "500.00", "n": [1, 2]},
                               evidence=[EvidenceRef("policy.pdf", "Section 4.2", 7)], warnings=["unclear limit"],
                               duration_ms=42, model="m", cached=True)
        db.save_agent_result(claim.id, original)
        loaded = db.latest_agent_results(claim.id)["policy"]
        assert loaded.output == original.output and loaded.evidence == original.evidence
        assert loaded.warnings == original.warnings and loaded.cached is True and loaded.created_at


# ---- reviews

def test_review_needs_a_claim_waiting_for_review():
    with temp_db() as (db, _):
        claim = make_claim(db)
        error = raises(WorkflowError, db.add_review, claim.id, ReviewDecision.APPROVE, "Sara Ahmed")
        assert "not waiting" in error.message
        assert db.list_reviews(claim.id) == []


def test_review_requires_a_reviewer_name():
    with temp_db() as (db, _):
        claim = make_claim(db)
        move_to_review(db, claim.id)
        raises(InvalidInputError, db.add_review, claim.id, ReviewDecision.APPROVE, " ")
        assert db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW


def test_each_decision_moves_the_claim_to_the_right_status():
    expected = {ReviewDecision.APPROVE: ClaimStatus.APPROVED, ReviewDecision.REQUEST_INFO: ClaimStatus.INFO_REQUESTED,
                ReviewDecision.ESCALATE: ClaimStatus.ESCALATED}
    for decision, status in expected.items():
        with temp_db() as (db, _):
            claim = make_claim(db)
            move_to_review(db, claim.id)
            review = db.add_review(claim.id, decision, "Sara Ahmed", "checked the quote")
            assert db.get_claim(claim.id).status == status
            assert review.decision == decision and review.notes == "checked the quote"


def test_a_decided_claim_cannot_be_decided_again():
    with temp_db() as (db, _):
        claim = make_claim(db)
        move_to_review(db, claim.id)
        db.add_review(claim.id, ReviewDecision.ESCALATE, "Sara Ahmed")
        raises(WorkflowError, db.add_review, claim.id, ReviewDecision.APPROVE, "Sara Ahmed")
        raises(WorkflowError, db.update_claim_status, claim.id, ClaimStatus.PROCESSING)


# ---- dashboard

def test_dashboard_stats_count_latest_results_only():
    with temp_db() as (db, _):
        a, b, _c = make_claim(db), make_claim(db), make_claim(db)
        move_to_review(db, a.id)
        db.save_agent_result(a.id, AgentResult("intake", StepStatus.FAILED, error="x"))
        db.save_agent_result(a.id, AgentResult("intake", StepStatus.DONE))       # fixed by a rerun
        db.save_agent_result(b.id, AgentResult("policy", StepStatus.WARNING, warnings=["w"]))
        db.save_agent_result(b.id, AgentResult("review", StepStatus.FAILED, error="y"))
        stats = db.dashboard_stats()
        assert stats["total"] == 3 and stats["awaiting_review"] == 1
        assert stats["claims_with_warnings"] == 1 and stats["failed_steps"] == 1
        assert stats["by_status"]["submitted"] == 2
