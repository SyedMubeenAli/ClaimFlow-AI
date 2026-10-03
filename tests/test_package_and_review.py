"""The final claim package PDF and the human-in-the-loop review audit."""

import io
import sqlite3
from datetime import date

from pypdf import PdfReader

from claimflow.db import Database
from claimflow.demo import load_demo_claims
from claimflow.errors import InvalidInputError
from claimflow.language import find_accusation_terms
from claimflow.models import (
    AgentResult, ClaimRecord, ClaimStatus, IncidentType, ReviewDecision, StepStatus, is_override,
)
from claimflow.package import build_claim_package, decision_state
from tests.helpers import make_claim, temp_container, temp_db


def pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    # Table cells wrap, so a name or a sentence can be split across lines in the extracted text.
    return " ".join(" ".join(page.extract_text().split()) for page in reader.pages)


def processed(c, name="Ayesha Khan"):
    claim = {cl.claimant_name: cl for cl, _ in load_demo_claims(c)}[name]
    summary = c.orchestrator.run(claim.id)
    return claim, summary


def package(c, claim):
    return build_claim_package(c.db.get_claim(claim.id), c.db.list_documents(claim.id), c.db.latest_agent_results(claim.id),
                               c.db.list_reviews(claim.id), "Demo mode (scripted offline model)", "2026-10-02T10:00:00+00:00")


def raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} was not raised")


# ---------------------------------------------------------------- the package

def test_a_draft_package_shows_the_recommendation_the_calculation_and_the_cited_sources():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim, _ = processed(c)
        data = package(c, claim)
        text = pdf_text(data)
        assert data[:5] == b"%PDF-" and claim.claim_number in text
        assert "DRAFT: no human decision has been recorded yet" in text
        assert "Approve for processing" in text and "No human decision has been recorded yet" in text
        assert "160000.00" in text and "185000.00" in text and "25000.00" in text           # the working, not just the answer
        assert "Page 3, Section 4. Excess" in text and "sample_motor_policy.pdf" in text    # the source of the deductible
        assert "word for word" in text and "fixed rules in code" in text
        for step in ("Claim intake", "Policy analysis", "Evidence check", "Compliance review"):
            assert step in text                                                              # the audit trail
        assert "Demo mode (scripted offline model)" in text and "Page 1 of" in text


def test_a_decided_package_is_final_and_records_who_decided_and_whether_they_agreed():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim, _ = processed(c)
        c.db.add_review(claim.id, ReviewDecision.APPROVE, "Sara Ahmed", "Checked the quotation.", "approve_for_processing")
        text = pdf_text(package(c, claim))
        assert "FINAL: approved for processing by a person" in text and "DRAFT" not in text
        assert "Sara Ahmed" in text and "Agreed with the AI" in text and "Checked the quotation." in text


def test_an_override_is_visible_in_the_package():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim, summary = processed(c, "Sara Ahmed")                                          # the AI recommends escalating
        assert summary.results["compliance"].output["recommendation"] == "escalate_for_investigation"
        c.db.add_review(claim.id, ReviewDecision.APPROVE, "Bilal Raza", "The date on the form is the correct one.",
                        "escalate_for_investigation")
        text = pdf_text(package(c, claim))
        assert "Overrode the AI" in text and "Escalate for investigation" in text and "The date on the form is the correct one." in text
        assert "approved for processing" in text


def test_the_banner_matches_the_claim_state():
    claim = ClaimRecord(1, "CF-1", ClaimStatus.SUBMITTED, "A B", "P1", date(2026, 1, 1), IncidentType.OTHER, "x" * 20, "t", "t")
    from dataclasses import replace
    states = {ClaimStatus.APPROVED: "FINAL: approved", ClaimStatus.ESCALATED: "FINAL: escalated",
              ClaimStatus.INFO_REQUESTED: "INTERIM", ClaimStatus.AWAITING_REVIEW: "DRAFT", ClaimStatus.SUBMITTED: "DRAFT",
              ClaimStatus.FAILED: "DRAFT", ClaimStatus.PROCESSING: "DRAFT"}
    for status, prefix in states.items():
        assert decision_state(replace(claim, status=status))[0].startswith(prefix), status


def test_a_claim_nothing_has_run_for_still_produces_a_valid_package_that_says_so():
    with temp_container() as (c, _):
        text = pdf_text(package(c, make_claim(c.db)))
        assert "have not produced a recommendation" in text and "did not run" in text and "Not run" in text


def test_hostile_and_unusual_text_cannot_break_the_pdf():
    with temp_container() as (c, _):
        claim = make_claim(c.db, claimant_name="Ayesha <b>Khan</b> & Co", description="Hit <script>alert(1)</script> & "
                           "more </para> \u0627\u0631\u062f\u0648 caf\u00e9 " + "very long word " * 200)
        c.db.save_agent_result(claim.id, AgentResult("intake", StepStatus.FAILED, "x <i>y</i>", error="boom & <b>"))
        text = pdf_text(package(c, claim))
        assert "Ayesha <b>Khan</b> & Co" in text and "<script>alert(1)</script>" in text      # shown literally, never interpreted
        assert "caf\u00e9" in text and "boom & <b>" in text


def test_nothing_in_any_demo_package_uses_accusatory_wording():
    with temp_container(DEMO_MODE="true") as (c, _):
        for claim, _ in load_demo_claims(c):
            c.orchestrator.run(claim.id)
            text = pdf_text(package(c, claim))
            assert find_accusation_terms(text) == [], (claim.claimant_name, find_accusation_terms(text))


def test_a_long_package_paginates_and_every_page_is_numbered():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim, _ = processed(c, "Sara Ahmed")
        reader = PdfReader(io.BytesIO(package(c, claim)))
        assert len(reader.pages) >= 2
        for number, page in enumerate(reader.pages, 1):
            assert f"Page {number} of {len(reader.pages)}" in page.extract_text()


# ---------------------------------------------------------------- review audit

def test_override_detection():
    for recommendation, decision, expected in (
            ("approve_for_processing", ReviewDecision.APPROVE, False), ("approve_for_processing", ReviewDecision.ESCALATE, True),
            ("request_more_information", ReviewDecision.REQUEST_INFO, False), ("request_more_information", ReviewDecision.APPROVE, True),
            ("escalate_for_investigation", ReviewDecision.ESCALATE, False), ("escalate_for_investigation", ReviewDecision.REQUEST_INFO, True),
            (None, ReviewDecision.APPROVE, False), ("unknown", ReviewDecision.APPROVE, False)):
        assert is_override(decision, recommendation) is expected, (recommendation, decision)


def to_review(db, claim_id):
    db.update_claim_status(claim_id, ClaimStatus.PROCESSING)
    db.update_claim_status(claim_id, ClaimStatus.AWAITING_REVIEW)


def test_deciding_against_the_ai_needs_a_written_reason_and_a_refused_decision_changes_nothing():
    with temp_db() as (db, _):
        claim = make_claim(db)
        to_review(db, claim.id)
        for notes in ("", "   ", "no"):
            error = raises(InvalidInputError, db.add_review, claim.id, ReviewDecision.APPROVE, "Sara Ahmed", notes, "escalate_for_investigation")
            assert "differs from the AI recommendation" in error.message
        assert db.list_reviews(claim.id) == [] and db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW
        review = db.add_review(claim.id, ReviewDecision.APPROVE, "Sara Ahmed", "Cleared by phone.", "escalate_for_investigation")
        assert review.ai_recommendation == "escalate_for_investigation" and db.get_claim(claim.id).status == ClaimStatus.APPROVED


def test_agreeing_with_the_ai_needs_no_note_and_the_recommendation_is_stored():
    with temp_db() as (db, _):
        claim = make_claim(db)
        to_review(db, claim.id)
        review = db.add_review(claim.id, ReviewDecision.REQUEST_INFO, "Sara Ahmed", "", "request_more_information")
        assert review.ai_recommendation == "request_more_information" and review.notes == ""
        assert db.list_reviews(claim.id)[0].ai_recommendation == "request_more_information"


def test_a_database_from_an_earlier_build_is_upgraded_and_its_old_reviews_still_load():
    with temp_db() as (db, tmp):
        claim = make_claim(db)
        old = Database(tmp / "old.db")
        old.init()
        with sqlite3.connect(old.path) as conn:                       # recreate the previous reviews table without the new column
            conn.executescript("PRAGMA foreign_keys=OFF; DROP TABLE reviews; CREATE TABLE reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                               "claim_id INTEGER NOT NULL, decision TEXT NOT NULL, reviewer TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', "
                               "created_at TEXT NOT NULL);")
        old_claim = make_claim(old)
        with sqlite3.connect(old.path) as conn:
            conn.execute("INSERT INTO reviews (claim_id, decision, reviewer, notes, created_at) VALUES (?,?,?,?,?)",
                         (old_claim.id, "approve", "Earlier Reviewer", "from the old build", "2026-09-01T00:00:00+00:00"))
        old.init()                                                    # the upgrade
        loaded = old.list_reviews(old_claim.id)
        assert loaded[0].reviewer == "Earlier Reviewer" and loaded[0].ai_recommendation is None
        old.init()                                                    # running it again is harmless
        to_review(old, old_claim.id) if old.get_claim(old_claim.id).status == ClaimStatus.SUBMITTED else None


def test_the_review_history_of_a_claim_keeps_every_decision_in_order():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim, _ = processed(c, "Bilal Raza")
        c.db.add_review(claim.id, ReviewDecision.REQUEST_INFO, "Sara Ahmed", "Send the quotation.", "request_more_information")
        c.orchestrator.rerun(claim.id, {"intake"})
        c.db.add_review(claim.id, ReviewDecision.ESCALATE, "Sara Ahmed", "Quotation still missing.", "request_more_information")
        history = c.db.list_reviews(claim.id)
        assert [r.decision for r in history] == [ReviewDecision.ESCALATE, ReviewDecision.REQUEST_INFO]
        assert is_override(history[0].decision, history[0].ai_recommendation) and not is_override(history[1].decision, history[1].ai_recommendation)
