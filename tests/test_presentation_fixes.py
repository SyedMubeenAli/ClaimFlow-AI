"""The presentation and wording fixes, the readable demo documents, and the successful demo path."""

import io
import subprocess
import sys
from dataclasses import replace
from datetime import date, timedelta

from pypdf import PdfReader

from claimflow import demo, extract
from claimflow.config import ROOT
from claimflow.decision import summarize
from claimflow.demo import DEMO_CLAIMS, create_upload_sample_claim, load_demo_claims, sample_documents, sample_form
from claimflow.errors import ClaimFlowError
from claimflow.extract import extract_pdf_text, has_text, read_pdf
from claimflow.models import ClaimStatus, DocumentKind, ReviewDecision, ReviewRecord
from claimflow.package import build_claim_package
from claimflow.schemas import (
    DAMAGE_DEMO_LABEL, RISK_INCOMPLETE, AssessmentOutput, ComplianceOutput, DamageOutput, EvidenceOutput, PolicyOutput, RiskOutput,
)
from claimflow.textutil import is_are, plural, quoted_note
from tests.helpers import POLICY_PAGES, attach, attach_photo, make_claim, make_pdf, png_bytes, temp_container
from tests.test_extract import FakeDocument, fake_pymupdf, with_fake

LIMITED = "limited demo analysis"


def pdf_text(data: bytes) -> str:
    return " ".join(" ".join(page.extract_text().split()) for page in PdfReader(io.BytesIO(data)).pages)


def package(c, claim, mode="Demo mode (scripted offline model)"):
    return build_claim_package(c.db.get_claim(claim.id), c.db.list_documents(claim.id), c.db.latest_agent_results(claim.id),
                               c.db.list_reviews(claim.id), mode, "2026-10-02T10:00:00+00:00")


def demo_claim(c, name):
    return {claim.claimant_name: claim for claim, _ in load_demo_claims(c)}[name]


# ---------------------------------------------------------------- 1. the human decision and the AI recommendation are separate things

def review(decision, recommendation, notes=""):
    return ReviewRecord(1, 1, decision, "Sara Ahmed", notes, "2026-10-02T10:00:00+00:00", recommendation)


def test_the_summary_keeps_the_human_decision_and_the_ai_recommendation_apart():
    pending = summarize("request_more_information", None)
    assert (pending.human, pending.human_tone, pending.ai) == ("Pending", "mute", "Request More Information") and pending.relation is None
    overrode = summarize("request_more_information", review(ReviewDecision.APPROVE, "request_more_information", "Quote received by email."))
    assert (overrode.human, overrode.ai, overrode.human_tone) == ("Approved for Processing", "Request More Information", "ok")
    assert overrode.relation == "The reviewer overrode the AI recommendation." and overrode.note == "Quote received by email."
    assert "Sara Ahmed" in overrode.human_detail and "2026-10-02 10:00:00 UTC" in overrode.human_detail
    agreed = summarize("approve_for_processing", review(ReviewDecision.APPROVE, "approve_for_processing"))
    assert agreed.relation == "The reviewer agreed with the AI recommendation."
    assert summarize(None, None).ai == "None" and "have not produced a recommendation" in summarize(None, None).ai_detail
    assert {summarize("x", review(d, None)).human for d in ReviewDecision} == {"Approved for Processing", "More Information Requested", "Escalated for Investigation"}


def test_a_recommendation_that_changed_after_the_decision_is_called_out():
    earlier = summarize("approve_for_processing", review(ReviewDecision.REQUEST_INFO, "request_more_information"))
    assert earlier.relation.startswith("The reviewer agreed") and "updated after that decision" in earlier.relation


def test_the_package_shows_both_labelled_separately_when_a_person_overrides_the_ai():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = demo_claim(c, "Bilal Raza")                                          # no quotation: the AI asks for more information
        c.orchestrator.run(claim.id)
        c.db.add_review(claim.id, ReviewDecision.APPROVE, "Sara Ahmed", "The quotation arrived by phone call.", "request_more_information")
        text = pdf_text(package(c, claim))
        assert "HUMAN DECISION Approved for Processing" in text
        assert "AI RECOMMENDATION Request More Information" in text
        assert "The reviewer overrode the AI recommendation." in text
        assert 'The reviewer\'s note: "The quotation arrived by phone call."' in text
        assert text.index("HUMAN DECISION") < text.index("AI RECOMMENDATION")
        assert "FINAL: approved for processing by a person" in text


def test_a_package_without_a_decision_says_pending_next_to_the_recommendation():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = demo_claim(c, "Ayesha Khan")
        c.orchestrator.run(claim.id)
        text = pdf_text(package(c, claim))
        assert "HUMAN DECISION Pending No human decision has been recorded yet." in text and "AI RECOMMENDATION Approve for Processing" in text


# ---------------------------------------------------------------- 2. risk wording when checks could not be performed

def test_the_package_never_says_no_indicators_were_found_when_checks_could_not_run():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = demo_claim(c, "Bilal Raza")                                          # the repair quotation is missing
        summary = c.orchestrator.run(claim.id)
        risk = RiskOutput.from_dict(summary.results["risk"].output)
        assert risk.assessment_complete is False and "Repair quotation" in " ".join(risk.incomplete_reasons)
        text = pdf_text(package(c, claim))
        risk_section = text[text.index("6. Risk indicators"): text.index("7. Damage analysis")]
        assert RISK_INCOMPLETE in risk_section and "No risk indicators were found" not in risk_section
        assert "Required evidence is missing or unusable" in risk_section


def test_a_complete_assessment_says_every_planned_check_was_performed_in_the_package_too():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = demo_claim(c, "Ayesha Khan")
        c.orchestrator.run(claim.id)
        text = pdf_text(package(c, claim))
        section = text[text.index("6. Risk indicators"): text.index("7. Damage analysis")]
        assert "No risk indicators were found. Every planned check was performed." in section and RISK_INCOMPLETE not in section


# ---------------------------------------------------------------- 3. demo damage analysis is labelled as limited

def test_demo_damage_analysis_is_labelled_limited_in_the_result_and_the_package():
    with temp_container(DEMO_MODE="true") as (c, _):
        claim = demo_claim(c, "Ayesha Khan")
        summary = c.orchestrator.run(claim.id)
        damage = DamageOutput.from_dict(summary.results["damage"].output)
        assert damage.analysis_mode == "demo" and DAMAGE_DEMO_LABEL in summary.results["damage"].summary
        assert "No vision model was used" in damage.disclaimer
        text = pdf_text(package(c, claim))
        assert "7. Damage analysis (limited demo analysis)" in text and "no vision model was used" in text
        assert "must not be treated as an assessment of the damage" in text


def test_a_real_vision_result_is_not_labelled_as_a_demo():
    from claimflow.agents.damage import DamageAnalysisAgent
    from claimflow.llm import FakeProvider
    from tests.helpers import ctx_for
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach_photo(c, claim, "dent.png", png_bytes())
        llm = FakeProvider({"TASK: describe damage photos": '{"photos": [{"photo": 1, "visible_damage": ["dent"], "affected_areas": ["door"], '
                            '"severity": "minor", "image_quality": "good"}], "overall_summary": "A small dent.", "consistent_with_description": "yes"}'})
        result = DamageAnalysisAgent().run(ctx_for(c, claim, llm=llm))
        assert DamageOutput.from_dict(result.output).analysis_mode == "vision" and DAMAGE_DEMO_LABEL not in result.summary
        c.db.save_agent_result(claim.id, result)
        text = pdf_text(package(c, claim, "Live AI (gemini, test)"))
        assert "7. Damage analysis" in text and "limited demo analysis" not in text


# ---------------------------------------------------------------- 4. the reviewer's note reads as a sentence

def test_a_reviewers_note_is_quoted_and_punctuated_so_it_cannot_run_into_the_next_sentence():
    assert quoted_note("Please send the repair quotation") == 'The reviewer\'s note: "Please send the repair quotation."'
    assert quoted_note("Please send the repair quotation.") == 'The reviewer\'s note: "Please send the repair quotation."'
    assert quoted_note("  Is the date right?  ") == 'The reviewer\'s note: "Is the date right?"'
    assert quoted_note("two\n\nlines   here") == 'The reviewer\'s note: "two lines here."'
    assert quoted_note("") == "" and quoted_note(None) == "" and quoted_note("   ") == ""


# ---------------------------------------------------------------- 5. singular and plural

def test_counts_use_the_right_form():
    assert [plural(n, "reason") for n in (0, 1, 2)] == ["0 reasons", "1 reason", "2 reasons"]
    assert plural(1, "policy", "policies") == "1 policy" and plural(2, "policy", "policies") == "2 policies"
    assert [is_are(n) for n in (0, 1, 2)] == ["are", "is", "are"]


def test_the_evidence_summary_says_is_for_one_and_are_for_the_rest():
    from claimflow.agents.evidence import EvidenceVerificationAgent
    from claimflow.models import DocumentKind
    from claimflow.schemas import IntakeOutput
    from tests.helpers import POLICY_PAGES, ctx_for, step_result
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach(c, claim, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
        one = EvidenceVerificationAgent().run(ctx_for(c, claim, {"intake": step_result("intake", IntakeOutput())}))
        assert one.summary.startswith("1 of 4 required items is present.")
        for kind, name, pages in ((DocumentKind.ACCIDENT_REPORT, "r.pdf", [["Accident report"]]), (DocumentKind.QUOTATION, "q.pdf", [["Total 5,000."]])):
            attach(c, claim, kind, name, pages)
        three = EvidenceVerificationAgent().run(ctx_for(c, claim, {"intake": step_result("intake", IntakeOutput(repair_estimate="5000.00"))}))
        assert three.summary.startswith("3 of 4 required items are present.")
        zero = EvidenceVerificationAgent().run(ctx_for(c, make_claim(c.db), {"intake": step_result("intake", IntakeOutput())}))
        assert zero.summary.startswith("0 of 4 required items are present.")


# ---------------------------------------------------------------- 6. professional wording for unreadable documents

def test_unreadable_documents_are_described_professionally_and_still_refused():
    from claimflow.agents.policy import PolicyAnalystAgent
    from claimflow.errors import AgentError
    from tests.helpers import ctx_for
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach(c, claim, DocumentKind.POLICY, "scan.pdf", [[]])                       # a PDF with no text layer
        try:
            PolicyAnalystAgent().run(ctx_for(c, claim))
        except AgentError as exc:
            assert exc.message == "The uploaded policy could not be parsed as readable text."
            assert "scanned image" in exc.user_message and "text-based PDF" in exc.user_message
        else:
            raise AssertionError("an unreadable policy must be refused, not guessed at")
        from claimflow.agents.evidence import build_items
        items = build_items(("policy",), {k: c.db.list_documents(claim.id) if k == DocumentKind.POLICY else [] for k in DocumentKind},
                            __import__("claimflow.schemas", fromlist=["IntakeOutput"]).IntakeOutput(), None)
        assert items[0].status == "unusable" and "could not be parsed as readable text" in items[0].detail


# ---------------------------------------------------------------- reading PDFs: a second reader before giving up

def test_a_pdf_that_pymupdf_finds_blank_is_given_to_pypdf_before_it_is_called_unreadable():
    blank_in_pymupdf = fake_pymupdf(FakeDocument(["", "  "]))
    result = with_fake(blank_in_pymupdf, lambda: read_pdf(make_pdf(POLICY_PAGES)))
    assert result.extractor == "pypdf" and has_text(result.pages[0].text)                  # pypdf could read it
    scan = make_pdf([[]])
    both_blank = with_fake(fake_pymupdf(FakeDocument([""])), lambda: read_pdf(scan))
    assert both_blank is not None and not any(has_text(p.text) for p in both_blank.pages)   # still a scan: still refused downstream
    assert with_fake(fake_pymupdf(FakeDocument([""])), lambda: extract_pdf_text(scan)).strip("\f ") == ""


# ---------------------------------------------------------------- readable, consistent sample documents

def test_the_sample_documents_are_readable_consistent_and_match_the_form():
    today = date(2026, 10, 2)
    files = {f.label: f for f in sample_documents(today)}
    form = sample_form(today)
    assert form["Incident date"] == "29 September 2026" and form["Policy number"] == "POL-1001"
    texts = {label: extract_pdf_text(f.data) for label, f in files.items() if f.mime == "application/pdf"}
    assert all(has_text(t) for t in texts.values()), "every sample PDF must be readable text"
    assert "POL-1001" in texts["Insurance policy"] and "Section 4. Excess" in texts["Insurance policy"]
    assert "Date of accident: 29 September 2026." in texts["Accident report"] and "Driver: Hamza Siddiqui" in texts["Accident report"]
    assert "Total 143,500." in texts["Repair quotation"] and "Customer: Hamza Siddiqui" in texts["Repair quotation"]
    assert files["Damage photo"].data[:2] == b"\xff\xd8" and files["Damage photo"].filename == "front_bumper_damage.jpg"
    assert sum(amount for _, amount in demo.UPLOAD_SAMPLE.items) == 143_500


def test_the_sample_files_uploaded_as_a_new_claim_give_the_full_successful_demo():
    with temp_container(DEMO_MODE="true") as (c, _):
        load_demo_claims(c)                                                          # the other demo claims exist too
        claim = create_upload_sample_claim(c)
        summary = c.orchestrator.run(claim.id)
        assert all(r.status == "done" or r.status.value == "done" for r in summary.results.values()), {k: v.status.value for k, v in summary.results.items()}
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW
        evidence = EvidenceOutput.from_dict(summary.results["evidence"].output)
        assert (evidence.present, evidence.required, evidence.missing) == (4, 4, [])
        policy = PolicyOutput.from_dict(summary.results["policy"].output)
        deductible = next(f for f in policy.findings if f["topic"] == "deductible")
        assert (deductible["value"], deductible["page"], deductible["section"]) == ("25000.00", 3, "Section 4. Excess")
        assert AssessmentOutput.from_dict(summary.results["assessment"].output).potential_amount == "118500.00"    # 143,500 - 25,000
        assert RiskOutput.from_dict(summary.results["risk"].output).assessment_complete and not summary.results["risk"].output["flags"]
        assert ComplianceOutput.from_dict(summary.results["compliance"].output).recommendation == "approve_for_processing"


def test_the_demo_loader_checks_its_own_documents_are_readable_and_says_so_if_they_are_not():
    original = demo.accident_report
    demo.accident_report = lambda spec, today: make_pdf([[]])                         # simulate a PDF reader that cannot read the file
    try:
        with temp_container(DEMO_MODE="true") as (c, _):
            try:
                load_demo_claims(c)
            except ClaimFlowError as exc:
                assert "could not be parsed as readable text" in exc.message and "PyMuPDF or pypdf" in exc.hint
            else:
                raise AssertionError("an unreadable demo document must stop the loader")
    finally:
        demo.accident_report = original


# ---------------------------------------------------------------- the one-command demo

def run_script(claim, out):
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_demo_workflow.py"), "--claim", claim, "--out", str(out)],
                          capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    return done.stdout


def test_the_demo_script_runs_each_story_offline_and_writes_a_package(tmp=None):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        clean = run_script("clean", folder)
        assert "Potential claim amount: 160000.00" in clean and "(page 3, Section 4. Excess)" in clean
        assert "Human decision: Approved for Processing" in clean and "No risk indicators were found. Every planned check was performed." in clean
        missing = run_script("missing", folder)
        assert "Re-running only the affected steps" in missing and "Potential claim amount: 67500.00" in missing
        upload = run_script("upload", folder)
        assert "Potential claim amount: 118500.00" in upload
        pdfs = sorted(Path(folder).glob("*.pdf"))
        assert {p.name.split("_")[1] for p in pdfs} == {"clean", "missing", "upload"} and len(pdfs) == 3
        assert all(p.read_bytes()[:5] == b"%PDF-" for p in pdfs)
