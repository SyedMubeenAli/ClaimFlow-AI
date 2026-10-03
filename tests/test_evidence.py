"""Evidence completeness: what each kind of claim needs, and whether it is attached and readable."""

from claimflow.agents.evidence import LABELS, REQUIRED, EvidenceVerificationAgent
from claimflow.llm import FakeProvider
from claimflow.models import DocumentKind, IncidentType, StepStatus
from claimflow.schemas import DamageOutput, DamagePhoto, EvidenceOutput, IntakeOutput
from tests.helpers import (
    POLICY_PAGES, QUOTATION_PAGES, REPORT_PAGES, attach, attach_photo, ctx_for, make_claim, step_result, temp_container,
)

GOOD_INTAKE = IntakeOutput(repair_estimate="50000.00", estimate_source="quote.pdf", police_report_number="KHI-4411")


def check(c, claim, intake=GOOD_INTAKE, damage=None):
    results = {"intake": step_result("intake", intake)}
    if damage is not None:
        results["damage"] = step_result("damage", damage)
    llm = FakeProvider()
    result = EvidenceVerificationAgent().run(ctx_for(c, claim, results, llm=llm))
    assert llm.calls == []                                            # fixed rules: no model involved
    return result, EvidenceOutput.from_dict(result.output)


def status_of(out, code):
    return next(i.status for i in out.items if i.code == code)


def full_collision(c, **claim_args):
    claim = make_claim(c.db, **claim_args)
    attach(c, claim, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
    attach(c, claim, DocumentKind.ACCIDENT_REPORT, "report.pdf", REPORT_PAGES)
    attach(c, claim, DocumentKind.QUOTATION, "quote.pdf", QUOTATION_PAGES)
    attach_photo(c, claim)
    return claim


def test_every_incident_type_has_requirements_and_every_requirement_has_a_label():
    assert set(REQUIRED) == set(IncidentType)
    assert all(code in LABELS for codes in REQUIRED.values() for code in codes)
    assert all("policy" in codes for codes in REQUIRED.values())      # nothing is decided without the policy


def test_a_complete_collision_claim_has_every_item():
    with temp_container() as (c, _):
        result, out = check(c, full_collision(c))
        assert result.status == StepStatus.DONE
        assert (out.required, out.present, out.missing) == (4, 4, [])


def test_missing_items_are_named_and_the_step_warns():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach(c, claim, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
        result, out = check(c, claim, intake=IntakeOutput())
        assert result.status == StepStatus.WARNING
        assert (out.required, out.present) == (4, 1)
        assert out.missing == ["Accident report", "Damage photos", "Repair quotation with a confirmed total"]
        assert any("No photos are attached" in w for w in result.warnings)
        assert "1 of 4 required items is present" in result.summary


def test_a_scanned_or_blank_document_is_unusable_not_present():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach(c, claim, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
        attach(c, claim, DocumentKind.ACCIDENT_REPORT, "scan.pdf", [[]])
        _, out = check(c, claim)
        assert status_of(out, "incident_report") == "unusable" and "scan" in out.items[1].detail


def test_a_quotation_counts_only_with_a_confirmed_total():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach(c, claim, DocumentKind.QUOTATION, "quote.pdf", QUOTATION_PAGES)
        _, out = check(c, claim, intake=IntakeOutput(repair_estimate=None))
        assert status_of(out, "repair_quotation") == "unusable" and "confirmed" in out.items[3].detail
        _, out = check(c, claim, intake=IntakeOutput(repair_estimate="50000.00"))
        assert status_of(out, "repair_quotation") == "present"


def test_an_estimate_found_only_in_the_description_is_not_a_quotation():
    with temp_container() as (c, _):
        _, out = check(c, make_claim(c.db), intake=IntakeOutput(repair_estimate="30000.00", estimate_source="claim description"))
        item = next(i for i in out.items if i.code == "repair_quotation")
        assert item.status == "missing" and "only in the description" in item.detail


def test_photos_all_rated_poor_are_unusable():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach_photo(c, claim)
        poor = DamageOutput(photos=[DamagePhoto(name="dent.png", image_quality="poor")])
        _, out = check(c, claim, damage=poor)
        assert status_of(out, "damage_photos") == "unusable"
        good = DamageOutput(photos=[DamagePhoto(name="dent.png", image_quality="good")])
        assert status_of(check(c, claim, damage=good)[1], "damage_photos") == "present"


def test_requirements_differ_by_incident_type():
    with temp_container() as (c, _):
        theft = make_claim(c.db, incident_type="theft")
        attach(c, theft, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
        attach(c, theft, DocumentKind.ACCIDENT_REPORT, "report.pdf", REPORT_PAGES)
        result, out = check(c, theft)                                  # theft needs a police report number, not photos
        assert [i.code for i in out.items] == ["policy", "incident_report", "police_report_number"]
        assert result.status == StepStatus.DONE
        _, out = check(c, theft, intake=IntakeOutput())
        assert out.missing == ["Police report number"]

        weather = make_claim(c.db, incident_type="weather")
        assert [i.code for i in check(c, weather)[1].items] == ["policy", "damage_photos", "repair_quotation"]
        vandal = make_claim(c.db, incident_type="vandalism")
        assert [i.code for i in check(c, vandal)[1].items] == ["policy", "damage_photos", "repair_quotation", "police_report_number"]
        other = make_claim(c.db, incident_type="other")
        assert [i.code for i in check(c, other)[1].items] == ["policy", "incident_report"]


def test_policy_number_check():
    with temp_container() as (c, _):
        claim = make_claim(c.db, policy_number="POL-1001")
        attach(c, claim, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
        assert check(c, claim)[1].checks[0].result == "match"
        other = make_claim(c.db, policy_number="XYZ 777")
        attach(c, other, DocumentKind.POLICY, "policy.pdf", POLICY_PAGES)
        found = check(c, other)[1].checks[0]
        assert found.result == "not_found" and "XYZ 777" in found.detail
        assert check(c, make_claim(c.db))[1].checks[0].result == "not_checked"       # no policy text to compare


def test_claimant_name_check_ignores_order_and_case():
    with temp_container() as (c, _):
        claim = make_claim(c.db, claimant_name="Ayesha Khan")
        attach(c, claim, DocumentKind.ACCIDENT_REPORT, "report.pdf", [["Driver: KHAN, AYESHA.", "Date: 1 September 2026."]])
        assert check(c, claim)[1].checks[1].result == "match"
        other = make_claim(c.db, claimant_name="Ayesha Khan")
        attach(c, other, DocumentKind.ACCIDENT_REPORT, "report.pdf", [["Driver: Bilal Raza."]])
        assert check(c, other)[1].checks[1].result == "not_found"
        assert check(c, make_claim(c.db))[1].checks[1].result == "not_checked"


def test_the_same_inputs_always_give_the_same_answer():
    with temp_container() as (c, _):
        claim = full_collision(c)
        assert check(c, claim)[0].output == check(c, claim)[0].output
