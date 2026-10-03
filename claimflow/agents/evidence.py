"""
Evidence verification agent: is the evidence this kind of claim needs attached,
readable and consistent? It uses fixed rules and no language model, so the same
inputs always give the same answer.

It reports facts. Turning a mismatch into a risk indicator is the risk agent's job.
"""

from __future__ import annotations

import re

from ..extract import has_text
from ..logging_setup import get_logger
from ..models import AgentResult, DocumentKind, DocumentRecord, IncidentType
from ..textutil import is_are, plural
from ..schemas import ConsistencyCheck, DamageOutput, EvidenceItem, EvidenceOutput, IntakeOutput
from .base import Agent, AgentContext

log = get_logger("agent.evidence")

LABELS = {
    "policy": "Insurance policy",
    "incident_report": "Accident report",
    "damage_photos": "Damage photos",
    "repair_quotation": "Repair quotation with a confirmed total",
    "police_report_number": "Police report number",
}

# What each kind of claim needs before a person can decide it.
REQUIRED: dict[IncidentType, tuple[str, ...]] = {
    IncidentType.COLLISION: ("policy", "incident_report", "damage_photos", "repair_quotation"),
    IncidentType.THEFT: ("policy", "incident_report", "police_report_number"),
    IncidentType.FIRE: ("policy", "incident_report", "damage_photos", "repair_quotation"),
    IncidentType.WEATHER: ("policy", "damage_photos", "repair_quotation"),
    IncidentType.VANDALISM: ("policy", "damage_photos", "repair_quotation", "police_report_number"),
    IncidentType.OTHER: ("policy", "incident_report"),
}


def _usable(doc: DocumentRecord) -> bool:
    return doc.extracted_text is not None and has_text(doc.extracted_text)


def _document_item(code: str, docs: list[DocumentRecord]) -> EvidenceItem:
    label = LABELS[code]
    if not docs:
        return EvidenceItem(code, label, "missing", "Nothing is attached.")
    if any(_usable(d) for d in docs):
        return EvidenceItem(code, label, "present", f"{plural(sum(_usable(d) for d in docs), 'readable file')}.")
    return EvidenceItem(code, label, "unusable",
                        "A file is attached, but it could not be parsed as readable text (it may be a scanned image or damaged).")


def build_items(codes: tuple[str, ...], docs_by_kind: dict[DocumentKind, list[DocumentRecord]],
                intake: IntakeOutput, damage: DamageOutput | None) -> list[EvidenceItem]:
    items = []
    for code in codes:
        if code == "policy":
            items.append(_document_item(code, docs_by_kind[DocumentKind.POLICY]))
        elif code == "incident_report":
            items.append(_document_item(code, docs_by_kind[DocumentKind.ACCIDENT_REPORT]))
        elif code == "damage_photos":
            photos = docs_by_kind[DocumentKind.DAMAGE_PHOTO]
            if not photos:
                items.append(EvidenceItem(code, LABELS[code], "missing", "No photos are attached."))
            elif damage and damage.photos and all(p.image_quality == "poor" for p in damage.photos):
                items.append(EvidenceItem(code, LABELS[code], "unusable", "Every photo was rated poor quality."))
            else:
                items.append(EvidenceItem(code, LABELS[code], "present", f"{plural(len(photos), 'photo')}."))
        elif code == "repair_quotation":
            quotes = docs_by_kind[DocumentKind.QUOTATION]
            if not quotes:
                detail = ("The estimate appears only in the description. A quotation is needed."
                          if intake.estimate_source == "claim description" else "No quotation is attached.")
                items.append(EvidenceItem(code, LABELS[code], "missing", detail))
            elif not any(_usable(q) for q in quotes):
                items.append(EvidenceItem(code, LABELS[code], "unusable", "The quotation could not be parsed as readable text."))
            elif intake.repair_estimate is None:
                items.append(EvidenceItem(code, LABELS[code], "unusable",
                                          "A total could not be confirmed in the quotation text."))
            else:
                items.append(EvidenceItem(code, LABELS[code], "present", f"Total {intake.repair_estimate} confirmed."))
        elif code == "police_report_number":
            if intake.police_report_number:
                items.append(EvidenceItem(code, LABELS[code], "present", intake.police_report_number))
            else:
                items.append(EvidenceItem(code, LABELS[code], "missing", "No police report number was found."))
    return items


def _alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def build_checks(claim_policy_number: str, claimant_name: str,
                 docs_by_kind: dict[DocumentKind, list[DocumentRecord]]) -> list[ConsistencyCheck]:
    checks = []

    policy_text = _alnum(" ".join(d.extracted_text for d in docs_by_kind[DocumentKind.POLICY] if _usable(d)))
    wanted = _alnum(claim_policy_number)
    if not policy_text or len(wanted) < 3:
        checks.append(ConsistencyCheck("policy_number", "Policy number on the form is in the policy", "not_checked",
                                       "No readable policy text to compare with."))
    elif wanted in policy_text:
        checks.append(ConsistencyCheck("policy_number", "Policy number on the form is in the policy", "match"))
    else:
        checks.append(ConsistencyCheck("policy_number", "Policy number on the form is in the policy", "not_found",
                                       f"{claim_policy_number} does not appear in the policy text."))

    report_text = " ".join(d.extracted_text for d in docs_by_kind[DocumentKind.ACCIDENT_REPORT] if _usable(d)).lower()
    tokens = [t for t in re.findall(r"[a-z]+", claimant_name.lower()) if len(t) >= 3]
    if not report_text or not tokens:
        checks.append(ConsistencyCheck("claimant_name", "Claimant name on the form is in the accident report",
                                       "not_checked", "No readable accident report to compare with."))
    elif all(t in report_text for t in tokens):
        checks.append(ConsistencyCheck("claimant_name", "Claimant name on the form is in the accident report", "match"))
    else:
        checks.append(ConsistencyCheck("claimant_name", "Claimant name on the form is in the accident report",
                                       "not_found", "The name on the form was not found in the report text. "
                                       "Spelling and name order can differ."))
    return checks


class EvidenceVerificationAgent(Agent):
    name = "evidence"
    label = "Evidence check"
    description = "Checks that the evidence this kind of claim needs is attached and readable."

    def run(self, ctx: AgentContext) -> AgentResult:
        claim = ctx.state.claim
        intake = ctx.state.typed("intake", IntakeOutput)
        damage = ctx.state.typed("damage", DamageOutput) if ctx.state.succeeded("damage") else None
        docs_by_kind = {kind: ctx.state.documents_of(kind) for kind in DocumentKind}

        items = build_items(REQUIRED[claim.incident_type], docs_by_kind, intake, damage)
        checks = build_checks(claim.policy_number, claim.claimant_name, docs_by_kind)
        missing = [i.label for i in items if i.status != "present"]
        output = EvidenceOutput(items=items, checks=checks, required=len(items),
                                present=sum(i.status == "present" for i in items), missing=missing)
        summary = f"{output.present} of {output.required} required items {is_are(output.present)} present."
        log.info("claim %s: %s", claim.id, summary)
        if missing:
            return self.warning(f"{summary} Missing or unusable: {', '.join(missing)}.",
                                [f"{i.label}: {i.detail}" for i in items if i.status != "present"], output.to_dict())
        return self.done(summary, output.to_dict())
