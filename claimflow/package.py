"""
The Final Claim Package: one PDF with everything a reviewer, an auditor or the claimant's
file needs. It is built only from stored results, so it shows exactly what the agents
found and what the person decided, and nothing is generated or changed here.

Contents: the decision (AI recommendation, human decision, whether the person agreed),
the claim, the cost calculation with its working, the policy findings with page and section,
the evidence check, the risk indicators, the damage description, the reasons and limits,
and the audit trail of every agent step.

Text is escaped for ReportLab's markup and limited to characters the built-in fonts can draw
(Latin-1). Other characters appear as "?". That keeps one odd character from breaking a PDF.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .decision import summarize
from .models import AgentResult, ClaimRecord, ClaimStatus, DocumentRecord, ReviewRecord, is_override
from .textutil import plural, quoted_note
from .pipeline import PIPELINE
from .schemas import (
    DAMAGE_DEMO_LABEL, RECOMMENDATIONS, RISK_INCOMPLETE, RISK_NOTICE, AssessmentOutput, ComplianceOutput, DamageOutput, EvidenceOutput, PolicyOutput,
    RiskOutput,
)

INK, ACCENT, MUTED, RULE = "#17212B", "#1D5C73", "#5B6670", "#C9D1D6"
TONES = {"done": "#1F6B42", "warning": "#8A5A00", "failed": "#9B1C1C", "skipped": "#5B6670", "pending": "#5B6670"}
TONE_COLORS = {"ok": "#1F6B42", "warn": "#8A5A00", "bad": "#9B1C1C", "mute": "#5B6670"}
SEVERITY_TONES = {"high": "#9B1C1C", "medium": "#8A5A00", "low": "#5B6670"}

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor(INK), spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.5, textColor=colors.HexColor(MUTED), spaceAfter=2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11, spaceAfter=0)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=2)
KICKER = ParagraphStyle("kicker", parent=BODY, fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor(ACCENT), spaceAfter=0)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5,
                    textColor=colors.HexColor(ACCENT))
BANNER = ParagraphStyle("banner", parent=BODY, fontName="Helvetica-Bold", fontSize=11, leading=14, spaceAfter=0)


def clean(text: object) -> str:
    """Safe for ReportLab: escaped markup, and only characters the built-in fonts can draw."""
    return escape(str(text if text is not None else "").encode("latin-1", "replace").decode("latin-1"))


def p(text: object, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(clean(text), style)


def rich(markup: str, style: ParagraphStyle = BODY) -> Paragraph:
    """Markup the caller built itself from clean() pieces."""
    return Paragraph(markup, style)


def stamp(iso: str) -> str:
    return iso.replace("T", " ").replace("+00:00", " UTC")


def table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    data = [[c if not isinstance(c, str) else Paragraph(clean(c), HEAD if header and i == 0 else CELL) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)))
    else:
        style.append(("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F4")))
    t.setStyle(TableStyle(style))
    return t


def pill(text: str, color: str) -> Paragraph:
    return rich(f'<font color="{color}"><b>{clean(text)}</b></font>', CELL)


class _NumberedCanvas(canvas.Canvas):
    """Draws 'Page x of y' on every page. The total is only known after the last page, hence two passes."""
    footer = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pages: list[dict] = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor(MUTED))
            self.drawString(18 * mm, 10 * mm, self.footer)
            self.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def decision_state(claim: ClaimRecord) -> tuple[str, str]:
    """(label, colour) for the banner at the top of the package."""
    return {
        ClaimStatus.APPROVED: ("FINAL: approved for processing by a person", "#1F6B42"),
        ClaimStatus.ESCALATED: ("FINAL: escalated for investigation by a person", "#9B1C1C"),
        ClaimStatus.INFO_REQUESTED: ("INTERIM: more information was requested by a person", "#8A5A00"),
    }.get(claim.status, ("DRAFT: no human decision has been recorded yet", "#8A5A00"))


def build_claim_package(claim: ClaimRecord, documents: list[DocumentRecord], results: dict[str, AgentResult],
                        reviews: list[ReviewRecord], mode_label: str, generated_at: str | None = None) -> bytes:
    generated = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    story: list = []

    # ------------------------------------------------------------ title and banner
    story += [rich("CLAIMFLOW AI", KICKER), p("Claim package", TITLE), p(f"{claim.claim_number}", ParagraphStyle(
        "num", parent=BODY, fontSize=12, textColor=colors.HexColor(MUTED)))]
    label, tone = decision_state(claim)
    banner = Table([[rich(f'<font color="{tone}">{clean(label)}</font>', BANNER)]], colWidths=[174 * mm])
    banner.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor(tone)), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F7F5")),
                                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [Spacer(1, 6), banner, Spacer(1, 8)]
    story.append(table([
        ["Claimant", claim.claimant_name], ["Policy number", claim.policy_number],
        ["Incident", f"{claim.incident_type.label}, {claim.incident_date:%d %B %Y}"],
        ["Claim status", claim.status.label], ["Package generated", stamp(generated)], ["AI mode", mode_label],
        ["Documents on file", ", ".join(f"{d.original_name} ({d.kind.label.lower()})" for d in documents) or "none"],
    ], [38, 136], header=False))

    # ------------------------------------------------------------ decision
    compliance = ComplianceOutput.from_dict(results["compliance"].output) if "compliance" in results and results["compliance"].succeeded else None
    story.append(p("1. Decision", H1))
    outcome = summarize(compliance.recommendation if compliance else None, reviews[0] if reviews else None)
    human_lines = f'<font color="{TONE_COLORS[outcome.human_tone]}"><b>{clean(outcome.human)}</b></font><br/>{clean(outcome.human_detail)}'
    if outcome.note:
        human_lines += "<br/>" + clean(quoted_note(outcome.note))
    ai_lines = f'<font color="{ACCENT}"><b>{clean(outcome.ai)}</b></font><br/>{clean(outcome.ai_detail)}'
    pair = Table([[rich("<b>HUMAN DECISION</b>", CELL), rich(human_lines, CELL)],
                  [rich("<b>AI RECOMMENDATION</b>", CELL), rich(ai_lines, CELL)]], colWidths=[46 * mm, 128 * mm])
    pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor(RULE)),
                              ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F4")),
                              ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [pair]
    if outcome.relation:
        story.append(p(outcome.relation, SMALL))
    if compliance and compliance.recommendation:
        story += [Spacer(1, 3), p("AI explanation: " + compliance.explanation)]
    if reviews:
        rows = [["When", "Reviewer", "Decision", "AI recommended at the time", "Result", "Notes"]]
        for r in reviews:
            recommended = RECOMMENDATIONS.get(r.ai_recommendation or "", "not recorded")
            result = "Overrode the AI" if is_override(r.decision, r.ai_recommendation) else \
                     "Agreed with the AI" if r.ai_recommendation else "not recorded"
            rows.append([stamp(r.created_at), r.reviewer, r.decision.outcome, recommended, result, r.notes or "-"])
        story += [Spacer(1, 3), p("Decision history", ParagraphStyle("hh", parent=SMALL, fontName="Helvetica-Bold")),
                  table(rows, [26, 24, 30, 30, 24, 40])]
    story.append(p("The AI recommends. A person decides. Nothing in this package was approved or rejected automatically.", SMALL))

    # ------------------------------------------------------------ claim
    story += [p("2. The claim", H1), p(claim.description)]

    # ------------------------------------------------------------ assessment
    story.append(p("3. Claim assessment", H1))
    assessment = AssessmentOutput.from_dict(results["assessment"].output) if "assessment" in results and results["assessment"].succeeded else None
    if assessment and assessment.calculated:
        story.append(table([["Step", "Amount"],
                            ["Repair estimate (confirmed in a document)", assessment.estimate],
                            ["Deductible (confirmed in the policy)", assessment.deductible],
                            ["Amount after the deductible", assessment.after_deductible],
                            ["Coverage limit (confirmed in the policy)", assessment.coverage_limit],
                            ["Potential claim amount" + (" (limit applied)" if assessment.limit_applied else ""), assessment.potential_amount]],
                           [110, 64]))
        for condition in assessment.conditions:
            story.append(rich(f'<font color="#8A5A00"><b>Condition:</b></font> {clean(condition)}', SMALL))
    elif assessment:
        story += [p("The potential claim amount could not be calculated. Missing: " + "; ".join(assessment.missing_inputs) + ".")]
    else:
        story.append(p("The assessment did not run."))
    story.append(p("Calculated with fixed rules in code, using exact decimal arithmetic. The language model does not take part. "
                   "Rule: " + AssessmentOutput().formula + ".", SMALL))

    # ------------------------------------------------------------ policy findings
    story.append(p("4. Policy findings and sources", H1))
    policy = PolicyOutput.from_dict(results["policy"].output) if "policy" in results and results["policy"].succeeded else None
    if policy and policy.findings:
        rows = [["Finding", "Value", "Source (page, section)", "Policy wording"]]
        for f in policy.findings:
            rows.append([f["label"], f["value"], f"Page {f['page']}" + (f", {f['section']}" if f["section"] else "") + f" ({f['source']})",
                         '"' + f["quote"] + '"'])
        story.append(table(rows, [30, 26, 44, 74]))
    elif policy is None:
        story.append(p("The policy analysis did not run: " + (results["policy"].summary if "policy" in results else "no result") ))
    for item in (policy.not_found if policy else []):
        story.append(rich(f'<font color="#8A5A00"><b>Not found:</b></font> {clean(item["message"])}', BODY))
    if policy and policy.rejected:
        story.append(p(f"{plural(len(policy.rejected), 'statement')} from the model could not be matched to the policy text and {'was' if len(policy.rejected) == 1 else 'were'} not used.", SMALL))
    story.append(p("Every finding above was checked by code against the policy text. A finding is shown only if its quote appears word for word on the cited page.", SMALL))

    # ------------------------------------------------------------ evidence
    story.append(p("5. Evidence check", H1))
    evidence = EvidenceOutput.from_dict(results["evidence"].output) if "evidence" in results and results["evidence"].succeeded else None
    if evidence:
        rows = [["Required item", "Status", "Detail"]] + [[i.label, i.status.capitalize(), i.detail or "-"] for i in evidence.items]
        story.append(table(rows, [60, 24, 90]))
        checks = [[c.label, c.result.replace("_", " ").capitalize(), c.detail or "-"] for c in evidence.checks]
        if checks:
            story += [Spacer(1, 4), table([["Consistency check", "Result", "Detail"]] + checks, [60, 24, 90])]
    else:
        story.append(p("The evidence check did not run."))

    # ------------------------------------------------------------ risk
    story.append(p("6. Risk indicators", H1))
    risk = RiskOutput.from_dict(results["risk"].output) if "risk" in results and results["risk"].succeeded else None
    if risk is None:
        story.append(rich(f'<font color="#8A5A00"><b>{clean(RISK_INCOMPLETE)}</b></font> The risk analysis did not run.'))
    else:
        if not risk.assessment_complete:
            story.append(rich(f'<font color="#8A5A00"><b>{clean(RISK_INCOMPLETE)}</b></font>'))
            story += [p("- " + reason, SMALL) for reason in risk.incomplete_reasons]
        if risk.flags:
            rows = [["Severity", "Indicator", "Detail", "Found by"]]
            for f in risk.flags:
                rows.append([pill(f.severity.capitalize(), SEVERITY_TONES.get(f.severity, MUTED)), f.title, f.detail,
                             "Model, quotes verified" if f.source == "model" else "Fixed rule"])
            story += [Spacer(1, 3), table(rows, [18, 48, 84, 24])]
        elif risk.assessment_complete:
            story.append(p("No risk indicators were found. Every planned check was performed."))
    story.append(p(RISK_NOTICE, SMALL))

    # ------------------------------------------------------------ damage
    damage = DamageOutput.from_dict(results["damage"].output) if "damage" in results and results["damage"].succeeded else None
    limited = damage is not None and damage.analysis_mode == "demo"
    story.append(p("7. Damage analysis" + (" (limited demo analysis)" if limited else ""), H1))
    if limited:
        story.append(rich(f'<font color="#8A5A00"><b>{clean(DAMAGE_DEMO_LABEL)}:</b></font> no vision model was used. '
                          "The photos were not looked at by an AI model. The description below comes from the file names and basic "
                          "image checks only, and must not be treated as an assessment of the damage."))
    if damage and damage.photos:
        rows = [["Photo", "Visible damage", "Areas", "Severity", "Image quality"]]
        rows += [[ph.name, "; ".join(ph.visible_damage) or "-", "; ".join(ph.affected_areas) or "-", ph.severity, ph.image_quality]
                 for ph in damage.photos]
        story += [Spacer(1, 3), table(rows, [38, 48, 42, 22, 24]), p(damage.overall_summary, SMALL), p(damage.disclaimer, SMALL)]
    else:
        reason = results["damage"].summary if "damage" in results else "The damage analysis did not run."
        story.append(p(f"No damage description is available. {reason}"))

    # ------------------------------------------------------------ reasons and limits
    story.append(p("8. Reasons and limitations", H1))
    if compliance and compliance.reasons:
        story += [p("Reasons for the recommendation:")] + [p("- " + r) for r in compliance.reasons]
        if compliance.limitations:
            story += [p("Limitations:")] + [p("- " + r) for r in compliance.limitations]
    else:
        story.append(p("No reasons were recorded."))

    # ------------------------------------------------------------ audit trail
    story.append(p("9. Audit trail: what each agent did", H1))
    rows = [["Step", "Status", "Result", "Time", "Model"]]
    for spec in PIPELINE:
        r = results.get(spec.name)
        if r is None:
            rows.append([spec.label, pill("Not run", TONES["pending"]), "-", "-", "-"])
        else:
            rows.append([spec.label, pill(r.status.label, TONES.get(r.status.value, MUTED)), r.error or r.summary or "-",
                         f"{r.duration_ms} ms", (r.model or "-") + (" (replayed)" if r.cached else "")])
    story.append(table(rows, [32, 18, 82, 16, 26]))
    story.append(KeepTogether([Spacer(1, 8), p("Generated by ClaimFlow AI. Agents extract, verify and recommend. A person makes the final decision. "
                                               "Figures come from fixed code. Policy statements come from the policy text.", SMALL)]))

    buffer = io.BytesIO()
    footer = f"ClaimFlow AI | {claim.claim_number} | generated {stamp(generated)}"
    canvas_class = type("Numbered", (_NumberedCanvas,), {"footer": clean(footer)})
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title=f"Claim package {claim.claim_number}", author="ClaimFlow AI").build(story, canvasmaker=canvas_class)
    return buffer.getvalue()
