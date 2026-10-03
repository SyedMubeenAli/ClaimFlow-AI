"""
Typed output schemas, one per agent. This is the contract between agents.

An agent builds its schema object and stores `.to_dict()` as AgentResult.output,
which is plain JSON in the database. A later agent reads it back with
`ctx.state.typed("intake", IntakeOutput)`. Reading is tolerant: if the earlier
step was skipped or failed, every field has a safe default, so a downstream agent
never crashes on a missing key. It checks `ctx.state.succeeded(step)` when it
needs to know whether the data is real.

Plain dataclasses on purpose: no extra dependency, and they behave identically on
every machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

SEVERITIES = ("low", "medium", "high")
RECOMMENDATIONS = {
    "approve_for_processing": "Approve for processing",
    "request_more_information": "Request more information",
    "escalate_for_investigation": "Escalate for investigation",
}
RECOMMENDATION_HEADLINES = {
    "approve_for_processing": "Approve for Processing",
    "request_more_information": "Request More Information",
    "escalate_for_investigation": "Escalate for Investigation",
}
RISK_INCOMPLETE = "Risk assessment incomplete: required evidence could not be fully assessed."
DAMAGE_DEMO_LABEL = "LIMITED DEMO ANALYSIS"
DAMAGE_DEMO_DISCLAIMER = ("Limited demo analysis. No vision model was used, so this description comes from the photo file "
                          "names and basic image checks. It is not a reading of the photos. Use Live AI with a vision "
                          "model for a real description.")
DAMAGE_DISCLAIMER = "A preliminary description of what is visible. It is not a professional damage assessment."
RISK_NOTICE = ("Risk indicators are signals that a person should look closer. They are not findings "
               "about anyone's honesty or conduct.")
COMPLIANCE_NOTICE = "This is a recommendation for human review. A person makes the final decision."


class Schema:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: Any):
        data = data if isinstance(data, dict) else {}
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in known})


def _each(items: Any, schema: type[Schema]) -> list:
    return [schema.from_dict(i) for i in items if isinstance(i, dict)] if isinstance(items, list) else []


# ---------------------------------------------------------------- intake and policy

@dataclass
class IntakeOutput(Schema):
    claimant_name: str = ""
    policy_number: str = ""
    incident_type: str = ""
    incident_date_on_form: str = ""
    incident_date_in_documents: str | None = None
    location: str | None = None
    vehicle: str | None = None
    police_report_number: str | None = None
    injuries_mentioned: bool | None = None
    repair_estimate: str | None = None          # a Decimal as text, only if found in the source text
    estimate_source: str | None = None
    summary: str = ""
    documents_read: list[str] = field(default_factory=list)
    documents_present: dict[str, int] = field(default_factory=dict)
    instruction_like_text: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class PolicyOutput(Schema):
    queries: list[str] = field(default_factory=list)
    chunks_considered: list[str] = field(default_factory=list)
    coverage_applies: str = "unclear"           # yes, no or unclear
    deductible: str | None = None
    deductible_kind: str | None = None          # fixed, percent or none
    coverage_limit: str | None = None
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    policy_documents: list[str] = field(default_factory=list)
    # Each verified finding with its source: topic, label, value, quote, page, section, source, doc_ref, chunk_id.
    findings: list[dict[str, Any]] = field(default_factory=list)
    # What the policy search could not find, stated explicitly: [{"topic": ..., "message": ...}].
    not_found: list[dict[str, str]] = field(default_factory=list)
    index_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- damage

@dataclass
class DamagePhoto(Schema):
    name: str = ""
    visible_damage: list[str] = field(default_factory=list)
    affected_areas: list[str] = field(default_factory=list)
    severity: str = "unclear"                   # minor, moderate, severe or unclear
    image_quality: str = "unclear"              # good, poor or unclear


@dataclass
class DamageOutput(Schema):
    photos: list[DamagePhoto] = field(default_factory=list)
    overall_summary: str = ""
    consistent_with_description: str = "unclear"  # yes, no or unclear (a model judgement)
    photos_not_analyzed: list[str] = field(default_factory=list)
    disclaimer: str = DAMAGE_DISCLAIMER
    analysis_mode: str = "vision"               # "vision" (a model looked at the photos) or "demo" (file names only)

    @classmethod
    def from_dict(cls, data: Any) -> "DamageOutput":
        out = super().from_dict(data)
        out.photos = _each(out.photos, DamagePhoto)
        return out


# ---------------------------------------------------------------- evidence

@dataclass
class EvidenceItem(Schema):
    code: str = ""
    label: str = ""
    status: str = "missing"                     # present, missing or unusable
    detail: str = ""


@dataclass
class ConsistencyCheck(Schema):
    code: str = ""
    label: str = ""
    result: str = "not_checked"                 # match, not_found or not_checked
    detail: str = ""


@dataclass
class EvidenceOutput(Schema):
    items: list[EvidenceItem] = field(default_factory=list)
    checks: list[ConsistencyCheck] = field(default_factory=list)
    required: int = 0
    present: int = 0
    missing: list[str] = field(default_factory=list)   # labels of items that are missing or unusable

    @classmethod
    def from_dict(cls, data: Any) -> "EvidenceOutput":
        out = super().from_dict(data)
        out.items = _each(out.items, EvidenceItem)
        out.checks = _each(out.checks, ConsistencyCheck)
        return out


# ---------------------------------------------------------------- risk

@dataclass
class RiskFlag(Schema):
    code: str = ""
    severity: str = "low"                       # low, medium or high
    category: str = ""                          # inconsistency, coverage, document or process
    title: str = ""
    detail: str = ""
    source: str = "rule"                        # rule (fixed code) or model (quotes verified)


@dataclass
class RiskOutput(Schema):
    flags: list[RiskFlag] = field(default_factory=list)
    highest_severity: str = "none"
    counts: dict[str, int] = field(default_factory=dict)
    model_check: str = "skipped"                # ran, skipped or failed
    notice: str = RISK_NOTICE
    assessment_complete: bool = True            # False when a required check could not be performed
    incomplete_reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "RiskOutput":
        out = super().from_dict(data)
        out.flags = _each(out.flags, RiskFlag)
        return out


# ---------------------------------------------------------------- assessment and compliance

@dataclass
class AssessmentOutput(Schema):
    calculated: bool = False
    estimate: str | None = None
    deductible: str | None = None
    after_deductible: str | None = None
    coverage_limit: str | None = None
    potential_amount: str | None = None
    limit_applied: bool = False
    steps: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    formula: str = "potential amount = smaller of (repair estimate minus deductible, but not below zero) and the coverage limit"


@dataclass
class ComplianceOutput(Schema):
    recommendation: str = ""
    recommendation_label: str = ""
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    explanation_source: str = "template"        # model or template
    open_items: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    human_decision_required: bool = True
    language_check_passed: bool = True
    notice: str = COMPLIANCE_NOTICE
