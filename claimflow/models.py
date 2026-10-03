"""
Plain data types shared by the database, the agents, the orchestrator and the UI.

These are standard-library dataclasses on purpose: no extra dependency, and
they behave the same everywhere. Agent outputs are stored as JSON, so
AgentResult knows how to turn itself into JSON and back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from .errors import InvalidInputError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- enums

class ClaimStatus(str, Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"
    INFO_REQUESTED = "info_requested"
    ESCALATED = "escalated"
    APPROVED = "approved_for_processing"
    FAILED = "processing_failed"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").capitalize()


# Which status may follow which. Escalated and approved are final: the AI
# pipeline cannot reopen a claim a human has already decided.
ALLOWED_TRANSITIONS: dict[ClaimStatus, set[ClaimStatus]] = {
    ClaimStatus.SUBMITTED: {ClaimStatus.PROCESSING},
    ClaimStatus.PROCESSING: {ClaimStatus.AWAITING_REVIEW, ClaimStatus.FAILED, ClaimStatus.SUBMITTED},
    ClaimStatus.AWAITING_REVIEW: {
        ClaimStatus.PROCESSING,
        ClaimStatus.APPROVED,
        ClaimStatus.INFO_REQUESTED,
        ClaimStatus.ESCALATED,
    },
    ClaimStatus.INFO_REQUESTED: {ClaimStatus.PROCESSING, ClaimStatus.SUBMITTED},
    ClaimStatus.FAILED: {ClaimStatus.PROCESSING},
    ClaimStatus.ESCALATED: set(),
    ClaimStatus.APPROVED: set(),
}


def can_transition(current: ClaimStatus, new: ClaimStatus) -> bool:
    return new in ALLOWED_TRANSITIONS[current]


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def label(self) -> str:
        return self.value.capitalize()


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REQUEST_INFO = "request_info"
    ESCALATE = "escalate"

    @property
    def label(self) -> str:
        return {
            "approve": "Approve for processing",
            "request_info": "Request more information",
            "escalate": "Escalate for investigation",
        }[self.value]

    @property
    def outcome(self) -> str:
        """What the decision was, for showing next to the AI recommendation."""
        return {"approve": "Approved for Processing", "request_info": "More Information Requested",
                "escalate": "Escalated for Investigation"}[self.value]


RECOMMENDATION_TO_DECISION = {
    "approve_for_processing": ReviewDecision.APPROVE,
    "request_more_information": ReviewDecision.REQUEST_INFO,
    "escalate_for_investigation": ReviewDecision.ESCALATE,
}


def is_override(decision: "ReviewDecision", ai_recommendation: str | None) -> bool:
    """True when a person decides differently from what the AI recommended."""
    expected = RECOMMENDATION_TO_DECISION.get(ai_recommendation or "")
    return expected is not None and expected != decision


DECISION_TO_STATUS = {
    ReviewDecision.APPROVE: ClaimStatus.APPROVED,
    ReviewDecision.REQUEST_INFO: ClaimStatus.INFO_REQUESTED,
    ReviewDecision.ESCALATE: ClaimStatus.ESCALATED,
}


class DocumentKind(str, Enum):
    POLICY = "policy"
    ACCIDENT_REPORT = "accident_report"
    DAMAGE_PHOTO = "damage_photo"
    QUOTATION = "quotation"
    SUPPORTING = "supporting"

    @property
    def label(self) -> str:
        return {
            "policy": "Insurance policy",
            "accident_report": "Accident report",
            "damage_photo": "Damage photo",
            "quotation": "Repair quotation",
            "supporting": "Supporting document",
        }[self.value]


class IncidentType(str, Enum):
    COLLISION = "collision"
    THEFT = "theft"
    FIRE = "fire"
    WEATHER = "weather"
    VANDALISM = "vandalism"
    OTHER = "other"

    @property
    def label(self) -> str:
        return {
            "collision": "Collision",
            "theft": "Theft",
            "fire": "Fire",
            "weather": "Weather or natural disaster",
            "vandalism": "Vandalism",
            "other": "Other",
        }[self.value]


# ---------------------------------------------------------------- claim input

_POLICY_NUMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _/-]{1,38}")


@dataclass(frozen=True)
class NewClaim:
    claimant_name: str
    policy_number: str
    incident_date: date
    incident_type: IncidentType
    description: str

    @classmethod
    def from_form(cls, claimant_name: str, policy_number: str, incident_date: Any,
                  incident_type: Any, description: str) -> "NewClaim":
        """Clean and check form values. Raises InvalidInputError listing every problem."""
        problems: list[str] = []

        name = (claimant_name or "").strip()
        if len(name) < 2:
            problems.append("Enter the claimant's name.")
        elif len(name) > 120:
            problems.append("The claimant's name is too long (120 characters at most).")

        policy = (policy_number or "").strip()
        if not policy:
            problems.append("Enter the policy number.")
        elif not _POLICY_NUMBER.fullmatch(policy):
            problems.append("The policy number can use letters, numbers, spaces and - _ / only.")

        if not isinstance(incident_date, date):
            problems.append("Choose the incident date.")
        elif incident_date > date.today():
            problems.append("The incident date cannot be in the future.")

        try:
            kind = incident_type if isinstance(incident_type, IncidentType) else IncidentType(incident_type)
        except ValueError:
            kind = IncidentType.OTHER
            problems.append("Choose an incident type.")

        text = (description or "").strip()
        if len(text) < 10:
            problems.append("Describe what happened (at least 10 characters).")
        elif len(text) > 5000:
            problems.append("The description is too long (5000 characters at most).")

        if problems:
            raise InvalidInputError(" ".join(problems))
        return cls(name, policy, incident_date, kind, text)


# ---------------------------------------------------------------- stored records

@dataclass(frozen=True)
class ClaimRecord:
    id: int
    claim_number: str
    status: ClaimStatus
    claimant_name: str
    policy_number: str
    incident_date: date
    incident_type: IncidentType
    description: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    claim_id: int
    kind: DocumentKind
    original_name: str
    stored_path: str  # relative to the upload folder
    mime: str
    size_bytes: int
    sha256: str
    extracted_text: str | None
    created_at: str


@dataclass(frozen=True)
class ReviewRecord:
    id: int
    claim_id: int
    decision: ReviewDecision
    reviewer: str
    notes: str
    created_at: str
    ai_recommendation: str | None = None   # what the AI recommended when the person decided


# ---------------------------------------------------------------- agent output

@dataclass(frozen=True)
class EvidenceRef:
    """Where a statement came from, for example a policy page or an uploaded file."""
    source: str
    detail: str
    page: int | None = None
    section: str | None = None


@dataclass
class AgentResult:
    step: str
    status: StepStatus
    summary: str = ""
    output: dict[str, Any] = field(default_factory=dict)  # must be JSON-serializable
    evidence: list[EvidenceRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0
    model: str | None = None
    cached: bool = False
    created_at: str = ""  # filled in when loaded from the database

    @property
    def succeeded(self) -> bool:
        return self.status in (StepStatus.DONE, StepStatus.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status.value,
            "summary": self.summary,
            "output": self.output,
            "evidence": [asdict(e) for e in self.evidence],
            "warnings": list(self.warnings),
            "error": self.error,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "cached": self.cached,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, text: str) -> "AgentResult":
        data = json.loads(text)
        return cls(
            step=data["step"],
            status=StepStatus(data["status"]),
            summary=data.get("summary", ""),
            output=data.get("output") or {},
            evidence=[EvidenceRef(**e) for e in data.get("evidence", [])],
            warnings=list(data.get("warnings", [])),
            error=data.get("error"),
            duration_ms=int(data.get("duration_ms", 0)),
            model=data.get("model"),
            cached=bool(data.get("cached", False)),
        )


@dataclass(frozen=True)
class SharedFile:
    """A file on this claim that is byte-for-byte the same as a file on another claim."""
    document_name: str
    kind: DocumentKind
    other_claims: tuple[str, ...]


@dataclass
class ClaimState:
    """
    The shared blackboard. An agent reads the claim, the documents and earlier
    agents' results from here, and returns one AgentResult of its own. Agents
    never talk to each other directly.

    `results` holds only steps that finished earlier in this run or in an earlier
    run. `typed()` reads a step's output as its schema, with safe defaults when the
    step produced nothing.
    """
    claim: ClaimRecord
    documents: list[DocumentRecord] = field(default_factory=list)
    results: dict[str, AgentResult] = field(default_factory=dict)
    shared_files: list[SharedFile] = field(default_factory=list)

    def result(self, step: str) -> AgentResult | None:
        return self.results.get(step)

    def succeeded(self, step: str) -> bool:
        found = self.results.get(step)
        return found is not None and found.succeeded

    def output(self, step: str) -> dict[str, Any]:
        found = self.results.get(step)
        return found.output if found else {}

    def typed(self, step: str, schema: Any):
        """The output of `step` as `schema` (for example IntakeOutput). Missing data gives defaults."""
        return schema.from_dict(self.output(step))

    def documents_of(self, kind: DocumentKind) -> list[DocumentRecord]:
        return [d for d in self.documents if d.kind == kind]


@dataclass(frozen=True)
class TraceEvent:
    step: str
    label: str
    status: StepStatus
    message: str
    at: str
