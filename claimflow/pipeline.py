"""
The shape of the workflow: which steps exist, which depend on which, and which
step a changed document should send back through.

This file is data and a few small functions. It imports nothing from the agents or
the orchestrator, so both can use it without a circular import.

requires  this step cannot run unless these steps finished (done or warning)
uses      this step reads these steps' output when it exists, but can run without it.
          It is also re-run when they change, so its conclusions stay current.

Only intake is a hard requirement for most steps on purpose. A claim with no policy
must still reach a person, who is told the policy is missing, instead of stalling
silently halfway through the workflow.

The list is ordered so every dependency comes before the step that needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import WorkflowError
from .models import DocumentKind


@dataclass(frozen=True)
class StepSpec:
    name: str
    label: str
    description: str
    requires: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()


PIPELINE: tuple[StepSpec, ...] = (
    StepSpec("intake", "Claim intake",
             "Turns the description, accident report and quotation into structured claim facts."),
    StepSpec("policy", "Policy analysis",
             "Finds coverage, exclusions, deductible and limit in the policy, with page citations."),
    StepSpec("damage", "Damage analysis",
             "Gives a preliminary, non-professional description of what the damage photos show."),
    StepSpec("evidence", "Evidence check",
             "Checks that the evidence this kind of claim needs is attached and readable.",
             requires=("intake",), uses=("policy", "damage")),
    StepSpec("risk", "Risk analysis",
             "Lists risk indicators and inconsistencies for a person to look at. It never judges anyone's conduct.",
             requires=("intake",), uses=("policy", "damage", "evidence")),
    StepSpec("assessment", "Claim assessment",
             "Works out the potential claim amount with fixed rules, not with the model.",
             requires=("intake",), uses=("policy",)),
    StepSpec("compliance", "Compliance review",
             "Combines every result into a recommended next action and a plain explanation.",
             requires=("intake", "evidence", "risk", "assessment"), uses=("policy", "damage")),
)

# A new or changed document of this kind sends these steps back through, plus what depends on them.
_DOCUMENT_CHANGES = {
    DocumentKind.POLICY: {"policy"},
    DocumentKind.ACCIDENT_REPORT: {"intake"},
    DocumentKind.QUOTATION: {"intake"},
    DocumentKind.SUPPORTING: {"intake"},
    DocumentKind.DAMAGE_PHOTO: {"damage"},
}


def validate_pipeline(pipeline: Iterable[StepSpec] = PIPELINE) -> None:
    """Every dependency must name a step that appears earlier in the list."""
    seen: set[str] = set()
    for spec in pipeline:
        if spec.name in seen:
            raise WorkflowError(f"Step '{spec.name}' appears twice in the pipeline.")
        for dep in spec.requires + spec.uses:
            if dep not in seen:
                raise WorkflowError(f"Step '{spec.name}' depends on '{dep}', which is not defined before it.")
        seen.add(spec.name)


def get_step(name: str, pipeline: Iterable[StepSpec] = PIPELINE) -> StepSpec:
    for spec in pipeline:
        if spec.name == name:
            return spec
    raise WorkflowError(f"Unknown workflow step '{name}'.")


def step_label(name: str, pipeline: Iterable[StepSpec] = PIPELINE) -> str:
    """A step's label, or the raw name for a step that no longer exists (for example in old history)."""
    return next((s.label for s in pipeline if s.name == name), name)


def dependents(name: str, pipeline: tuple[StepSpec, ...] = PIPELINE) -> list[str]:
    """Steps that read this step's output directly."""
    return [s.name for s in pipeline if name in s.requires or name in s.uses]


def affected_steps(changed: Iterable[str], pipeline: tuple[StepSpec, ...] = PIPELINE) -> list[str]:
    """
    The changed steps plus everything downstream of them, in pipeline order. This is
    what makes a rerun selective: new evidence changes one step, and only that step
    and its dependents run again.
    """
    changed = set(changed)
    unknown = changed - {s.name for s in pipeline}
    if unknown:
        raise WorkflowError(f"Unknown workflow step(s): {', '.join(sorted(unknown))}.")
    affected = set(changed)
    for spec in pipeline:
        if spec.name not in affected and any(dep in affected for dep in spec.requires + spec.uses):
            affected.add(spec.name)
    return [s.name for s in pipeline if s.name in affected]


def steps_changed_by_document(kind: DocumentKind) -> set[str]:
    return set(_DOCUMENT_CHANGES[kind])
