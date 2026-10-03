"""
Shared Streamlit helpers: the cached app container, page styling, status pills,
error handling, the claim picker and the step result renderer.

Only widely supported Streamlit calls are used here, and width options are left
at their defaults on purpose: `use_container_width` is being removed upstream.
"""

from __future__ import annotations

import html
import re
from contextlib import contextmanager
from typing import Iterator

import pandas as pd
import streamlit as st

from .container import AppContainer, build_container
from .errors import ClaimFlowError
from .logging_setup import get_logger
from .extract import render_page_png, split_pages
from .models import AgentResult, ClaimRecord, ClaimStatus, DocumentRecord, StepStatus
from .pipeline import StepSpec, dependents, step_label
from .decision import summarize
from .rag import split_on_quote
from .textutil import quoted_note
from .schemas import (
    DAMAGE_DEMO_LABEL, AssessmentOutput, ComplianceOutput, DamageOutput, EvidenceOutput, PolicyOutput,
    RiskOutput,
)
from .uploads import resolve_stored_path

log = get_logger("ui")

# Page paths, relative to app.py, as registered with st.navigation.
PAGE_DASHBOARD = "views/dashboard.py"
PAGE_NEW_CLAIM = "views/new_claim.py"
PAGE_WORKSPACE = "views/workspace.py"
PAGE_POLICY = "views/policy.py"
PAGE_REVIEW = "views/review.py"
PAGE_SETTINGS = "views/settings.py"

CLAIM_TONES = {
    ClaimStatus.SUBMITTED: "mute", ClaimStatus.PROCESSING: "info", ClaimStatus.AWAITING_REVIEW: "warn",
    ClaimStatus.INFO_REQUESTED: "warn", ClaimStatus.ESCALATED: "bad", ClaimStatus.APPROVED: "ok",
    ClaimStatus.FAILED: "bad",
}
STEP_TONES = {
    StepStatus.DONE: "ok", StepStatus.WARNING: "warn", StepStatus.FAILED: "bad",
    StepStatus.SKIPPED: "mute", StepStatus.PENDING: "mute", StepStatus.RUNNING: "info",
}
STEP_ICONS = {
    StepStatus.DONE: "✓", StepStatus.WARNING: "⚠", StepStatus.FAILED: "✗",
    StepStatus.SKIPPED: "–", StepStatus.PENDING: "○", StepStatus.RUNNING: "…",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=Source+Serif+4:wght@600&display=swap');
.stApp { font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    font-weight: 600; letter-spacing: -0.01em; color: #17212B;
}
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #D8DCE0; border-left: 4px solid #1D5C73;
    border-radius: 4px; padding: 0.7rem 1rem;
}
.cf-pill {
    display: inline-block; padding: 2px 11px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600; border: 1px solid transparent;
}
.cf-ok   { color: #1F6B42; background: #E4F2EA; border-color: #BBDCC9; }
.cf-warn { color: #8A5A00; background: #FBF0D9; border-color: #EBD29B; }
.cf-bad  { color: #9B1C1C; background: #FBE5E3; border-color: #EFBDB8; }
.cf-info { color: #1D5C73; background: #E1EEF3; border-color: #B7D3DE; }
.cf-mute { color: #4B5563; background: #ECEEF0; border-color: #D4D8DC; }
"""


# ------------------------------------------------------------------ app container

@st.cache_resource(show_spinner=False)
def get_container() -> AppContainer:
    """Built once per server process and shared by every page and session."""
    return build_container()


def load_container() -> AppContainer:
    """Return the container, or show the configuration problem and stop the page."""
    try:
        return get_container()
    except ClaimFlowError as exc:
        st.error(f"ClaimFlow could not start. {exc.user_message}")
        st.stop()
    except Exception:
        log.exception("Unexpected error while starting")
        st.error("ClaimFlow could not start. The details are in logs/claimflow.log.")
        st.stop()


# ------------------------------------------------------------------ small helpers

def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def md_escape(text: str) -> str:
    """Show user-typed text literally inside markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", text or "")


def pill(text: str, tone: str) -> str:
    return f'<span class="cf-pill cf-{tone}">{html.escape(text)}</span>'


def show_pill(text: str, tone: str) -> None:
    st.markdown(pill(text, tone), unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def flash(kind: str, text: str) -> None:
    """Queue a message to show after the next rerun (a rerun would erase one shown now)."""
    st.session_state["_flash"] = (kind, text)


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if item:
        kind, text = item
        {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}.get(kind, st.info)(text)


@contextmanager
def handle_errors(action: str) -> Iterator[None]:
    """
    Show a readable message instead of a stack trace. Streamlit's own stop and
    rerun signals are not Exceptions, so they pass straight through.
    """
    try:
        yield
    except ClaimFlowError as exc:
        st.error(exc.user_message)
    except Exception:
        log.exception("Unexpected error while %s", action)
        st.error(f"Something went wrong while {action}. The details are in logs/claimflow.log.")


def go_to(page: str) -> None:
    st.switch_page(page)


# ------------------------------------------------------------------ sidebar

def mode_label(container: AppContainer) -> str:
    settings = container.settings
    if settings.demo_mode:
        return "Demo mode (scripted offline model)"
    return f"Live AI ({settings.provider}, {settings.model})" if settings.provider != "fake" else "Offline test provider"


def _toggle_demo() -> None:
    container = get_container()
    problem = container.set_demo_mode(bool(st.session_state.get("demo_toggle")))
    if problem:
        st.session_state["demo_toggle"] = container.settings.demo_mode   # put the switch back
        flash("warning", problem)


def render_sidebar(container: AppContainer) -> None:
    settings = container.settings
    with st.sidebar:
        st.markdown("**ClaimFlow AI**")
        st.caption("Agents investigate and recommend. People decide.")
        st.toggle("Demo mode (works offline)", value=settings.demo_mode, key="demo_toggle", on_change=_toggle_demo,
                  help="On: a scripted offline model runs the whole workflow with no internet. "
                       "Off: the live AI provider from .env is used.")
        if settings.demo_mode:
            show_pill("Demo mode: scripted offline model", "info")
        elif settings.provider == "fake":
            show_pill("Offline test provider", "mute")
        elif settings.has_api_key():
            show_pill(f"Live AI: {settings.provider}, {settings.model}", "ok")
        else:
            show_pill(f"{settings.provider}: API key missing", "bad")


def show_info_request(container: AppContainer, claim: ClaimRecord) -> None:
    """When a person asked for more information, say what was asked and what to do next."""
    if claim.status != ClaimStatus.INFO_REQUESTED:
        return
    reviews = container.db.list_reviews(claim.id)
    notes = next((r.notes for r in reviews if r.decision.value == "request_info" and r.notes), "")
    st.warning("More information was requested for this claim. " + (quoted_note(notes) + " " if notes else "")
               + "Attach the missing documents in the claim workspace, then re-run the steps they affect.")


# ------------------------------------------------------------------ claim picker

def current_claim_id() -> int | None:
    return st.session_state.get("claim_id")


def set_current_claim(claim_id: int) -> None:
    st.session_state["claim_id"] = claim_id


def pick_claim(container: AppContainer, key: str, label: str = "Claim") -> ClaimRecord | None:
    """
    A claim selector that follows the claim chosen on another page. Streamlit
    deletes a widget's state when its page is not shown, so the shared choice
    lives in session_state["claim_id"] and is copied into the widget on arrival.
    """
    claims = container.db.list_claims()
    if not claims:
        st.info("There are no claims yet. Create the first one on the New claim page.")
        return None
    by_id = {c.id: c for c in claims}
    ids = list(by_id)

    current = current_claim_id()
    synced = st.session_state.get(f"{key}_synced")
    if current in by_id and (key not in st.session_state or current != synced):
        st.session_state[key] = current

    chosen = st.selectbox(
        label, ids, key=key,
        format_func=lambda i: f"{by_id[i].claim_number}   {by_id[i].claimant_name}   ({by_id[i].status.label})",
    )
    set_current_claim(chosen)
    st.session_state[f"{key}_synced"] = chosen
    return by_id[chosen]


def claim_header(claim: ClaimRecord) -> None:
    st.subheader(claim.claim_number)
    show_pill(claim.status.label, CLAIM_TONES[claim.status])
    st.caption(f"{md_escape(claim.claimant_name)}, policy {md_escape(claim.policy_number)}, "
               f"{claim.incident_type.label.lower()} on {claim.incident_date:%d %b %Y}")


# ------------------------------------------------------------------ agent results

SEVERITY_TONES = {"high": "bad", "medium": "warn", "low": "mute"}
RECOMMENDATION_TONES = {"approve_for_processing": "ok", "request_more_information": "warn",
                        "escalate_for_investigation": "bad"}


def show_step_result(spec: StepSpec, result: AgentResult | None) -> None:
    """Render one agent's result: status, summary, warnings, typed details, evidence and raw output."""
    if result is None:
        st.info(f"{spec.label} has not run for this claim yet. Run the agents from the Agent workflow tab.")
        return
    show_pill(f"{STEP_ICONS[result.status]} {result.status.label}", STEP_TONES[result.status])
    if result.summary:
        st.write(result.summary)
    if result.error:
        st.error(result.error)
    for warning in result.warnings:
        if warning not in result.summary:      # do not say the same sentence twice in a row
            st.warning(warning)
    if result.succeeded:
        _show_typed(spec.name, result)
    if result.evidence:
        st.markdown("**Evidence**")
        st.dataframe(pd.DataFrame([{"Source": e.source, "Section": e.section or "", "Detail": e.detail,
                                    "Page": str(e.page or "")} for e in result.evidence]), hide_index=True)
    if result.output:
        with st.expander("Structured output"):
            st.json(result.output)
    meta = [f"{result.duration_ms} ms"]
    if result.model:
        meta.append(result.model)
    if result.cached:
        meta.append("replayed from the response cache")
    st.caption(", ".join(meta))


def _bullets(items: list[str]) -> None:
    for item in items:
        st.write(f"- {md_escape(item)}")


def _show_typed(step: str, result: AgentResult) -> None:
    """Step-specific views of the typed output. Anything unknown falls back to the raw output."""
    if step == "policy":
        out = PolicyOutput.from_dict(result.output)
        if out.findings:
            st.dataframe(pd.DataFrame([{"Finding": f["label"], "Value": f["value"], "Section": f["section"],
                                        "Page": f["page"], "Quote": f["quote"]} for f in out.findings]), hide_index=True)
        for item in out.not_found:
            st.warning(item["message"])
    elif step == "evidence":
        out = EvidenceOutput.from_dict(result.output)
        st.dataframe(pd.DataFrame([{"Required item": i.label, "Status": i.status, "Detail": i.detail}
                                   for i in out.items]), hide_index=True)
        if out.checks:
            st.dataframe(pd.DataFrame([{"Consistency check": c.label, "Result": c.result.replace("_", " "),
                                        "Detail": c.detail} for c in out.checks]), hide_index=True)
    elif step == "risk":
        out = RiskOutput.from_dict(result.output)
        if not out.assessment_complete:
            _bullets(out.incomplete_reasons)           # the incomplete message itself is already shown above
        if out.flags:
            st.dataframe(pd.DataFrame([{"Severity": f.severity, "Indicator": f.title, "Detail": f.detail,
                                        "Found by": "model, quotes verified" if f.source == "model" else "fixed rule"}
                                       for f in out.flags]), hide_index=True)
        st.caption(out.notice)
    elif step == "assessment":
        out = AssessmentOutput.from_dict(result.output)
        _bullets(out.steps if out.calculated else out.missing_inputs)
        st.caption("Calculated with fixed rules in code. The model is not involved. " + out.formula + ".")
    elif step == "damage":
        out = DamageOutput.from_dict(result.output)
        if out.analysis_mode == "demo":
            st.warning(f"{DAMAGE_DEMO_LABEL}: no vision model was used. The photos were not looked at by an AI model. "
                       "The description comes from the file names and basic image checks only.")
        if out.photos:
            st.dataframe(pd.DataFrame([{"Photo": p.name, "Visible damage": "; ".join(p.visible_damage),
                                        "Areas": "; ".join(p.affected_areas), "Severity": p.severity,
                                        "Image quality": p.image_quality} for p in out.photos]), hide_index=True)
        st.caption(out.disclaimer)
    elif step == "compliance":
        out = ComplianceOutput.from_dict(result.output)
        if out.recommendation:
            show_pill(out.recommendation_label, RECOMMENDATION_TONES.get(out.recommendation, "mute"))
            st.write(out.explanation)
            scripted = get_container().settings.demo_mode
            st.caption(("Explanation written by the scripted demo model and checked by code." if scripted
                        else "Explanation written by the model and checked by code.") if out.explanation_source == "model"
                       else "Explanation produced from the recorded reasons.")
            st.markdown("**Reasons**")
            _bullets(out.reasons)
            if out.limitations:
                st.markdown("**Limitations**")
                _bullets(out.limitations)
            st.caption(out.notice)


def workflow_overview(results: dict[str, AgentResult], pipeline) -> pd.DataFrame:
    """One row per step: what it needs, how it ended, how long it took."""
    rows = []
    for spec in pipeline:
        result = results.get(spec.name)
        rows.append({
            "Step": spec.label,
            "Status": f"{STEP_ICONS[result.status]} {result.status.label}" if result else f"{STEP_ICONS[StepStatus.PENDING]} Not run",
            "Needs": ", ".join(step_label(n) for n in spec.requires) or "nothing",
            "Also reads": ", ".join(step_label(n) for n in spec.uses) or "nothing",
            "Time (ms)": result.duration_ms if result else "",
            "Model": (result.model or "") if result else "",
        })
    return pd.DataFrame(rows)


def feeds(spec: StepSpec) -> str:
    names = dependents(spec.name)
    return ", ".join(step_label(n) for n in names) if names else "nobody (it is the last step)"


# ------------------------------------------------------------------ running agents with a live trace

def run_with_trace(container: AppContainer, claim: ClaimRecord, action) -> None:
    """
    Run the orchestrator with a live trace, then refresh the page. `action` receives the trace callback
    and returns the RunSummary. Errors are shown, never raised, so one failure cannot crash the page.
    """
    with handle_errors("running the agents"):
        with st.status("Running agents", expanded=True) as status:
            def on_event(event):
                if event.status == StepStatus.RUNNING:
                    status.write(f"Running {event.label}")
                else:
                    status.write(f"{STEP_ICONS[event.status]} {event.label}: {event.message}")

            summary = action(on_event)
            status.update(label="Run finished", state="complete")
        st.session_state.get("changed_steps", {}).pop(claim.id, None)
        flash("success", f"Run finished. The claim is now {summary.final_status.label.lower()}.")
        st.rerun()


# ------------------------------------------------------------------ policy sources

def document_for_ref(documents: list[DocumentRecord], doc_ref: str) -> DocumentRecord | None:
    """The stored document a chunk or finding came from. Its doc_ref is "d" plus the document id."""
    return next((d for d in documents if f"d{d.id}" == doc_ref), None)


def show_source_page(container: AppContainer, doc: DocumentRecord | None, page: int, quote: str | None = None) -> None:
    """The cited page as an image when PyMuPDF can render it, plus its text with the quoted words in bold."""
    if doc is None:
        st.caption("The source file is no longer attached to this claim.")
        return
    png = None
    try:
        png = render_page_png(resolve_stored_path(container.settings.upload_dir, doc.stored_path).read_bytes(), page)
    except Exception:
        log.warning("Could not load %s to show page %s", doc.original_name, page)
    pages = split_pages(doc.extracted_text) if doc.extracted_text else []
    text = pages[page - 1].text if 1 <= page <= len(pages) else ""

    def markdown_of(raw: str) -> str:
        parts = split_on_quote(raw, quote) if quote else None
        body = (md_escape(parts[0]) + "**" + md_escape(parts[1]) + "**" + md_escape(parts[2])) if parts else md_escape(raw)
        return body.replace("\n", "  \n")

    st.caption(f"{doc.original_name}, page {page}")
    if png is not None:
        image_column, text_column = st.columns(2)
        image_column.image(png)
        text_column.markdown(markdown_of(text) if text else "No text on this page.")
    else:
        st.markdown(markdown_of(text) if text else "No text on this page.")
        st.caption("The page image needs PyMuPDF. The extracted text is shown instead.")


def describe_embedder(embedder) -> str:
    describe = getattr(embedder, "describe", None)
    return describe() if callable(describe) else f"{embedder.name} ({embedder.model}, {embedder.dim} dimensions)"


def show_decision_summary(recommendation: str | None, latest) -> None:
    """The person's decision and the AI's recommendation, side by side and clearly separate."""
    outcome = summarize(recommendation, latest)
    human, ai = st.columns(2)
    with human:
        st.markdown("**HUMAN DECISION**")
        show_pill(outcome.human, outcome.human_tone)
        st.caption(outcome.human_detail)
        if outcome.note:
            st.caption(quoted_note(outcome.note))
    with ai:
        st.markdown("**AI RECOMMENDATION**")
        show_pill(outcome.ai, "info" if outcome.ai != "None" else "mute")
        st.caption(outcome.ai_detail)
    if outcome.relation:
        st.caption(outcome.relation)
