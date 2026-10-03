"""
Claim workspace: everything about one claim. The five tabs cover documents and
evidence, policy analysis, the agent workflow, risk and verification, and the assessment.

The Agent workflow tab is where the multi-agent system is visible: one row per step
with its status and inputs, a live trace while it runs, and the full result of any step.
"""

import pandas as pd
import streamlit as st

from claimflow import demo, ui
from claimflow.errors import UploadError
from claimflow.models import ClaimStatus, DocumentKind, StepStatus, can_transition
from claimflow.pipeline import PIPELINE, affected_steps, get_step, step_label, steps_changed_by_document
from claimflow.textutil import plural
from claimflow.uploads import ALLOWED_TYPES, resolve_stored_path, store_document

container = ui.load_container()
settings = container.settings
ui.page_header("Claim workspace", "Documents, agent results and the audit trail for one claim.")
ui.show_flash()

claim = ui.pick_claim(container, key="workspace_picker")
if claim is None:
    st.stop()
ui.claim_header(claim)
ui.show_info_request(container, claim)

results = container.db.latest_agent_results(claim.id)
documents = container.db.list_documents(claim.id)
is_final = claim.status in (ClaimStatus.APPROVED, ClaimStatus.ESCALATED)
runnable = can_transition(claim.status, ClaimStatus.PROCESSING)


if st.session_state.get("autorun_claim") == claim.id:     # set by the New claim page: create, then run, in one click
    st.session_state.pop("autorun_claim")
    if runnable:
        ui.run_with_trace(container, claim, lambda on_event: container.orchestrator.run(claim.id, on_event=on_event))

tab_docs, tab_policy, tab_flow, tab_risk, tab_assess = st.tabs(
    ["Documents and evidence", "Policy analysis", "Agent workflow", "Risk and verification", "Assessment"])

# ---------------------------------------------------------------- documents
with tab_docs:
    st.write(f"**Claim description.** {ui.md_escape(claim.description)}")
    if documents:
        st.dataframe(pd.DataFrame([{
            "Type": d.kind.label, "File": d.original_name, "Size": f"{d.size_bytes / 1024:.0f} KB",
            "SHA-256": d.sha256[:12], "Added": d.created_at.replace("T", " ").replace("+00:00", " UTC"),
        } for d in documents]), hide_index=True)
        photos = [d for d in documents if d.kind == DocumentKind.DAMAGE_PHOTO]
        if photos:
            st.markdown("**Damage photos**")
            for column, photo in zip(st.columns(min(len(photos), 4)), photos[:4]):
                try:
                    column.image(str(resolve_stored_path(settings.upload_dir, photo.stored_path)),
                                 caption=photo.original_name)
                except Exception:
                    column.caption(f"{photo.original_name} could not be displayed.")
    else:
        st.info("No documents are attached yet.")

    held = None if is_final else demo.held_back_quotation(claim, documents)
    if held is not None:
        st.info("Demo helper: this claim's repair quotation was held back so you can show the request-more-information "
                "flow. Attach it, then re-run only the steps it affects.")
        if st.button("Attach the repair quotation (demo)", key="attach_demo_quotation"):
            with ui.handle_errors("attaching the quotation"):
                store_document(container.db, settings.upload_dir, claim.id, DocumentKind.QUOTATION, held[0], held[1],
                               settings.max_upload_bytes)
                st.session_state.setdefault("changed_steps", {}).setdefault(claim.id, set()).update(
                    steps_changed_by_document(DocumentKind.QUOTATION))
                ui.flash("success", "The quotation was attached. Open the Agent workflow tab to re-run only the steps it affects.")
                st.rerun()

    if is_final:
        st.caption("This claim has a final decision, so no more documents can be added.")
    else:
        with st.form("add_evidence_form", clear_on_submit=True):
            st.markdown("**Add evidence**")
            kind = st.selectbox("Document type", list(DocumentKind), format_func=lambda k: k.label)
            files = st.file_uploader("Files", type=[e.lstrip(".") for e in ALLOWED_TYPES],
                                     accept_multiple_files=True, key="add_evidence_files")
            attach = st.form_submit_button("Attach to claim")
        if attach:
            with ui.handle_errors("attaching the files"):
                if not files:
                    st.warning("Choose at least one file.")
                attached = 0
                for file in files or []:
                    try:
                        store_document(container.db, settings.upload_dir, claim.id, kind,
                                       file.name, file.getvalue(), settings.max_upload_bytes)
                        attached += 1
                    except UploadError as exc:
                        st.warning(f"{file.name} was not attached. {exc.user_message}")
                if attached:
                    changed = st.session_state.setdefault("changed_steps", {}).setdefault(claim.id, set())
                    changed |= steps_changed_by_document(kind)
                    ui.flash("success", f"Attached {plural(attached, 'file')}. Open the Agent workflow tab to re-run "
                                        "only the steps this evidence affects.")
                    st.rerun()

# ---------------------------------------------------------------- policy
with tab_policy:
    ui.show_step_result(get_step("policy"), results.get("policy"))

# ---------------------------------------------------------------- workflow
with tab_flow:
    changed = st.session_state.get("changed_steps", {}).get(claim.id, set())
    if changed and runnable:
        todo = affected_steps(changed)
        st.info("New evidence was added. These steps will run again: "
                + ", ".join(step_label(n) for n in todo) + ". The other steps keep their results.")
        if st.button(f"Re-run {plural(len(todo), 'affected step')}", type="primary", key="rerun_affected"):
            ui.run_with_trace(container, claim, lambda on_event: container.orchestrator.rerun(claim.id, changed, on_event=on_event))

    run_label = "Run all agents again" if results else "Run agents"
    if st.button(run_label, type="primary" if not changed else "secondary", disabled=not runnable, key="run_agents"):
        ui.run_with_trace(container, claim, lambda on_event: container.orchestrator.run(claim.id, on_event=on_event))
    if not runnable:
        st.caption("This claim has a final decision, so the agents cannot run again.")

    st.markdown("#### Workflow")
    st.dataframe(ui.workflow_overview(results, PIPELINE), hide_index=True)

    selected = st.session_state.get("selected_step", PIPELINE[0].name)
    if selected not in {s.name for s in PIPELINE}:
        selected = PIPELINE[0].name
    for column, spec in zip(st.columns(len(PIPELINE)), PIPELINE):
        result = results.get(spec.name)
        icon = ui.STEP_ICONS[result.status] if result else ui.STEP_ICONS[StepStatus.PENDING]
        if column.button(f"{icon} {spec.label}", key=f"step_{spec.name}",
                         type="primary" if spec.name == selected else "secondary"):
            st.session_state["selected_step"] = selected = spec.name

    spec = get_step(selected)
    st.markdown(f"#### {spec.label}")
    st.caption(spec.description)
    st.caption("Needs: " + (", ".join(step_label(n) for n in spec.requires) or "nothing")
               + ". Also reads: " + (", ".join(step_label(n) for n in spec.uses) or "nothing")
               + ". Passes its result to: " + ui.feeds(spec) + ".")
    ui.show_step_result(spec, results.get(selected))

    history = container.db.agent_history(claim.id)
    if history:
        with st.expander(f"Run history ({len(history)} step runs)"):
            st.dataframe(pd.DataFrame([{
                "When": r.created_at.replace("T", " ").replace("+00:00", " UTC"), "Step": step_label(r.step),
                "Status": r.status.label, "Time (ms)": r.duration_ms, "Model": r.model or "",
                "Replayed": "yes" if r.cached else "",
            } for r in reversed(history)]), hide_index=True)

# ---------------------------------------------------------------- risk and verification
with tab_risk:
    st.markdown("#### Evidence check")
    ui.show_step_result(get_step("evidence"), results.get("evidence"))
    st.markdown("#### Risk analysis")
    ui.show_step_result(get_step("risk"), results.get("risk"))

# ---------------------------------------------------------------- assessment
with tab_assess:
    st.markdown("#### Damage analysis")
    ui.show_step_result(get_step("damage"), results.get("damage"))
    st.markdown("#### Claim assessment")
    ui.show_step_result(get_step("assessment"), results.get("assessment"))
    st.markdown("#### Compliance review")
    ui.show_step_result(get_step("compliance"), results.get("compliance"))
