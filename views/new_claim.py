"""
New claim: the claim form and document uploads. Agents run from the claim workspace.

The page has two states. With no claim just created it shows the form. Right after
a claim is created it shows a confirmation instead, so the form cannot be
submitted twice by accident.
"""

from datetime import date

import pandas as pd
import streamlit as st

from claimflow import demo, ui
from claimflow.errors import UploadError
from claimflow.models import DocumentKind, IncidentType, NewClaim
from claimflow.textutil import plural
from claimflow.uploads import store_document

container = ui.load_container()
settings = container.settings
ui.page_header("New claim", "Enter the claim and attach what the claimant provided.")

created = st.session_state.get("last_created")
if created:
    st.success(f"Claim {created['claim']} was created with {plural(len(created['stored']), 'document')}.")
    for name in created["stored"]:
        st.write(f"Attached: {ui.md_escape(name)}")
    for problem in created["rejected"]:
        st.warning(f"Not attached. {problem}")
    if created["rejected"]:
        st.caption("You can attach corrected files from the claim workspace.")
    run_column, open_column, another_column = st.columns(3)
    if run_column.button("Open the workspace and run the agents", type="primary"):
        st.session_state["autorun_claim"] = created["id"]
        st.session_state.pop("last_created", None)
        ui.go_to(ui.PAGE_WORKSPACE)
    if open_column.button("Open the claim workspace"):
        st.session_state.pop("last_created", None)
        ui.go_to(ui.PAGE_WORKSPACE)
    if another_column.button("Create another claim"):
        st.session_state.pop("last_created", None)
        st.rerun()
    st.stop()

with st.expander("Sample documents for a live upload demo"):
    st.write("Four readable files that agree with each other: the policy, an accident report, a repair quotation and a "
             "damage photo. Download them, type these values into the form, and upload the files below.")
    st.dataframe(pd.DataFrame([{"Field": k, "Value": v} for k, v in demo.sample_form().items()]), hide_index=True)
    for sample in demo.sample_documents():
        st.download_button(f"Download: {sample.label} ({sample.filename})", data=sample.data, file_name=sample.filename,
                           mime=sample.mime, key=f"sample_{sample.filename}")
    st.caption("Expected result: Approve for processing, with a potential claim amount of 118,500.00 (the 143,500 repair "
               "estimate minus the 25,000 collision excess). Upload each sample file to one claim only: the same file on "
               "two claims is reported as a risk indicator.")

# (kind, label, accepted extensions, allow several files)
UPLOAD_SLOTS = [
    (DocumentKind.POLICY, "Insurance policy (PDF)", ["pdf"], False),
    (DocumentKind.ACCIDENT_REPORT, "Accident report (PDF)", ["pdf"], False),
    (DocumentKind.DAMAGE_PHOTO, "Damage photos (PNG or JPG)", ["png", "jpg", "jpeg"], True),
    (DocumentKind.QUOTATION, "Repair quotation (PDF)", ["pdf"], False),
    (DocumentKind.SUPPORTING, "Other supporting documents (PDF, PNG or JPG)", ["pdf", "png", "jpg", "jpeg"], True),
]

with st.form("new_claim_form"):
    left, right = st.columns(2)
    claimant_name = left.text_input("Claimant name")
    policy_number = right.text_input("Policy number")
    incident_date = left.date_input("Incident date", value=date.today(), max_value=date.today())
    incident_type = right.selectbox("Incident type", list(IncidentType), format_func=lambda t: t.label)
    description = st.text_area("What happened", height=140, max_chars=5000,
                               help="Write it as the claimant would. Dates, places and amounts help the agents.")
    st.markdown("**Documents**")
    st.caption(f"Each file can be up to {settings.max_upload_mb} MB. Files are checked before they are stored.")
    uploads = {}
    for kind, label, extensions, several in UPLOAD_SLOTS:
        uploads[kind] = st.file_uploader(label, type=extensions, accept_multiple_files=several,
                                         key=f"new_claim_{kind.value}")
    submitted = st.form_submit_button("Create claim", type="primary")

if submitted:
    with ui.handle_errors("creating the claim"):
        new = NewClaim.from_form(claimant_name, policy_number, incident_date, incident_type, description)
        claim = container.db.create_claim(new)
        stored, rejected = [], []
        for kind, uploaded in uploads.items():
            files = uploaded if isinstance(uploaded, list) else ([uploaded] if uploaded else [])
            for file in files:
                try:
                    store_document(container.db, settings.upload_dir, claim.id, kind,
                                   file.name, file.getvalue(), settings.max_upload_bytes)
                    stored.append(f"{file.name} ({kind.label.lower()})")
                except UploadError as exc:
                    rejected.append(f"{file.name}: {exc.user_message}")
        ui.set_current_claim(claim.id)
        st.session_state["last_created"] = {"claim": claim.claim_number, "id": claim.id, "stored": stored, "rejected": rejected}
        st.rerun()
