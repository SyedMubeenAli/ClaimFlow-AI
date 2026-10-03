"""
Review and package: where a person makes the decision, and where the final claim package is produced.

The three buttons record a human decision and are enabled only while a claim is waiting for review.
The decision is stored together with what the AI recommended. Deciding differently from the AI
needs a written reason, and the package shows whether the person agreed or overrode.
"""

import pandas as pd
import streamlit as st

from claimflow import ui
from claimflow.models import ClaimStatus, ReviewDecision, is_override
from claimflow.package import build_claim_package
from claimflow.pipeline import get_step
from claimflow.schemas import RECOMMENDATIONS, ComplianceOutput

container = ui.load_container()
ui.page_header("Review and package", "The agents recommend. The decision on this page is yours.")
ui.show_flash()

claim = ui.pick_claim(container, key="review_picker")
if claim is None:
    st.stop()
ui.claim_header(claim)
ui.show_info_request(container, claim)

results = container.db.latest_agent_results(claim.id)
documents = container.db.list_documents(claim.id)
reviews = container.db.list_reviews(claim.id)
can_decide = claim.status == ClaimStatus.AWAITING_REVIEW

compliance = results.get("compliance")
has_recommendation = compliance is not None and compliance.succeeded
ai_recommendation = ComplianceOutput.from_dict(compliance.output).recommendation if has_recommendation else None

st.subheader("Decision status")
ui.show_decision_summary(ai_recommendation, reviews[0] if reviews else None)

st.subheader("Summary")
st.write(ui.md_escape(claim.description))

st.subheader("AI recommendation")
if has_recommendation:
    ui.show_step_result(get_step("compliance"), compliance)
    st.caption("This is advice. It has no effect until a person records a decision below.")
else:
    st.info("There is no recommendation yet. Run the agents from the claim workspace first.")

for name, title in [("policy", "Policy findings"), ("evidence", "Evidence check"), ("risk", "Risk indicators"),
                    ("damage", "Damage summary"), ("assessment", "Claim assessment")]:
    with st.expander(title):
        ui.show_step_result(get_step(name), results.get(name))

st.subheader("Your decision")
if not can_decide:
    if claim.status in (ClaimStatus.APPROVED, ClaimStatus.ESCALATED):
        st.info(f"This claim already has a final decision: {claim.status.label.lower()}.")
    elif claim.status != ClaimStatus.INFO_REQUESTED:
        st.info("This claim is not waiting for review. Decisions open once the agents have finished.")
elif ai_recommendation:
    st.caption(f"The AI recommends: {RECOMMENDATIONS[ai_recommendation]}. A decision that differs from it needs a note "
               "explaining why.")

reviewer = st.text_input("Your name", key="reviewer_name", disabled=not can_decide)
notes = st.text_area("Notes for the file", key=f"reviewer_notes_{claim.id}", disabled=not can_decide, height=100,
                     help="Required if you decide differently from the AI recommendation. If you request more "
                          "information, say what is needed.")

chosen = None
for column, decision in zip(st.columns(3), ReviewDecision):
    if column.button(decision.label, key=f"decide_{decision.value}", disabled=not can_decide,
                     type="primary" if decision == ReviewDecision.APPROVE else "secondary"):
        chosen = decision
if chosen is not None:
    with ui.handle_errors("recording the decision"):
        container.db.add_review(claim.id, chosen, reviewer, notes, ai_recommendation)
        ui.flash("success", f"Decision recorded: {chosen.outcome}."
                 + (" You overrode the AI recommendation, and the package will show that." if is_override(chosen, ai_recommendation) else ""))
        st.rerun()

if reviews:
    st.subheader("Decision history")
    st.dataframe(pd.DataFrame([{
        "When": r.created_at.replace("T", " ").replace("+00:00", " UTC"), "Decision": r.decision.outcome,
        "Reviewer": r.reviewer, "AI recommended": RECOMMENDATIONS.get(r.ai_recommendation or "", "not recorded"),
        "Result": ("Overrode the AI" if is_override(r.decision, r.ai_recommendation) else
                   "Agreed with the AI" if r.ai_recommendation else ""), "Notes": r.notes,
    } for r in reviews]), hide_index=True)

st.subheader("Final claim package")
if not has_recommendation:
    st.info("The package can be prepared once the agents have finished this claim.")
else:
    st.write("One PDF with the decision, the cost calculation, the policy findings and their source pages, the evidence "
             "check, the risk indicators, the damage description and the audit trail of every agent step.")
    if st.button("Prepare the claim package", type="primary", key="prepare_package"):
        st.session_state.setdefault("package_ready", {})[claim.id] = True
    if st.session_state.get("package_ready", {}).get(claim.id):
        with ui.handle_errors("building the claim package"):
            pdf = build_claim_package(claim, documents, results, reviews, ui.mode_label(container))
            st.download_button("Download the claim package (PDF)", data=pdf, mime="application/pdf",
                               file_name=f"{claim.claim_number}_claim_package.pdf", key="download_package")
            state = "final" if claim.status in (ClaimStatus.APPROVED, ClaimStatus.ESCALATED) else "draft"
            st.caption(f"This is the {state} package ({len(pdf) / 1024:.0f} KB). It is rebuilt from the stored results "
                       "each time, so it always includes the latest decision.")
