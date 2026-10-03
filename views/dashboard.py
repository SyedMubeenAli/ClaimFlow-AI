"""Dashboard: totals, claims by status, the review queue and every claim."""

import pandas as pd
import streamlit as st

from claimflow import demo, ui
from claimflow.models import ClaimStatus
from claimflow.textutil import plural

container = ui.load_container()
ui.page_header("Dashboard", "Every claim in the system, and what is waiting for a person.")
ui.show_flash()

stats = container.db.dashboard_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total claims", stats["total"])
c2.metric("Awaiting review", stats["awaiting_review"])
c3.metric("With warnings", stats["claims_with_warnings"],
          help="Claims where the latest run of at least one agent ended in a warning or a failure.")
c4.metric("Failed steps", stats["failed_steps"])



def load_demo() -> None:
    with ui.handle_errors("loading the demo claims"):
        with st.spinner("Creating the demo claims and their documents"):
            loaded = demo.load_demo_claims(container)
        created = sum(1 for _, new in loaded if new)
        ui.flash("success", f"{plural(created, 'demo claim')} created. Open one in the claim workspace and press Run agents."
                 if created else "The demo claims already exist.")
        st.rerun()


claims = container.db.list_claims()
if not claims:
    st.info("There are no claims yet. Load the three demo claims to see the whole workflow, or create your own.")
    left, right = st.columns(2)
    if left.button("Load the demo claims", type="primary"):
        load_demo()
    if right.button("Create a claim"):
        ui.go_to(ui.PAGE_NEW_CLAIM)
    st.stop()

st.subheader("Claims by status")
by_status = {ClaimStatus(key).label: count for key, count in stats["by_status"].items()}
st.bar_chart(pd.DataFrame({"Claims": list(by_status.values())}, index=list(by_status.keys())))

waiting = [c for c in claims if c.status == ClaimStatus.AWAITING_REVIEW]
st.subheader("Waiting for a human decision")
if not waiting:
    st.caption("Nothing is waiting. Claims appear here after the agents finish.")
for claim in waiting:
    left, right = st.columns([4, 1])
    left.write(f"{ui.md_escape(claim.claim_number)}   {ui.md_escape(claim.claimant_name)}, "
               f"{claim.incident_type.label.lower()}")
    if right.button("Open for review", key=f"review_{claim.id}"):
        ui.set_current_claim(claim.id)
        ui.go_to(ui.PAGE_REVIEW)

st.subheader("All claims")
st.dataframe(
    pd.DataFrame([{
        "Claim": c.claim_number, "Claimant": c.claimant_name, "Incident": c.incident_type.label,
        "Status": c.status.label, "Updated": c.updated_at.replace("T", " ").replace("+00:00", " UTC"),
    } for c in claims]),
    hide_index=True,
)

with st.expander("Demo data"):
    st.write("Three made-up claims with real documents, each showing a different path through the workflow.")
    st.dataframe(pd.DataFrame([{"Claim": d.title, "Claimant": d.claimant, "What should happen": d.expectation}
                               for d in demo.DEMO_CLAIMS]), hide_index=True)
    if st.button("Load the demo claims", key="load_demo_again"):
        load_demo()

with st.expander("How the AI is used in this app"):
    st.markdown("""
- **Generative AI.** A language model reads the accident report and quotation, describes the damage photos, and writes the explanation of a recommendation.
- **RAG.** The policy is split into sections, embedded and searched. Every policy finding cites the page and section it came from, and says "not found" when the policy does not answer. See Policy search.
- **Multi-agent system.** Seven agents with typed inputs and outputs share one claim state and never call each other. See the Agent workflow tab.
- **Agentic AI.** The policy agent writes its own search queries for each claim. When evidence is missing the workflow says what is needed, and when new evidence arrives only the affected agents run again.
- **AI workflow and process automation.** An orchestrator runs the steps in order, skips what cannot run, contains failures, records every step, and ends with a claim package PDF.
- **Human in the loop.** Nothing is approved or rejected by the AI. A person decides on the Review page, and overriding the AI needs a written reason.
- **Guardrails.** Money is calculated in code. Policy quotes are checked word for word. Risk analysis never judges anyone's honesty.
""")

st.subheader("Open a claim")
picked = ui.pick_claim(container, key="dashboard_picker")
if picked is not None and st.button("Open claim workspace", type="primary"):
    ui.go_to(ui.PAGE_WORKSPACE)
