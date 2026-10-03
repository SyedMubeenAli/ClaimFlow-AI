"""
Policy search: the policy RAG subsystem in one place.

  1. Upload a policy PDF (or use the sample). It is read, split into sections, embedded and indexed.
  2. See how it was indexed: pages, chunks, sections, which reader and embedding model were used.
  3. Ask the policy a question. The answer is shown only with quotes that were found, word for word,
     in the policy. If the policy does not say, the page says "not found".
  4. See the policy analyst's findings for this claim, each with the page that supports it.
"""

import pandas as pd
import streamlit as st

from claimflow import ui
from claimflow.config import ROOT
from claimflow.errors import IndexingError
from claimflow.models import ClaimStatus, DocumentKind, StepStatus, can_transition
from claimflow.pipeline import steps_changed_by_document
from claimflow.rag import NOT_FOUND, answer_question
from claimflow.schemas import PolicyOutput
from claimflow.textutil import plural
from claimflow.uploads import store_document

container = ui.load_container()
settings = container.settings
SAMPLE = ROOT / "samples" / "sample_motor_policy.pdf"
EXAMPLES = ["", "What is the excess for a collision claim?", "Is racing covered?",
            "How long do I have to report a theft to the police?", "What is the sum insured?",
            "Does the policy cover wear and tear?", "Is a rental car provided while my car is repaired?"]

ui.page_header("Policy search", "Upload a policy, see how it was indexed, ask it questions, and check every finding "
                                "against the page it came from.")
ui.show_flash()

claim = ui.pick_claim(container, key="policy_picker")
if claim is None:
    st.stop()
ui.claim_header(claim)

documents = container.db.list_documents(claim.id)
policies = [d for d in documents if d.kind == DocumentKind.POLICY]
results = container.db.latest_agent_results(claim.id)
is_final = claim.status in (ClaimStatus.APPROVED, ClaimStatus.ESCALATED)


def attach_policy(name: str, data: bytes) -> None:
    """Store the PDF, index it, and remember that the policy step now needs to run again."""
    record = store_document(container.db, settings.upload_dir, claim.id, DocumentKind.POLICY, name, data,
                            settings.max_upload_bytes)
    st.session_state.setdefault("changed_steps", {}).setdefault(claim.id, set()).update(
        steps_changed_by_document(DocumentKind.POLICY))
    try:
        with st.spinner("Reading and indexing the policy"):
            info = container.rag.ensure(record)
        ui.flash("success", f"{name} was uploaded and indexed: {info['pages']} pages, {info['chunks']} chunks, "
                            f"{len(info['sections'])} sections. Open the Agent workflow tab to re-run the steps "
                            "this policy affects.")
    except IndexingError as exc:
        ui.flash("warning", f"{name} was uploaded but could not be indexed. {exc.user_message}")
    st.rerun()


def show_where(page: int, section: str, source: str) -> str:
    return f"Page {page}" + (f", {section}" if section else "") + f" ({source})"


# ---------------------------------------------------------------- 1. upload
st.subheader("1. Policy document")
if is_final:
    st.caption("This claim has a final decision, so no more documents can be added.")
else:
    with st.form("policy_upload_form", clear_on_submit=True):
        uploaded = st.file_uploader("Policy PDF", type=["pdf"], key="policy_upload_file")
        send = st.form_submit_button("Upload and index", type="primary")
    if send:
        with ui.handle_errors("uploading the policy"):
            if uploaded is None:
                st.warning("Choose a PDF first.")
            else:
                attach_policy(uploaded.name, uploaded.getvalue())
    if SAMPLE.exists():
        if st.button("Use the sample policy", key="use_sample_policy"):
            with ui.handle_errors("attaching the sample policy"):
                attach_policy(SAMPLE.name, SAMPLE.read_bytes())
        st.caption("The sample is a fictional motor policy for testing. It is in the samples folder.")
if policies:
    st.dataframe(pd.DataFrame([{"File": d.original_name, "Size": f"{d.size_bytes / 1024:.0f} KB",
                                "Added": d.created_at.replace("T", " ").replace("+00:00", " UTC")} for d in policies]),
                 hide_index=True)

# ---------------------------------------------------------------- 2. indexed information
st.subheader("2. Indexed policy information")
if not policies:
    st.info("No policy is attached to this claim yet.")
for doc in policies:
    info = container.rag.info(doc)
    st.markdown(f"**{ui.md_escape(doc.original_name)}**")
    if info is None:
        st.warning("This policy is not indexed yet.")
        if st.button("Build the index", key=f"index_{doc.id}"):
            with ui.handle_errors("indexing the policy"):
                with st.spinner("Reading and indexing the policy"):
                    container.rag.ensure(doc)
                st.rerun()
        continue
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pages", info["pages"])
    c2.metric("Pages with text", info["pages_with_text"])
    c3.metric("Chunks", info["chunks"])
    c4.metric("Sections and clauses", len(info["sections"]))
    embedder = info["embedder"]
    meta = info.get("pdf_metadata", {})
    st.dataframe(pd.DataFrame([
        {"Item": "Text read by", "Value": info["extractor"]},
        {"Item": "Embeddings", "Value": f"{embedder['name']}, {embedder['model']}, {embedder['dim']} dimensions"},
        {"Item": "Vector store", "Value": info["vector_store"]},
        {"Item": "Indexed at", "Value": info["indexed_at"].replace("T", " ").replace("+00:00", " UTC")},
        {"Item": "Document title", "Value": meta.get("title", "not set")},
        {"Item": "Author", "Value": meta.get("author", "not set")},
    ]), hide_index=True)
    if info["pages"] > info["pages_indexed"]:
        st.warning(f"Only the first {info['pages_indexed']} of {info['pages']} pages were indexed.")
    if embedder["name"] == "hashing":
        st.caption("These embeddings match on shared wording, not on meaning. Install fastembed for semantic search. "
                   "See the Settings page.")
    with st.expander("Sections found"):
        st.dataframe(pd.DataFrame([{"Section": s["title"], "First page": s["page"]} for s in info["sections"]]),
                     hide_index=True)
    if st.button("Rebuild the index", key=f"rebuild_{doc.id}"):
        with ui.handle_errors("rebuilding the index"):
            with st.spinner("Reading and indexing the policy"):
                container.rag.rebuild(doc)
            st.rerun()

# ---------------------------------------------------------------- 3. ask the policy
st.subheader("3. Ask the policy")


def show_answer(answer) -> None:
    st.markdown(f"**Question:** {ui.md_escape(answer.question)}")
    if answer.status == "answered":
        st.success(ui.md_escape(answer.answer))
        st.markdown("**Sources**")
        for i, cite in enumerate(answer.citations, 1):
            st.markdown(f"{i}. **{ui.md_escape(show_where(cite.page, cite.section, cite.source))}**")
            st.markdown("> " + ui.md_escape(cite.quote))
            with st.expander(f"View source page {cite.page}"):
                ui.show_source_page(container, ui.document_for_ref(documents, cite.doc_ref), cite.page, cite.quote)
    elif answer.status == "not_found":
        st.warning(NOT_FOUND)
        if answer.note:
            st.caption(answer.note)
    else:
        st.info(answer.note)
    if answer.passages and not answer.found:
        with st.expander("Closest passages searched (these are not an answer)"):
            for hit in answer.passages:
                st.markdown(f"**{ui.md_escape(show_where(hit.chunk.page, hit.chunk.section, hit.chunk.source))}**")
                st.markdown("> " + ui.md_escape(hit.chunk.text))


if not policies:
    st.caption("Upload a policy to ask it questions.")
else:
    with st.form("policy_question_form"):
        typed = st.text_input("Your question", placeholder="For example: What is the excess for a collision claim?")
        example = st.selectbox("Or pick an example", EXAMPLES, format_func=lambda text: text or "(none)")
        ask = st.form_submit_button("Ask the policy", type="primary")
    if ask:
        with ui.handle_errors("answering the question"):
            readable = []
            for doc in policies:
                try:
                    container.rag.ensure(doc)
                    readable.append(doc)
                except IndexingError as exc:
                    st.warning(f"{doc.original_name} was not searched. {exc.user_message}")
            if not readable:
                st.error("None of the attached policies can be searched.")
            else:
                with st.spinner("Searching the policy"):
                    answer = answer_question(typed.strip() or example, container.rag.corpus(readable), container.llm)
                st.session_state.setdefault("policy_answers", {})[claim.id] = answer
    saved = st.session_state.get("policy_answers", {}).get(claim.id)
    if saved is not None:
        show_answer(saved)
    st.caption("Answers come from the policy text only. A model writes the wording, and code checks that every "
               "quote is really in the policy. Questions here are not part of the claim workflow.")

# ---------------------------------------------------------------- 4. findings for this claim
st.subheader("4. Policy findings for this claim")
runnable = can_transition(claim.status, ClaimStatus.PROCESSING)
result = results.get("policy")
if result is None:
    st.info("The policy analyst has not run for this claim yet.")
    if st.button("Run the policy analysis", type="primary", disabled=not runnable, key="run_policy_analysis"):
        ui.run_with_trace(container, claim, lambda on_event: container.orchestrator.rerun(claim.id, {"policy"}, on_event=on_event))
else:
    ui.show_pill(f"{ui.STEP_ICONS[result.status]} {result.status.label}", ui.STEP_TONES[result.status])
    st.write(result.summary)
    if result.error:
        st.error(result.error)
    out = PolicyOutput.from_dict(result.output)
    stated = {item["message"] for item in out.not_found}
    for warning in result.warnings:
        if warning not in stated:
            st.warning(warning)
    for item in out.not_found:
        st.warning(item["message"])
    for finding in out.findings:
        with st.container(border=True):
            st.markdown(f"**{ui.md_escape(finding['label'])}:** {ui.md_escape(finding['value'])}")
            st.caption(show_where(finding["page"], finding["section"], finding["source"]))
            st.markdown("> " + ui.md_escape(finding["quote"]))
            if finding.get("detail"):
                st.caption(f"Reason given: {finding['detail']}")
            with st.expander(f"View source page {finding['page']}"):
                ui.show_source_page(container, ui.document_for_ref(documents, finding["doc_ref"]),
                                    finding["page"], finding["quote"])
    if out.rejected:
        with st.expander(f"Statements that could not be verified and were not used ({len(out.rejected)})"):
            for item in out.rejected:
                st.write(f"- {ui.md_escape(item['item'])}: {ui.md_escape(item['why'])}")
    if out.queries:
        with st.expander("How the policy was searched"):
            st.write("Search queries used (the agent chose some of these):")
            for query in out.queries:
                st.write(f"- {ui.md_escape(query)}")
            st.write(f"{plural(len(out.chunks_considered), 'chunk')} {'was' if len(out.chunks_considered) == 1 else 'were'} given to the model, from an index of "
                     f"{out.index_info.get('chunks', '?')} chunks.")
    if runnable:
        if st.button("Re-run the policy analysis and what depends on it", key="rerun_policy_analysis"):
            ui.run_with_trace(container, claim, lambda on_event: container.orchestrator.rerun(claim.id, {"policy"}, on_event=on_event))
