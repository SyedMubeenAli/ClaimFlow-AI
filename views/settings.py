"""Settings: what the app is configured with, and a live check of the AI provider."""

import pandas as pd
import streamlit as st

from claimflow import ui
from claimflow.config import KEY_VARS
from claimflow.extract import extractor_name
from claimflow.llm import LLMRequest
from claimflow.rag import store as vector_store

container = ui.load_container()
settings = container.settings
ui.page_header("Settings", "Read from .env and the environment. Change them there and restart the app.")

st.subheader("Current configuration")
st.dataframe(pd.DataFrame([{"Setting": k, "Value": v} for k, v in settings.safe_summary().items()]),
             hide_index=True)

st.subheader("API keys")
st.dataframe(pd.DataFrame([{"Provider": provider, "Variable": variable,
                            "Status": "set" if settings.has_api_key(provider) else "not set"}
                           for provider, variable in KEY_VARS.items()]), hide_index=True)
st.caption("Keys are never shown. They stay in .env or in your host's secrets.")

st.subheader("Policy search (RAG)")
embedder = container.rag.embedder
indexed = len(list(settings.index_dir.glob("doc*/manifest.json"))) if settings.index_dir.exists() else 0
st.dataframe(pd.DataFrame([
    {"Part": "PDF reader", "Status": f"{extractor_name()} first, then pypdf if that fails"},
    {"Part": "Embeddings", "Status": ui.describe_embedder(embedder)},
    {"Part": "Vector store", "Status": "FAISS" if vector_store.faiss is not None else "NumPy (faiss-cpu is not installed)"},
    {"Part": "Index folder", "Status": f"{settings.index_dir} ({indexed} policy index(es) saved)"},
]), hide_index=True)
if getattr(embedder, "fallback_reason", None):
    st.warning(embedder.fallback_reason)
if extractor_name() != "pymupdf":
    st.caption("PyMuPDF is not installed, so pypdf reads the PDFs and page images are not shown. "
               "Run pip install -r requirements.txt to add it.")

st.subheader("Check the AI provider")
if settings.demo_mode:
    st.info("Demo mode is on, so the app uses the scripted offline model and never calls a provider. "
            "Turn the Demo mode switch in the sidebar off to use live AI. That needs a provider and key in .env.")
elif settings.provider == "fake":
    st.info("The offline test provider is selected. Set LLM_PROVIDER to gemini, groq or openrouter "
            "and add that provider's key to use a real model.")
elif st.button("Send a test request", type="primary"):
    with ui.handle_errors("checking the provider"):
        live = getattr(container.llm, "inner", container.llm)  # skip the cache: this must be a real call
        with st.spinner("Contacting the provider"):
            response = live.complete(LLMRequest(prompt="Reply with the single word: ready", max_tokens=20))
        st.success(f"{response.provider} answered with model {response.model} in {response.latency_ms} ms.")
        st.code(response.text)

st.subheader("Switching providers")
st.write("Set LLM_PROVIDER to gemini, groq, openrouter or fake, add the matching key, and restart. "
         "If a model name is rejected, set LLM_MODEL to a name from the provider's current model list.")
