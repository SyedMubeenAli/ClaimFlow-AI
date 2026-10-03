"""
ClaimFlow AI: the Streamlit entry point and navigation shell.

Run with:  streamlit run app.py

This file does three things on every rerun: sets the page, builds the shared
container, and registers the five pages. Each page is a script in views/.
"""

import streamlit as st

st.set_page_config(page_title="ClaimFlow AI", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

from claimflow import ui  # noqa: E402  (must come after set_page_config)

ui.inject_css()
container = ui.load_container()

pages = [
    st.Page(ui.PAGE_DASHBOARD, title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page(ui.PAGE_NEW_CLAIM, title="New claim", icon=":material/note_add:"),
    st.Page(ui.PAGE_WORKSPACE, title="Claim workspace", icon=":material/folder_open:"),
    st.Page(ui.PAGE_POLICY, title="Policy search", icon=":material/menu_book:"),
    st.Page(ui.PAGE_REVIEW, title="Review and package", icon=":material/fact_check:"),
    st.Page(ui.PAGE_SETTINGS, title="Settings", icon=":material/settings:"),
]
current_page = st.navigation(pages)
ui.render_sidebar(container)
current_page.run()
