"""
ui/engine_select.py — engine chooser + switch control.

The app routes to one of two interfaces via st.session_state["_engine"]:
  "esm3" → conversational ESM3 design (full generation-parameters sidebar)
  "rfd"  → RFdiffusion / MPNN backbone design (clean, minimal sidebar)

When no engine is chosen, render_engine_chooser() shows a clean selection screen.
"""

from __future__ import annotations

import streamlit as st


def render_engine_chooser() -> None:
    """Full-page, understated selection screen. Sets _engine and reruns on choice."""
    st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;margin-bottom:0.4rem'>"
        "<div style='font-size:1.35rem;font-weight:600;color:#141414;letter-spacing:-0.01em'>"
        "Choose a design engine</div>"
        "<div style='font-size:0.85rem;color:#767676;margin-top:0.4rem'>"
        "Two ways to design proteins. You can switch anytime.</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:2.5vh'></div>", unsafe_allow_html=True)

    left, mid, right = st.columns([1, 6, 1])
    with mid:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("##### 💬 ESM3")
                st.markdown(
                    "<span style='color:#666666;font-size:0.84rem'>"
                    "Conversational design. Describe your goal in plain English — "
                    "ESM3 proposes candidate sequences with fitness scoring, all in a chat flow."
                    "</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                if st.button("Use ESM3", key="pick_esm3", use_container_width=True):
                    st.session_state["_engine"] = "esm3"
                    st.rerun()
        with c2:
            with st.container(border=True):
                st.markdown("##### 🧬 RFdiffusion / MPNN")
                st.markdown(
                    "<span style='color:#666666;font-size:0.84rem'>"
                    "Structure-based design. Keep or generate a backbone, design sequences with "
                    "the ProteinMPNN family, and QC by folding. Runs as asynchronous GPU jobs."
                    "</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                if st.button("Use RFdiffusion / MPNN", key="pick_rfd", use_container_width=True):
                    st.session_state["_engine"] = "rfd"
                    st.rerun()


def render_engine_switch() -> None:
    """Small control (place in the sidebar) to return to the engine chooser."""
    current = st.session_state.get("_engine")
    label = {"esm3": "ESM3", "rfd": "RFdiffusion / MPNN"}.get(current, "—")
    st.caption(f"Engine · **{label}**")
    if st.button("⇄ Switch engine", key="switch_engine", use_container_width=True):
        st.session_state.pop("_engine", None)
        st.rerun()
