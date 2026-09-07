"""
ui/engine_select.py — engine chooser + switch control.

The app routes to one of two interfaces via st.session_state["_engine"]:
  "esm3" → conversational ESM3 design (full generation-parameters sidebar)
  "rfd"  → RFdiffusion / MPNN backbone design (clean, minimal sidebar)

When no engine is chosen, render_engine_chooser() shows a clean selection screen.
"""

from __future__ import annotations

import streamlit as st


# Streamlit stretches the COLUMN to the tallest sibling but leaves the bordered container at its
# natural height, so cards with shorter copy float with their buttons at different baselines. This
# makes each card fill its column and pins the button to the bottom, so the four line up whatever
# the text length. Scoped to this page — it is a full-page view with nothing else on it.
#
# The `:has(style)` rule is not cosmetic: an injected <style> still gets its own
# stElementContainer, which consumes height in the surrounding flex layout and pushes the page
# down. That has bitten this app before.
_CARD_CSS = """
<style>
  div[data-testid="stElementContainer"]:has(> div > style) { display: none !important; }

  /* card fills its (already stretched) column */
  div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"],
  div[data-testid="stColumn"] div[data-testid="stLayoutWrapper"] { height: 100%; }

  /* card body becomes a column so the button can be pushed to the foot */
  div[data-testid="stColumn"] div[data-testid="stLayoutWrapper"]
    > div[data-testid="stVerticalBlock"] {
      height: 100%;
      display: flex;
      flex-direction: column;
  }
  div[data-testid="stColumn"] div[data-testid="stLayoutWrapper"]
    div[data-testid="stElementContainer"]:last-child { margin-top: auto; }

  /* Reserve the height a WRAPPED title actually occupies (measured: 53px vs 38px for one
     line), so one- and two-line names start their body copy level. This costs nothing in
     total height — the tallest card already carries a two-line title. */
  div[data-testid="stColumn"] div[data-testid="stLayoutWrapper"] h5 {
      min-height: 4em;
      display: flex;
      align-items: flex-start;
  }
</style>
"""


def render_engine_chooser() -> None:
    """Full-page, understated selection screen. Sets _engine and reruns on choice."""
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.markdown("<div style='height:0.5vh'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;margin-bottom:0.4rem'>"
        "<div style='font-size:1.35rem;font-weight:600;color:#141414;letter-spacing:-0.01em'>"
        "Choose a design engine</div>"
        "<div style='font-size:0.85rem;color:#767676;margin-top:0.4rem'>"
        "Four ways to design proteins. You can switch anytime.</div></div>",
        unsafe_allow_html=True,
    )
    left, mid, right = st.columns([1, 8, 1])
    with mid:
        c1, c2, c3, c4 = st.columns(4, gap="large")
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
        with c3:
            with st.container(border=True):
                st.markdown("##### 🧩 Compose Graft")
                st.markdown(
                    "<span style='color:#666666;font-size:0.84rem'>"
                    "Borrowed Bodies. Graft a catalytic mount onto a stable torso, pose them in a "
                    "shared 3D frame, and export a graft package to run through RFdiffusion."
                    "</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                if st.button("Compose a graft", key="pick_compose", use_container_width=True):
                    st.session_state["_engine"] = "compose"
                    st.rerun()
        with c4:
            with st.container(border=True):
                st.markdown("##### 🧷 Concatemer")
                st.markdown(
                    "<span style='color:#666666;font-size:0.84rem'>"
                    "Peptide payloads. Assemble bioactive peptides into a secretable carrier, "
                    "then screen which are likely to express and to give the peptides back on "
                    "digestion."
                    "</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                if st.button("Compose a concatemer", key="pick_concat", use_container_width=True):
                    st.session_state["_engine"] = "concatemer"
                    st.rerun()


def render_engine_switch() -> None:
    """Small control (place in the sidebar) to return to the engine chooser."""
    current = st.session_state.get("_engine")
    label = {"esm3": "ESM3", "rfd": "RFdiffusion / MPNN", "compose": "Compose Graft",
             "concatemer": "Concatemer"}.get(current, "—")
    st.caption(f"Engine · **{label}**")
    if st.button("⇄ Switch engine", key="switch_engine", use_container_width=True):
        st.session_state.pop("_engine", None)
        st.rerun()
