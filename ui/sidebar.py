"""
ui/sidebar.py — Streamlit sidebar: generation parameters, backend toggle, status.
"""

from __future__ import annotations

import os

import streamlit as st

_MODEL_OPTIONS = [
    "esm3-small-2024-08",
    "esm3-medium-2024-08",
    "esm3-large-2024-08",
]
_MODEL_LABELS = {
    "esm3-small-2024-08":  "Small  — fast, exploratory runs",
    "esm3-medium-2024-08": "Medium — standard engineering (default)",
    "esm3-large-2024-08":  "Large  — complex / de novo design",
}


def render_sidebar() -> dict:
    """
    Render the settings sidebar and return the current runtime configuration.

    Returns:
        Dict with keys: anthropic_key, forge_token, use_local, forge_model,
                        n_candidates, temperature, num_steps
    """
    env_anthropic = os.getenv("ANTHROPIC_API_KEY", "")
    env_forge     = os.getenv("FORGE_API_TOKEN", "")

    # Read keys — production uses env vars, local uses session state inputs
    anthropic_key = env_anthropic or st.session_state.get("anthropic_key", "")
    forge_token   = env_forge     or st.session_state.get("forge_token", "")

    with st.sidebar:
        st.title("⚙️ Settings")

        # ── 1. Generation Parameters ───────────────────────────────────────────
        st.subheader("Generation Parameters")

        n_candidates = st.slider(
            "Candidates per prompt",
            min_value=1,
            max_value=50,
            value=st.session_state.get("n_candidates", 5),
            help="Number of candidate proteins to generate per request.",
        )
        if n_candidates > 20:
            _est_min = round(n_candidates * 2 * 4.5 / 60, 1)
            st.caption(
                f"⏱️ ~{_est_min} min expected — {n_candidates * 2} API calls "
                f"(2× over-generate, then deduplicate). Keep the tab active."
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.5,
            value=st.session_state.get("temperature", 0.7),
            step=0.05,
            help=(
                "Controls diversity. Lower (0.3–0.5) = conservative/high-fidelity. "
                "Higher (0.8–1.2) = diverse/exploratory."
            ),
        )

        num_steps = st.slider(
            "Generation steps",
            min_value=4,
            max_value=20,
            value=st.session_state.get("num_steps", 8),
            help="More steps = slower but potentially better designs.",
        )

        # ── Forge model selector (inline with generation params) ───────────────
        recommended = st.session_state.get("recommended_model", "esm3-medium-2024-08")
        default_idx = _MODEL_OPTIONS.index(recommended) if recommended in _MODEL_OPTIONS else 1

        forge_model = st.selectbox(
            "ESM3 Model",
            options=_MODEL_OPTIONS,
            format_func=lambda m: _MODEL_LABELS.get(m, m),
            index=default_idx,
            key="forge_model_selector",
            help=(
                "Claude auto-selects based on prompt complexity. "
                "Override here to force a specific model. "
                "Larger = better quality, slower, higher cost."
            ),
        )
        if recommended and recommended != forge_model:
            st.caption(f"💡 Claude recommends: **{_MODEL_LABELS.get(recommended, recommended)}**")

        st.divider()

        # ── 2. Backend toggle ──────────────────────────────────────────────────
        use_local = st.toggle(
            "Use local ESM3-open (1.4B)",
            value=st.session_state.get(
                "use_local",
                os.getenv("USE_LOCAL_ESM3", "false").lower() == "true",
            ),
            help=(
                "OFF: uses Forge API (recommended, higher quality). "
                "ON: uses local ESM3-open model (free, needs GPU)."
            ),
        )

        st.divider()

        # ── 3. Status ──────────────────────────────────────────────────────────
        st.subheader("Status")
        _render_status(use_local, forge_token, anthropic_key)

        st.divider()

        # ── New Design button ──────────────────────────────────────────────────
        if st.button("🔄 New Design", use_container_width=True,
                     help="Clear current results and inputs to start a fresh design session."):
            st.session_state["_show_new_design_dialog"] = True
            st.rerun()

        st.caption("📋 All sessions and generation requests are logged for compliance.")

        # Persist to session state
        st.session_state["anthropic_key"] = anthropic_key
        st.session_state["forge_token"]   = forge_token
        st.session_state["use_local"]     = use_local
        st.session_state["n_candidates"]  = n_candidates
        st.session_state["temperature"]   = temperature
        st.session_state["num_steps"]     = num_steps

    return {
        "anthropic_key": anthropic_key or None,
        "forge_token":   forge_token or None,
        "use_local":     use_local,
        "forge_model":   forge_model,
        "n_candidates":  n_candidates,
        "temperature":   temperature,
        "num_steps":     num_steps,
    }


def _render_status(use_local: bool, forge_token: str, anthropic_key: str) -> None:
    """Four compact status lines with tick/cross indicators."""

    def _line(label: str, ok: bool, detail: str = "") -> None:
        icon   = "✅" if ok else "❌"
        status = "Connected" if ok else "Not configured"
        text   = f"{icon} **{label}:** {status}"
        if detail:
            text += f" · {detail}"
        st.markdown(text)

    _line("Anthropic Key", bool(anthropic_key))

    _line("Forge API",     bool(forge_token) and not use_local)

    _line("Claude API",    bool(anthropic_key))

    if use_local:
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                _line("Backend", True, f"Local ESM3 · {gpu}")
            else:
                st.markdown("❌ **Backend:** Local ESM3 · No GPU detected")
        except ImportError:
            st.markdown("⚠️ **Backend:** Local ESM3 · torch not installed")
    else:
        _line("Backend", bool(forge_token), "Forge API" if forge_token else "")
