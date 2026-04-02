"""
ui/sidebar.py — Streamlit sidebar: settings, API keys, and backend controls.

In production (Railway), API keys are set via environment variables and shown
as read-only status badges. In local/Colab mode, editable input fields are shown.
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
        Dict with keys:
          - anthropic_key: str | None
          - forge_token: str | None
          - use_local: bool
          - forge_model: str
          - n_candidates: int
          - temperature: float
          - num_steps: int
    """
    with st.sidebar:
        st.title("⚙️ Settings")

        # ── API Keys ───────────────────────────────────────────────────────────
        st.subheader("API Keys")

        # In production (env var set), show badge. In local/Colab, show input.
        env_anthropic = os.getenv("ANTHROPIC_API_KEY", "")
        if env_anthropic:
            st.success("Anthropic: connected via environment ✅", icon="🔑")
            anthropic_key = env_anthropic
        else:
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=st.session_state.get("anthropic_key", ""),
                type="password",
                help="Required for NL parsing. Get yours at console.anthropic.com",
            )

        st.divider()
        st.subheader("ESM3 Backend")

        use_local = st.toggle(
            "Use local ESM3-open (1.4B)",
            value=st.session_state.get("use_local", os.getenv("USE_LOCAL_ESM3", "false").lower() == "true"),
            help=(
                "When OFF: uses Forge API (requires token below, higher quality). "
                "When ON: uses local ESM3-open model (free, needs GPU)."
            ),
        )

        env_forge = os.getenv("FORGE_API_TOKEN", "")
        if env_forge and not use_local:
            st.success("Forge API: connected via environment ✅", icon="🔑")
            forge_token = env_forge
        else:
            forge_token = st.text_input(
                "Forge API Token",
                value=st.session_state.get("forge_token", ""),
                type="password",
                disabled=use_local,
                help="Get yours at forge.evolutionaryscale.ai",
            )

        # ── Model selector (Forge only) ────────────────────────────────────────
        forge_model = "esm3-medium-2024-08"
        if not use_local and forge_token:
            # Default to Claude's recommendation from the last parsed prompt (if any)
            recommended = st.session_state.get("recommended_model", "esm3-medium-2024-08")
            default_idx = _MODEL_OPTIONS.index(recommended) if recommended in _MODEL_OPTIONS else 1

            forge_model = st.selectbox(
                "Forge Model",
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

            # Nudge if Claude recommended a different model than current selection
            if recommended and recommended != forge_model:
                st.caption(
                    f"💡 Claude recommends: **{_MODEL_LABELS.get(recommended, recommended)}**"
                )

        # ── Backend status ─────────────────────────────────────────────────────
        _render_backend_status(use_local, forge_token, anthropic_key)

        st.divider()

        # ── Generation Parameters ──────────────────────────────────────────────
        st.subheader("Generation Parameters")

        n_candidates = st.slider(
            "Candidates per prompt",
            min_value=1,
            max_value=10,
            value=st.session_state.get("n_candidates", 5),
            help="Number of candidate proteins to generate per request.",
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

        st.divider()
        st.caption("ESM3 Protein Engineering Prompter")
        st.caption("Built with ESM3 + Claude · EvolutionaryScale")

        # Persist to session state
        st.session_state["anthropic_key"] = anthropic_key
        st.session_state["forge_token"] = forge_token
        st.session_state["use_local"] = use_local
        st.session_state["n_candidates"] = n_candidates
        st.session_state["temperature"] = temperature
        st.session_state["num_steps"] = num_steps

    return {
        "anthropic_key": anthropic_key or None,
        "forge_token": forge_token or None,
        "use_local": use_local,
        "forge_model": forge_model,
        "n_candidates": n_candidates,
        "temperature": temperature,
        "num_steps": num_steps,
    }


def _render_backend_status(use_local: bool, forge_token: str, anthropic_key: str):
    """Show coloured status indicators for the current backend configuration."""
    st.markdown("**Status:**")

    if anthropic_key:
        st.success("Claude API: connected", icon="✅")
    else:
        st.error("Claude API: key missing", icon="❌")

    if use_local:
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                if "A100" in gpu:
                    st.success(f"Backend: Local ESM3 · {gpu}", icon="🚀")
                else:
                    st.warning(f"Backend: Local ESM3 · {gpu} (A100 recommended)", icon="⚠️")
            else:
                st.error("Backend: Local ESM3 · No GPU detected (will be very slow)", icon="🐌")
        except ImportError:
            st.warning("Backend: Local ESM3 · torch not installed", icon="⚠️")
    else:
        if forge_token:
            st.success("Backend: Forge API · connected", icon="✅")
        else:
            st.error("Backend: Forge API · token missing", icon="❌")
