"""
app.py — Protein Engineering Prompter: Streamlit entrypoint.

Run with:
    streamlit run app.py

Or on Colab Pro A100 (see colab_launcher.ipynb for full setup):
    !streamlit run app.py &
    !npx localtunnel --port 8501
"""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="ESM3 Protein Engineering Prompter",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from auth import check_auth, render_user_badge

# ── Auth gate — must be first thing after page config ──────────────────────────
check_auth()

from ui.sidebar import render_sidebar
from ui.chat import (
    render_chat_history,
    render_input_area,
    add_user_message,
    add_assistant_message,
    show_prompt_summary,
    get_conversation_history,
)
from ui.results_panel import render_results

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🧬 ESM3 Protein Engineering Prompter")
st.caption(
    "Describe what you want in plain English. "
    "Claude interprets your request → ESM3 generates candidates → "
    "ESM2 scores fitness → refine iteratively with chain-of-thought."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
settings = render_sidebar()
render_user_badge()   # shows signed-in email + sign-out button at bottom of sidebar

if settings["anthropic_key"]:
    os.environ["ANTHROPIC_API_KEY"] = settings["anthropic_key"]
if settings["forge_token"]:
    os.environ["FORGE_API_TOKEN"] = settings["forge_token"]
os.environ["USE_LOCAL_ESM3"] = "true" if settings["use_local"] else "false"
os.environ["FORGE_MODEL"] = settings["forge_model"]

# ── Session state init ─────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "messages": [],
        "generation_history": [],   # list of round dicts
        "viewing_round": 0,         # index into generation_history
        "refine_request": None,     # set by results_panel when user clicks Refine
        "pdb_bytes": None,
        "pdb_filename": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ── Welcome message ────────────────────────────────────────────────────────────
if not st.session_state["messages"]:
    with st.chat_message("assistant", avatar="🧬"):
        st.markdown(
            "**Welcome!** Describe the protein you want to engineer. For example:\n\n"
            "- *\"Generate 6 GFP variants keeping T65, Y66, G67, R96, E222 fixed. "
            "Aim for high fluorescence.\"*\n"
            "- *\"Design a compact serine protease (~150 residues) with a catalytic triad.\"*\n"
            "- *\"Redesign this zinc finger sequence keeping cysteines at positions 3, 6, 20, 23.\"*\n\n"
            "After generation, click **🔬 Refine** on any top-5 candidate to iterate "
            "with chain-of-thought refinement — add improvement keywords, SS8/SASA hints, "
            "or condense the scaffold."
        )


# ════════════════════════════════════════════════════════════════════════════════
# Pipeline functions — defined before the layout code that calls them
# ════════════════════════════════════════════════════════════════════════════════

def _parse_prompt(
    user_text: str,
    history: list,
    pdb_bytes: bytes | None,
    pdb_filename: str | None,
    settings: dict,
):
    """Parse user text → PromptSpec via Claude. Returns None on error."""
    try:
        from core.nl_parser import NLParser
        from config import get_anthropic_client

        parser = NLParser(anthropic_client=get_anthropic_client())
        spec = parser.parse(
            user_message=user_text,
            conversation_history=history,
            pdb_uploaded=(pdb_bytes is not None),
            pdb_filename=pdb_filename,
        )
        spec.num_candidates = settings["n_candidates"]
        spec.generation_temperature = settings["temperature"]
        spec.num_steps = settings["num_steps"]
        # Propagate Claude's model recommendation to the sidebar
        st.session_state["recommended_model"] = spec.recommended_model
        return spec

    except Exception as e:
        msg = f"**Error parsing request:** {e}"
        st.error(msg)
        add_assistant_message(msg)
        return None


def _build_and_generate(spec, pdb_bytes: bytes | None, settings: dict):
    """
    Build ESMProtein → run ESM3 generation → run ESM2 scoring → process results.
    Returns list[CandidateResult] or None on error.
    """
    # Build ESMProtein prompt
    with st.spinner("Building ESM3 prompt…"):
        try:
            from core.prompt_builder import build_esm_protein
            esm_protein = build_esm_protein(spec, pdb_source=pdb_bytes)
        except Exception as e:
            msg = f"**Error building ESM3 prompt:** {e}"
            st.error(msg)
            add_assistant_message(msg)
            return None

    # ESM3 generation
    progress_bar = st.empty()

    def update_gen_progress(current: int, total: int):
        if total > 0:
            progress_bar.progress(
                current / total,
                text=f"ESM3 generating candidate {current + 1} of {total}…",
            )

    try:
        from core.esm_backend import choose_generation_strategy
        from config import get_esm_client

        generate_fn = choose_generation_strategy(spec)
        with st.spinner(
            f"ESM3 generating {spec.num_candidates} candidates "
            f"({spec.num_steps} steps, T={spec.generation_temperature:.2f})…"
        ):
            raw_results = generate_fn(
                esm_protein=esm_protein,
                spec=spec,
                client=get_esm_client(model_name=settings.get("forge_model")),
                progress_callback=update_gen_progress,
            )
        progress_bar.empty()

    except Exception as e:
        progress_bar.empty()
        msg = f"**Error during ESM3 generation:** {e}"
        st.error(msg)
        add_assistant_message(msg)
        return None

    # ESM2 scoring
    esm2_progress = st.empty()

    def update_esm2_progress(current: int, total: int):
        if total > 0:
            esm2_progress.progress(
                current / total,
                text=f"ESM2 scoring candidate {current + 1} of {total}…",
            )

    try:
        from core.result_processor import process_results
        with st.spinner("ESM2 scoring (fitness estimation)…"):
            candidates = process_results(
                raw_results,
                spec=spec,
                run_esm2_scoring=True,
                esm2_mode="pseudo",
                progress_callback=update_esm2_progress,
            )
        esm2_progress.empty()

    except Exception as e:
        esm2_progress.empty()
        msg = f"**Error processing results:** {e}"
        st.error(msg)
        add_assistant_message(msg)
        return None

    return candidates


def _store_round(
    candidates,
    spec,
    round_num: int,
    user_prompt: str,
    refined_from: int | None,
):
    """Append this generation round to the session history and point the view at it."""
    st.session_state["generation_history"].append({
        "round": round_num,
        "candidates": candidates,
        "spec": spec,
        "user_prompt": user_prompt,
        "refined_from": refined_from,
    })
    # Always show the latest round
    st.session_state["viewing_round"] = len(st.session_state["generation_history"]) - 1


def _show_generation_summary(candidates, round_num: int):
    """Show a compact success message and add it to chat history."""
    if not candidates:
        msg = "Generation completed but no candidates were returned."
        st.warning(msg)
        add_assistant_message(msg)
        return

    best = candidates[0]
    msg = (
        f"✅ **Round {round_num}** — {len(candidates)} candidates generated. "
        f"Best: Score={best.composite_score:.3f} · "
        f"pTM={best.ptm:.3f} · pLDDT={best.mean_plddt:.1f} · "
        f"ESM2={best.esm2_score:.3f}. "
        f"Results on the right →"
    )
    st.success(msg)
    add_assistant_message(msg)


def _run_generation_round(
    user_text: str,
    settings: dict,
    pdb_bytes: bytes | None,
    pdb_filename: str | None,
):
    """Run a full fresh generation round from a user prompt."""
    with st.chat_message("assistant", avatar="🧬"):

        # Step 1 — NL parsing
        with st.spinner("Interpreting your request with Claude…"):
            spec = _parse_prompt(
                user_text=user_text,
                history=get_conversation_history()[:-1],
                pdb_bytes=pdb_bytes,
                pdb_filename=pdb_filename,
                settings=settings,
            )
            if spec is None:
                return

        show_prompt_summary(spec, pdb_provided=(pdb_bytes is not None))

        # Step 2 → 4 — Build, generate, score
        candidates = _build_and_generate(spec, pdb_bytes, settings)
        if candidates is None:
            return

        # Step 5 — Store round + summary message
        round_num = len(st.session_state["generation_history"]) + 1
        _store_round(
            candidates=candidates,
            spec=spec,
            round_num=round_num,
            user_prompt=user_text,
            refined_from=None,
        )
        _show_generation_summary(candidates, round_num)


def _run_refinement_round(req: dict, settings: dict, pdb_bytes: bytes | None):
    """Run a refinement generation round from a previous candidate."""
    candidate = req["candidate"]
    options = req["options"]
    from_round = req["from_round"]
    from_rank = req["from_rank"]

    # Get the PromptSpec from the round we're refining
    history = st.session_state["generation_history"]
    source_entry = next((e for e in history if e["round"] == from_round), None)
    if source_entry is None:
        st.error("Could not find the source generation round. Please try again.")
        return

    original_spec = source_entry["spec"]

    with st.chat_message("assistant", avatar="🔬"):
        from core.refiner import build_refinement_spec, describe_refinement

        round_num = len(history) + 1
        refine_summary = describe_refinement(options, round_num)

        with st.spinner(f"Building refinement spec for Round {round_num}…"):
            try:
                refined_spec = build_refinement_spec(
                    candidate=candidate,
                    original_spec=original_spec,
                    options=options,
                )
                # Override generation settings from sidebar
                refined_spec.num_candidates = settings["n_candidates"]
                # Temperature is already decreased by refiner; clamp to sidebar max
                refined_spec.generation_temperature = min(
                    settings["temperature"],
                    refined_spec.generation_temperature,
                )
                refined_spec.num_steps = settings["num_steps"]
            except Exception as e:
                st.error(f"**Refinement spec error:** {e}")
                add_assistant_message(f"Refinement failed: {e}")
                return

        st.markdown(refine_summary)
        show_prompt_summary(refined_spec, pdb_provided=(pdb_bytes is not None))

        add_assistant_message(
            f"Starting Round {round_num} — refining from candidate #{from_rank} "
            f"(Round {from_round}). {refine_summary}"
        )

        # Generate
        candidates = _build_and_generate(refined_spec, pdb_bytes, settings)
        if candidates is None:
            return

        _store_round(
            candidates=candidates,
            spec=refined_spec,
            round_num=round_num,
            user_prompt=f"[Refinement from Round {from_round} candidate #{from_rank}]",
            refined_from=from_rank,
        )
        _show_generation_summary(candidates, round_num)


# ── Two-column layout ──────────────────────────────────────────────────────────
chat_col, results_col = st.columns([1, 1], gap="large")

# ════════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN: Chat interface
# ════════════════════════════════════════════════════════════════════════════════
with chat_col:
    render_chat_history()
    user_text, pdb_bytes, pdb_filename = render_input_area()

    # ── Check for refinement request from the results panel ───────────────────
    # This fires when the user clicks "Generate refined candidates" in results_panel
    if st.session_state.get("refine_request") is not None:
        req = st.session_state.pop("refine_request")
        _run_refinement_round(req, settings, pdb_bytes)

    # ── Handle fresh user prompt ───────────────────────────────────────────────
    if user_text:
        add_user_message(user_text)
        with st.chat_message("user"):
            st.markdown(user_text)

        _run_generation_round(
            user_text=user_text,
            settings=settings,
            pdb_bytes=pdb_bytes,
            pdb_filename=pdb_filename,
        )


# ════════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN: Results
# ════════════════════════════════════════════════════════════════════════════════
with results_col:
    history = st.session_state["generation_history"]
    viewing_idx = st.session_state.get("viewing_round", len(history) - 1)
    viewing_idx = max(0, min(viewing_idx, len(history) - 1))

    if history:
        entry = history[viewing_idx]
        render_results(
            candidates=entry["candidates"],
            spec=entry["spec"],
            generation_history=history,
            current_round=entry["round"],
        )
    else:
        st.markdown("### Results will appear here")
        st.info(
            "Submit a prompt in the chat on the left to generate protein candidates.\n\n"
            "Candidates are ranked by a composite score:\n"
            "**0.5 × pTM + 0.3 × pLDDT + 0.2 × ESM2 log-likelihood**",
            icon="⬅️",
        )
