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
    page_title="Phyx44 Guided Protein Design Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from auth import check_auth, render_user_badge

# ── Auth gate — must be first thing after page config ──────────────────────────
check_auth()

# ── Audit: log session start on first render after OAuth ──────────────────────
if st.session_state.pop("_log_session_start", False):
    from utils.audit_log import log_session_start
    log_session_start()

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
# ── Top banner ─────────────────────────────────────────────────────────────────
import base64 as _b64
_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "phyx44_logo.png")
with open(_logo_path, "rb") as _lf:
    _logo_b64 = _b64.b64encode(_lf.read()).decode()

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

    /* ── Global font — text-bearing elements only (keeps Material Icons intact) ── */
    html, body, .stApp,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th,
    h1, h2, h3, h4, h5, h6, p, label,
    button, input, textarea, select,
    .stButton > button, .stDownloadButton > button {{
        font-family: 'Montserrat', system-ui, -apple-system, sans-serif !important;
    }}

    /* ── Two font sizes only ── */
    h1, h2, h3 {{
        font-size: 1.15rem !important;
        font-weight: 400 !important;
        letter-spacing: -0.01em;
        color: #141414 !important;
    }}
    h4, h5, h6, p, label, input, textarea,
    .stMarkdown p, .stMarkdown li,
    .stButton > button, .stDownloadButton > button,
    .stCaption, [data-testid="stCaptionContainer"] p,
    .stTextInput label, .stTextArea label,
    .stSelectbox label, .stMultiSelect label,
    [data-testid="stSlider"] label,
    .stCheckbox label, .stToggle label {{
        font-size: 0.82rem !important;
    }}

    /* ── White background ── */
    .stApp {{ background-color: #ffffff !important; }}
    .main .block-container {{ background-color: #ffffff !important; padding-top: 1.5rem; }}
    [data-testid="stHeader"] {{ background-color: #ffffff !important; border-bottom: 1px solid #eeeeee; }}
    [data-testid="stToolbar"] {{ background-color: #ffffff !important; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: #fafafa !important;
        border-right: 1px solid #eeeeee !important;
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {{
        color: #444444 !important;
    }}

    /* ── Body text & markdown ── */
    .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {{ color: #333333 !important; }}
    .stCaption, [data-testid="stCaptionContainer"] p {{ color: #888888 !important; }}

    /* ── Buttons ── */
    .stButton > button {{
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.45rem 1.1rem !important;
        transition: background-color 0.2s ease !important;
    }}
    .stButton > button:hover {{ background-color: #000000 !important; }}
    .stButton > button:active {{ background-color: #000000 !important; }}
    .stButton > button[disabled] {{
        background-color: #e5e5e5 !important;
        color: #999999 !important;
    }}
    /* Download buttons — ghost style */
    .stDownloadButton > button {{
        background-color: transparent !important;
        color: #1a1a1a !important;
        border: 1px solid #1a1a1a !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
    }}
    .stDownloadButton > button:hover {{
        background-color: rgba(0,0,0,0.05) !important;
    }}

    /* ── Expanders ── */
    [data-testid="stExpander"] {{
        background-color: #fafafa !important;
        border: 1px solid #eeeeee !important;
        border-radius: 4px !important;
        margin-bottom: 6px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: #333333 !important;
        font-weight: 500 !important;
    }}
    [data-testid="stExpander"] summary:hover {{ color: #1a1a1a !important; }}
    [data-testid="stExpander"] > div:last-child {{
        background-color: #fafafa !important;
    }}

    /* ── Text inputs / textareas ── */
    .stTextInput input, .stTextArea textarea {{
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
        color: #111111 !important;
        border-radius: 4px !important;
        line-height: 1.45 !important;
    }}
    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: #1a1a1a !important;
        box-shadow: 0 0 0 2px rgba(0,0,0,0.06) !important;
    }}
    .stTextInput label, .stTextArea label {{ color: #555555 !important; }}

    /* ── Selectbox / multiselect ── */
    .stSelectbox > div > div, .stMultiSelect > div > div {{
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 4px !important;
        color: #111111 !important;
    }}
    .stSelectbox label, .stMultiSelect label {{ color: #555555 !important; }}

    /* ── Sliders ── */
    [data-testid="stSlider"] label {{ color: #555555 !important; }}
    [data-testid="stSlider"] [role="slider"] {{
        background-color: #1a1a1a !important;
        border: 2px solid #1a1a1a !important;
    }}
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {{ color: #1a1a1a !important; }}

    /* ── Checkboxes / toggles ── */
    .stCheckbox label, .stToggle label {{ color: #444444 !important; }}
    [data-testid="stCheckbox"] svg, [data-testid="stToggle"] svg {{ color: #1a1a1a !important; }}

    /* ── Alerts ── */
    [data-testid="stAlert"] {{
        border-radius: 4px !important;
        background-color: #f9fafb !important;
        border-left: 3px solid #1a1a1a !important;
    }}

    /* ── Dataframe / tables ── */
    [data-testid="stDataFrame"] {{ border-radius: 4px !important; overflow: hidden; }}
    [data-testid="stDataFrame"] > div {{ background-color: #fafafa !important; }}

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {{
        background-color: #f9fafb !important;
        border: 1px solid #eeeeee !important;
        border-radius: 4px !important;
    }}
    [data-testid="stChatInput"] > div {{
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 4px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: #111111 !important;
        background-color: #ffffff !important;
    }}

    /* ── Dividers ── */
    hr {{ border-color: #eeeeee !important; }}

    /* ── Code blocks ── */
    .stCode, [data-testid="stCodeBlock"] {{
        background-color: #f5f5f5 !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 4px !important;
    }}
    code {{ color: #444444 !important; }}

    /* ── Spinner ── */
    [data-testid="stSpinner"] p {{ color: #666666 !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{ background-color: #f5f5f5 !important; border-bottom: 1px solid #eeeeee !important; }}
    .stTabs [data-baseweb="tab"] {{ color: #666666 !important; }}
    .stTabs [aria-selected="true"] {{ color: #1a1a1a !important; border-bottom-color: #1a1a1a !important; }}

    /* ── Metric tiles ── */
    [data-testid="stMetric"] {{
        background-color: #fafafa !important;
        border: 1px solid #eeeeee !important;
        border-radius: 4px !important;
        padding: 12px 16px !important;
    }}
    [data-testid="stMetricLabel"] p {{ color: #888888 !important; text-transform: uppercase; letter-spacing: 0.05em; }}
    [data-testid="stMetricValue"] {{ color: #111111 !important; }}

    /* ── Scrollbars ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #f5f5f5; }}
    ::-webkit-scrollbar-thumb {{ background: #cccccc; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #aaaaaa; }}

    /* ══════════════ PHYX44 BANNER ══════════════ */
    .phyx-banner {{
        display: flex;
        align-items: center;
        background: #ffffff;
        border-radius: 0;
        padding: 12px 0 18px 0;
        margin-bottom: 24px;
        border-bottom: 1px solid #eeeeee;
        gap: 20px;
        flex-wrap: wrap;
    }}
    .phyx-logo {{
        height: 44px;
        width: auto;
        max-width: 200px;
        flex-shrink: 0;
    }}
    .phyx-divider {{
        width: 1px;
        height: 44px;
        background: #eeeeee;
        flex-shrink: 0;
    }}
    .phyx-text {{ flex: 1; min-width: 200px; }}
    .phyx-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: #111111;
        letter-spacing: -0.01em;
        line-height: 1.3;
        margin-bottom: 3px;
        font-family: 'Montserrat', system-ui, sans-serif;
    }}
    .phyx-desc {{
        font-size: 0.76rem;
        color: #888888;
        line-height: 1.5;
        font-family: 'Montserrat', system-ui, sans-serif;
    }}
    @media (max-width: 768px) {{
        .phyx-divider {{ display: none; }}
        .phyx-logo {{ height: 32px; max-width: 140px; }}
    }}
    </style>
    <div class="phyx-banner">
        <img class="phyx-logo"
             src="data:image/png;base64,{_logo_b64}"
             alt="PHYX44" />
        <div class="phyx-divider"></div>
        <div class="phyx-text">
            <div class="phyx-title">Guided Protein Design Tool</div>
            <div class="phyx-desc">
                Design novel proteins — conversational sequence generation, or structure-based
                backbone design with folding-based quality control.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state init (runs before engine routing below) ─────────────────────
def _init_session():
    import time as _time
    defaults = {
        "messages": [],
        "generation_history": [],   # list of round dicts
        "viewing_round": 0,         # index into generation_history
        "refine_request": None,     # set by results_panel when user clicks Refine
        "pdb_bytes": None,
        "pdb_filename": None,
        # Unique prefix for all downloaded files in this session (YYMMDD_HHMMSS)
        "_session_file_prefix": _time.strftime("%y%m%d_%H%M%S"),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ── New Design: reset helper + confirmation dialog ─────────────────────────────

def _reset_session():
    """Clear all generation state, inputs and uploads. Preserve sidebar settings."""
    keep = {
        "_auth_email", "_auth_name", "_audit_session_id",
        "n_candidates", "temperature", "num_steps",
        "anthropic_key", "forge_token", "use_local",
        "forge_model_selector",
    }
    for key in list(st.session_state.keys()):
        if key not in keep:
            del st.session_state[key]
    # Increment uploader key so Streamlit resets the file_uploader widget
    st.session_state["pdb_uploader_key"] = st.session_state.get("pdb_uploader_key", 0) + 1
    st.rerun()


@st.dialog("Start a new design?")
def _confirm_new_design_dialog():
    st.write(
        "This will clear all generated candidates, uploaded files, and structured inputs. "
        "Sidebar settings (candidates, temperature, steps) will be preserved."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear & start fresh", type="primary", use_container_width=True):
            _reset_session()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("_show_new_design_dialog", None)
            st.rerun()


if st.session_state.get("_show_new_design_dialog"):
    _confirm_new_design_dialog()


# ── Engine routing (engine-select first, D3) ──────────────────────────────────
from ui.engine_select import render_engine_chooser, render_engine_switch

_engine = st.session_state.get("_engine")

# No engine chosen yet → clean full-page selection screen.
if not _engine:
    render_engine_chooser()
    st.stop()

# RFdiffusion / MPNN — clean interface with a minimal sidebar (no ESM3 params/status).
if _engine == "rfd":
    with st.sidebar:
        render_engine_switch()
    render_user_badge()
    from ui.rfd_panel import render_rfd_engine
    _user_email = st.session_state.get("_auth_email") or "local@dev"
    render_rfd_engine(_user_email)
    st.stop()

# ESM3 — conversational interface with the full generation-parameters sidebar.
settings = render_sidebar()
with st.sidebar:
    st.divider()
    render_engine_switch()
render_user_badge()

if settings["anthropic_key"]:
    os.environ["ANTHROPIC_API_KEY"] = settings["anthropic_key"]
if settings["forge_token"]:
    os.environ["FORGE_API_TOKEN"] = settings["forge_token"]
os.environ["USE_LOCAL_ESM3"] = "true" if settings["use_local"] else "false"
os.environ["FORGE_MODEL"] = settings["forge_model"]


# ── Welcome message ────────────────────────────────────────────────────────────
if not st.session_state["messages"]:
    with st.chat_message("assistant", avatar="🧬"):
        st.markdown(
            "**Welcome.** Describe the protein you want to engineer. "
            "Upload a PDB for structure-conditioned design, or use **🔩 Condense scaffold** "
            "to generate a shorter protein that preserves your key residues' backbone geometry."
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
    selected_keywords: list[str] | None = None,
    structured_inputs: dict | None = None,
    ai_infer_keywords: bool = True,
):
    """Parse user text → PromptSpec via Claude. Returns None on error."""
    try:
        from core.nl_parser import NLParser
        from config import get_anthropic_client

        # Determine what to pass to parser as the keyword override:
        # - AI inference ON + no selections: None → Haiku infers freely
        # - AI inference ON + selections: pass selections → Haiku is told to use them
        # - AI inference OFF: always pass selections (may be empty list) → no inference
        if not ai_infer_keywords:
            override_keywords: list[str] | None = selected_keywords or []
        elif selected_keywords:
            override_keywords = selected_keywords
        else:
            override_keywords = None

        parser = NLParser(anthropic_client=get_anthropic_client())
        spec = parser.parse(
            user_message=user_text,
            conversation_history=history,
            pdb_uploaded=(pdb_bytes is not None),
            pdb_filename=pdb_filename,
            pdb_bytes=pdb_bytes,
            structured_inputs=structured_inputs,
            override_keywords=override_keywords,
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

    Over-generates 2× the requested candidates to compensate for deduplication
    and input-sequence filtering, then trims to the original N requested.
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

    # Over-generate: request 2× candidates so we have enough after dedup
    requested_n = spec.num_candidates
    spec.num_candidates = min(requested_n * 2, 50)  # cap over-generation within Cloud Run timeout

    # ESM3 generation
    progress_bar = st.empty()

    def update_gen_progress(current: int, total: int):
        if total > 0:
            progress_bar.progress(
                current / total,
                text=f"ESM3 generating candidate {current + 1} of {total} (→ top {requested_n} kept)…",
            )

    try:
        from core.esm_backend import choose_generation_strategy
        from config import get_esm_client

        generate_fn = choose_generation_strategy(spec)
        with st.spinner(
            f"ESM3 generating {spec.num_candidates} candidates "
            f"(→ top {requested_n} after dedup) · "
            f"{spec.num_steps} steps · T={spec.generation_temperature:.2f}…"
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
    finally:
        # Restore the original requested count
        spec.num_candidates = requested_n

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

    # Trim to the originally requested number of candidates
    if len(candidates) > requested_n:
        candidates = candidates[:requested_n]
        # Re-assign ranks after trim
        for rank, c in enumerate(candidates, start=1):
            c.rank = rank

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
    struct_part = (
        f"pTM={best.ptm:.3f} · pLDDT={best.mean_plddt:.1f} · "
        if best.has_structure_scores else ""
    )
    msg = (
        f"✅ **Round {round_num}** — {len(candidates)} candidates generated. "
        f"Best: Score={best.composite_score:.3f} · "
        f"{struct_part}"
        f"ESM2={best.esm2_score:.3f}. "
        f"Results below ↓"
    )
    st.success(msg)
    add_assistant_message(msg)


def _run_generation_round(
    user_text: str,
    settings: dict,
    pdb_bytes: bytes | None,
    pdb_filename: str | None,
    selected_keywords: list[str] | None = None,
    structured_inputs: dict | None = None,
    ai_infer_keywords: bool = True,
):
    """Run a full fresh generation round from a user prompt."""
    from utils.audit_log import log_generation_request, log_generation_result

    with st.chat_message("assistant", avatar="🧬"):

        # Step 1 — NL parsing
        with st.spinner("Interpreting your request…"):
            spec = _parse_prompt(
                user_text=user_text,
                history=get_conversation_history()[:-1],
                pdb_bytes=pdb_bytes,
                pdb_filename=pdb_filename,
                settings=settings,
                selected_keywords=selected_keywords,
                structured_inputs=structured_inputs,
                ai_infer_keywords=ai_infer_keywords,
            )
            if spec is None:
                return

        # Audit log — record what is being sent to ESM Forge
        request_id = log_generation_request(
            user_prompt=user_text,
            spec=spec,
            pdb_filename=pdb_filename,
            forge_model=settings.get("forge_model", ""),
            selected_keywords=selected_keywords,
            ai_infer_keywords=ai_infer_keywords,
        )

        show_prompt_summary(spec, pdb_provided=(pdb_bytes is not None))

        # Step 2 → 4 — Build, generate, score
        candidates = _build_and_generate(spec, pdb_bytes, settings)

        # Audit log — record the outcome
        log_generation_result(
            request_id=request_id,
            candidates=candidates,
            error=None if candidates is not None else "generation failed",
        )

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




# ════════════════════════════════════════════════════════════════════════════════
# TOP: Chat interface (inputs)
# ════════════════════════════════════════════════════════════════════════════════
render_chat_history()
user_text, pdb_bytes, pdb_filename, selected_keywords, structured_inputs, ai_infer_keywords = render_input_area()

# ── Handle condensation request (button in condense expander) ─────────────────
if st.session_state.pop("condense_request", False):
    si = st.session_state.get("structured_inputs_cache", structured_inputs)
    tgt_len = si.get("condense_target_length", "?")
    key_res = si.get("condense_key_residues", "") or "no key residues specified"
    auto_text = (
        f"Condense scaffold to {tgt_len} residues, "
        f"preserving: {key_res}"
    )
    add_user_message(auto_text)
    with st.chat_message("user"):
        st.markdown(auto_text)
    _run_generation_round(
        user_text=auto_text,
        settings=settings,
        pdb_bytes=pdb_bytes,
        pdb_filename=pdb_filename,
        selected_keywords=selected_keywords,
        structured_inputs=si,
        ai_infer_keywords=False,
    )

# ── Handle fresh user prompt ──────────────────────────────────────────────────
elif user_text:
    add_user_message(user_text)
    with st.chat_message("user"):
        st.markdown(user_text)

    _run_generation_round(
        user_text=user_text,
        settings=settings,
        pdb_bytes=pdb_bytes,
        pdb_filename=pdb_filename,
        selected_keywords=selected_keywords,
        structured_inputs=structured_inputs,
        ai_infer_keywords=ai_infer_keywords,
    )


# ════════════════════════════════════════════════════════════════════════════════
# BELOW: Results
# ════════════════════════════════════════════════════════════════════════════════
history = st.session_state["generation_history"]
viewing_idx = st.session_state.get("viewing_round", len(history) - 1)
viewing_idx = max(0, min(viewing_idx, len(history) - 1))

if history:
    st.markdown("---")
    entry = history[viewing_idx]
    render_results(
        candidates=entry["candidates"],
        spec=entry["spec"],
        generation_history=history,
        current_round=entry["round"],
    )
