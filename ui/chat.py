"""
ui/chat.py — Chat interface components.

Handles rendering of chat history, user input, PDB file upload, and
the ESM3 prompt summary shown before generation begins.
"""

from __future__ import annotations

import streamlit as st

from core.prompt_builder import describe_prompt


def render_chat_history():
    """Render all messages stored in st.session_state['messages']."""
    messages = st.session_state.get("messages", [])
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
        elif role == "system_info":
            # System information cards (prompt summary, etc.)
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(content)


# ── Curated valid ESM3 / InterPro function keywords ───────────────────────────
# Grouped for the multiselect UI. These are verified to be accepted by the
# Forge API FunctionAnnotation validator.
_KEYWORD_GROUPS: dict[str, list[str]] = {
    "🌿 Plant / Sweet proteins": [
        "thaumatin family",
        "pathogenesis-related protein",
        "plant defensin",
        "osmotin",
    ],
    "🔬 Fluorescence / Optical": [
        "fluorescence",
        "green fluorescent protein",
        "chromophore",
        "bioluminescence",
    ],
    "⚗️ Enzymes": [
        "serine protease activity",
        "cysteine protease activity",
        "metalloprotease activity",
        "kinase activity",
        "phosphatase activity",
        "oxidoreductase activity",
        "transferase activity",
        "hydrolase activity",
        "lyase activity",
        "isomerase activity",
        "ligase activity",
        "alpha/beta hydrolase",
        "trypsin fold",
        "catalytic triad",
    ],
    "🔗 Binding": [
        "DNA binding",
        "RNA binding",
        "zinc finger",
        "calcium binding",
        "ATP binding",
        "GTP binding",
        "heme binding",
        "metal ion binding",
        "lipid binding",
        "carbohydrate binding",
    ],
    "🏗️ Structure / Fold": [
        "beta barrel",
        "TIM barrel",
        "immunoglobulin fold",
        "beta propeller",
        "Rossmann fold",
        "coiled coil",
        "armadillo repeat",
        "WD40 repeat",
    ],
    "🔌 Membrane / Transport": [
        "MFS transporter",
        "ion channel",
        "transmembrane",
        "G protein-coupled receptor",
        "ABC transporter",
    ],
    "🧫 Signalling / Immune": [
        "transcription factor",
        "signal transduction",
        "immunoglobulin",
        "antibody",
        "SH2 domain",
        "SH3 domain",
        "PDZ domain",
    ],
}

# Flat sorted list for the multiselect options
_ALL_KEYWORDS: list[str] = [
    kw for group in _KEYWORD_GROUPS.values() for kw in group
]


def render_input_area() -> tuple[str | None, bytes | None, str | None, list[str], dict, bool]:
    """
    Render the chat input box, optional PDB upload, function keyword selector,
    and structured protein inputs.

    Returns:
        Tuple of (user_text, pdb_bytes, pdb_filename, selected_keywords, structured_inputs,
        ai_infer_keywords).
        user_text is None if the user hasn't submitted anything.
        pdb_bytes / pdb_filename are None if no file was uploaded.
        selected_keywords is a list of InterPro-valid function keywords (may be empty).
        structured_inputs is a dict with keys "sequence", "fixed_residues", "mask_regions"
        (all str, empty string means not provided).
        ai_infer_keywords: if True, Claude infers keywords from the prompt (existing behaviour).
        If False, only the explicitly selected keywords are used — empty selection = no keywords.
    """
    # PDB upload in a small expander above the chat input
    pdb_bytes: bytes | None = None
    pdb_filename: str | None = None

    with st.expander("📎 Upload PDB file (optional — for structure motif constraints)", expanded=False):
        uploaded = st.file_uploader(
            "Upload a .pdb file",
            type=["pdb"],
            help=(
                "Upload a reference PDB to preserve backbone coordinates of an active site or "
                "binding pocket. Mention which residues to fix in your prompt."
            ),
            key=f"pdb_uploader_{st.session_state.get('pdb_uploader_key', 0)}",
        )
        if uploaded is not None:
            pdb_bytes = uploaded.read()
            pdb_filename = uploaded.name
            st.session_state["pdb_bytes"] = pdb_bytes
            st.session_state["pdb_filename"] = pdb_filename
            st.success(f"Loaded: {pdb_filename} ({len(pdb_bytes):,} bytes)")

    # If a PDB was previously uploaded this session, keep it
    if pdb_bytes is None and "pdb_bytes" in st.session_state:
        pdb_bytes = st.session_state["pdb_bytes"]
        pdb_filename = st.session_state.get("pdb_filename")

    # ── Structured protein inputs ──────────────────────────────────────────────
    with st.expander("🔬 Structured protein inputs (optional — bypasses automatic masking logic)", expanded=False):
        st.caption(
            "Provide sequence and residue constraints directly. "
            "Masking is built deterministically — no prompt interpretation errors. "
            "Unspecified residues are open to redesign by default."
        )
        seq_input = st.text_area(
            "Protein sequence",
            value=st.session_state.get("struct_sequence", ""),
            height=80,
            placeholder="Paste full amino acid sequence (single-letter codes, e.g. ATFEIVNRCS…)",
            key="struct_sequence_input",
            help="The reference sequence. Required when specifying fixed residues.",
        )
        col1, col2 = st.columns(2)
        with col1:
            fixed_input = st.text_input(
                "Fixed residues (kept unchanged)",
                value=st.session_state.get("struct_fixed", ""),
                placeholder="K67, R82  or  67, 82  or  1M, 2A, 3K  (1-based)",
                key="struct_fixed_input",
                help=(
                    "Residues to keep unchanged. Accepts: K67, 67K, 1M (any order of letter+number), "
                    "or number-only (67). 1-based. Comma or space separated. "
                    "Everything else will be open to redesign."
                ),
            )
        with col2:
            mask_input = st.text_input(
                "Mask / redesign regions",
                value=st.session_state.get("struct_mask", ""),
                placeholder="18-25, 66-72  (1-based ranges)",
                key="struct_mask_input",
                help=(
                    "Specific regions to redesign. Format: start-end ranges, 1-based inclusive. "
                    "Use instead of (or alongside) fixed residues."
                ),
            )

        # Persist to session state
        st.session_state["struct_sequence"] = seq_input
        st.session_state["struct_fixed"] = fixed_input
        st.session_state["struct_mask"] = mask_input

        # Show a live summary when inputs are provided
        if seq_input.strip() or fixed_input.strip() or mask_input.strip():
            parts = []
            if seq_input.strip():
                parts.append(f"Sequence: {len(seq_input.strip())} aa")
            if fixed_input.strip():
                parts.append(f"Fixed: {fixed_input.strip()}")
            if mask_input.strip():
                parts.append(f"Mask regions: {mask_input.strip()}")
            st.info("  ·  ".join(parts))

    structured_inputs: dict = {
        "sequence":       seq_input.strip(),
        "fixed_residues": fixed_input.strip(),
        "mask_regions":   mask_input.strip(),
        # Condensation fields populated below
        "condense_enabled":      False,
        "condense_key_residues": "",
        "condense_target_length": 0,
    }

    # ── Function keyword selector ──────────────────────────────────────────────
    with st.expander("🏷️ Function keywords (optional)", expanded=False):
        ai_infer_keywords: bool = st.checkbox(
            "Allow inference of keywords from prompt",
            value=st.session_state.get("ai_infer_keywords", True),
            key="ai_infer_keywords_cb",
            help=(
                "When ON (default): InterPro function keywords are automatically inferred "
                "from your prompt. Any keywords you select below are added on top.\n\n"
                "When OFF: only the keywords you select below are used. Leave the selector "
                "empty to send no function keywords at all."
            ),
        )
        st.session_state["ai_infer_keywords"] = ai_infer_keywords

        if ai_infer_keywords:
            st.caption(
                "Keywords will be automatically interpreted and used for generation based on your prompt. "
                "Select keywords below to override or supplement its choices."
            )
        else:
            st.caption(
                "Keyword inference is OFF — only your selections below are sent to ESM3. "
                "Leave the selector empty for **no function keywords**."
            )

        selected_keywords: list[str] = st.multiselect(
            "Keywords",
            options=_ALL_KEYWORDS,
            default=st.session_state.get("selected_function_keywords", []),
            format_func=lambda x: x,
            key="function_keyword_selector",
            label_visibility="collapsed",
            help="Up to 3 keywords recommended.",
        )
        if len(selected_keywords) > 3:
            st.warning("ESM3 works best with ≤ 3 function keywords — consider narrowing your selection.")
        if not ai_infer_keywords and not selected_keywords:
            st.info("No function keywords will be passed to ESM3.")

        st.session_state["selected_function_keywords"] = selected_keywords

    # ── Condense scaffold ──────────────────────────────────────────────────────
    with st.expander("🔩 Condense scaffold (optional — requires PDB)", expanded=False):
        condense_enabled: bool = st.checkbox(
            "Enable scaffold condensation",
            value=st.session_state.get("condense_enabled", False),
            key="condense_enabled_cb",
            help=(
                "Generates a shorter protein that preserves the backbone geometry of your key "
                "residues (active site, binding residues). Based on the ESM3 paper Fig 4D approach: "
                "key residue coordinates are extracted from the PDB and anchored at "
                "proportionally-remapped positions in the shorter scaffold."
            ),
        )
        st.session_state["condense_enabled"] = condense_enabled

        if condense_enabled:
            if pdb_bytes is None:
                st.warning("⚠️ A PDB file is required for scaffold condensation. Upload one above.")
            else:
                # Get original length from PDB (cached in session to avoid re-parsing)
                _pdb_cache_key = f"condense_pdb_len_{pdb_filename}"
                if _pdb_cache_key not in st.session_state:
                    try:
                        from utils.pdb_utils import get_sequence_from_pdb
                        _pdb_seq = get_sequence_from_pdb(pdb_bytes)
                        st.session_state[_pdb_cache_key] = len(_pdb_seq)
                    except Exception:
                        st.session_state[_pdb_cache_key] = 200
                orig_len = st.session_state[_pdb_cache_key]

                condense_key_res = st.text_input(
                    "Key residues to preserve",
                    value=st.session_state.get("condense_key_residues", ""),
                    placeholder="K67, R82, 65-72  (1-based, same format as fixed residues)",
                    key="condense_key_res_input",
                    help=(
                        "The active site, binding residues, or catalytic residues you want to keep. "
                        "Their backbone geometry is extracted from the PDB and anchored at "
                        "proportionally-remapped positions in the shorter protein. "
                        "Leave empty to generate a shorter protein with no anchored residues."
                    ),
                )
                st.session_state["condense_key_residues"] = condense_key_res

                # Parse number of key residues for slider min
                import re as _re
                _n_key = max(1, len([
                    t for t in _re.split(r"[,\s]+", condense_key_res.strip()) if t.strip()
                ])) if condense_key_res.strip() else 1
                _slider_min = max(5, _n_key + 2)
                _slider_max = max(_slider_min + 1, orig_len - 1)
                _default_tgt = max(_slider_min, int(orig_len * 0.75))
                _default_tgt = min(_default_tgt, _slider_max)

                condense_target_len = st.slider(
                    "Target length (residues)",
                    min_value=_slider_min,
                    max_value=_slider_max,
                    value=st.session_state.get("condense_target_length", _default_tgt),
                    key="condense_target_len_slider",
                    help="How many residues the condensed protein should have.",
                )
                st.session_state["condense_target_length"] = condense_target_len

                _pct = condense_target_len / orig_len * 100
                st.caption(
                    f"**{orig_len} → {condense_target_len} residues** "
                    f"({_pct:.0f}% of original)"
                )

                # Update structured_inputs with condensation fields
                structured_inputs["condense_enabled"] = True
                structured_inputs["condense_key_residues"] = condense_key_res
                structured_inputs["condense_target_length"] = condense_target_len

                st.markdown("---")
                _can_condense = condense_target_len > 0
                if st.button(
                    "🔩 Generate condensed scaffold →",
                    key="condense_submit_btn",
                    disabled=not _can_condense,
                    use_container_width=True,
                    help="Generate without needing a text prompt — condensation is fully specified above.",
                ):
                    # Store structured inputs for the condense handler in app.py
                    st.session_state["structured_inputs_cache"] = dict(structured_inputs)
                    st.session_state["condense_request"] = True
                    st.rerun()

    user_text = st.chat_input(
        "Describe the protein you want to design… "
        "(e.g. 'Generate GFP variants with T65, Y66, G67 fixed. I want bright fluorescence.')"
    )

    return user_text, pdb_bytes, pdb_filename, selected_keywords, structured_inputs, ai_infer_keywords


def add_user_message(text: str):
    """Append a user message to the chat history."""
    _ensure_messages()
    st.session_state["messages"].append({"role": "user", "content": text})


def add_assistant_message(text: str):
    """Append an assistant message to the chat history."""
    _ensure_messages()
    st.session_state["messages"].append({"role": "assistant", "content": text})


def add_system_info(text: str):
    """Append a system info card (non-user, non-assistant) to the chat history."""
    _ensure_messages()
    st.session_state["messages"].append({"role": "system_info", "content": text})


def show_prompt_summary(spec, pdb_provided: bool = False):
    """
    Display a collapsible card showing what ESM3 will be prompted with.
    Called after parsing, before generation.
    """
    summary = describe_prompt(spec, pdb_provided=pdb_provided)
    if spec.notes_to_user:
        summary = (
            f"**Interpretation & construction of PLM prompt based on inputs**:\n{spec.notes_to_user}"
            f"\n\n---\n\n"
        ) + summary

    with st.expander("📋 ESM3 Prompt Summary", expanded=True):
        st.markdown(summary)
        if spec.sequence_template:
            st.code(spec.sequence_template, language=None)


def render_generation_progress(current: int, total: int):
    """Render a progress bar during generation."""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"Generating candidate {current + 1} of {total}…")


def _ensure_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def get_conversation_history() -> list[dict]:
    """Return the chat history in a format suitable for the Claude API."""
    messages = st.session_state.get("messages", [])
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
