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


def render_input_area() -> tuple[str | None, bytes | None, str | None]:
    """
    Render the chat input box and optional PDB upload.

    Returns:
        Tuple of (user_text, pdb_bytes, pdb_filename).
        user_text is None if the user hasn't submitted anything.
        pdb_bytes / pdb_filename are None if no file was uploaded.
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
            key="pdb_uploader",
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

    user_text = st.chat_input(
        "Describe the protein you want to design… "
        "(e.g. 'Generate GFP variants with T65, Y66, G67 fixed. I want bright fluorescence.')"
    )

    return user_text, pdb_bytes, pdb_filename


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
        summary = f"**Claude's interpretation:**\n{spec.notes_to_user}\n\n---\n\n" + summary

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
