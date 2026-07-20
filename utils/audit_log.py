"""
utils/audit_log.py — Structured audit logging for compliance.

All generation requests and their payloads are logged as structured JSON
to stdout, which Cloud Run automatically forwards to Google Cloud Logging.

Query logs in GCP Console → Logging → Log Explorer:
    resource.type="cloud_run_revision"
    jsonPayload.log_type="prot_prompt_audit"

Or via gcloud:
    gcloud logging read 'jsonPayload.log_type="prot_prompt_audit"' \
      --project phyx44-pp-codonlm-v1 --limit 50 --format json
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.nl_parser import PromptSpec

# Use a dedicated logger so audit records are distinguishable from app logs.
_logger = logging.getLogger("prot_prompt.audit")

# Ensure the logger emits even if root logger level is higher.
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def _session_id() -> str:
    """Return a stable per-Streamlit-session ID (created once, stored in session_state)."""
    import streamlit as st
    if "_audit_session_id" not in st.session_state:
        st.session_state["_audit_session_id"] = uuid.uuid4().hex[:12]
    return st.session_state["_audit_session_id"]


def _user_email() -> str:
    try:
        import streamlit as st
        return st.session_state.get("_auth_email", "anonymous")
    except Exception:
        return "anonymous"


def _spec_summary(spec: "PromptSpec") -> dict:
    """Extract the fields being sent to ESM Forge — what matters for compliance."""
    template = getattr(spec, "sequence_template", "") or ""
    mask_count = template.count("_")
    total_len = getattr(spec, "protein_length", 0)

    return {
        "protein_length": total_len,
        "mask_count": mask_count,
        "fixed_position_count": len(getattr(spec, "fixed_positions", {}) or {}),
        "function_keywords": list(getattr(spec, "function_keywords", []) or []),
        "use_structure_motif": bool(getattr(spec, "use_structure_motif", False)),
        "motif_residue_count": len(getattr(spec, "motif_residue_indices", []) or []),
        "condense_mode": bool(getattr(spec, "motif_source_indices", [])),
        "num_candidates": getattr(spec, "num_candidates", 0),
        "temperature": getattr(spec, "generation_temperature", 0),
        "num_steps": getattr(spec, "num_steps", 0),
        "recommended_model": getattr(spec, "recommended_model", ""),
        # Full sequence template logged for ESM payload audit
        "sequence_template": template,
        "original_sequence_length": len(getattr(spec, "original_sequence", "") or ""),
    }


def log_generation_request(
    user_prompt: str,
    spec: "PromptSpec",
    pdb_filename: str | None,
    forge_model: str,
    selected_keywords: list[str] | None,
    ai_infer_keywords: bool,
) -> str:
    """
    Log a generation request. Returns the request_id for correlating with the result.
    Called immediately after NL parsing, before ESM3 generation starts.
    """
    request_id = uuid.uuid4().hex[:16]
    record = {
        "log_type": "prot_prompt_audit",
        "event": "generation_request",
        "request_id": request_id,
        "session_id": _session_id(),
        "user_email": _user_email(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_prompt": user_prompt,
        "pdb_filename": pdb_filename or None,
        "forge_model": forge_model,
        "keyword_inference_enabled": ai_infer_keywords,
        "user_selected_keywords": list(selected_keywords or []),
        "esm_payload": _spec_summary(spec),
    }
    _logger.info(json.dumps(record))
    return request_id


def log_generation_result(
    request_id: str,
    candidates: list | None,
    error: str | None = None,
) -> None:
    """
    Log the outcome of a generation request.
    Called after ESM3 generation + ESM2 scoring completes (or fails).
    """
    if candidates:
        scores = [round(c.composite_score, 4) for c in candidates]
        result_summary = {
            "status": "success",
            "candidates_returned": len(candidates),
            "composite_scores": scores,
            "top_score": max(scores) if scores else None,
            "has_structure_scores": candidates[0].has_structure_scores if candidates else False,
        }
    else:
        result_summary = {
            "status": "error" if error else "empty",
            "candidates_returned": 0,
            "error": error,
        }

    record = {
        "log_type": "prot_prompt_audit",
        "event": "generation_result",
        "request_id": request_id,
        "session_id": _session_id(),
        "user_email": _user_email(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": result_summary,
    }
    _logger.info(json.dumps(record))


def log_session_start() -> None:
    """Log when a user successfully authenticates. Called from check_auth after login."""
    record = {
        "log_type": "prot_prompt_audit",
        "event": "session_start",
        "session_id": _session_id(),
        "user_email": _user_email(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _logger.info(json.dumps(record))
