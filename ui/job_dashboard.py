"""
ui/job_dashboard.py — persistent job history + results for the RFdiffusion/MPNN engine.

Lists the signed-in user's own jobs from Firestore (durable — survives refresh /
re-login, B7), shows live status with a clear done/completed indicator (B8 — no
push notifications; the user checks back), and renders results for finished jobs.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

_STATUS_BADGE = {
    "queued": "⏳ Queued",
    "running": "🔄 Running",
    "done": "✅ Done",
    "failed": "❌ Failed",
}


def render_job_dashboard(user_email: str) -> None:
    st.subheader("🗂️ My jobs")

    from proteinredesign.submit import backend_configured
    if not backend_configured():
        st.info(
            "The RFdiffusion/MPNN backend isn't configured in this environment "
            "(GCP project + buckets not set). Jobs can't be listed here."
        )
        return

    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("🔄 Refresh", key="rfd_refresh_jobs", use_container_width=True):
            st.rerun()

    try:
        from proteinredesign import jobstore
        jobs = jobstore.list_jobs_for_user(user_email, limit=50)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load jobs: {exc}")
        return

    if not jobs:
        st.caption("No jobs yet. Submit one above — it will appear here and update as it runs.")
        return

    for job in jobs:
        _render_job_row(job)


def _render_job_row(job) -> None:
    badge = _STATUS_BADGE.get(job.status, job.status)
    age = _ago(job.created_at)
    header = f"{badge} · {job.title or job.preset} · {age}"

    with st.expander(header, expanded=(job.status in ("running", "queued"))):
        if job.status == "running":
            st.progress(min(max(job.progress, 0.0), 1.0), text=job.stage or "Running…")
        elif job.status == "queued":
            st.caption("Queued — waiting for a worker to pick it up.")
        elif job.status == "failed":
            st.error(f"Failed: {job.error or 'unknown error'}")
        elif job.status == "done":
            st.success("Completed — results below.")
            _render_results(job)

        st.caption(f"Job `{job.job_id}`")


def _render_results(job) -> None:
    try:
        from proteinredesign import storage
        results = storage.read_json(job.result_uri)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Results not available yet: {exc}")
        return

    cands = results.get("candidates", [])
    if not cands:
        st.warning("No candidates passed QC. Try loosening constraints or increasing outputs.")
        return

    df = pd.DataFrame([
        {
            "Rank": c["rank"],
            "Score": round(c.get("composite_score", 0.0), 3),
            "ESM2": round(c.get("esm2_score", 0.0), 3),
            "pLDDT": round(c.get("plddt", 0.0), 1),
            "RMSD Å": round(c.get("rmsd_to_design", 0.0), 2),
            "Length": len(c.get("sequence", "")),
            "Sequence (preview)": c.get("sequence", "")[:40] + ("…" if len(c.get("sequence", "")) > 40 else ""),
        }
        for c in cands
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Bulk downloads — FASTA (sequences) and a zip of every candidate's ESMFold
    # structure (both already persisted to GCS by the worker).
    pdb_names = tuple(c["pdb"] for c in cands if c.get("pdb"))
    col_fasta, col_pdb = st.columns(2)

    with col_fasta:
        try:
            from proteinredesign import storage
            fasta = storage.download_bytes(storage.output_uri(job.job_id, "candidates.fasta"))
            st.download_button(
                "⬇️ Download all as FASTA", data=fasta,
                file_name=f"{job.job_id}_candidates.fasta", mime="text/plain",
                key=f"dl_fasta_{job.job_id}", use_container_width=True,
            )
        except Exception:
            pass

    with col_pdb:
        if pdb_names:
            try:
                zip_bytes = _build_pdb_zip(job.job_id, pdb_names)
                st.download_button(
                    "⬇️ Download all PDBs (.zip)", data=zip_bytes,
                    file_name=f"{job.job_id}_pdbs.zip", mime="application/zip",
                    key=f"dl_pdbzip_{job.job_id}", use_container_width=True,
                )
            except Exception:
                pass


@st.cache_data(show_spinner=False)
def _build_pdb_zip(job_id: str, pdb_names: tuple[str, ...]) -> bytes:
    """
    Fetch each candidate's ESMFold PDB from GCS and bundle them into a zip.
    Cached by (job_id, pdb_names) so the GCS reads happen once per job rather
    than on every dashboard re-render/poll.
    """
    import io
    import zipfile

    from proteinredesign import storage

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in pdb_names:
            pdb = storage.download_bytes(storage.output_uri(job_id, name))
            # Prefix with the job id so extracting multiple jobs' zips into one
            # folder doesn't collide (candidate_1.pdb -> <job>_candidate_1.pdb).
            zf.writestr(f"{job_id}_{name}", pdb)
    return buf.getvalue()


def _ago(ts: float) -> str:
    secs = max(0, int(time.time() - ts))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
