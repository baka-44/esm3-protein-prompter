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
        st.warning("This run produced no candidates at all — check the job logs.")
        return

    # Candidates below the QC gate are returned flagged rather than withheld. The gates are not
    # user-adjustable, so hiding the output would leave nothing to act on but "generate more and
    # hope"; the structures and sequences are still worth inspecting, and seeing how far off they
    # were is the only way to judge whether the design or the gate is the problem.
    n_pass = results.get("passed_qc")
    n_below = results.get("reported_below_gate") or 0
    if n_pass == 0 and n_below:
        reasons: dict[str, int] = {}
        for c in cands:
            for f in c.get("gate_failures", []):
                reasons[f.split()[0]] = reasons.get(f.split()[0], 0) + 1
        summary = ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        st.warning(
            f"**No candidate passed QC** — all {n_below} are shown below for inspection, "
            f"with structures and sequences downloadable as usual.\n\n"
            f"Most common shortfall: {summary or 'n/a'}. The per-candidate reason is in the "
            f"**QC** column."
        )
    elif n_below:
        st.info(f"{n_pass} passed QC; {n_below} shown below the gate for inspection.")

    # "Diversity" (RF3 drift-from-input, Å) is only meaningful for RF3 presets
    # (scaffold diversification) — show the column only when candidates carry it.
    def _is_num(v) -> bool:
        return isinstance(v, (int, float)) and v == v  # not None, not NaN

    has_diversity = any(_is_num(c.get("diversity_from_input")) for c in cands)
    has_motif = any(_is_num(c.get("motif_rmsd")) for c in cands)
    design_names = tuple(dict.fromkeys(c["design_pdb"] for c in cands if c.get("design_pdb")))
    has_design = bool(design_names)

    def _row(c: dict) -> dict:
        row = {
            "Rank": c["rank"],
            "QC": ("pass" if c.get("passed_gate", True)
                   else "; ".join(c.get("gate_failures", [])) or "below gate"),
            "Score": round(c.get("composite_score", 0.0), 3),
            "ESM2": round(c.get("esm2_score", 0.0), 3),
            "pLDDT": round(c.get("plddt", 0.0), 1),
            "RMSD Å": round(c.get("rmsd_to_design", 0.0), 2),
        }
        if has_diversity:
            d = c.get("diversity_from_input")
            row["Diversity Å"] = round(d, 2) if _is_num(d) else "—"
        if has_motif:
            mr = c.get("motif_rmsd")
            row["Motif Å"] = round(mr, 2) if _is_num(mr) else "—"
        if has_design:
            # m candidates share one backbone; showing it makes that grouping visible, so a
            # cluster of similar RMSDs reads as "same backbone" rather than as a coincidence.
            d = c.get("design_pdb")
            row["Backbone"] = d.replace("design_", "#").replace(".pdb", "") if d else "—"
        row["Length"] = len(c.get("sequence", ""))
        row["Sequence (preview)"] = c.get("sequence", "")[:40] + ("…" if len(c.get("sequence", "")) > 40 else "")
        return row

    df = pd.DataFrame([_row(c) for c in cands])
    if has_diversity:
        st.caption("**RMSD Å** = self-consistency vs the generated backbone (lower = better). "
                   "**Diversity Å** = drift from the original input backbone (higher = more novel).")
    if has_motif:
        st.caption("**RMSD Å** = self-consistency vs the generated scaffold (lower = better). "
                   "**Motif Å** = catalytic-geometry fidelity vs the parent enzyme (lower = better).")
    st.dataframe(df, hide_index=True, use_container_width=True)

    if has_design:
        # Without this, the ESMFold refold is the only structure on offer and reads as "the
        # candidate" — so a design gets judged by a prediction that never saw its pose.
        st.info(
            "**Two structures per candidate — they are not interchangeable.**\n\n"
            "**Designed backbone** is what RFdiffusion actually built. It holds your composed "
            "pose exactly, including the torso-to-mount placement you set by hand. "
            "*This is the design — evaluate it, and superimpose it on your composite.*\n\n"
            "**ESMFold prediction** is re-folded from the sequence alone and never sees the "
            "pose. On multi-domain grafts it often places the domains far from where they were "
            "designed, so it is a QC measurement (pLDDT, the RMSD gates) rather than the design "
            "itself."
        )

    # Bulk downloads — sequences, the DESIGNED backbones, and the ESMFold refolds
    # (all already persisted to GCS by the worker).
    pdb_names = tuple(c["pdb"] for c in cands if c.get("pdb"))
    cols = st.columns(3 if has_design else 2)
    col_fasta, col_pdb = cols[0], cols[-1]

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

    if has_design:
        with cols[1]:
            try:
                st.download_button(
                    "⬇️ Designed backbones (.zip)",
                    data=_build_pdb_zip(job.job_id, design_names),
                    file_name=f"{job.job_id}_designed_backbones.zip", mime="application/zip",
                    key=f"dl_designzip_{job.job_id}", use_container_width=True,
                    help="What RFdiffusion built — holds your composed pose. Start here.",
                )
            except Exception:
                pass

    with col_pdb:
        if pdb_names:
            try:
                zip_bytes = _build_pdb_zip(job.job_id, pdb_names)
                st.download_button(
                    "⬇️ ESMFold predictions (.zip)" if has_design else "⬇️ Download all PDBs (.zip)",
                    data=zip_bytes,
                    file_name=f"{job.job_id}_esmfold_predictions.zip" if has_design
                              else f"{job.job_id}_pdbs.zip",
                    mime="application/zip",
                    key=f"dl_pdbzip_{job.job_id}", use_container_width=True,
                    help="QC refolds from sequence alone — these do not carry the composed pose."
                         if has_design else None,
                )
            except Exception:
                pass


@st.cache_data(show_spinner=False)
def _build_pdb_zip(job_id: str, pdb_names: tuple[str, ...]) -> bytes:
    """
    Fetch each named PDB from GCS and bundle them into a zip — used for both the
    designed backbones and the ESMFold refolds.
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
