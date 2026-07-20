"""
ui/results_panel.py — Candidate results display.

Renders:
  - Round navigation breadcrumb (for multi-round sessions)
  - Summary metrics (best pTM, pLDDT, ESM2, diversity)
  - Ranked results table with ESM2 score column
  - Per-candidate expandable detail: full sequence, pLDDT chart, 3D viewer,
    FASTA/PDB downloads
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.result_processor import CandidateResult, candidates_to_fasta, diversity_summary
from ui.refinement_panel import render_round_breadcrumb


def render_results(
    candidates: list[CandidateResult],
    spec,
    generation_history: list[dict],
    current_round: int,
):
    """
    Render the full results panel.

    Args:
        candidates:          Ranked list of CandidateResult for the current view.
        spec:                PromptSpec that generated these candidates.
        generation_history:  Full list of generation round dicts (for breadcrumb).
        current_round:       1-based round number currently being displayed.
    """
    if not candidates:
        st.warning("No candidates were generated. Try adjusting your prompt or parameters.")
        return

    # ── Round breadcrumb navigation ────────────────────────────────────────────
    render_round_breadcrumb(generation_history)

    st.markdown("---")
    col_heading, col_newdesign = st.columns([5, 1])
    with col_heading:
        st.subheader(
            f"🧬 Round {current_round} — "
            f"{len(candidates)} Candidate{'s' if len(candidates) != 1 else ''}"
        )
    with col_newdesign:
        st.markdown("<div style='padding-top:0.5rem'>", unsafe_allow_html=True)
        if st.button("🔄 New Design", key="new_design_results_top",
                     help="Clear current results and start a fresh design session."):
            st.session_state["_show_new_design_dialog"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Summary metrics row ────────────────────────────────────────────────────
    best = candidates[0]
    div = diversity_summary(candidates)
    esm2_available = any(c.esm2_score != 0.0 for c in candidates)
    struct_available = any(c.has_structure_scores for c in candidates)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Best pTM",
        f"{best.ptm:.3f}" if best.has_structure_scores else "—",
        help="Predicted TM-score (fold quality, 0–1). Fold a candidate to populate." if not struct_available else "Predicted TM-score (fold quality, 0–1)",
    )
    col2.metric(
        "Best pLDDT",
        f"{best.mean_plddt:.1f}" if best.has_structure_scores else "—",
        help="Mean per-residue confidence (0–100). Fold a candidate to populate." if not struct_available else "Mean per-residue confidence of top candidate (0–100)",
    )
    col3.metric(
        "Best ESM2",
        f"{best.esm2_score:.3f}" if esm2_available else "N/A",
        help="ESM2 masked marginal log-likelihood (fitness proxy, higher=better)",
    )
    composite_help = (
        "ESM2 fitness score only — fold candidates to unlock full composite (0.5×pTM + 0.3×pLDDT + 0.2×ESM2)"
        if not struct_available else
        "Composite: 0.5×pTM + 0.3×pLDDT + 0.2×ESM2 (normalised)"
    )
    col4.metric("Best Score", f"{best.composite_score:.3f}", help=composite_help)
    col5.metric("Mean Diversity", f"{div * 100:.1f}%",
                help="Mean pairwise sequence diversity across candidates")

    if not struct_available:
        st.caption(
            "💡 **pTM and pLDDT are not available for sequence-only generation.** "
            "Click **🔬 Fold structure** on any candidate to run ESMFold and get structure confidence scores."
        )

    st.markdown("---")

    # ── Score explanation expander ─────────────────────────────────────────────
    with st.expander("ℹ️ Understanding the scores", expanded=False):
        st.markdown(
            """
| Score | What it measures | Range | Higher means |
|---|---|---|---|
| **pTM** | Predicted TM-score — overall structural fold quality | 0–1 | Better fold |
| **pLDDT** | Per-residue confidence in predicted structure | 0–100 | More confident |
| **ESM2** | Masked marginal log-likelihood — how "natural" the sequence is | ~−3 to 0 | More likely to fold & function |
| **Novelty %** | Sequence distance from reference/template | 0–100% | More novel design |
| **Score** | ESM2 only (sequence generation) · or full 0.5×pTM + 0.3×pLDDT + 0.2×ESM2 (after folding) | 0–1 | Overall better candidate |

**Note:** pTM and pLDDT are only available after folding. Click **🔬 Fold structure** on any candidate to run ESMFold and unlock the full composite score.

**Tip:** For improving a known protein, prefer lower Novelty % with high ESM2 score.
For exploring new sequence space, fold top candidates and compare pTM/pLDDT.
            """
        )

    # ── Ranked table ───────────────────────────────────────────────────────────
    df = _build_results_df(candidates, esm2_available)

    col_config = {
        "Rank": st.column_config.NumberColumn(width="small"),
        "pTM": st.column_config.TextColumn(width="small"),
        "pLDDT": st.column_config.TextColumn(width="small"),
        "Score ▼": st.column_config.ProgressColumn(
            "Score ▼", format="%.3f", min_value=0.0, max_value=1.0,
            help="ESM2 fitness only (fold to get full composite with pTM + pLDDT)" if not struct_available else "Composite: 0.5×pTM + 0.3×pLDDT + 0.2×ESM2",
        ),
        "Novelty %": st.column_config.TextColumn(width="small",
                                                   help="% positions different from reference. '—' when no reference was available."),
        "Sequence (preview)": st.column_config.TextColumn(width="large"),
    }
    if esm2_available:
        col_config["ESM2 LL"] = st.column_config.NumberColumn(format="%.3f", width="small",
                                                               help="ESM2 log-likelihood (higher=better)")

    st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_config)

    # ── Bulk FASTA download ────────────────────────────────────────────────────
    fasta_str = candidates_to_fasta(candidates)
    pfx = st.session_state.get("_session_file_prefix", "download")
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        st.download_button(
            label="⬇️ Download all as FASTA",
            data=fasta_str,
            file_name=f"{pfx}_candidates.fasta",
            mime="text/plain",
            key=f"download_fasta_all_r{current_round}",
            use_container_width=True,
        )
    with col_dl2:
        pdb_candidates = [c for c in candidates if c.pdb_string]
        if pdb_candidates:
            import io
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for c in pdb_candidates:
                    zf.writestr(f"{pfx}_candidate_{c.rank}.pdb", c.pdb_string)
            st.download_button(
                label="⬇️ Download all PDBs (.zip)",
                data=buf.getvalue(),
                file_name=f"{pfx}_pdbs.zip",
                mime="application/zip",
                key=f"download_pdb_zip_r{current_round}",
                use_container_width=True,
            )
        else:
            st.caption("PDB structures not available — run with structure conditioning or fold candidates individually.")

    st.markdown("---")

    # ── Per-candidate detail ───────────────────────────────────────────────────
    st.subheader("Candidate Details")

    for candidate in candidates:
        _render_candidate_detail(
            candidate=candidate,
            spec=spec,
            current_round=current_round,
        )


def _render_candidate_detail(
    candidate: CandidateResult,
    spec,
    current_round: int,
):
    """Render an expandable detail section for a single candidate."""
    esm2_str = f"ESM2={candidate.esm2_score:.3f} · " if candidate.esm2_score != 0.0 else ""
    ptm_str = f"pTM: {candidate.ptm:.3f}" if candidate.has_structure_scores else "pTM: —"
    plddt_str = f"pLDDT: {candidate.mean_plddt:.1f}" if candidate.has_structure_scores else "pLDDT: —"
    novelty_str = f"Novelty: {candidate.novelty_pct:.1f}%" if candidate.has_novelty_ref else "Novelty: —"
    label = (
        f"**#{candidate.rank}** — "
        f"Score: {candidate.composite_score:.3f} · "
        f"{ptm_str} · "
        f"{plddt_str} · "
        f"{esm2_str}"
        f"{novelty_str}"
        + (" ✨ Top pick" if candidate.rank == 1 else "")
    )
    with st.expander(label, expanded=(candidate.rank == 1)):

        # ── Sequence + downloads ───────────────────────────────────────────────
        col_seq, col_dl = st.columns([3, 1])
        with col_seq:
            st.markdown("**Full sequence:**")
            st.code(candidate.sequence, language=None)

        with col_dl:
            pfx = st.session_state.get("_session_file_prefix", "download")
            single_fasta = f">{candidate.fasta_header()}\n{candidate.sequence}\n"
            st.download_button(
                "⬇️ FASTA",
                data=single_fasta,
                file_name=f"{pfx}_candidate_{candidate.rank}.fasta",
                mime="text/plain",
                key=f"fasta_r{current_round}_c{candidate.rank}_{candidate.index}",
            )
            if candidate.pdb_string:
                st.download_button(
                    "⬇️ PDB",
                    data=candidate.pdb_string,
                    file_name=f"{pfx}_candidate_{candidate.rank}.pdb",
                    mime="chemical/x-pdb",
                    key=f"pdb_r{current_round}_c{candidate.rank}_{candidate.index}",
                )

        # ── 3D viewer + fold on-demand ─────────────────────────────────────────
        fold_key = f"fold_pdb_r{current_round}_c{candidate.rank}_{candidate.index}"
        pdb_to_show = candidate.pdb_string or st.session_state.get(fold_key)

        if pdb_to_show:
            _render_3d_viewer(pdb_to_show, key_suffix=fold_key)
            # Show PDB download for on-demand folded structures (generation-time
            # structures already have a download button in col_dl above)
            if not candidate.pdb_string:
                st.download_button(
                    "⬇️ PDB (folded)",
                    data=pdb_to_show,
                    file_name=f"r{current_round}_candidate_{candidate.rank}_folded.pdb",
                    mime="chemical/x-pdb",
                    key=f"dl_fold_{fold_key}",
                )
        else:
            if st.button("🔬 Fold structure (ESMFold)", key=f"fold_btn_{fold_key}"):
                with st.spinner("Folding with ESMFold via Forge API (~10–15s)…"):
                    try:
                        from core.esm_backend import fold_sequence
                        from config import get_esm_client
                        pdb = fold_sequence(candidate.sequence, client=get_esm_client())
                        if pdb:
                            st.session_state[fold_key] = pdb
                            st.rerun()
                        else:
                            st.error(
                                "Folding returned no structure. "
                                "Download the FASTA and fold in ColabFold or ESMFold server."
                            )
                    except Exception as exc:
                        st.error(f"Folding failed: {exc}")

        # ── pLDDT chart ────────────────────────────────────────────────────────
        if candidate.plddt_per_residue:
            _render_plddt_chart(candidate, current_round)


def _render_3d_viewer(pdb_string: str, key_suffix: str = ""):
    st.markdown("**Predicted 3D structure:**")
    try:
        import py3Dmol
        from streamlit.components.v1 import html as _st_html

        view = py3Dmol.view(width=900, height=480)
        view.addModel(pdb_string, "pdb")
        view.setStyle({"cartoon": {"colorscheme": "ssJmol"}})
        view.addSurface(
            py3Dmol.VDW,
            {"opacity": 0.12, "color": "white"},
            {"hetflag": False},
        )
        view.zoomTo()
        # py3Dmol._make_html() returns a self-contained HTML snippet with
        # the 3Dmol.js viewer embedded — render via Streamlit's HTML component.
        _st_html(view._make_html(), height=492, scrolling=False)
    except ImportError:
        st.info(
            "3D viewer requires `py3Dmol`: `pip install py3Dmol`. "
            "Download the PDB to view in PyMOL or ChimeraX.",
            icon="ℹ️",
        )
    except Exception as exc:
        st.warning(f"3D viewer error: {exc}. Download the PDB above to view locally.")


def _render_plddt_chart(candidate: CandidateResult, round_num: int):
    """Per-residue pLDDT confidence bar chart with confidence-level colour bands."""
    st.markdown("**Per-residue pLDDT confidence:**")
    plddt = candidate.plddt_per_residue
    df = pd.DataFrame({
        "Residue": list(range(1, len(plddt) + 1)),
        "pLDDT": plddt,
    })
    # Colour by confidence zone
    df["Zone"] = pd.cut(
        df["pLDDT"],
        bins=[0, 50, 70, 90, 100],
        labels=["Low (<50)", "OK (50–70)", "Good (70–90)", "High (>90)"],
    )
    st.bar_chart(df.set_index("Residue")["pLDDT"], height=140, color="#4c9be8")
    low_pct = (df["pLDDT"] < 70).mean() * 100
    st.caption(
        f"Mean: {candidate.mean_plddt:.1f} · "
        f"{low_pct:.0f}% of residues below pLDDT 70 (regeneration candidates)"
    )


def _build_results_df(
    candidates: list[CandidateResult],
    include_esm2: bool,
) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row = {
            "Rank": c.rank,
            "pTM": f"{c.ptm:.3f}" if c.has_structure_scores else "—",
            "pLDDT": f"{c.mean_plddt:.1f}" if c.has_structure_scores else "—",
            "Score ▼": c.composite_score,
            "Novelty %": f"{c.novelty_pct:.1f}%" if c.has_novelty_ref else "—",
            "Length": len(c.sequence),
            "Sequence (preview)": c.sequence[:40] + ("…" if len(c.sequence) > 40 else ""),
        }
        if include_esm2:
            row["ESM2 LL"] = c.esm2_score
        rows.append(row)

    col_order = ["Rank", "Score ▼", "pTM", "pLDDT"]
    if include_esm2:
        col_order.append("ESM2 LL")
    col_order += ["Novelty %", "Length", "Sequence (preview)"]

    return pd.DataFrame(rows)[col_order]
