"""
ui/results_panel.py — Candidate results display with refinement controls.

Renders:
  - Round navigation breadcrumb (for multi-round chain-of-thought sessions)
  - Summary metrics (best pTM, pLDDT, ESM2, diversity)
  - Ranked results table with ESM2 score column
  - Per-candidate expandable detail: full sequence, pLDDT chart, 3D viewer,
    FASTA/PDB downloads, and the full refinement panel for top-5 candidates
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.result_processor import CandidateResult, candidates_to_fasta, diversity_summary
from ui.refinement_panel import render_refinement_panel, render_round_breadcrumb

# How many candidates get the refinement panel
REFINE_TOP_N = 5


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
    st.subheader(
        f"🧬 Round {current_round} — "
        f"{len(candidates)} Candidate{'s' if len(candidates) != 1 else ''}"
    )

    # ── Summary metrics row ────────────────────────────────────────────────────
    best = candidates[0]
    div = diversity_summary(candidates)
    esm2_available = any(c.esm2_score != 0.0 for c in candidates)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Best pTM", f"{best.ptm:.3f}",
                help="Predicted TM-score (fold quality, 0–1)")
    col2.metric("Best pLDDT", f"{best.mean_plddt:.1f}",
                help="Mean per-residue confidence of top candidate (0–100)")
    col3.metric("Best ESM2", f"{best.esm2_score:.3f}" if esm2_available else "N/A",
                help="ESM2 masked marginal log-likelihood (fitness proxy, higher=better)")
    col4.metric("Best Score", f"{best.composite_score:.3f}",
                help="Composite: 0.5×pTM + 0.3×pLDDT + 0.2×ESM2 (normalised)")
    col5.metric("Mean Diversity", f"{div * 100:.1f}%",
                help="Mean pairwise sequence diversity across candidates")

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
| **Composite** | Weighted combination: 0.5×pTM + 0.3×pLDDT + 0.2×ESM2 | 0–1 | Overall better candidate |

**Tip:** For improving a known protein, prefer lower Novelty % with high ESM2 score.
For exploring new sequence space, balance novelty with pTM/pLDDT quality.
            """
        )

    # ── Ranked table ───────────────────────────────────────────────────────────
    df = _build_results_df(candidates, esm2_available)

    col_config = {
        "Rank": st.column_config.NumberColumn(width="small"),
        "pTM": st.column_config.NumberColumn(format="%.3f", width="small"),
        "pLDDT": st.column_config.NumberColumn(format="%.1f", width="small"),
        "Composite ▼": st.column_config.ProgressColumn(
            "Composite ▼", format="%.3f", min_value=0.0, max_value=1.0
        ),
        "Novelty %": st.column_config.NumberColumn(format="%.1f%%", width="small"),
        "Sequence (preview)": st.column_config.TextColumn(width="large"),
    }
    if esm2_available:
        col_config["ESM2 LL"] = st.column_config.NumberColumn(format="%.3f", width="small",
                                                               help="ESM2 log-likelihood (higher=better)")

    st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_config)

    # ── Bulk FASTA download ────────────────────────────────────────────────────
    fasta_str = candidates_to_fasta(candidates)
    st.download_button(
        label="⬇️ Download all as FASTA",
        data=fasta_str,
        file_name=f"esm3_round{current_round}_candidates.fasta",
        mime="text/plain",
        key=f"download_fasta_all_r{current_round}",
    )

    st.markdown("---")

    # ── Per-candidate detail + refinement panels ───────────────────────────────
    st.subheader("Candidate Details & Refinement")
    st.caption(
        f"Top {min(REFINE_TOP_N, len(candidates))} candidates have refinement controls. "
        "Click 'Refine' to start the next generation round from any candidate."
    )

    for candidate in candidates:
        _render_candidate_detail(
            candidate=candidate,
            spec=spec,
            current_round=current_round,
            show_refine=(candidate.rank <= REFINE_TOP_N),
        )


def _render_candidate_detail(
    candidate: CandidateResult,
    spec,
    current_round: int,
    show_refine: bool,
):
    """Render an expandable detail section for a single candidate."""
    esm2_str = f"ESM2={candidate.esm2_score:.3f} · " if candidate.esm2_score != 0.0 else ""
    label = (
        f"**#{candidate.rank}** — "
        f"Score: {candidate.composite_score:.3f} · "
        f"pTM: {candidate.ptm:.3f} · "
        f"pLDDT: {candidate.mean_plddt:.1f} · "
        f"{esm2_str}"
        f"Novelty: {candidate.novelty_pct:.1f}%"
        + (" ✨ Top pick" if candidate.rank == 1 else "")
    )
    with st.expander(label, expanded=(candidate.rank == 1)):

        # ── Sequence + downloads ───────────────────────────────────────────────
        col_seq, col_dl = st.columns([3, 1])
        with col_seq:
            st.markdown("**Full sequence:**")
            st.code(candidate.sequence, language=None)

        with col_dl:
            single_fasta = f">{candidate.fasta_header()}\n{candidate.sequence}\n"
            st.download_button(
                "⬇️ FASTA",
                data=single_fasta,
                file_name=f"r{current_round}_candidate_{candidate.rank}.fasta",
                mime="text/plain",
                key=f"fasta_r{current_round}_c{candidate.rank}_{candidate.index}",
            )
            if candidate.pdb_string:
                st.download_button(
                    "⬇️ PDB",
                    data=candidate.pdb_string,
                    file_name=f"r{current_round}_candidate_{candidate.rank}.pdb",
                    mime="chemical/x-pdb",
                    key=f"pdb_r{current_round}_c{candidate.rank}_{candidate.index}",
                )

        # ── 3D viewer ──────────────────────────────────────────────────────────
        if candidate.pdb_string:
            _render_3d_viewer(candidate, current_round)
        else:
            st.caption("3D structure not available — use Forge API structure track for full coordinates.")

        # ── pLDDT chart ────────────────────────────────────────────────────────
        if candidate.plddt_per_residue:
            _render_plddt_chart(candidate, current_round)

        # ── Refinement panel (top N only) ──────────────────────────────────────
        if show_refine:
            st.markdown("---")
            panel_key = f"r{current_round}_c{candidate.rank}"

            # Toggle button to show/hide the refinement panel
            toggle_key = f"show_refine_{panel_key}"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False

            if st.button(
                f"🔬 Refine from candidate #{candidate.rank}",
                key=f"toggle_refine_{panel_key}",
                use_container_width=False,
            ):
                st.session_state[toggle_key] = not st.session_state[toggle_key]

            if st.session_state[toggle_key]:
                refine_options = render_refinement_panel(
                    candidate=candidate,
                    original_spec=spec,
                    round_num=current_round + 1,
                    panel_key=panel_key,
                )
                if refine_options is not None:
                    # Signal app.py to start a refinement generation round
                    st.session_state["refine_request"] = {
                        "candidate": candidate,
                        "options": refine_options,
                        "from_round": current_round,
                        "from_rank": candidate.rank,
                    }
                    st.session_state[toggle_key] = False
                    st.rerun()


def _render_3d_viewer(candidate: CandidateResult, round_num: int):
    st.markdown("**Predicted 3D structure:**")
    try:
        import py3Dmol
        import stmol

        view = py3Dmol.view(width=600, height=380)
        view.addModel(candidate.pdb_string, "pdb")
        view.setStyle({"cartoon": {"colorscheme": "ssJmol"}})
        view.addSurface(
            py3Dmol.VDW,
            {"opacity": 0.15, "color": "white"},
            {"hetflag": False},
        )
        view.zoomTo()
        stmol.showmol(view, height=380)
    except ImportError:
        st.info(
            "Install `stmol` + `py3Dmol` for 3D viewer: `pip install stmol py3Dmol`. "
            "Download the PDB to view in PyMOL or ChimeraX.",
            icon="ℹ️",
        )


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
            "pTM": c.ptm,
            "pLDDT": c.mean_plddt,
            "Composite ▼": c.composite_score,
            "Novelty %": c.novelty_pct,
            "Length": len(c.sequence),
            "Sequence (preview)": c.sequence[:40] + ("…" if len(c.sequence) > 40 else ""),
        }
        if include_esm2:
            row["ESM2 LL"] = c.esm2_score
        rows.append(row)

    col_order = ["Rank", "Composite ▼", "pTM", "pLDDT"]
    if include_esm2:
        col_order.append("ESM2 LL")
    col_order += ["Novelty %", "Length", "Sequence (preview)"]

    return pd.DataFrame(rows)[col_order]
