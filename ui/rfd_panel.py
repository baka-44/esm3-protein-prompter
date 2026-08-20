"""
ui/rfd_panel.py — RFdiffusion/MPNN engine UI (engine-select first, D3).

Increment 1 exposes preset #1 (fixed-backbone redesign) only; the other MVP
presets appear disabled ("coming soon"). Submits async jobs to the proteinredesign backend
and shows the persistent job dashboard below.
"""

from __future__ import annotations

import streamlit as st

from ui.job_dashboard import render_job_dashboard

# MVP presets (D6). User-facing labels carry no internal preset numbers.
_PRESETS = [
    ("Fixed-backbone redesign", True,
     "Keep the input backbone; redesign the sequence while pinning chosen residues."),
    ("Ligand-aware redesign", True,
     "Redesign a protein around a bound ligand — upload a protein–ligand complex PDB."),
    ("Motif scaffolding", True,
     "Keep chosen blocks of a structure fixed and generate the connecting regions (RF3 inpainting)."),
    ("Enzyme active-site scaffolding", True,
     "Scaffold a new protein around a catalytic site (RF3 all-atom), holding the catalytic geometry."),
    ("Scaffold diversification", True,
     "Generate structurally-diverse variants of a backbone (RF3 partial diffusion), same length and fold."),
]


def render_rfd_engine(user_email: str) -> None:
    st.subheader("🧬 RFdiffusion / MPNN engine")
    st.caption(
        "Backbone-based design: keep or generate a structure, design sequences with the "
        "ProteinMPNN family, and QC every candidate by folding it back (ESMFold). "
        "Jobs run asynchronously — submit, then check the dashboard below."
    )

    preset_label = st.selectbox(
        "Task",
        options=[p[0] for p in _PRESETS],
        format_func=lambda l: l + ("" if dict((p[0], p[1]) for p in _PRESETS)[l] else "  (coming soon)"),
        key="rfd_preset",
    )
    enabled = dict((p[0], p[1]) for p in _PRESETS)[preset_label]
    desc = dict((p[0], p[2]) for p in _PRESETS)[preset_label]
    st.caption(desc)

    if not enabled:
        st.info("This task ships in a later increment.")
    elif preset_label == "Fixed-backbone redesign":
        _render_preset1_form(user_email)
    elif preset_label == "Ligand-aware redesign":
        _render_preset2_form(user_email)
    elif preset_label == "Scaffold diversification":
        _render_preset5_form(user_email)
    elif preset_label == "Enzyme active-site scaffolding":
        _render_preset6_form(user_email)
    elif preset_label == "Motif scaffolding":
        _render_preset3_form(user_email)

    st.divider()
    render_job_dashboard(user_email)
    _footer()


def _render_preset1_form(user_email: str) -> None:
    from proteinredesign.config_builders.preset1 import ConfigError, build_preset1_config
    from proteinredesign.submit import backend_configured, submit_preset1

    uploaded = st.file_uploader(
        "Input PDB", type=["pdb"],
        help="The protein whose backbone you want to keep and redesign the sequence for.",
        key="rfd_p1_pdb",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        fixed_str = st.text_input(
            "Residues to keep fixed",
            placeholder="K67, R82  or  67, 82  (PDB author numbering)",
            help="Residues held unchanged (e.g. functional/binding residues). Everything else is redesigned.",
            key="rfd_p1_fixed",
        )
    with col2:
        chain = st.text_input("Chain", placeholder="auto", key="rfd_p1_chain",
                              help="Leave blank to use the first chain.").strip() or None

    num_outputs = st.slider("Outputs (QC-passed, ranked)", 1, 10, 10, key="rfd_p1_n")

    pdb_bytes = uploaded.read() if uploaded is not None else None

    # Live pre-validation: show the author→sequential mapping and catch errors early.
    if pdb_bytes and fixed_str.strip():
        try:
            cfg = build_preset1_config(pdb_bytes, fixed_str, chain_id=chain)
            st.success(f"Fixed residues: {cfg.mapping_summary}  (chain {cfg.chain_id})")
            for w in cfg.warnings:
                st.warning(w)
        except ConfigError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the PDB: {e}")

    submit_disabled = not (pdb_bytes and fixed_str.strip())
    if st.button("🚀 Submit design job", type="primary", disabled=submit_disabled,
                 use_container_width=True, key="rfd_p1_submit"):
        if not backend_configured():
            st.error("The generation backend isn't configured (GCP project + buckets). "
                     "Deploy the proteinredesign backend first.")
            return
        try:
            rec, cfg = submit_preset1(
                pdb_bytes=pdb_bytes,
                pdb_filename=uploaded.name,
                fixed_residues_str=fixed_str,
                chain_id=chain,
                user_email=user_email,
                num_outputs=num_outputs,
            )
            st.success(f"Submitted job `{rec.job_id}` — it will appear in the dashboard below.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Submission failed: {e}")


def _render_preset2_form(user_email: str) -> None:
    from proteinredesign.config_builders.preset2 import ConfigError, build_preset2_config
    from proteinredesign.submit import backend_configured, submit_preset2
    from utils.pdb_utils import get_hetatm_groups

    uploaded = st.file_uploader(
        "Input PDB (protein–ligand complex)", type=["pdb"],
        help=(
            "A protein bound to the ligand you want to design around (e.g. from "
            "the PDB / X-ray). The ligand's HETATM records give LigandMPNN real "
            "atomic coordinates — no separate ligand file needed."
        ),
        key="rfd_p2_pdb",
    )
    pdb_bytes = uploaded.read() if uploaded is not None else None

    ligand_key = None
    if pdb_bytes:
        try:
            candidates = get_hetatm_groups(pdb_bytes)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the PDB: {e}")
            candidates = []

        if not candidates:
            st.warning(
                "No ligand-like HETATM groups found (after excluding waters, ions, and "
                "common crystallization additives). If this protein has no ligand to "
                "condition on, use **Fixed-backbone redesign** instead."
            )
        else:
            labels = [c.label() for c in candidates]
            picked = st.selectbox("Ligand to condition on", options=labels, key="rfd_p2_ligand")
            ligand_key = candidates[labels.index(picked)].key

    col1, col2 = st.columns([3, 1])
    with col1:
        fixed_str = st.text_input(
            "Residues to keep fixed (optional)",
            placeholder="K67, R82  or  67, 82  (PDB author numbering)",
            help=(
                "Optional — the ligand's atomic context already guides design of the "
                "pocket. Add specific residues here only if you also want them pinned."
            ),
            key="rfd_p2_fixed",
        )
    with col2:
        chain = st.text_input("Chain", placeholder="auto", key="rfd_p2_chain",
                              help="Protein chain to redesign. Leave blank to use the first chain.").strip() or None

    num_outputs = st.slider("Outputs (QC-passed, ranked)", 1, 10, 10, key="rfd_p2_n")

    # Live pre-validation: show the ligand + any fixed-residue mapping, catch errors early.
    cfg = None
    if pdb_bytes and ligand_key:
        try:
            cfg = build_preset2_config(pdb_bytes, ligand_key, fixed_str, chain_id=chain)
            summary = f"Ligand: **{cfg.ligand.label()}**  ·  chain {cfg.chain_id}"
            if cfg.mapping_summary:
                summary += f"  ·  Fixed: {cfg.mapping_summary}"
            st.success(summary)
            for w in cfg.warnings:
                st.warning(w)
        except ConfigError as e:
            st.error(str(e))

    submit_disabled = cfg is None
    if st.button("🚀 Submit design job", type="primary", disabled=submit_disabled,
                 use_container_width=True, key="rfd_p2_submit"):
        if not backend_configured():
            st.error("The generation backend isn't configured (GCP project + buckets). "
                     "Deploy the proteinredesign backend first.")
            return
        try:
            rec, cfg = submit_preset2(
                pdb_bytes=pdb_bytes,
                pdb_filename=uploaded.name,
                ligand_key=ligand_key,
                fixed_residues_str=fixed_str,
                chain_id=chain,
                user_email=user_email,
                num_outputs=num_outputs,
            )
            st.success(f"Submitted job `{rec.job_id}` — it will appear in the dashboard below.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Submission failed: {e}")


def _render_preset5_form(user_email: str) -> None:
    from proteinredesign.config_builders.preset5 import (
        ConfigError, K_MAX, M_MAX, PARTIAL_T_MAX, PARTIAL_T_MIN, build_preset5_config,
    )
    from proteinredesign.submit import backend_configured, submit_preset5

    uploaded = st.file_uploader(
        "Input PDB", type=["pdb"],
        help="The backbone to diversify. RF3 keeps its length and overall fold, "
             "generating structurally-varied relatives.",
        key="rfd_p5_pdb",
    )
    chain = st.text_input("Chain", placeholder="auto", key="rfd_p5_chain",
                          help="Chain to diversify. Leave blank to use the first chain.").strip() or None

    col_k, col_m = st.columns(2)
    with col_k:
        k = st.slider("K — backbones (RF3 variants)", 1, K_MAX, 5, key="rfd_p5_k",
                      help="How many diverse backbones RF3 generates by partial diffusion.")
    with col_m:
        m = st.slider("M — sequences per backbone", 1, M_MAX, 3, key="rfd_p5_m",
                      help="ProteinMPNN designs this many sequences on each backbone.")

    partial_t = st.slider(
        "Diversity — partial-diffusion noise (Å)",
        PARTIAL_T_MIN, PARTIAL_T_MAX, 8.0, step=0.5, key="rfd_p5_pt",
        help="RF3 partial_t: more Å → more divergence from the input (5–15 Å typical). "
             "Start small for tight variants.",
    )

    # Compute budget warning (D10.5) — every one of K×M designs is folded by ESMFold
    # (the slow QC stage), so surface an estimate when the fan-out is large, mirroring
    # the ESM3 app's >20-candidate warning.
    total = k * m
    st.caption(f"Total designs: **{k}×{m} = {total}** (each folded by ESMFold for QC, then ranked).")
    if total > 40:
        st.warning(
            f"{total} designs is a large run — RF3 + {total} ESMFold folds may take many "
            f"minutes and could approach the job time limit for larger proteins. Consider "
            f"lowering K or M. Keep the tab open; results appear in the dashboard below."
        )

    pdb_bytes = uploaded.read() if uploaded is not None else None

    cfg = None
    if pdb_bytes:
        try:
            cfg = build_preset5_config(pdb_bytes, partial_t, k, m, chain_id=chain)
            st.success(
                f"Diversifying chain **{cfg.chain_id}** ({cfg.length} aa) · contig `{cfg.contig}` · "
                f"{cfg.k}×{cfg.m} designs · {cfg.partial_t:g} Å"
            )
            for w in cfg.warnings:
                st.warning(w)
        except ConfigError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the PDB: {e}")

    if st.button("🚀 Submit diversification job", type="primary", disabled=cfg is None,
                 use_container_width=True, key="rfd_p5_submit"):
        if not backend_configured():
            st.error("The generation backend isn't configured (GCP project + buckets). "
                     "Deploy the proteinredesign backend first.")
            return
        try:
            rec, cfg = submit_preset5(
                pdb_bytes=pdb_bytes,
                pdb_filename=uploaded.name,
                partial_t=partial_t,
                k=k,
                m=m,
                chain_id=chain,
                user_email=user_email,
            )
            st.success(f"Submitted job `{rec.job_id}` — it will appear in the dashboard below.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Submission failed: {e}")


def _render_preset3_form(user_email: str) -> None:
    from proteinredesign.config_builders.preset3 import (
        ConfigError, K_MAX, M_MAX, build_preset3_config,
    )
    from proteinredesign.submit import backend_configured, submit_preset3

    uploaded = st.file_uploader(
        "Input PDB", type=["pdb"],
        help="The structure whose chosen blocks you want to keep fixed while the connecting "
             "regions are regenerated.",
        key="rfd_p3_pdb",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        keep_str = st.text_input(
            "Blocks to keep (fixed)",
            placeholder="1-20, 50-80, 130-160, 190-200  (PDB author numbering)",
            help="Discontiguous blocks held at their exact coordinates. RF3 regenerates the "
                 "gaps between them (original gap lengths, so total length is preserved).",
            key="rfd_p3_keep",
        )
    with col2:
        chain = st.text_input("Chain", placeholder="auto", key="rfd_p3_chain",
                              help="Chain the blocks are on.").strip() or None

    col_k, col_m = st.columns(2)
    with col_k:
        k = st.slider("K — backbones", 1, K_MAX, 5, key="rfd_p3_k",
                      help="How many RF3 structures to generate (varied bridges).")
    with col_m:
        m = st.slider("M — sequences per backbone", 1, M_MAX, 3, key="rfd_p3_m")

    pdb_bytes = uploaded.read() if uploaded is not None else None

    cfg = None
    if pdb_bytes and keep_str.strip():
        try:
            cfg = build_preset3_config(pdb_bytes, keep_str, k, m, chain_id=chain)
            blocks = ", ".join(f"{a}-{b}" for a, b in cfg.keep_ranges)
            gaps = ", ".join(str(g) for g in cfg.gaps)
            st.success(f"Chain **{cfg.chain_id}** · keep [{blocks}] · generate bridges [{gaps}] · "
                       f"contig `{cfg.contig}`")
            for w in cfg.warnings:
                st.warning(w)
        except ConfigError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the PDB: {e}")

    total = k * m
    st.caption(f"Total designs: **{k}×{m} = {total}** (each folded by ESMFold, motif-RMSD checked, ranked).")
    if total > 40:
        st.warning(f"{total} designs is a large run — consider lowering K or M. Keep the tab open.")

    if st.button("🚀 Submit inpainting job", type="primary", disabled=cfg is None,
                 use_container_width=True, key="rfd_p3_submit"):
        if not backend_configured():
            st.error("The generation backend isn't configured (GCP project + buckets).")
            return
        try:
            rec, cfg = submit_preset3(
                pdb_bytes=pdb_bytes, pdb_filename=uploaded.name,
                keep_ranges_str=keep_str, k=k, m=m, chain_id=chain, user_email=user_email,
            )
            st.success(f"Submitted job `{rec.job_id}` — it will appear in the dashboard below.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Submission failed: {e}")


def _render_preset6_form(user_email: str) -> None:
    from proteinredesign.config_builders.preset6 import (
        ConfigError, FIXED_ATOM_MODES, K_MAX, LENGTH_MAX, LENGTH_MIN, M_MAX,
        build_preset6_config,
    )
    from proteinredesign.submit import backend_configured, submit_preset6
    from utils.pdb_utils import get_hetatm_groups

    uploaded = st.file_uploader(
        "Parent enzyme PDB", type=["pdb"],
        help="The enzyme whose catalytic site you want to transplant onto a new scaffold. "
             "Include the cofactor as HETATM if the site needs one.",
        key="rfd_p6_pdb",
    )
    pdb_bytes = uploaded.read() if uploaded is not None else None

    col1, col2 = st.columns([3, 1])
    with col1:
        catalytic_str = st.text_input(
            "Catalytic residues",
            placeholder="H57, D102, S195  (PDB author numbering)",
            help="Active-site residues to preserve. Their geometry is held while RF3 builds a "
                 "new body around them.",
            key="rfd_p6_cat",
        )
    with col2:
        chain = st.text_input("Chain", placeholder="auto", key="rfd_p6_chain",
                              help="Chain the catalytic residues are on.").strip() or None

    col_a, col_b = st.columns(2)
    with col_a:
        atoms_mode = st.selectbox(
            "Fixed atoms", options=list(FIXED_ATOM_MODES), index=0, key="rfd_p6_atoms",
            help="TIP = catalytic tip atoms (tightest hold, most freedom); "
                 "BKBN = backbone; ALL = every atom.",
        )
    with col_b:
        pass

    # Optional cofactor picker (same HETATM detect/confirm as ligand-aware redesign).
    ligand_key = None
    if pdb_bytes:
        try:
            candidates = get_hetatm_groups(pdb_bytes)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the PDB: {e}")
            candidates = []
        if candidates:
            labels = ["(none)"] + [c.label() for c in candidates]
            picked = st.selectbox("Cofactor / ligand (optional)", options=labels, key="rfd_p6_lig")
            if picked != "(none)":
                ligand_key = candidates[labels.index(picked) - 1].key

    col_l, col_r = st.columns(2)
    with col_l:
        length = st.slider("Scaffold length (aa)", LENGTH_MIN, LENGTH_MAX, (140, 200),
                           key="rfd_p6_len",
                           help="RF3 builds a new body of this length range around the site.")
    with col_r:
        colk, colm = st.columns(2)
        with colk:
            k = st.slider("K — scaffolds", 1, K_MAX, 5, key="rfd_p6_k")
        with colm:
            m = st.slider("M — seqs/scaffold", 1, M_MAX, 3, key="rfd_p6_m")

    total = k * m
    st.caption(f"Total designs: **{k}×{m} = {total}** (each folded by ESMFold, motif-RMSD checked, ranked).")
    if total > 40:
        st.warning(
            f"{total} designs is a large run — RF3 all-atom + {total} ESMFold folds may take many "
            f"minutes. Consider lowering K or M. Keep the tab open; results appear below."
        )

    cfg = None
    if pdb_bytes and catalytic_str.strip():
        try:
            cfg = build_preset6_config(
                pdb_bytes, catalytic_str, fixed_atoms_mode=atoms_mode, ligand_key=ligand_key,
                length_min=length[0], length_max=length[1], k=k, m=m, chain_id=chain,
            )
            lig = f"  ·  cofactor **{cfg.ligand.resname}**" if cfg.ligand is not None else ""
            st.success(f"Catalytic: {cfg.mapping_summary}  ·  chain {cfg.chain_id}{lig}  ·  "
                       f"{cfg.length_min}-{cfg.length_max} aa")
            for w in cfg.warnings:
                st.warning(w)
        except ConfigError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the PDB: {e}")

    if st.button("🚀 Submit scaffolding job", type="primary", disabled=cfg is None,
                 use_container_width=True, key="rfd_p6_submit"):
        if not backend_configured():
            st.error("The generation backend isn't configured (GCP project + buckets). "
                     "Deploy the proteinredesign backend first.")
            return
        try:
            rec, cfg = submit_preset6(
                pdb_bytes=pdb_bytes, pdb_filename=uploaded.name,
                catalytic_residues_str=catalytic_str, fixed_atoms_mode=atoms_mode,
                ligand_key=ligand_key, length_min=length[0], length_max=length[1],
                k=k, m=m, chain_id=chain, user_email=user_email,
            )
            st.success(f"Submitted job `{rec.job_id}` — it will appear in the dashboard below.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Submission failed: {e}")


def _footer() -> None:
    st.divider()
    st.caption(
        "Models: **RFdiffusion3** (CC BY 4.0) · **ProteinMPNN / LigandMPNN** (MIT) · "
        "**ESMFold** (MIT). RFdiffusion3 is used under CC BY 4.0 — attribution to the "
        "Institute for Protein Design / Baker Lab."
    )
