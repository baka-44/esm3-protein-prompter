"""
ui/rfd_panel.py — RFdiffusion/MPNN engine UI (engine-select first, D3).

Increment 1 exposes preset #1 (fixed-backbone redesign) only; the other MVP
presets appear disabled ("coming soon"). Submits async jobs to the cofold backend
and shows the persistent job dashboard below.
"""

from __future__ import annotations

import streamlit as st

from ui.job_dashboard import render_job_dashboard

# MVP presets (D6). Only #1 is enabled in increment 1.
_PRESETS = [
    ("#1 Fixed-backbone redesign", True,
     "Keep the input backbone; redesign the sequence while pinning chosen residues."),
    ("#2 Ligand-aware redesign", False, "Redesign around a bound ligand (LigandMPNN). Coming soon."),
    ("#3 Motif scaffolding", False, "Build a new protein around a fixed motif (RFdiffusion3). Coming soon."),
    ("#6 Enzyme active-site scaffolding", False, "Scaffold an enzyme around a catalytic site (RF3 all-atom). Coming soon."),
    ("#8 Scaffold diversification", False, "Generate variants of a structure (partial diffusion). Coming soon."),
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
        st.info("This task ships in a later increment. Preset #1 is available now.")
    else:
        _render_preset1_form(user_email)

    st.divider()
    render_job_dashboard(user_email)
    _footer()


def _render_preset1_form(user_email: str) -> None:
    from cofold.config_builders.preset1 import ConfigError, build_preset1_config
    from cofold.submit import backend_configured, submit_preset1

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
                     "Deploy the cofold backend first.")
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


def _footer() -> None:
    st.divider()
    st.caption(
        "Models: **RFdiffusion3** (CC BY 4.0) · **ProteinMPNN / LigandMPNN** (MIT) · "
        "**ESMFold** (MIT). RFdiffusion3 is used under CC BY 4.0 — attribution to the "
        "Institute for Protein Design / Baker Lab."
    )
