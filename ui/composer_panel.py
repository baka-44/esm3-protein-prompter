"""
ui/composer_panel.py — Borrowed Bodies "Compose Graft" cockpit (Phase-1 cockpit-lite).

Typed operations (retain / cut) + snap-to-fit pose → compose a graft in a shared 3D frame; a
right-hand panel shows live metrics with tooltips (CC12) and an export-gated download (CC13).
The interactive drag/click 3D editor is Phase 2; here the viewer is view-only and manipulation is
by typed inputs. Exported `.graft` packages import into the RFdiffusion "Borrowed Bodies" preset.
"""

from __future__ import annotations

import streamlit as st


def _parse_ranges(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in (t.strip() for t in s.split(",") if t.strip()):
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(tok), int(tok)))
    return out


def _view_composite(pdb_text: str, repack_tokens: set[str], height: int = 420) -> None:
    """View-only 3D render of the composite (torso = grey, mount = orange, repack = sticks)."""
    try:
        import py3Dmol
        from streamlit.components.v1 import html as _st_html
    except Exception:  # noqa: BLE001
        st.caption("3D viewer unavailable (py3Dmol not installed).")
        return
    view = py3Dmol.view(width=560, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"chain": "A"}, {"cartoon": {"color": "#8a8a8a"}})
    view.setStyle({"chain": "B"}, {"cartoon": {"color": "#e08a2b"}})
    for tok in repack_tokens:
        try:
            view.addStyle({"chain": tok[0], "resi": int(tok[1:])},
                          {"stick": {"colorscheme": "yellowCarbon", "radius": 0.2}})
        except ValueError:
            pass
    view.zoomTo()
    _st_html(view._make_html(), height=height + 10, scrolling=False)


def _view_structure(pdb_bytes: bytes, color: str, height: int = 260) -> None:
    """Single-protein preview (uploaded torso or mount, before composing)."""
    try:
        import py3Dmol
        from streamlit.components.v1 import html as _st_html
    except Exception:  # noqa: BLE001
        return
    view = py3Dmol.view(width=560, height=height)
    view.addModel(pdb_bytes.decode(errors="ignore"), "pdb")
    view.setStyle({}, {"cartoon": {"color": color}})
    view.zoomTo()
    _st_html(view._make_html(), height=height + 10, scrolling=False)


def _chain_info(pdb_bytes: bytes) -> str:
    """Human-readable chain/residue-range summary so users can pick valid cut/keep numbers."""
    from utils.pdb_utils import get_residues
    try:
        residues = get_residues(pdb_bytes, chain_id=None)
    except Exception:  # noqa: BLE001
        return ""
    chains: dict[str, list[int]] = {}
    for r in residues:
        chains.setdefault(r.get_parent().id, []).append(r.id[1])
    return " · ".join(f"chain **{c}**: residues {min(ns)}–{max(ns)} ({len(ns)} aa)"
                      for c, ns in chains.items())


def render_composer(user_email: str) -> None:
    st.subheader("🧩 Compose Graft — Borrowed Bodies")
    st.caption(
        "Graft a catalytic **mount** onto a stable **torso**: keep the mount's functional residues, "
        "cut the torso open, and pose them together. Export a graft package to run through the "
        "RFdiffusion **Borrowed Bodies** preset."
    )

    from proteinredesign.composer import compose_graft
    from proteinredesign.graft import to_engine_params
    from proteinredesign.graft_metrics import compute_metrics, critical_failures

    main, panel = st.columns([2, 1], gap="large")

    with main:
        st.markdown("##### 1 · Torso (the stable body)")
        t_pdb = st.file_uploader("Torso PDB", type=["pdb"], key="cmp_torso")
        if t_pdb:
            info = _chain_info(t_pdb.getvalue())
            if info:
                st.caption(info)
        c1, c2, c3 = st.columns(3)
        with c1:
            t_chain = st.text_input("Chain", placeholder="auto", key="cmp_tchain").strip() or None
        with c2:
            cut1 = st.number_input("Cut point 1", min_value=1, value=15, step=1, key="cmp_cut1")
        with c3:
            cut2 = st.number_input("Cut point 2", min_value=1, value=30, step=1, key="cmp_cut2")
        st.caption("The residues **between** the two cut points are excised → TORSO-1 + TORSO-2 (CC5).")

        st.markdown("##### 2 · Mount (the catalytic insert)")
        m_pdb = st.file_uploader("Mount PDB", type=["pdb"], key="cmp_mount")
        if m_pdb:
            info = _chain_info(m_pdb.getvalue())
            if info:
                st.caption(info)
        mc1, mc2 = st.columns([1, 3])
        with mc1:
            m_chain = st.text_input("Chain", placeholder="auto", key="cmp_mchain").strip() or None
        with mc2:
            keep = st.text_input("Residues to keep (with side chains)",
                                 placeholder="10-18  or  57, 102, 195",
                                 key="cmp_keep",
                                 help="Kept as an all-atom rigid motif (catalytic geometry held).")

        st.markdown("##### 3 · Placement & fan-out")
        repose = st.checkbox(
            "Re-pose the mount (snap-to-fit)", value=True, key="cmp_repose",
            help="ON: snap the mount's termini onto the torso cut ends. Turn OFF when the mount is "
                 "already in the right frame — e.g. reinserting a loop from the SAME PDB — so its "
                 "native geometry is kept (snap-to-fit would displace it).",
        )
        fk, fm = st.columns(2)
        with fk:
            k = st.slider("K — backbones", 1, 10, 4, key="cmp_k")
        with fm:
            m = st.slider("M — sequences per backbone", 1, 10, 2, key="cmp_m")

        composed = None
        if t_pdb and m_pdb and keep.strip():
            try:
                composed = compose_graft(
                    torso_pdb=t_pdb.getvalue(), mount_pdb=m_pdb.getvalue(),
                    torso_cut=(int(cut1), int(cut2)), mount_keep=_parse_ranges(keep),
                    torso_chain=t_chain, mount_chain=m_chain, repose=repose, k=k, m=m,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not compose: {e}")

        if composed is not None:
            frags = " → ".join(s.label for s in composed.spec.fragments())
            st.success(f"Composed (snap-to-fit): **{frags}**  ·  contig "
                       f"`{to_engine_params(composed)['contig']}`")
            # Exactly ONE py3Dmol viewer on the page at a time (multiple conflict → blank).
            choice = st.radio("View", ["Composite", "Torso", "Mount"], horizontal=True,
                              key="cmp_view", label_visibility="collapsed")
            if choice == "Composite":
                repack = {f"{r['chain']}{r['author_num']}" for r in composed.spec.repack_residues}
                _view_composite(composed.composite_pdb.decode(errors="ignore"), repack)
                st.caption("Torso = grey · Mount = orange · Repack shell = yellow sticks (CC9).")
            elif choice == "Torso" and t_pdb:
                _view_structure(t_pdb.getvalue(), "#8a8a8a", height=420)
            elif choice == "Mount" and m_pdb:
                _view_structure(m_pdb.getvalue(), "#e08a2b", height=420)
            st.download_button(
                "⬇️ Download composite PDB", data=composed.composite_pdb,
                file_name="composite.pdb", mime="chemical/x-pdb", key="cmp_dlpdb",
                help="If the 3D view is blank, download and open in PyMOL / ChimeraX.",
            )

    with panel:
        st.markdown("##### Live metrics")
        if composed is None:
            st.info("Upload a torso + mount and list the mount residues to keep. "
                    "Metrics and export appear here.")
            return

        metrics = compute_metrics(composed)
        for mt in metrics:
            flag = "" if mt.ok else " ⚠️"
            tip = f"{mt.what}\n\n{mt.meaning}\n\n**Desired:** {mt.desired}"
            st.metric(mt.label + flag, f"{mt.value:g} {mt.unit}".strip(), help=tip)

        st.divider()
        crit = critical_failures(metrics)
        if crit:
            st.error("Cannot export — fix: " + ", ".join(crit))
            st.button("⬇️ Export graft package", disabled=True, use_container_width=True,
                      key="cmp_export_disabled")
        else:
            soft = [mt.label for mt in metrics if not mt.critical and not mt.ok]
            if soft:
                st.warning("Exportable, but consider: " + ", ".join(soft))
            st.download_button(
                "⬇️ Export graft package", data=composed.to_bytes(),
                file_name="graft_package.graft", mime="application/zip",
                use_container_width=True, key="cmp_export",
                help="Downloads a .graft package — import it in the RFdiffusion Borrowed Bodies preset.",
            )
