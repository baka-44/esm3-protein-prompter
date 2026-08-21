"""
ui/composer_panel.py — Borrowed Bodies "Compose Graft" cockpit (Phase-1 cockpit-lite + Phase-2
manual pose).

Layout: a thin top header (grid toggle · engine switch · sign out), a collapsible LEFT inputs
panel (torso/mount/cut/keep/fan-out), a wide black 3D canvas in the centre, and a collapsible
RIGHT panel — pose sliders (top) + live metrics & export (bottom). The viewer is view-only
(drag-in-3D is the Phase-2 custom component); manipulation is by typed inputs + pose sliders.
Exported `.graft` packages import into the RFdiffusion "Borrowed Bodies" preset.
"""

from __future__ import annotations

import streamlit as st

_BG = "0x0d0d0d"  # near-black canvas


def _parse_ranges(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in (t.strip() for t in s.split(",") if t.strip()):
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(tok), int(tok)))
    return out


def _chain_info(pdb_bytes: bytes) -> str:
    from utils.pdb_utils import get_residues
    try:
        residues = get_residues(pdb_bytes, chain_id=None)
    except Exception:  # noqa: BLE001
        return ""
    chains: dict[str, list[int]] = {}
    for r in residues:
        chains.setdefault(r.get_parent().id, []).append(r.id[1])
    return " · ".join(f"chain **{c}**: {min(ns)}–{max(ns)} ({len(ns)} aa)" for c, ns in chains.items())


def _grid_lines(view, pdb_text: str) -> None:
    """A thin mild-grey reference grid in the XZ plane under the structure."""
    try:
        xs, ys, zs = [], [], []
        for line in pdb_text.splitlines():
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                xs.append(float(line[30:38])); ys.append(float(line[38:46])); zs.append(float(line[46:54]))
        if not xs:
            return
        cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
        span = max(max(xs) - min(xs), max(zs) - min(zs), 20.0) / 2 + 15.0
        y0, step = min(ys) - 5.0, 8.0
        n = int(span // step)
        for i in range(-n, n + 1):
            off = i * step
            view.addLine({"start": {"x": cx - span, "y": y0, "z": cz + off},
                          "end": {"x": cx + span, "y": y0, "z": cz + off}, "color": "0x333333"})
            view.addLine({"start": {"x": cx + off, "y": y0, "z": cz - span},
                          "end": {"x": cx + off, "y": y0, "z": cz + span}, "color": "0x333333"})
    except Exception:  # noqa: BLE001
        pass


def _view(pdb_text: str, *, repack_tokens: set[str] | None = None, single_color: str | None = None,
          grid: bool = True, height: int = 640) -> None:
    try:
        import py3Dmol
        from streamlit.components.v1 import html as _st_html
    except Exception:  # noqa: BLE001
        st.caption("3D viewer unavailable (py3Dmol not installed).")
        return
    view = py3Dmol.view(width=1050, height=height)
    view.setBackgroundColor(_BG)
    view.addModel(pdb_text, "pdb")
    if single_color:
        view.setStyle({}, {"cartoon": {"color": single_color}})
    else:
        view.setStyle({"chain": "A"}, {"cartoon": {"color": "#9aa0a6"}})
        view.setStyle({"chain": "B"}, {"cartoon": {"color": "#e08a2b"}})
        for tok in (repack_tokens or set()):
            try:
                view.addStyle({"chain": tok[0], "resi": int(tok[1:])},
                              {"stick": {"colorscheme": "yellowCarbon", "radius": 0.2}})
            except ValueError:
                pass
    if grid:
        _grid_lines(view, pdb_text)
    view.zoomTo()
    _st_html(view._make_html(), height=height + 10, scrolling=False)


def _handle_component_event(ev) -> None:
    """Dispatch a value from the 3D component: a residue pick (M1) or a drag-pose delta (M2)."""
    if not isinstance(ev, dict):
        return
    kind = ev.get("kind")
    if kind == "pose":
        # Ignore already-applied events (the component returns the same value until a new drag).
        if st.session_state.get("_cmp_pose_ts") == ev.get("ts"):
            return
        st.session_state["_cmp_pose_ts"] = ev.get("ts")
        dn, dr = ev.get("dnudge", [0, 0, 0]), ev.get("drotate", [0, 0, 0])
        for k, v in zip(("cmp_dnx", "cmp_dny", "cmp_dnz"), dn):
            st.session_state[k] = _clamp(st.session_state.get(k, 0.0) + float(v), -60.0, 60.0)
        for k, v in zip(("cmp_drx", "cmp_dry", "cmp_drz"), dr):
            st.session_state[k] = _clamp(st.session_state.get(k, 0.0) + float(v), -360.0, 360.0)
        st.rerun()
    else:
        _apply_pick(ev)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _apply_pick(pick) -> None:
    """Show the last picked residue (from the interactive viewer) + apply it to cut/keep (M1)."""
    if not isinstance(pick, dict) or pick.get("resi") is None:
        return
    chain, resi, resn = pick.get("chain"), int(pick["resi"]), pick.get("resn")
    st.caption(f"Picked **{chain}{resi}** ({resn}) — apply to:")
    b1, b2, b3 = st.columns(3)
    if b1.button("→ Cut 1", key="cmp_pk_c1", use_container_width=True):
        st.session_state["cmp_cut1"] = resi
        st.rerun()
    if b2.button("→ Cut 2", key="cmp_pk_c2", use_container_width=True):
        st.session_state["cmp_cut2"] = resi
        st.rerun()
    if b3.button("+ Keep", key="cmp_pk_keep", use_container_width=True):
        cur = st.session_state.get("cmp_keep", "").strip()
        st.session_state["cmp_keep"] = f"{cur}, {resi}" if cur else str(resi)
        st.rerun()


def render_composer(user_email: str) -> None:
    from ui.mol_component import mol_viewer
    st.session_state.setdefault("cmp_grid", True)

    # ── thin top header (nav bar) ──
    h1, h2, h3, h4 = st.columns([6.0, 1.3, 1.4, 1.4])
    with h1:
        st.markdown("🧩 **Compose Graft** — Borrowed Bodies")
    with h2:
        if st.button(("▦ Grid" if st.session_state["cmp_grid"] else "▧ Grid"),
                     key="cmp_grid_btn", use_container_width=True, help="Toggle the canvas grid."):
            st.session_state["cmp_grid"] = not st.session_state["cmp_grid"]
            st.rerun()
    with h3:
        if st.button("⇄ Engine", key="cmp_switch", use_container_width=True):
            st.session_state.pop("_engine", None)
            st.rerun()
    with h4:
        if st.button("Sign out", key="cmp_signout", use_container_width=True):
            st.session_state.pop("_auth_email", None)
            st.session_state.pop("_auth_name", None)
            st.rerun()
    st.divider()

    from proteinredesign.composer import compose_graft
    from proteinredesign.graft import to_engine_params
    from proteinredesign.graft_metrics import compute_metrics, critical_failures

    left, centre, right = st.columns([2.0, 7.0, 2.4], gap="medium")

    # ── LEFT · inputs (collapsible; expander body always runs → values persist) ──
    with left:
        with st.expander("◤ Inputs", expanded=True):
            st.markdown("**Torso** (stable body)")
            t_pdb = st.file_uploader("Torso PDB", type=["pdb"], key="cmp_torso", label_visibility="collapsed")
            if t_pdb:
                st.caption(_chain_info(t_pdb.getvalue()))
            t_chain = st.text_input("Torso chain", placeholder="auto", key="cmp_tchain").strip() or None
            cc1, cc2 = st.columns(2)
            with cc1:
                cut1 = st.number_input("Cut 1", min_value=1, value=15, step=1, key="cmp_cut1")
            with cc2:
                cut2 = st.number_input("Cut 2", min_value=1, value=30, step=1, key="cmp_cut2")

            st.markdown("**Mount** (catalytic insert)")
            m_pdb = st.file_uploader("Mount PDB", type=["pdb"], key="cmp_mount", label_visibility="collapsed")
            if m_pdb:
                st.caption(_chain_info(m_pdb.getvalue()))
            m_chain = st.text_input("Mount chain", placeholder="auto", key="cmp_mchain").strip() or None
            keep = st.text_input("Residues to keep", placeholder="10-18  or  57,102,195", key="cmp_keep")
            repose = st.checkbox("Re-pose (snap-to-fit)", value=True, key="cmp_repose",
                                 help="OFF keeps the mount's input coords (same-PDB reinsertion).")
            kk, mm = st.columns(2)
            with kk:
                k = st.slider("K", 1, 10, 4, key="cmp_k")
            with mm:
                m = st.slider("M", 1, 10, 2, key="cmp_m")

    # Drag-to-pose accumulator (plain session keys, NOT widget keys — the 3D canvas drag adds to
    # these; the sliders are an additional fine offset. Kept separate to avoid Streamlit's
    # "can't modify a widget's state after it's created" error).
    for _dk in ("cmp_dnx", "cmp_dny", "cmp_dnz", "cmp_drx", "cmp_dry", "cmp_drz"):
        st.session_state.setdefault(_dk, 0.0)
    dnx, dny, dnz = (st.session_state[k] for k in ("cmp_dnx", "cmp_dny", "cmp_dnz"))
    drx, dry, drz = (st.session_state[k] for k in ("cmp_drx", "cmp_dry", "cmp_drz"))

    # ── RIGHT (top) · pose — drag on the canvas (Move/Rotate) or fine-tune here ──
    with right:
        with st.expander("◥ Pose — drag on canvas or nudge", expanded=True):
            if any(abs(v) > 1e-6 for v in (dnx, dny, dnz, drx, dry, drz)):
                st.caption(f"drag: ({dnx:+.1f}, {dny:+.1f}, {dnz:+.1f}) Å · "
                           f"({drx:+.0f}, {dry:+.0f}, {drz:+.0f})°")
                if st.button("↺ Reset drag", key="cmp_reset_drag", use_container_width=True):
                    for _dk in ("cmp_dnx", "cmp_dny", "cmp_dnz", "cmp_drx", "cmp_dry", "cmp_drz"):
                        st.session_state[_dk] = 0.0
                    st.rerun()
            tx = st.slider("Move X (Å)", -25.0, 25.0, 0.0, 0.5, key="cmp_tx")
            ty = st.slider("Move Y (Å)", -25.0, 25.0, 0.0, 0.5, key="cmp_ty")
            tz = st.slider("Move Z (Å)", -25.0, 25.0, 0.0, 0.5, key="cmp_tz")
            rx = st.slider("Rotate X (°)", -180, 180, 0, 5, key="cmp_rx")
            ry = st.slider("Rotate Y (°)", -180, 180, 0, 5, key="cmp_ry")
            rz = st.slider("Rotate Z (°)", -180, 180, 0, 5, key="cmp_rz")

    composed, err = None, None
    if t_pdb and m_pdb and keep.strip():
        try:
            composed = compose_graft(
                torso_pdb=t_pdb.getvalue(), mount_pdb=m_pdb.getvalue(),
                torso_cut=(int(cut1), int(cut2)), mount_keep=_parse_ranges(keep),
                torso_chain=t_chain, mount_chain=m_chain, repose=repose,
                nudge=(tx + dnx, ty + dny, tz + dnz),
                rotate=(float(rx) + drx, float(ry) + dry, float(rz) + drz), k=k, m=m,
            )
        except Exception as e:  # noqa: BLE001
            err = str(e)

    # ── CENTRE · the canvas (max space) ──
    with centre:
        if composed is not None:
            frags = " → ".join(s.label for s in composed.spec.fragments())
            st.caption(f"{frags} · contig `{to_engine_params(composed)['contig']}` · "
                       "torso grey · mount orange · repack sticks · **click a residue to pick it**")
            repack = [f"{r['chain']}{r['author_num']}" for r in composed.spec.repack_residues]
            pick = mol_viewer(composed.composite_pdb.decode(errors="ignore"), repack=repack, key="cmp_mol")
            _handle_component_event(pick)
        elif t_pdb or m_pdb:
            if err:
                st.error(f"Could not compose: {err}")
            src = t_pdb or m_pdb
            st.caption(("Torso" if t_pdb else "Mount") + " · **click a residue to pick it** "
                       "(→ set cut points / add to keep)")
            pick = mol_viewer(src.getvalue().decode(errors="ignore"), key="cmp_mol")
            _handle_component_event(pick)
        else:
            st.info("In the **Inputs** panel: upload a torso + mount, set the cut points and the "
                    "mount residues to keep — the composite appears here.")

    # ── RIGHT (bottom) · metrics & export ──
    with right:
        with st.expander("◢ Metrics & export", expanded=True):
            if composed is None:
                st.caption("Compose a graft to see metrics + export.")
            else:
                for mt in compute_metrics(composed):
                    flag = "" if mt.ok else " ⚠️"
                    st.metric(mt.label + flag, f"{mt.value:g} {mt.unit}".strip(),
                              help=f"{mt.what}\n\n{mt.meaning}\n\n**Desired:** {mt.desired}")
                if critical_failures(compute_metrics(composed)):
                    st.error("Can't export — resolve the ⚠️ metrics.")
                    st.button("⬇️ Export graft package", disabled=True, use_container_width=True, key="cmp_exp0")
                else:
                    st.download_button("⬇️ Export graft package", data=composed.to_bytes(),
                                       file_name="graft_package.graft", mime="application/zip",
                                       use_container_width=True, key="cmp_exp")
                st.download_button("⬇️ Composite PDB", data=composed.composite_pdb,
                                   file_name="composite.pdb", mime="chemical/x-pdb",
                                   use_container_width=True, key="cmp_dlpdb")
