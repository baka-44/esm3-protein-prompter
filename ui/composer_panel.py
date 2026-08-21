"""
ui/composer_panel.py — Borrowed Bodies "Compose Graft" cockpit (full-screen · Phase-2 live pose).

Layout: a compact top nav (logo · title · engine switch · sign out), ALL controls in one
collapsible LEFT sidebar (inputs + K/M + metrics + export), and a full-bleed 3D canvas. The
mount is posed *live* on the canvas (rigid client-side transform, no server round-trip); the
released transform feeds compose_graft for metrics + export. Exported `.graft` packages import
into the RFdiffusion "Borrowed Bodies" preset.
"""

from __future__ import annotations

import hashlib
import time

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
    """Dispatch a value from the 3D canvas: a residue pick (Camera mode) or a cumulative rigid
    mount transform (Pose mode). The transform is stored and fed to compose_graft for metrics +
    export; the live view is already correct client-side, so we don't re-render the structure."""
    if not isinstance(ev, dict):
        return
    kind = ev.get("kind")
    if kind == "pose_xform":
        if st.session_state.get("_cmp_pose_ts") == ev.get("ts"):
            return   # same value returned until the next gesture — apply once
        st.session_state["_cmp_pose_ts"] = ev.get("ts")
        st.session_state["cmp_xform"] = {
            "rot": [float(x) for x in ev.get("rot", [1, 0, 0, 0, 1, 0, 0, 0, 1])],
            "tran": [float(x) for x in ev.get("tran", [0, 0, 0])],
            "about": [float(x) for x in ev.get("about", [0, 0, 0])],
        }
        st.rerun()
    elif kind == "cut_span":
        # scissor-by-click (M3): two torso clicks set the excision span. Stash in PENDING keys —
        # the Cut 1/Cut 2 number_inputs already rendered this run, so we apply before they render
        # next run (see the pending block at the top of render_composer).
        if st.session_state.get("_cmp_cut_ts") == ev.get("ts"):
            return
        st.session_state["_cmp_cut_ts"] = ev.get("ts")
        st.session_state["_pending_cut1"] = int(ev["cut1"])
        st.session_state["_pending_cut2"] = int(ev["cut2"])
        st.rerun()
    else:
        _apply_pick(ev)


def _connection_points(base) -> list[dict]:
    """Exit vectors (M3): for every linker junction, the two OPEN ENDS it will bridge — a fixed
    fragment's C-end and the next fragment's N-start. Each end carries its CA position and the
    OUTWARD backbone tangent (the direction the excised loop was flowing), in the base-composite
    frame. Ends on the mount chain ("B") are flagged `m` so the canvas rotates/moves them with the
    live pose; the canvas then glows the pair by how well the two arrows point at each other."""
    from proteinredesign.graft import Fragment, Linker
    from utils.pdb_utils import get_residues
    ca: dict[tuple[str, int], list[float]] = {}
    for r in get_residues(base.composite_pdb, chain_id=None):
        if "CA" in r:
            ca[(r.get_parent().id, r.id[1])] = [float(x) for x in r["CA"].coord]

    def end(chain: str, resi: int, neighbor: int, is_mount: bool):
        """CA at `resi` + outward tangent (CA[resi] - CA[neighbor], pointing away from the body)."""
        p, q = ca.get((chain, resi)), ca.get((chain, neighbor))
        if p is None:
            return None
        d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]] if q is not None else [0.0, 0.0, 0.0]
        return {"pos": p, "dir": d, "m": is_mount}

    order = base.spec.chain_order
    pairs: list[dict] = []
    for i, seg in enumerate(order):
        if isinstance(seg, Linker) and 0 < i < len(order) - 1:
            prev, nxt = order[i - 1], order[i + 1]
            if isinstance(prev, Fragment) and isinstance(nxt, Fragment):
                a = end(prev.chain, prev.end, prev.end - 1, prev.chain == "B")     # C-side end
                b = end(nxt.chain, nxt.start, nxt.start + 1, nxt.chain == "B")     # N-side start
                if a and b:
                    pairs.append({"a": a, "b": b})
    return pairs


def _is_identity_xform(x: dict | None) -> bool:
    if not x:
        return True
    I = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    return (all(abs(a - b) < 1e-6 for a, b in zip(x.get("rot", I), I))
            and all(abs(v) < 1e-6 for v in x.get("tran", [0, 0, 0])))


def _apply_pick(pick) -> None:
    """Show the last picked residue (from the interactive viewer) + apply it to cut/keep (M1)."""
    if not isinstance(pick, dict) or pick.get("resi") is None:
        return
    chain, resi, resn = pick.get("chain"), int(pick["resi"]), pick.get("resn")
    st.caption(f"Picked **{chain}{resi}** ({resn}) — apply to:")
    b1, b2, b3 = st.columns(3)
    # Route through PENDING keys (applied before the widgets render next run) — Streamlit forbids
    # mutating a widget's key after its widget has already been instantiated this run.
    if b1.button("→ Cut 1", key="cmp_pk_c1", use_container_width=True):
        st.session_state["_pending_cut1"] = resi
        st.rerun()
    if b2.button("→ Cut 2", key="cmp_pk_c2", use_container_width=True):
        st.session_state["_pending_cut2"] = resi
        st.rerun()
    if b3.button("+ Keep", key="cmp_pk_keep", use_container_width=True):
        cur = st.session_state.get("cmp_keep", "").strip()
        st.session_state["_pending_keep"] = f"{cur}, {resi}" if cur else str(resi)
        st.rerun()


import base64 as _b64
import functools
import os


@functools.lru_cache(maxsize=1)
def _logo_data_uri() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "phyx44_logo.png")
    try:
        with open(path, "rb") as fh:
            return "data:image/png;base64," + _b64.b64encode(fh.read()).decode()
    except Exception:  # noqa: BLE001
        return ""


_FULLBLEED_CSS = """
<style>
  /* Hide the big global PHYX44 banner — we render a compact top nav instead. */
  .phyx-banner { display: none !important; }
  /* Streamlit's header overlays the top of the page and was clipping our nav row. Keep it thin
     + transparent (it still holds the sidebar >> expand control) and pad content down to clear it. */
  header[data-testid="stHeader"] { height: 2.6rem !important; background: transparent !important; }
  /* Edge-to-edge canvas: drop Streamlit's centered max-width; pad the top past the header. */
  .main .block-container, .block-container {
      max-width: 100% !important; padding: 3rem 0.9rem 0 !important; }
  /* Tighten the sidebar so it reads as a control rail, and let it collapse to nothing. */
  section[data-testid="stSidebar"] { width: 340px !important; }
  section[data-testid="stSidebar"] .block-container { padding-top: 0.6rem !important; }

  /* ── compact top nav ── */
  .cmp-nav { display: flex; align-items: center; gap: 14px; height: 34px; }
  .cmp-nav img { height: 26px; width: auto; }
  .cmp-nav .t { font-weight: 600; font-size: 0.9rem; color: #111; letter-spacing: -0.01em;
                white-space: nowrap; }
  .cmp-nav .d { font-size: 0.72rem; color: #8a8a8a; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; }
  /* keep the nav-row buttons small + right-aligned */
  div[data-testid="stHorizontalBlock"]:has(#cmp-nav-anchor) button { padding: 2px 10px; min-height: 30px; }
</style>
"""


def render_composer(user_email: str) -> None:
    from ui.mol_component import mol_viewer
    from proteinredesign.composer import compose_graft
    from proteinredesign.graft import to_engine_params
    from proteinredesign.graft_metrics import compute_metrics, critical_failures

    st.markdown(_FULLBLEED_CSS, unsafe_allow_html=True)

    # Apply residue picks / scissor cuts staged by the 3D canvas BEFORE the Cut/Keep widgets
    # instantiate below (Streamlit forbids mutating a widget's key after its widget renders).
    for pend, widget in (("_pending_cut1", "cmp_cut1"), ("_pending_cut2", "cmp_cut2"),
                         ("_pending_keep", "cmp_keep")):
        if pend in st.session_state:
            st.session_state[widget] = st.session_state.pop(pend)

    # ── compact top nav: logo + title/description on the left, actions on the right ──
    nav_l, nav_r = st.columns([7, 2.2], gap="small")
    with nav_l:
        st.markdown(
            f'<div class="cmp-nav"><span id="cmp-nav-anchor"></span>'
            f'<img src="{_logo_data_uri()}" alt="PHYX44"/>'
            f'<span class="t">Compose Graft — Borrowed Bodies</span>'
            f'<span class="d">graft a catalytic mount onto a stable torso</span></div>',
            unsafe_allow_html=True)
    with nav_r:
        b1, b2 = st.columns(2)
        if b1.button("⇄ Engine", key="cmp_switch", use_container_width=True):
            st.session_state.pop("_engine", None)
            st.rerun()
        if b2.button("Sign out", key="cmp_signout", use_container_width=True):
            st.session_state.pop("_auth_email", None)
            st.session_state.pop("_auth_name", None)
            st.rerun()

    # ── ALL controls live in the collapsible sidebar → collapse it and the canvas is the
    #    whole screen. (Streamlit has no native right sidebar; the pose/selection/metrics
    #    read-outs live as overlays inside the canvas instead.) ──
    with st.sidebar:
        st.markdown("**Torso** (stable body · grey)")
        t_pdb = st.file_uploader("Torso PDB", type=["pdb"], key="cmp_torso", label_visibility="collapsed")
        if t_pdb:
            st.caption(_chain_info(t_pdb.getvalue()))
        t_chain = st.text_input("Torso chain", placeholder="auto", key="cmp_tchain").strip() or None
        st.caption("**Excision span** — type it, or **✂ Cut** two torso residues on the canvas")
        cc1, cc2 = st.columns(2)
        with cc1:
            cut1 = st.number_input("from", min_value=1, value=15, step=1, key="cmp_cut1")
        with cc2:
            cut2 = st.number_input("to", min_value=1, value=30, step=1, key="cmp_cut2")

        st.markdown("**Mount** (catalytic insert · orange · movable)")
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

    # Reset the live pose whenever the inputs that define the base composite change (a new base
    # means the client rebuilds and its pose resets — keep Python's stored transform in step).
    sig = None
    if t_pdb and m_pdb and keep.strip():
        sig = hashlib.md5(
            t_pdb.getvalue() + m_pdb.getvalue()
            + f"|{cut1}|{cut2}|{keep}|{t_chain}|{m_chain}|{repose}".encode()
        ).hexdigest()
    if sig != st.session_state.get("_cmp_sig"):
        st.session_state["_cmp_sig"] = sig
        st.session_state.pop("cmp_xform", None)

    xform = st.session_state.get("cmp_xform")

    # base = snapped composite WITHOUT the live transform (stable → the canvas never rebuilds while
    # posing). posed = same + the live mount transform → drives metrics + export only.
    base, posed, err = None, None, None
    if t_pdb and m_pdb and keep.strip():
        common = dict(
            torso_pdb=t_pdb.getvalue(), mount_pdb=m_pdb.getvalue(),
            torso_cut=(int(cut1), int(cut2)), mount_keep=_parse_ranges(keep),
            torso_chain=t_chain, mount_chain=m_chain, repose=repose, k=k, m=m,
        )
        try:
            base = compose_graft(**common)
            posed = base if _is_identity_xform(xform) else compose_graft(**common, mount_transform=xform)
        except Exception as e:  # noqa: BLE001
            err = str(e)

    # ── metrics + export → sidebar bottom (all Streamlit chrome in one collapsible rail) ──
    with st.sidebar:
        if posed is not None:
            st.divider()
            if not _is_identity_xform(xform):
                if st.button("↺ Reset pose", key="cmp_reset_pose", use_container_width=True):
                    st.session_state.pop("cmp_xform", None)
                    st.session_state["cmp_reset_ts"] = time.time()
                    st.rerun()
            metrics = compute_metrics(posed)
            if critical_failures(metrics):
                st.error("Can't export — resolve the ⚠️ metrics.")
                st.button("⬇️ Export graft package", disabled=True, use_container_width=True, key="cmp_exp0")
            else:
                st.download_button("⬇️ Export graft package", data=posed.to_bytes(),
                                   file_name="graft_package.graft", mime="application/zip",
                                   use_container_width=True, key="cmp_exp")
            st.download_button("⬇️ Composite PDB", data=posed.composite_pdb,
                               file_name="composite.pdb", mime="chemical/x-pdb",
                               use_container_width=True, key="cmp_dlpdb")
        elif err:
            st.error(f"Could not compose: {err}")

    # ── MAIN · the canvas, edge to edge ──
    if base is not None:
        repack = [f"{r['chain']}{r['author_num']}" for r in base.spec.repack_residues]
        hud_metrics = [{"label": mt.label, "value": f"{mt.value:g}", "unit": mt.unit, "ok": mt.ok}
                       for mt in compute_metrics(posed)]
        ev = mol_viewer(base.composite_pdb.decode(errors="ignore"), repack=repack,
                        metrics=hud_metrics, mount_chain="B",
                        connections=_connection_points(base),
                        reset_ts=st.session_state.get("cmp_reset_ts"), height=760, key="cmp_mol")
        _handle_component_event(ev)
    elif t_pdb or m_pdb:
        src = t_pdb or m_pdb
        ev = mol_viewer(src.getvalue().decode(errors="ignore"), height=760, key="cmp_mol")
        _handle_component_event(ev)
    else:
        hint, _ = st.columns([3, 4])
        with hint:
            st.info("← In the sidebar: upload a **torso** + **mount** and set the mount residues "
                    "to keep. On the canvas: **✂ Cut** = click two torso residues to set the "
                    "excision span, **✋ Pose Mount** = drag-rotate / two-finger-scroll the mount. "
                    "The **arrows** at each open end show the backbone's exit direction — pose the "
                    "mount until each pair points head-to-head and glows **green** (a linker can "
                    "bridge them). Two-finger scroll in **Camera** mode pans the whole scene.")
