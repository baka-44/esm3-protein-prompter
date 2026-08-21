"""
ui/mol_component.py — thin Python wrapper around the static interactive 3D component
(ui/molcomponent/index.html). Phase-2 milestone 1: click-to-pick.

The component renders a structure (3Dmol.js) with an in-canvas grid toggle and click-to-pick;
clicking a residue returns {chain, resi, resn, ts} to Python (the `ts` makes repeated clicks on
the same residue distinct so Streamlit reruns each time). No npm build — the bare Streamlit
component postMessage protocol is implemented in the HTML, so it ships in the Cloud Run image.
"""

from __future__ import annotations

import os

import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "molcomponent")
_component = components.declare_component("bb_mol_viewer", path=_DIR)


def mol_viewer(pdb_text: str, *, repack: list[str] | None = None,
               metrics: list[dict] | None = None, mount_chain: str = "B",
               connections: list[dict] | None = None,
               reset_ts: float | None = None, height: int = 620, key: str | None = None):
    """
    Render `pdb_text` interactively in a full-bleed canvas. `pdb_text` is the *base* (snapped)
    composite; the mount (`mount_chain`, default "B") is posed live client-side — the base never
    changes during posing, so there is no server round-trip lag. Returns the last component event:
      - {"kind":"pick", "chain","resi","resn","ts"}  — a residue was clicked (Camera mode)
      - {"kind":"pose_xform", "rot":[9], "tran":[3], "about":[3], "ts"}  — cumulative rigid mount
        transform (rotation matrix row-major + translation, about the mount centroid) on release
    or None. `repack` = ["A15","B10",...] sticks (Camera mode only). `metrics` = [{label,value,
    unit,ok}] shown as a read-only HUD. Bump `reset_ts` to reset the on-canvas pose to base.
    """
    return _component(pdb=pdb_text, repack=repack or [], metrics=metrics or [],
                      mount_chain=mount_chain, connections=connections or [], reset_ts=reset_ts,
                      height=height, key=key, default=None)
