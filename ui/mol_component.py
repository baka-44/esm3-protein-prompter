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
               metrics: list[dict] | None = None, height: int = 620, key: str | None = None):
    """
    Render `pdb_text` interactively in a full-bleed canvas. Returns the last component event:
      - {"kind":"pick", "chain","resi","resn","ts"}  — a residue was clicked (Camera mode)
      - {"kind":"pose", "dnudge":[x,y,z], "drotate":[rx,ry,rz], "ts"}  — the mount was dragged/scrolled
    or None. `repack` = ["A15","B10",...] shown as sticks. `metrics` = [{label,value,unit,ok}]
    shown as a read-only HUD in the top-right of the canvas.
    """
    return _component(pdb=pdb_text, repack=repack or [], metrics=metrics or [],
                      height=height, key=key, default=None)
