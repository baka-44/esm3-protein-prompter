"""
Tests for gap-derived linker sizing (CC18) and the parent→output repack mapping.

Both come out of job 3b1d4fa7b70a, where every candidate came back with the catalytic
domain intact but the torso rotated ~160° and 53 Å off its designed position:

  * the contig offered `3-8` for both connectors and RF3 picked 3 for a junction whose
    measured gap was 13.7 Å — 90% of a 3-mer's fully extended reach. A taut linker has no
    conformational freedom, so the fold relaxes it by hinging the flanking domain away.
  * `repack_residues` are authored in parent numbering, but MPNN's --fixed_residues is
    written in RF3's OUTPUT numbering. Diffing the two dropped 44/60 refs outright and
    unfixed the wrong residue for the other 16.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.composer import (  # noqa: E402
    CA_CA_MAX, LINKER_CAP, LINKER_FLOOR, compose_fusion, linker_span,
)
from proteinredesign.graft import Linker  # noqa: E402
from proteinredesign.worker import _map_refs_to_output, _mpnn_fixed_tokens  # noqa: E402

from tests.test_proteinredesign_fusion import MOUNT, TORSO  # noqa: E402


# ── linker_span ────────────────────────────────────────────────────────────────

def _extension(gap: float, length: int) -> float:
    """Fraction of the linker's fully extended reach the gap consumes."""
    return gap / ((length + 1) * CA_CA_MAX)


@pytest.mark.parametrize("gap", [4.0, 9.9, 13.7, 20.0, 30.0])
def test_min_linker_is_never_taut(gap):
    """The shortest offered linker must sit well below full extension."""
    lo, _ = linker_span(gap)
    assert _extension(gap, lo) <= 0.75, (
        f"gap {gap} Å -> Linker({lo}, ...) is {_extension(gap, lo):.0%} extended"
    )


def test_the_junction_that_failed_now_gets_slack():
    """J1 of 3b1d4fa7b70a: 13.7 Å used to be handed a 3-mer at 90% extension."""
    assert _extension(13.7, 3) > 0.85          # what the old fixed (3, 8) allowed
    lo, hi = linker_span(13.7)
    assert lo == 6 and hi == 11
    assert _extension(13.7, lo) < 0.55


def test_every_offered_length_can_actually_reach():
    for gap in (5.0, 12.0, 25.0, 40.0):
        lo, hi = linker_span(gap)
        assert lo <= hi
        if lo < LINKER_CAP:                     # uncapped → the whole range must reach
            assert (lo + 1) * CA_CA_MAX >= gap


def test_degenerate_gaps_fall_back_to_the_floor():
    for bad in (float("inf"), float("nan"), 0.0, -3.0):
        assert linker_span(bad) == (LINKER_FLOOR, LINKER_FLOOR + 5)


def test_cap_leaves_absurd_gaps_unbridgeable():
    """
    Auto-sizing must not paper over a bad pose: past the cap the junction stays
    infeasible so the closure metric can fail it, rather than silently growing a tether.
    """
    lo, hi = linker_span(500.0)
    assert (lo, hi) == (LINKER_CAP, LINKER_CAP)
    assert hi * CA_CA_MAX < 500.0               # still cannot reach → closure fails


# ── wiring: the composer sizes from the pose it actually produced ──────────────

def test_fusion_sizes_the_linker_from_the_measured_gap():
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO)
    lk = next(s for s in pkg.spec.chain_order if isinstance(s, Linker))
    gap = pkg.spec.provenance["linker_gaps_a"]["J1"]
    assert pkg.spec.provenance["linker_sizing"] == "auto"
    assert gap is not None
    assert (lk.length_min, lk.length_max) == linker_span(gap)
    assert _extension(gap, lk.length_min) <= 0.75


def test_explicit_linker_length_still_wins():
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, linker_length=(5, 9))
    lk = next(s for s in pkg.spec.chain_order if isinstance(s, Linker))
    assert (lk.length_min, lk.length_max) == (5, 9)
    assert pkg.spec.provenance["linker_sizing"] == "explicit"


# ── repack refs must be translated into RF3 output numbering ───────────────────

class _Design:
    """A design .pdb path whose sibling .json carries a diffused_index_map."""

    def __init__(self, tmp_path, idx_map):
        import json
        self.pdb = str(tmp_path / "design.pdb")
        open(self.pdb, "w").write("END\n")
        json.dump({"diffused_index_map": idx_map}, open(str(tmp_path / "design.json"), "w"))


# real slice of job 3b1d4fa7b70a's map: torso A17-79 -> A1-63, mount B114+ -> A67+
IDX = {"A17": "A1", "A67": "A51", "A79": "A63", "A93": "A413",
       "B114": "A67", "B125": "A78", "B280": "A233"}


def test_repack_refs_are_mapped_into_output_space(tmp_path):
    d = _Design(tmp_path, IDX)
    assert _map_refs_to_output(["A67", "B114", "B280"], d.pdb) == ["A51", "A67", "A233"]


def test_unmapped_refs_are_dropped_not_passed_through(tmp_path):
    """A ref RF3 never kept was never fixed, so unfixing it is a no-op — but leaving it in
    parent space is not: it collides with an unrelated output residue."""
    d = _Design(tmp_path, IDX)
    assert _map_refs_to_output(["B114", "B999"], d.pdb) == ["A67"]


def test_without_a_map_refs_pass_through(tmp_path):
    import json
    pdb = str(tmp_path / "d.pdb")
    open(pdb, "w").write("END\n")
    json.dump({}, open(str(tmp_path / "d.json"), "w"))
    assert _map_refs_to_output(["A5", "B7"], pdb) == ["A5", "B7"]


def test_end_to_end_the_right_residue_gets_unfixed(tmp_path):
    """
    The regression itself. Fixed set is in OUTPUT space; asking to repack parent A67 must
    free output A51 — NOT output A67, which is parent B114 (a catalytic-domain residue).
    """
    d = _Design(tmp_path, IDX)
    fixed_out = ["A51", "A63", "A67", "A78"]          # output-numbered, from the index map
    tokens = _mpnn_fixed_tokens(fixed_out, _map_refs_to_output(["A67"], d.pdb)).split()
    assert "A51" not in tokens                         # parent A67 freed, as asked
    assert "A67" in tokens                             # parent B114 still pinned

    naive = _mpnn_fixed_tokens(fixed_out, ["A67"]).split()   # the old behaviour
    assert "A67" not in naive and "A51" in naive             # exactly backwards


# ── resizing packages exported before CC18 ─────────────────────────────────────

def _load_resize():
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "resize_graft_linkers.py")
    spec = importlib.util.spec_from_file_location("resize_graft_linkers", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resize


def test_resize_rewrites_stale_linkers_from_the_composite():
    """
    The contig is built from the package's stored chain_order at submit time, so a package
    exported with the old fixed (3, 8) re-runs with (3, 8) even on a fixed backend. Resizing
    must read the gaps out of the package's own composite.
    """
    resize = _load_resize()
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO)
    gap = pkg.spec.provenance["linker_gaps_a"]["J1"]

    stale = pkg.spec.to_dict()
    for seg in stale["chain_order"]:
        if seg.get("kind") == "linker":
            seg["length_min"], seg["length_max"] = 3, 8          # what old exports carry

    fixed, changes = resize(stale, pkg.composite_pdb)
    assert len(changes) == 1
    lk = next(s for s in fixed["chain_order"] if s.get("kind") == "linker")
    assert (lk["length_min"], lk["length_max"]) == linker_span(gap)
    assert fixed["provenance"]["linker_sizing"].startswith("auto")


def test_resize_is_idempotent():
    resize = _load_resize()
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO)
    once, changes = resize(pkg.spec.to_dict(), pkg.composite_pdb)
    assert changes == []                                          # already auto-sized
    twice, again = resize(once, pkg.composite_pdb)
    assert again == [] and twice == once
