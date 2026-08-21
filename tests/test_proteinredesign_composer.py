"""
Unit tests for proteinredesign.composer — the Borrowed Bodies compose geometry (cut torso,
keep mount, snap-to-fit pose, repack shell) → GraftPackage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.composer import compose_graft, cut_torso  # noqa: E402
from proteinredesign.graft import GraftPackage, to_engine_params  # noqa: E402


def _atom(serial, name, resname, chain, num, x, y, z):
    return ("%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % ("ATOM", serial, name, "", resname, chain, num, "", x, y, z, 1.0, 0.0, name[0]))


def _backbone_pdb(n, chain="A", origin=(0.0, 0.0, 0.0), spacing=3.8):
    """A linear poly-Ala chain: N, CA, C, O per residue, CA spaced along x from origin."""
    ox, oy, oz = origin
    lines, serial = [], 1
    for i in range(1, n + 1):
        cx = ox + spacing * i
        for name, dx, dy in (("N", -0.5, 0.5), ("CA", 0.0, 0.0), ("C", 0.5, 0.5), ("O", 0.6, 1.0)):
            lines.append(_atom(serial, name, "ALA", chain, i, cx + dx, oy + dy, oz))
            serial += 1
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


TORSO = _backbone_pdb(40, chain="A", origin=(0.0, 0.0, 0.0))
MOUNT = _backbone_pdb(20, chain="A", origin=(0.0, 20.0, 0.0))   # offset so it's a separate body


def test_cut_torso_excises_middle():
    (a1, b1), (a2, b2) = cut_torso(list(range(1, 41)), 15, 25)
    assert (a1, b1) == (1, 15) and (a2, b2) == (25, 40)   # 16-24 excised


def test_cut_points_out_of_range_raise():
    with pytest.raises(ValueError):
        cut_torso(list(range(1, 41)), 0, 25)
    with pytest.raises(ValueError):
        cut_torso(list(range(1, 41)), 15, 15)   # not distinct


def test_compose_produces_valid_package_and_contig():
    pkg = compose_graft(
        torso_pdb=TORSO, mount_pdb=MOUNT,
        torso_cut=(15, 25), mount_keep=[(1, 20)],
        linker_lengths=((5, 5), (5, 5)), k=3, m=2,
    )
    assert isinstance(pkg, GraftPackage)
    assert pkg.spec.validate() == []
    p = to_engine_params(pkg)
    # TORSO-1 (A1-15) / L5 / MOUNT-1 (B1-20, all-atom) / L5 / TORSO-2 (A25-40)
    assert p["contig"] == "A1-15,5,B1-20,5,A25-40"
    assert p["select_fixed_atoms"] == {f"B{n}": "ALL" for n in range(1, 21)}
    assert p["k"] == 3 and p["m"] == 2
    # motif = all fixed residues: 15 + 20 + 16
    assert len(p["motif_residues"]) == 15 + 20 + 16


def test_compose_fragmented_mount_makes_multiple_mount_segments():
    pkg = compose_graft(
        torso_pdb=TORSO, mount_pdb=MOUNT,
        torso_cut=(15, 25), mount_keep=[(1, 5), (12, 16)],   # two kept blocks
        linker_lengths=((5, 5), (5, 5)),
    )
    labels = [s.label for s in pkg.spec.fragments()]
    assert labels == ["TORSO-1", "MOUNT-1", "MOUNT-2", "TORSO-2"]
    contig = to_engine_params(pkg)["contig"]
    assert "B1-5" in contig and "B12-16" in contig


def test_compose_repack_shell_nonempty_and_within_fixed():
    pkg = compose_graft(
        torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25), mount_keep=[(1, 20)],
    )
    repack = pkg.spec.repack_residues
    assert repack  # junction residues at least
    fixed = {(r["chain_id"], r["author_num"]) for r in to_engine_params(pkg)["motif_residues"]}
    for r in repack:
        assert (r["chain"], r["author_num"]) in fixed   # repack ⊆ fixed fragments


def test_compose_roundtrips_through_package_bytes():
    pkg = compose_graft(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25), mount_keep=[(1, 20)])
    back = GraftPackage.from_bytes(pkg.to_bytes())
    assert to_engine_params(back)["contig"] == to_engine_params(pkg)["contig"]
    assert back.composite_pdb == pkg.composite_pdb


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_manual_nudge_translates_the_mount():
    import numpy as np
    from proteinredesign.graft_metrics import compute_metrics
    base = compose_graft(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25), mount_keep=[(1, 20)])
    nudged = compose_graft(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25), mount_keep=[(1, 20)],
                           nudge=(0.0, 0.0, 30.0))
    gap_base = {m.key: m for m in compute_metrics(base)}["max_linker_gap"].value
    gap_nudged = {m.key: m for m in compute_metrics(nudged)}["max_linker_gap"].value
    assert gap_nudged > gap_base + 10   # a 30 Å push opens the junction gaps


def test_manual_rotation_is_rigid():
    # A pure rotation about the mount centre preserves the mount's internal geometry (bond lengths).
    import numpy as np
    from utils.pdb_utils import get_residues
    rotated = compose_graft(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25), mount_keep=[(1, 20)],
                            rotate=(0.0, 90.0, 0.0))
    res = [r for r in get_residues(rotated.composite_pdb, chain_id=None) if r.get_parent().id == "B"]
    cas = [r["CA"].coord for r in res if "CA" in r]
    d = np.linalg.norm(np.array(cas[1]) - np.array(cas[0]))
    assert 3.0 < d < 4.5   # consecutive CA distance stays ~3.8 Å (rigid)


def test_connection_points_are_the_two_linker_junctions():
    # The exit-vector arrows (M3) are one per linker: TORSO-1.end → MOUNT-1.start and
    # MOUNT-last.end → TORSO-2.start, with the mount-side endpoint flagged so the canvas moves
    # it with the live pose. Mirror the extraction that ui/composer_panel._connection_points does.
    from proteinredesign.graft import Fragment, Linker
    from utils.pdb_utils import get_residues
    base = compose_graft(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25), mount_keep=[(1, 20)])
    ca = {(r.get_parent().id, r.id[1]): r["CA"].coord
          for r in get_residues(base.composite_pdb, chain_id=None) if "CA" in r}
    order = base.spec.chain_order
    conns = []
    for i, seg in enumerate(order):
        if isinstance(seg, Linker) and 0 < i < len(order) - 1:
            prev, nxt = order[i - 1], order[i + 1]
            if isinstance(prev, Fragment) and isinstance(nxt, Fragment):
                conns.append((prev.chain == "B", nxt.chain == "B",
                              (prev.chain, prev.end) in ca, (nxt.chain, nxt.start) in ca))
    assert len(conns) == 2
    # first junction: torso→mount ; second: mount→torso ; all four endpoints resolve to a CA.
    assert conns[0][:2] == (False, True)
    assert conns[1][:2] == (True, False)
    assert all(c[2] and c[3] for c in conns)


def test_mount_transform_matches_client_math():
    # The live-canvas transform (R about `about` + t) must reproduce, atom-for-atom, the exact
    # coordinates the 3D component applies: new = R·(base − c) + c + t. This is what keeps the
    # exported/metric geometry identical to what the user posed on screen. Agreement is to PDB
    # write precision (coords stored to 3 decimals), so the tolerance is ~1e-3 Å.
    import numpy as np
    from utils.pdb_utils import get_residues
    common = dict(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25), mount_keep=[(1, 20)])

    def mount_coords(pkg):
        res = [r for r in get_residues(pkg.composite_pdb, chain_id=None) if r.get_parent().id == "B"]
        return np.array([a.coord for r in res for a in r])

    base = compose_graft(**common)
    bc = mount_coords(base)
    c = bc.mean(0)
    th = np.radians(37.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    t = np.array([4.0, -2.0, 3.5])
    expected = (R @ (bc - c).T).T + c + t
    posed = compose_graft(**common, mount_transform={
        "rot": R.flatten().tolist(), "tran": t.tolist(), "about": c.tolist()})
    assert np.abs(mount_coords(posed) - expected).max() < 2e-3
    # torso (chain A) is untouched by the mount transform.
    ta = np.array([a.coord for r in get_residues(base.composite_pdb, chain_id=None)
                   if r.get_parent().id == "A" for a in r])
    tb = np.array([a.coord for r in get_residues(posed.composite_pdb, chain_id=None)
                   if r.get_parent().id == "A" for a in r])
    assert np.abs(ta - tb).max() < 1e-6
