"""
Tests for compose_fusion — end-to-end fusion with no excision.

The mode exists because an enzyme with an N-terminal propeptide cannot be loop-inserted:
autocatalytic cleavage would sever the chain and drop the upstream flank. So both bodies stay
whole, the chassis sits on a terminus, and RF3 designs only the single connection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from proteinredesign.composer import compose_fusion  # noqa: E402
from proteinredesign.graft import Fragment, Linker, to_engine_params  # noqa: E402


def _atom(serial, name, resname, chain, num, x, y, z):
    return ("%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % ("ATOM", serial, name, "", resname, chain, num, "", x, y, z, 1.0, 0.0, name[0]))


def _pdb(n, chain="A", origin=(0.0, 0.0, 0.0), spacing=3.8):
    ox, oy, oz = origin
    lines, serial = [], 1
    for i in range(1, n + 1):
        cx = ox + spacing * i
        for nm, dx, dy in (("N", -0.5, 0.5), ("CA", 0.0, 0.0), ("C", 0.5, 0.5), ("O", 0.6, 1.0)):
            lines.append(_atom(serial, nm, "ALA", chain, i, cx + dx, oy + dy, oz))
            serial += 1
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


MOUNT = _pdb(30, chain="A", origin=(0.0, 0.0, 0.0))
TORSO = _pdb(20, chain="A", origin=(0.0, 40.0, 0.0))


def test_both_bodies_are_kept_whole_with_one_linker():
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO)
    order = pkg.spec.chain_order
    assert len(order) == 3
    assert isinstance(order[0], Fragment) and isinstance(order[1], Linker) and isinstance(order[2], Fragment)
    frags = pkg.spec.fragments()
    assert {f.label for f in frags} == {"MOUNT-1", "TORSO-1"}
    # nothing excised: full spans retained
    mount = next(f for f in frags if f.label == "MOUNT-1")
    torso = next(f for f in frags if f.label == "TORSO-1")
    assert (mount.start, mount.end) == (1, 30)
    assert (torso.start, torso.end) == (1, 20)


def test_default_puts_the_chassis_on_the_c_terminus():
    """N-terminal-propeptide enzymes require this: autoprocessing must not sever the chassis."""
    order = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO).spec.chain_order
    assert order[0].label == "MOUNT-1" and order[2].label == "TORSO-1"


def test_chassis_terminus_n_reverses_the_order():
    order = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, chassis_terminus="N").spec.chain_order
    assert order[0].label == "TORSO-1" and order[2].label == "MOUNT-1"


def test_invalid_terminus_is_rejected():
    with pytest.raises(ValueError, match="chassis_terminus"):
        compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, chassis_terminus="middle")


def test_contig_holds_both_bodies_and_generates_only_the_linker():
    params = to_engine_params(compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO,
                                             linker_length=(5, 9)))
    # e.g. "B1-30,5-9,A1-20": chain-labelled blocks are fixed, the bare range is designed
    assert params["contig"] == "B1-30,5-9,A1-20"


def _coords(pkg, chain):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in pkg.composite_pdb.decode().splitlines()
                     if l.startswith("ATOM") and l[21] == chain])


def test_auto_separate_places_the_junction_within_linker_reach():
    """What matters is the gap the LINKER must close, not the centroid separation. Separating by
    centroid puts elongated bodies absurdly far apart at the junction."""
    from proteinredesign.graft_metrics import compute_metrics
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, linker_length=(4, 12))
    m = {x.label: x for x in compute_metrics(pkg)}
    # the longest linker (12 res) can span ~12 x 3.3 A; the default pose must land inside that
    assert m["Max linker gap"].value <= 12 * 3.3
    assert m["Closure feasibility"].ok          # no unclosable junction
    assert m["Core clashes"].ok                 # and the bodies are not interpenetrating
    assert pkg.spec.provenance["posed_by"] == "auto_separate"


def test_auto_separate_does_not_leave_the_bodies_overlapping():
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO)
    a, b = _coords(pkg, "A"), _coords(pkg, "B")
    assert len(a) and len(b)
    assert np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min() > 2.5


def test_manual_pose_overrides_auto_separation():
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, nudge=(5.0, 0.0, 0.0))
    assert pkg.spec.provenance["posed_by"] == "manual"


def test_catalytic_site_rides_along_when_supplied():
    site = {"nucleophile": ["A", 385], "base": ["A", 213], "acid": ["A", 175]}
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, catalytic_site=site)
    assert to_engine_params(pkg)["catalytic_site"] == site
    # and stays absent when not supplied
    assert "catalytic_site" not in to_engine_params(compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO))


def test_multi_block_input_is_rejected_with_a_pointer_to_compose_graft():
    with pytest.raises(ValueError, match="compose_graft"):
        compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, mount_keep=[(1, 5), (20, 30)])


def test_mode_is_recorded_in_provenance():
    assert compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO).spec.provenance["mode"] == "fusion"
