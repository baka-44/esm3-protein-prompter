"""Unit tests for proteinredesign.graft_metrics (compose scorecard + export gate)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinredesign.composer import compose_graft  # noqa: E402
from proteinredesign.graft_metrics import compute_metrics, critical_failures  # noqa: E402


def _atom(serial, name, resname, chain, num, x, y, z):
    return ("%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % ("ATOM", serial, name, "", resname, chain, num, "", x, y, z, 1.0, 0.0, name[0]))


def _backbone_pdb(n, chain="A", origin=(0.0, 0.0, 0.0)):
    ox, oy, oz = origin
    lines, serial = [], 1
    for i in range(1, n + 1):
        cx = ox + 3.8 * i
        for name, dx, dy in (("N", -0.5, 0.5), ("CA", 0.0, 0.0), ("C", 0.5, 0.5), ("O", 0.6, 1.0)):
            lines.append(_atom(serial, name, "ALA", chain, i, cx + dx, oy + dy, oz))
            serial += 1
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


TORSO = _backbone_pdb(40, origin=(0.0, 0.0, 0.0))
MOUNT = _backbone_pdb(20, origin=(0.0, 20.0, 0.0))


def _metrics(**kw):
    pkg = compose_graft(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(15, 25),
                        mount_keep=[(1, 20)], **kw)
    return pkg, {m.key: m for m in compute_metrics(pkg)}


def test_all_metrics_present_with_tooltip_fields():
    _, m = _metrics()
    for key in ("max_linker_gap", "closure", "clash", "rg"):
        assert key in m
        met = m[key]
        assert met.what and met.meaning and met.desired   # tooltip triple (CC12)


def test_snap_to_fit_closes_the_junctions():
    # snap-to-fit places the mount termini on the torso cut ends → small gaps, feasible closure.
    _, m = _metrics()
    assert m["closure"].value == 0            # all junctions feasible
    assert m["closure"].ok is True


def test_critical_flags_drive_export_gate():
    _, m = _metrics()
    # closure + clash are the critical metrics.
    assert m["closure"].critical and m["clash"].critical
    assert not m["rg"].critical and not m["max_linker_gap"].critical


def test_far_pose_is_infeasible_and_gates_export():
    import numpy as np
    # Explicit pose that shoves the mount 200 Å away → gaps can't close → critical failure.
    pkg, m = _metrics(pose=(np.eye(3), np.array([0.0, 0.0, 200.0])))
    assert m["closure"].value >= 1 and m["closure"].ok is False
    assert critical_failures(compute_metrics(pkg))   # non-empty → export disabled


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))


def test_interface_contact_not_counted_as_core_clash():
    # Snap-to-fit places the mount touching the torso — the interface (repack shell) is in
    # contact, but that's expected (MPNN redesigns it). Core-core clash must be 0 for a clean
    # composition, so it stays exportable.
    _, m = _metrics()
    assert m["clash"].value == 0
    assert m["clash"].ok is True
