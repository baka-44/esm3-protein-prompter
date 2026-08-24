"""
Tests for explicit repack / fixed residue overrides on the compose paths.

Why these exist: `_auto_repack` selects only residues within the contact shell of the other
body plus junctions. A surface previously buried by a deleted domain contacts nothing — so the
residues that most need redesigning are precisely the ones the automatic rule cannot see.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.composer import (  # noqa: E402
    _normalise_residue_refs, compose_fusion, compose_graft,
)


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


MOUNT = _pdb(30, origin=(0.0, 0.0, 0.0))
TORSO = _pdb(20, origin=(0.0, 40.0, 0.0))


def _repack_set(pkg):
    return {(r["chain"], r["author_num"]) for r in pkg.spec.repack_residues}


# ── reference normalisation ───────────────────────────────────────────────────

def test_all_reasonable_reference_shapes_are_accepted():
    got = _normalise_residue_refs(["B280", ("B", 282), {"chain": "B", "author_num": 284}, 286], "B")
    assert got == {("B", 280), ("B", 282), ("B", 284), ("B", 286)}


def test_bare_integers_default_to_the_mount_chain():
    assert _normalise_residue_refs([12], "B") == {("B", 12)}


def test_unrecognised_reference_is_rejected():
    with pytest.raises(ValueError, match="Unrecognised"):
        _normalise_residue_refs([3.5], "B")


# ── the behaviour that matters ────────────────────────────────────────────────

def test_extra_repack_adds_residues_the_contact_shell_cannot_see():
    """A residue far from the other body is never in the auto shell; the override must add it."""
    base = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO)
    assert ("B", 3) not in _repack_set(base)          # nowhere near the torso
    with_override = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, extra_repack=["B3"])
    assert ("B", 3) in _repack_set(with_override)
    # and it is additive — the automatic shell survives
    assert _repack_set(base) <= _repack_set(with_override)


def test_extra_fixed_removes_a_residue_the_shell_would_have_repacked():
    base = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO)
    auto = _repack_set(base)
    assert auto, "expected the automatic shell to select something"
    victim = sorted(auto)[0]
    pinned = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO,
                            extra_fixed=[f"{victim[0]}{victim[1]}"])
    assert victim not in _repack_set(pinned)


def test_fixed_wins_when_a_residue_is_listed_in_both():
    pkg = compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO,
                         extra_repack=["B5"], extra_fixed=["B5"])
    assert ("B", 5) not in _repack_set(pkg)


def test_unknown_residue_reference_is_rejected_with_a_chain_hint():
    with pytest.raises(ValueError, match="not present in the composite"):
        compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, extra_repack=["B9999"])


def test_wrong_chain_letter_is_caught_rather_than_silently_ignored():
    """Torso is relabelled A and mount B; a plausible-but-wrong letter must not pass silently."""
    with pytest.raises(ValueError, match="chain letter"):
        compose_fusion(mount_pdb=MOUNT, torso_pdb=TORSO, extra_repack=["Z5"])


def test_overrides_work_on_the_insertion_path_too():
    pkg = compose_graft(torso_pdb=TORSO, mount_pdb=MOUNT, torso_cut=(5, 12),
                        mount_keep=[(1, 30)], extra_repack=["B7", "B9"])
    got = _repack_set(pkg)
    assert ("B", 7) in got and ("B", 9) in got
