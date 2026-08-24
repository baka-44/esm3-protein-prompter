"""
Tests for optional catalytic-geometry reporting on generated candidates (graft Option A).

The governing requirement: the spec is OPT-IN. A graft package without one must design exactly
as before — no geometry keys, no extra work, and above all no failure. These tests pin that
down first, then the reporting behaviour itself.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinredesign import worker  # noqa: E402
from proteinredesign.graft import (  # noqa: E402
    Fragment, GraftPackage, GraftSpec, Linker, to_engine_params,
)

SITE = {"nucleophile": ["A", 385], "base": ["A", 213], "acid": ["A", 175],
        "oxyanion": [["A", 314]],
        "metal_sites": {"Ca2": [["A", 277], ["A", 320], ["A", 350]]}}


def _spec(site=None):
    s = GraftSpec(chain_order=[Fragment("T1", "torso", "A", 1, 15), Linker(4, 8),
                               Fragment("M1", "mount", "B", 5, 20), Linker(4, 8),
                               Fragment("T2", "torso", "A", 30, 46)])
    if site:
        s.catalytic_site = site
    return s


def _atom(serial, name, resname, num, x, y, z, b=90.0):
    return ("%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % ("ATOM", serial, name, "", resname, "A", num, "", x, y, z, 1.0, b, name[0]))


def _triad_pdb(n_nuc=3, n_base=2, n_acid=1):
    """Intact Ser-His-Asp relay at arbitrary residue numbers."""
    return "\n".join([
        _atom(1, "OG", "SER", n_nuc, 0, 0, 0), _atom(2, "CA", "SER", n_nuc, 0, -1.4, 0),
        _atom(3, "NE2", "HIS", n_base, 3.0, 0, 0), _atom(4, "ND1", "HIS", n_base, 4.4, 1.1, 0),
        _atom(5, "CA", "HIS", n_base, 3.0, -1.4, 0),
        _atom(6, "OD1", "ASP", n_acid, 6.9, 1.6, 0), _atom(7, "CA", "ASP", n_acid, 6.9, 0.2, 0),
        "END",
    ]) + "\n"


class _Cand:
    def __init__(self, pdb="", positions=None, rank=1):
        self.pdb, self.catalytic_pred_positions, self.rank = pdb, positions or [], rank


# ── the opt-in contract ───────────────────────────────────────────────────────

def test_package_without_site_produces_no_catalytic_site_param():
    params = to_engine_params(GraftPackage(spec=_spec(), composite_pdb=b"ATOM\n"))
    assert "catalytic_site" not in params


def test_package_with_site_passes_it_through():
    params = to_engine_params(GraftPackage(spec=_spec(SITE), composite_pdb=b"ATOM\n"))
    assert params["catalytic_site"] == SITE


def test_spec_roundtrips_through_serialisation():
    d = _spec(SITE).to_dict()
    assert GraftSpec.from_dict(d).catalytic_site == SITE
    # and a spec without one round-trips to an empty dict, not None
    assert GraftSpec.from_dict(_spec().to_dict()).catalytic_site == {}


def test_missing_spec_makes_geometry_a_noop(tmp_path):
    assert worker._candidate_geometry(_Cand(_triad_pdb()), {}, [], {}, str(tmp_path)) == {}
    assert worker._reference_geometry({}, "does_not_exist.pdb") == {}


def test_reference_measurement_on_unreadable_file_is_survivable():
    # a missing/broken composite must degrade to "no reference", not raise
    assert worker._reference_geometry(SITE, "/nonexistent/path.pdb") == {}


def test_geometry_errors_are_captured_not_raised(tmp_path):
    c = _Cand("NOT A PDB", positions=[1, 2, 3])
    out = worker._candidate_geometry(c, SITE, [("A", 385), ("A", 213), ("A", 175)], {}, str(tmp_path))
    assert isinstance(out, dict)          # never raises; either an error note or empty metrics
    assert "metrics" in out or "error" in out or "note" in out


# ── remapping into the design's numbering ─────────────────────────────────────

def test_remap_translates_parent_numbering_to_design_positions():
    keys = [("A", 385), ("A", 213), ("A", 175)]
    mapped = worker._remap_site({"nucleophile": ["A", 385], "base": ["A", 213], "acid": ["A", 175]},
                                keys, [3, 2, 1])
    assert mapped["nucleophile"] == ["A", 3] and mapped["base"] == ["A", 2]


def test_remap_gives_up_when_a_required_residue_is_unmapped():
    # position 0 means RF3 did not place it — a partial map would measure the wrong atoms
    keys = [("A", 385), ("A", 213), ("A", 175)]
    assert worker._remap_site({"nucleophile": ["A", 385], "base": ["A", 213], "acid": ["A", 175]},
                              keys, [3, 0, 1]) is None


def test_remap_drops_incomplete_metal_sites_rather_than_scoring_them():
    keys = [("A", 385), ("A", 213), ("A", 175), ("A", 277), ("A", 320)]
    mapped = worker._remap_site(SITE, keys, [3, 2, 1, 4, 5])   # Ca2 needs 350 too — absent
    assert mapped is not None and mapped["metal_sites"] == {}


# ── the report itself ─────────────────────────────────────────────────────────

def test_report_carries_value_reference_and_deviation(tmp_path):
    keys = [("A", 385), ("A", 213), ("A", 175)]
    c = _Cand(_triad_pdb(), positions=[3, 2, 1])
    ref = {"nucleophile–base": 2.50, "base–acid": 2.68}
    out = worker._candidate_geometry(c, {"nucleophile": ["A", 385], "base": ["A", 213],
                                         "acid": ["A", 175]}, keys, ref, str(tmp_path))
    by = {m["label"]: m for m in out["metrics"]}
    nb = by["nucleophile–base"]
    assert nb["value"] is not None and nb["reference"] == 2.50
    assert nb["deviation"] == abs(nb["value"] - 2.50)
    assert out["geometry_deviation"] is not None     # scalar for ranking
    assert out["all_within_band"] is True


def test_report_is_produced_even_with_no_reference(tmp_path):
    """No reference (e.g. unreadable composite) still yields measurements, just no deltas."""
    keys = [("A", 385), ("A", 213), ("A", 175)]
    out = worker._candidate_geometry(_Cand(_triad_pdb(), positions=[3, 2, 1]),
                                     {"nucleophile": ["A", 385], "base": ["A", 213],
                                      "acid": ["A", 175]}, keys, {}, str(tmp_path))
    assert out["metrics"] and all(m["reference"] is None for m in out["metrics"])
    assert out["geometry_deviation"] is None
