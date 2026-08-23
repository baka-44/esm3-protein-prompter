"""
Unit tests for proteinredesign.catalytic_geometry — the tier-1 active-site geometry filter.

Built on a synthetic Ser-His-Asp charge relay so the tests are deterministic and need no
network or reference structure download.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.catalytic_geometry import (  # noqa: E402
    CatalyticSite, Structure, measure, site_confidence, verdict,
)


def _atom(serial, name, resname, chain, num, x, y, z, b=90.0):
    el = name[0]
    return ("%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % ("ATOM", serial, name, "", resname, chain, num, "", x, y, z, 1.0, b, el))


def _triad_pdb(path, og=(0.0, 0.0, 0.0), ne2=(3.0, 0.0, 0.0), nd1=(4.4, 1.1, 0.0),
               od1=(6.9, 1.6, 0.0), nd2_asn=(0.0, 4.2, 0.0), b=90.0):
    """Minimal Ser/His/Asp/Asn placed so the relay geometry is in-band by construction."""
    lines = [
        _atom(1, "OG", "SER", "A", 385, *og, b),
        _atom(2, "CA", "SER", "A", 385, og[0], og[1] - 1.4, og[2], b),
        _atom(3, "NE2", "HIS", "A", 213, *ne2, b),
        _atom(4, "ND1", "HIS", "A", 213, *nd1, b),
        _atom(5, "CA", "HIS", "A", 213, ne2[0], ne2[1] - 1.4, ne2[2], b),
        _atom(6, "OD1", "ASP", "A", 175, *od1, b),
        _atom(7, "OD2", "ASP", "A", 175, od1[0] + 1.0, od1[1], od1[2], b),
        _atom(8, "CA", "ASP", "A", 175, od1[0], od1[1] - 1.4, od1[2], b),
        _atom(9, "ND2", "ASN", "A", 314, *nd2_asn, b),
        _atom(10, "CA", "ASN", "A", 314, nd2_asn[0], nd2_asn[1] - 1.4, nd2_asn[2], b),
        "END",
    ]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


SITE = CatalyticSite("test", ("A", 385), ("A", 213), ("A", 175), [("A", 314)])


def test_intact_triad_passes(tmp_path):
    s = Structure(_triad_pdb(tmp_path / "ok.pdb"))
    ok, fails = verdict(measure(s, SITE))
    assert ok, f"intact triad should pass, failed: {fails}"


def test_broken_nucleophile_distance_is_caught(tmp_path):
    # push the serine 9 A away from the histidine — the relay cannot function
    s = Structure(_triad_pdb(tmp_path / "far.pdb", og=(-9.0, 0.0, 0.0)))
    fs = measure(s, SITE)
    ok, fails = verdict(fs)
    assert not ok and "nucleophile–base" in fails


def test_broken_base_acid_distance_is_caught(tmp_path):
    s = Structure(_triad_pdb(tmp_path / "acid.pdb", od1=(14.0, 1.6, 0.0)))
    ok, fails = verdict(measure(s, SITE))
    assert not ok and "base–acid" in fails


def test_missing_oxyanion_residue_reports_not_ok(tmp_path):
    # oxyanion residue far from the nucleophile => outside the band
    s = Structure(_triad_pdb(tmp_path / "ox.pdb", nd2_asn=(0.0, 30.0, 0.0)))
    fs = measure(s, SITE)
    ox = [f for f in fs if f.label.startswith("oxyanion")]
    assert ox and not ox[0].ok


def test_confidence_is_read_from_bfactor(tmp_path):
    s = Structure(_triad_pdb(tmp_path / "conf.pdb", b=42.0))
    conf = site_confidence(s, SITE)
    assert conf["nucleophile"] == pytest.approx(42.0, abs=1e-6)
    assert min(v for v in conf.values() if v is not None) == pytest.approx(42.0, abs=1e-6)


def test_metal_cluster_spread_flags_collapsed_site(tmp_path):
    p = tmp_path / "metal.pdb"
    lines = [
        _atom(1, "OG", "SER", "A", 385, 0, 0, 0), _atom(2, "NE2", "HIS", "A", 213, 3, 0, 0),
        _atom(3, "ND1", "HIS", "A", 213, 4.4, 1.1, 0), _atom(4, "OD1", "ASP", "A", 175, 6.9, 1.6, 0),
        # three Ca ligands splayed far apart => site has collapsed
        _atom(5, "OD1", "ASP", "A", 135, 0, 0, 0), _atom(6, "OD1", "ASP", "A", 184, 30, 0, 0),
        _atom(7, "OD1", "ASP", "A", 227, 0, 30, 0), "END",
    ]
    p.write_text("\n".join(lines) + "\n")
    site = CatalyticSite("m", ("A", 385), ("A", 213), ("A", 175), [],
                         {"Ca1": [("A", 135), ("A", 184), ("A", 227)]})
    fs = measure(Structure(str(p)), site)
    metal = [f for f in fs if "metal site" in f.label]
    assert metal and not metal[0].ok


def test_esmfold_0_to_1_confidence_is_normalised_to_0_100(tmp_path):
    """ESMFold writes pLDDT as 0-1; AlphaFold as 0-100. Both must report on the same scale."""
    esm = Structure(_triad_pdb(tmp_path / "esm.pdb", b=0.87))
    af = Structure(_triad_pdb(tmp_path / "af.pdb", b=87.0))
    assert site_confidence(esm, SITE)["nucleophile"] == pytest.approx(87.0, abs=1e-6)
    assert site_confidence(af, SITE)["nucleophile"] == pytest.approx(87.0, abs=1e-6)
    assert esm.mean_confidence() == pytest.approx(af.mean_confidence(), abs=1e-6)
