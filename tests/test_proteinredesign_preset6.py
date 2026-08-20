"""
Unit tests for proteinredesign.config_builders.preset6 (enzyme active-site
scaffolding — RF3 all-atom, unindexed catalytic motif).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.config_builders.preset6 import (  # noqa: E402
    ConfigError,
    K_MAX,
    LENGTH_MAX,
    M_MAX,
    build_preset6_config,
)


# ── PDB fixture helpers (mirror the preset #2 fixture) ──────────────────────────

def _line(record, serial, atomname, resname, chain, resseq, x, y, z):
    return (
        "%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
        % (record, serial, atomname, "", resname, chain, resseq, "", x, y, z, 1.0, 0.0, atomname[0])
    )


def _make_pdb(protein_residues, hetatms=(), chain="A"):
    lines, serial = [], 1
    for num, resname in protein_residues:
        lines.append(_line("ATOM", serial, "CA", resname, chain, num, float(serial), 0.0, 0.0))
        serial += 1
    for resname, het_chain, resseq, natoms in hetatms:
        for a in range(natoms):
            lines.append(_line("HETATM", serial, f"C{a+1}", resname, het_chain, resseq,
                                10.0 + a, 10.0, 10.0))
            serial += 1
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


# A 60-residue chain with a catalytic triad placed at author positions 57/50/59.
PROTEIN = [(i, "ALA") for i in range(1, 61)]
PROTEIN[56] = (57, "HIS")   # H57
PROTEIN[49] = (50, "ASP")   # D50
PROTEIN[58] = (59, "SER")   # S59

PDB = _make_pdb(PROTEIN)
PDB_WITH_LIG = _make_pdb(PROTEIN, hetatms=[("NAI", "A", 201, 5), ("HOH", "A", 301, 1)])


_LEN = dict(length_min=120, length_max=160, k=2, m=2)


# ── happy paths ─────────────────────────────────────────────────────────────────

def test_valid_config_builds_rf3_spec():
    cfg = build_preset6_config(PDB, "H57, D50, S59", **_LEN)
    assert cfg.chain_id == "A"
    assert cfg.rf3_unindex() == "A57,A50,A59"
    assert cfg.rf3_select_fixed_atoms() == {"A57": "TIP", "A50": "TIP", "A59": "TIP"}
    assert cfg.total_designs == 4
    assert cfg.ligand is None and cfg.filtered_pdb_bytes is None


def test_fixed_atoms_mode_applied():
    cfg = build_preset6_config(PDB, "H57", fixed_atoms_mode="BKBN", **_LEN)
    assert cfg.rf3_select_fixed_atoms() == {"A57": "BKBN"}


def test_ligand_conditioning_and_filter():
    cfg = build_preset6_config(PDB_WITH_LIG, "H57, D50", ligand_key=("NAI", "A", 201), **_LEN)
    assert cfg.ligand is not None and cfg.ligand.resname == "NAI"
    fa = cfg.rf3_select_fixed_atoms()
    assert fa["NAI"] == "ALL"                     # cofactor fully fixed
    assert fa["A57"] == "TIP"
    # Filtered PDB keeps the ligand, drops water.
    from utils.pdb_utils import get_hetatm_groups
    groups = get_hetatm_groups(cfg.filtered_pdb_bytes)
    assert len(groups) == 1 and groups[0].resname == "NAI"


def test_params_roundtrip():
    cfg = build_preset6_config(PDB, "H57, D50, S59", **_LEN)
    p = cfg.to_params()
    assert p["unindex"] == "A57,A50,A59"
    assert p["length"] == "120-160"
    assert p["select_fixed_atoms"] == {"A57": "TIP", "A50": "TIP", "A59": "TIP"}
    assert p["ligand"] is None
    assert p["k"] == 2 and p["m"] == 2


# ── validation guards ─────────────────────────────────────────────────────────────

def test_no_catalytic_residues_raises():
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "  ,  ", **_LEN)


def test_bad_fixed_atoms_mode_raises():
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "H57", fixed_atoms_mode="SIDECHAIN", **_LEN)


def test_residue_aa_mismatch_raises():
    # residue 57 is HIS, not LYS — the same author-vs-sequential guard as #1/#2.
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "K57", **_LEN)


def test_length_out_of_range_raises():
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "H57", length_min=10, length_max=20, k=1, m=1)   # below LENGTH_MIN
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "H57", length_min=200, length_max=100, k=1, m=1)  # min>max
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "H57", length_min=120, length_max=LENGTH_MAX + 1, k=1, m=1)


def test_k_m_bounds_raise():
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "H57", length_min=120, length_max=160, k=K_MAX + 1, m=1)
    with pytest.raises(ConfigError):
        build_preset6_config(PDB, "H57", length_min=120, length_max=160, k=1, m=M_MAX + 1)


def test_unknown_ligand_raises():
    with pytest.raises(ConfigError):
        build_preset6_config(PDB_WITH_LIG, "H57", ligand_key=("XXX", "A", 999), **_LEN)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ── submit routing (D11: #6 → rf3 job) ───────────────────────────────────────────

def test_enzyme_routes_to_rf3_job(monkeypatch):
    from proteinredesign.manifest import Preset
    from proteinredesign.submit import _job_name_for_preset
    monkeypatch.delenv("PROTEINREDESIGN_RF3_JOB_NAME", raising=False)
    assert _job_name_for_preset(Preset.ENZYME_ACTIVE_SITE) == "proteinredesign-rf3-worker"
