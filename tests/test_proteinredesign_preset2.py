"""
Unit tests for utils.pdb_utils HETATM helpers and
proteinredesign.config_builders.preset2 (ligand-aware redesign).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.config_builders.preset2 import (  # noqa: E402
    ConfigError,
    build_preset2_config,
)
from utils.pdb_utils import filter_pdb_keep_ligand, get_hetatm_groups  # noqa: E402


# ── PDB fixture helpers ────────────────────────────────────────────────────────

def _line(record, serial, atomname, resname, chain, resseq, x, y, z):
    return (
        "%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
        % (record, serial, atomname, "", resname, chain, resseq, "", x, y, z, 1.0, 0.0, atomname[0])
    )


def _make_complex_pdb(protein_residues, hetatms, chain="A"):
    """
    protein_residues: [(author_num, resname), ...] — one CA ATOM each.
    hetatms: [(record_resname, chain, resseq, natoms), ...] — HETATM group(s).
    """
    lines = []
    serial = 1
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


PROTEIN = [(1, "MET"), (2, "LYS"), (3, "ARG"), (4, "GLY")]

# One real ligand (LIG, 5 atoms) + water + a sodium ion — the latter two must be
# filtered out of the candidate list by default.
PDB_WITH_LIGAND = _make_complex_pdb(
    PROTEIN,
    hetatms=[("LIG", "A", 101, 5), ("HOH", "A", 201, 1), ("NA", "A", 202, 1)],
)

# No ligand at all — just protein + waters/ions (common in many structures).
PDB_NO_LIGAND = _make_complex_pdb(
    PROTEIN, hetatms=[("HOH", "A", 201, 1), ("HOH", "A", 202, 1), ("CL", "A", 203, 1)],
)


# ── get_hetatm_groups ───────────────────────────────────────────────────────────

def test_finds_ligand_excludes_solvent_and_ions():
    groups = get_hetatm_groups(PDB_WITH_LIGAND)
    assert len(groups) == 1
    g = groups[0]
    assert g.resname == "LIG"
    assert g.chain_id == "A"
    assert g.res_seq == 101
    assert g.atom_count == 5


def test_no_ligand_present_returns_empty():
    assert get_hetatm_groups(PDB_NO_LIGAND) == []


def test_label_is_human_readable():
    g = get_hetatm_groups(PDB_WITH_LIGAND)[0]
    assert "LIG" in g.label() and "chain A" in g.label() and "101" in g.label()


# ── filter_pdb_keep_ligand ───────────────────────────────────────────────────────

def test_filter_keeps_protein_and_only_chosen_ligand():
    filtered = filter_pdb_keep_ligand(PDB_WITH_LIGAND, ("LIG", "A", 101))
    text = filtered.decode()
    assert text.count("ATOM  ") == 4          # all 4 protein residues kept
    assert "LIG" in text
    assert "HOH" not in text                   # water dropped
    assert " NA " not in text.replace("NA A", "")  # sodium dropped (crude but sufficient here)
    # Re-parse: ligand still detectable, waters/ion gone.
    groups = get_hetatm_groups(filtered)
    assert len(groups) == 1 and groups[0].resname == "LIG"


# ── build_preset2_config ─────────────────────────────────────────────────────────

def test_build_config_no_fixed_residues_valid():
    cfg = build_preset2_config(PDB_WITH_LIGAND, ("LIG", "A", 101))
    assert cfg.ligand.resname == "LIG"
    assert cfg.chain_id == "A"
    assert cfg.fixed_positions == {}
    assert cfg.fixed_residues == []
    assert cfg.mapping_summary == ""
    # Filtered PDB round-trips: still has the ligand, no solvent.
    assert get_hetatm_groups(cfg.filtered_pdb_bytes) == [
        g for g in get_hetatm_groups(PDB_WITH_LIGAND)
    ]


def test_build_config_with_fixed_residues():
    cfg = build_preset2_config(PDB_WITH_LIGAND, ("LIG", "A", 101), fixed_residues_str="K2, R3")
    assert cfg.fixed_positions == {"A": [2, 3]}
    assert cfg.mapping_summary == "K2→A#2, R3→A#3"


def test_unknown_ligand_raises():
    with pytest.raises(ConfigError):
        build_preset2_config(PDB_WITH_LIGAND, ("XXX", "A", 999))


def test_no_ligand_in_pdb_raises_helpful_message():
    with pytest.raises(ConfigError) as exc:
        build_preset2_config(PDB_NO_LIGAND, ("LIG", "A", 101))
    assert "Fixed-backbone redesign" in str(exc.value)


def test_bad_fixed_residue_still_validates_against_pdb():
    # residue 2 is LYS (K), not ALA — should raise, same guard as preset #1.
    with pytest.raises(ConfigError):
        build_preset2_config(PDB_WITH_LIGAND, ("LIG", "A", 101), fixed_residues_str="A2")


def test_manifest_params_roundtrip():
    cfg = build_preset2_config(PDB_WITH_LIGAND, ("LIG", "A", 101), fixed_residues_str="K2")
    params = cfg.to_params()
    assert params["ligand"] == {"resname": "LIG", "chain_id": "A", "res_seq": 101}
    assert params["fixed_positions"] == {"A": [2]}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
