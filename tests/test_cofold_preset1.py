"""
Unit tests for cofold.manifest and cofold.config_builders.preset1.

Runnable via `python -m pytest tests/test_cofold_preset1.py` or directly
(`python tests/test_cofold_preset1.py`).
"""

import os
import sys

# Make the repo root importable whether run via pytest or directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from cofold.config_builders.preset1 import (  # noqa: E402
    ConfigError,
    build_preset1_config,
    parse_fixed_residue_tokens,
)
from cofold.manifest import JobManifest, Preset  # noqa: E402


# ── PDB fixture helpers ───────────────────────────────────────────────────────

def _atom_line(serial, atomname, resname, chain, resseq, x, y, z):
    return (
        "ATOM  %5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
        % (serial, atomname, "", resname, chain, resseq, "", x, y, z, 1.0, 0.0, atomname[0])
    )


def _make_multichain_pdb(chains):
    """chains: dict chain_id -> list of (author_num, resname). One CA atom per residue."""
    lines = []
    serial = 1
    for chain, residues in chains.items():
        for num, resname in residues:
            lines.append(_atom_line(serial, "CA", resname, chain, num, float(serial), 0.0, 0.0))
            serial += 1
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


def _make_pdb(residues, chain="A"):
    """residues: list of (author_num, resname). One CA atom per residue."""
    return _make_multichain_pdb({chain: residues})


# Author numbering starts at 10 (NOT 1-based) → exercises author→sequential mapping.
# author 10→seq1(A), 11→seq2(K), 12→seq3(R), 13→seq4(D)
PDB = _make_pdb([(10, "ALA"), (11, "LYS"), (12, "ARG"), (13, "ASP")])

# Two-chain PDB to test chain selection / warning.
PDB_2CHAIN = _make_multichain_pdb(
    {"A": [(1, "MET"), (2, "LYS")], "B": [(1, "GLY"), (2, "ARG")]}
)


# ── Token parsing ─────────────────────────────────────────────────────────────

def test_parse_tokens_mixed_forms():
    assert parse_fixed_residue_tokens("K11, 12, 13R") == [("K", 11), (None, 12), ("R", 13)]


def test_parse_tokens_whitespace_and_commas():
    assert parse_fixed_residue_tokens("10  11,12") == [(None, 10), (None, 11), (None, 12)]


def test_parse_tokens_bad_token_raises():
    with pytest.raises(ConfigError):
        parse_fixed_residue_tokens("K11, foo")


# ── Config building: author→sequential mapping ────────────────────────────────

def test_author_numbering_maps_to_sequential():
    cfg = build_preset1_config(PDB, "K11, R12")
    # author 11 (K) → seq 2, author 12 (R) → seq 3
    assert cfg.fixed_positions == {"A": [2, 3]}
    assert cfg.mapping_summary == "K11→A#2, R12→A#3"
    assert cfg.chain_id == "A"


def test_first_residue_maps_to_seq_one():
    cfg = build_preset1_config(PDB, "A10")
    assert cfg.fixed_positions == {"A": [1]}


def test_number_only_uses_pdb_aa():
    cfg = build_preset1_config(PDB, "11")
    assert cfg.fixed_positions == {"A": [2]}
    assert cfg.fixed_residues[0].aa == "K"


def test_positions_sorted_and_deduped():
    cfg = build_preset1_config(PDB, "13, 10, 10")
    assert cfg.fixed_positions == {"A": [1, 4]}  # author 10→1, 13→4; dupe dropped


# ── Validation: residue# ↔ AA ─────────────────────────────────────────────────

def test_residue_aa_mismatch_raises():
    # author 11 is LYS(K), not ALA(A) — should be rejected, not silently fixed.
    with pytest.raises(ConfigError) as exc:
        build_preset1_config(PDB, "A11")
    assert "is K" in str(exc.value)


def test_unknown_residue_number_raises():
    with pytest.raises(ConfigError):
        build_preset1_config(PDB, "K99")


def test_empty_input_raises():
    with pytest.raises(ConfigError):
        build_preset1_config(PDB, "   ")


# ── Chain selection ───────────────────────────────────────────────────────────

def test_multichain_defaults_first_and_warns():
    cfg = build_preset1_config(PDB_2CHAIN, "M1")
    assert cfg.chain_id == "A"
    assert cfg.warnings and "2 chains" in cfg.warnings[0]


def test_explicit_chain_selection():
    cfg = build_preset1_config(PDB_2CHAIN, "R2", chain_id="B")
    assert cfg.fixed_positions == {"B": [2]}


def test_missing_chain_raises():
    with pytest.raises(ConfigError):
        build_preset1_config(PDB_2CHAIN, "M1", chain_id="Z")


# ── Manifest round-trips ──────────────────────────────────────────────────────

def test_manifest_roundtrip_json():
    cfg = build_preset1_config(PDB, "K11, R12")
    m = JobManifest(
        preset=Preset.FIXED_BACKBONE_REDESIGN,
        user_email="scientist@phyx44.com",
        pdb_uri="gs://bucket/in/x.pdb",
        params=cfg.to_params(),
        num_outputs=10,
    )
    m2 = JobManifest.from_json(m.to_json())
    assert m2.preset is Preset.FIXED_BACKBONE_REDESIGN
    assert m2.params["fixed_positions"] == {"A": [2, 3]}
    assert m2.job_id == m.job_id
    assert m2.output_prefix() == f"jobs/{m.job_id}"


def test_requires_rfdiffusion_flag():
    m1 = JobManifest(Preset.FIXED_BACKBONE_REDESIGN, "u@phyx44.com", "gs://b/x.pdb")
    m3 = JobManifest(Preset.MOTIF_SCAFFOLDING, "u@phyx44.com", "gs://b/x.pdb")
    assert m1.requires_rfdiffusion() is False
    assert m3.requires_rfdiffusion() is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
