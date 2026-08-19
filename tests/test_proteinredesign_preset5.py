"""
Unit tests for proteinredesign.config_builders.preset5 (scaffold diversification),
the preset→job routing in submit.py, and that the worker's pure ranking still holds
with the new RF3 candidate fields.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.config_builders.preset5 import (  # noqa: E402
    ConfigError,
    K_MAX,
    M_MAX,
    PARTIAL_T_MAX,
    build_preset5_config,
)
from proteinredesign.manifest import Preset  # noqa: E402


# ── PDB fixture helpers ────────────────────────────────────────────────────────

def _ca_line(serial, resname, chain, resseq, x):
    return (
        "%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
        % ("ATOM", serial, "CA", "", resname, chain, resseq, "", x, 0.0, 0.0, 1.0, 0.0, "C")
    )


def _make_pdb(residues, chain="A"):
    """residues: [(author_num, resname), ...] — one CA ATOM each."""
    lines = [_ca_line(i + 1, rn, chain, num, float(i + 1))
             for i, (num, rn) in enumerate(residues)]
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


PROTEIN = [(1, "MET"), (2, "LYS"), (3, "ARG"), (4, "GLY"), (5, "ALA")]
PDB = _make_pdb(PROTEIN)
# Two-chain PDB to exercise the default-chain warning.
PDB_2CHAIN = (_make_pdb(PROTEIN, chain="A")[:-4] + _make_pdb(PROTEIN, chain="B"))


# ── build_preset5_config ─────────────────────────────────────────────────────────

def test_valid_config_builds_contig_and_length():
    cfg = build_preset5_config(PDB, partial_t=8.0, k=5, m=3)
    assert cfg.chain_id == "A"
    assert cfg.length == 5
    assert cfg.contig == "A1-5"
    assert cfg.partial_t == 8.0
    assert cfg.k == 5 and cfg.m == 3
    assert cfg.total_designs == 15


def test_params_roundtrip():
    cfg = build_preset5_config(PDB, partial_t=10.0, k=2, m=4, chain_id="A")
    params = cfg.to_params()
    assert params == {
        "chain_id": "A", "length": 5, "contig": "A1-5",
        "partial_t": 10.0, "k": 2, "m": 4,
    }


def test_partial_t_out_of_range_raises():
    with pytest.raises(ConfigError):
        build_preset5_config(PDB, partial_t=PARTIAL_T_MAX + 1, k=1, m=1)
    with pytest.raises(ConfigError):
        build_preset5_config(PDB, partial_t=0.5, k=1, m=1)


def test_k_and_m_bounds_raise():
    with pytest.raises(ConfigError):
        build_preset5_config(PDB, partial_t=8.0, k=K_MAX + 1, m=1)
    with pytest.raises(ConfigError):
        build_preset5_config(PDB, partial_t=8.0, k=1, m=M_MAX + 1)
    with pytest.raises(ConfigError):
        build_preset5_config(PDB, partial_t=8.0, k=0, m=1)


def test_multichain_defaults_first_with_warning():
    cfg = build_preset5_config(PDB_2CHAIN, partial_t=8.0, k=1, m=1)
    assert cfg.chain_id == "A"
    assert any("chains" in w for w in cfg.warnings)


def test_explicit_chain_selects_it():
    cfg = build_preset5_config(PDB_2CHAIN, partial_t=8.0, k=1, m=1, chain_id="B")
    assert cfg.chain_id == "B"
    assert cfg.contig == "B1-5"


def test_unknown_chain_raises():
    with pytest.raises(ConfigError):
        build_preset5_config(PDB, partial_t=8.0, k=1, m=1, chain_id="Z")


# ── submit routing (D11: preset → job) ───────────────────────────────────────────

def test_preset_job_routing(monkeypatch):
    from proteinredesign.submit import _job_name_for_preset

    monkeypatch.delenv("PROTEINREDESIGN_JOB_NAME", raising=False)
    monkeypatch.delenv("PROTEINREDESIGN_RF3_JOB_NAME", raising=False)

    # MPNN-only presets → the default mpnn-worker job.
    assert _job_name_for_preset(Preset.FIXED_BACKBONE_REDESIGN) == "proteinredesign-worker"
    assert _job_name_for_preset(Preset.LIGAND_AWARE_REDESIGN) == "proteinredesign-worker"
    # RF3 presets → the rf3-worker job.
    assert _job_name_for_preset(Preset.SCAFFOLD_DIVERSIFICATION) == "proteinredesign-rf3-worker"
    assert _job_name_for_preset(Preset.MOTIF_SCAFFOLDING) == "proteinredesign-rf3-worker"


def test_preset_job_routing_env_override(monkeypatch):
    from proteinredesign.submit import _job_name_for_preset

    monkeypatch.setenv("PROTEINREDESIGN_JOB_NAME", "custom-mpnn")
    monkeypatch.setenv("PROTEINREDESIGN_RF3_JOB_NAME", "custom-rf3")
    assert _job_name_for_preset(Preset.FIXED_BACKBONE_REDESIGN) == "custom-mpnn"
    assert _job_name_for_preset(Preset.SCAFFOLD_DIVERSIFICATION) == "custom-rf3"


# ── worker: pure ranking unaffected by the new RF3 fields ─────────────────────────

def test_ranking_handles_rf3_candidate_fields():
    from proteinredesign.worker import Candidate, select_top_candidates

    # Two RF3-style candidates (carry parent_backbone_path + diversity_from_input),
    # both passing the structural gate. Ranking must still work and be deterministic.
    c1 = Candidate(sequence="AAAA", mpnn_scores={"proteinmpnn": -1.0}, esm2_score=-0.5,
                   plddt=90.0, rmsd_to_design=1.0, parent_backbone_path="/bb1.pdb",
                   diversity_from_input=3.2)
    c2 = Candidate(sequence="CCCC", mpnn_scores={"proteinmpnn": -2.0}, esm2_score=-1.5,
                   plddt=85.0, rmsd_to_design=1.5, parent_backbone_path="/bb2.pdb",
                   diversity_from_input=5.1)
    top = select_top_candidates([c1, c2], num_outputs=10)
    assert [c.rank for c in top] == [1, 2]
    assert top[0].sequence == "AAAA"  # better ESM2 + lower (better) mpnn nll
    # Diversity metric is carried, not used as a gate.
    assert top[0].diversity_from_input == 3.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
