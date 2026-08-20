"""
Unit tests for proteinredesign.config_builders.preset3 (motif scaffolding / inpainting —
RF3 indexed multi-segment contig).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.config_builders.preset3 import (  # noqa: E402
    ConfigError,
    K_MAX,
    M_MAX,
    build_preset3_config,
)


def _ca_line(serial, resname, chain, resseq):
    return (
        "%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
        % ("ATOM", serial, "CA", "", resname, chain, resseq, "", float(serial), 0.0, 0.0, 1.0, 0.0, "C")
    )


def _make_pdb(n=200, chain="A"):
    lines = [_ca_line(i, "ALA", chain, i) for i in range(1, n + 1)]
    lines.append("END")
    return ("\n".join(lines) + "\n").encode()


PDB = _make_pdb(200)


def test_valid_multisegment_contig():
    cfg = build_preset3_config(PDB, "1-20, 50-80, 130-160, 190-200", k=5, m=3)
    assert cfg.chain_id == "A"
    assert cfg.keep_ranges == [(1, 20), (50, 80), (130, 160), (190, 200)]
    # gaps: 21-49=29, 81-129=49, 161-189=29
    assert cfg.gaps == [29, 49, 29]
    assert cfg.contig == "A1-20,29,A50-80,49,A130-160,29,A190-200"
    assert cfg.total_designs == 15


def test_motif_residues_cover_all_kept():
    cfg = build_preset3_config(PDB, "1-3, 10-11", k=1, m=1)
    nums = [r["author_num"] for r in cfg.motif_residues]
    assert nums == [1, 2, 3, 10, 11]
    assert all(r["chain_id"] == "A" for r in cfg.motif_residues)


def test_params_roundtrip():
    cfg = build_preset3_config(PDB, "1-20, 50-80", k=2, m=2)
    p = cfg.to_params()
    assert p["contig"] == "A1-20,29,A50-80"
    assert p["gaps"] == [29]
    assert p["keep_ranges"] == [[1, 20], [50, 80]]
    assert len(p["motif_residues"]) == 20 + 31   # 1-20 (20 residues) + 50-80 (31 residues)


def test_ranges_sorted_regardless_of_input_order():
    cfg = build_preset3_config(PDB, "50-80, 1-20", k=1, m=1)
    assert cfg.contig == "A1-20,29,A50-80"


def test_single_block_raises():
    with pytest.raises(ConfigError):
        build_preset3_config(PDB, "1-20", k=1, m=1)


def test_overlapping_blocks_raise():
    with pytest.raises(ConfigError):
        build_preset3_config(PDB, "1-20, 15-40", k=1, m=1)


def test_touching_blocks_raise():
    # 1-20 and 21-40 leave no gap to fill.
    with pytest.raises(ConfigError):
        build_preset3_config(PDB, "1-20, 21-40", k=1, m=1)


def test_residue_out_of_chain_raises():
    with pytest.raises(ConfigError):
        build_preset3_config(PDB, "1-20, 50-250", k=1, m=1)  # 250 > 200


def test_bad_range_string_raises():
    with pytest.raises(ConfigError):
        build_preset3_config(PDB, "1-20, fifty", k=1, m=1)


def test_k_m_bounds_raise():
    with pytest.raises(ConfigError):
        build_preset3_config(PDB, "1-20, 50-80", k=K_MAX + 1, m=1)
    with pytest.raises(ConfigError):
        build_preset3_config(PDB, "1-20, 50-80", k=1, m=M_MAX + 1)


def test_routes_to_rf3_job(monkeypatch):
    from proteinredesign.manifest import Preset
    from proteinredesign.submit import _job_name_for_preset
    monkeypatch.delenv("PROTEINREDESIGN_RF3_JOB_NAME", raising=False)
    assert _job_name_for_preset(Preset.MOTIF_SCAFFOLDING) == "proteinredesign-rf3-worker"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
