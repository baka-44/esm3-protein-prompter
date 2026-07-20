"""
Unit tests for cofold.worker.select_top_candidates — the pure QC-gate + ranking
logic (D2 hard gate + ESM2 soft floor + B3 metadata ranking). No ML deps needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cofold.worker import Candidate, select_top_candidates  # noqa: E402


def _c(seq, plddt=80.0, rmsd=1.0, esm2=-0.5, pmpnn=0.5, soluble=0.5):
    return Candidate(
        sequence=seq,
        mpnn_scores={"proteinmpnn": pmpnn, "soluble_mpnn": soluble},
        esm2_score=esm2,
        plddt=plddt,
        rmsd_to_design=rmsd,
    )


def test_structural_gate_filters():
    good = _c("AAA", plddt=80, rmsd=1.0)
    low_plddt = _c("BBB", plddt=60, rmsd=1.0)
    high_rmsd = _c("CCC", plddt=80, rmsd=3.0)
    out = select_top_candidates([good, low_plddt, high_rmsd], num_outputs=10,
                                plddt_gate=70, rmsd_gate=2.0)
    assert [c.sequence for c in out] == ["AAA"]


def test_empty_when_none_pass():
    out = select_top_candidates([_c("X", plddt=10, rmsd=9.0)], num_outputs=10)
    assert out == []


def test_caps_at_num_outputs_and_assigns_ranks():
    cands = [_c(f"S{i}", esm2=-float(i) / 100.0) for i in range(15)]
    out = select_top_candidates(cands, num_outputs=10, esm2_drop_fraction=0.0)
    assert len(out) == 10
    assert [c.rank for c in out] == list(range(1, 11))


def test_ranking_prefers_good_esm2_and_good_mpnn():
    # MPNN score: LOWER is better. ESM2: HIGHER is better.
    best = _c("BEST", esm2=-0.4, pmpnn=0.4, soluble=0.4)
    good_esm2_bad_mpnn = _c("A", esm2=-0.4, pmpnn=1.2, soluble=1.2)
    bad_esm2_good_mpnn = _c("B", esm2=-2.5, pmpnn=0.4, soluble=0.4)
    out = select_top_candidates([good_esm2_bad_mpnn, bad_esm2_good_mpnn, best],
                                num_outputs=3, esm2_drop_fraction=0.0)
    assert out[0].sequence == "BEST"
    assert out[0].rank == 1


def test_soft_floor_drops_unnatural_tail_with_headroom():
    cands = [_c(f"ok{i}", esm2=-0.5) for i in range(11)]
    outlier = _c("OUTLIER", esm2=-5.0)  # clearly-unnatural tail
    out = select_top_candidates(cands + [outlier], num_outputs=10, esm2_drop_fraction=0.10)
    assert len(out) == 10
    assert "OUTLIER" not in [c.sequence for c in out]


def test_soft_floor_never_starves_output():
    # Exactly num_outputs pass the structural gate → soft floor must NOT prune,
    # even the low one, because that would drop below the requested count.
    cands = [_c(f"ok{i}", esm2=-0.5) for i in range(9)] + [_c("LOW", esm2=-5.0)]
    out = select_top_candidates(cands, num_outputs=10, esm2_drop_fraction=0.10)
    assert len(out) == 10
    assert "LOW" in [c.sequence for c in out]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
