"""
Unit tests for proteinredesign.worker.select_top_candidates — the pure QC-gate + ranking
logic (D2 hard gate + ESM2 soft floor + B3 metadata ranking). No ML deps needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinredesign.worker import Candidate, select_top_candidates  # noqa: E402


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


def test_failing_candidates_are_reported_flagged_not_discarded():
    """An empty result tells you only that something failed — not how badly, or on which axis.

    When nothing clears the gate the candidates come back ranked and flagged, so the run is
    still inspectable and the gate's distribution is visible enough to calibrate.
    """
    out = select_top_candidates([_c("X", plddt=10, rmsd=9.0)], num_outputs=10)
    assert len(out) == 1
    c = out[0]
    assert c.passed_gate is False
    assert any("pLDDT" in f for f in c.gate_failures)
    assert any("RMSD-to-design" in f for f in c.gate_failures)
    assert c.rank == 1                      # still ranked, so "best of a bad set" is readable


def test_passing_candidates_are_marked_clean():
    out = select_top_candidates([_c("OK", plddt=90, rmsd=1.0)], num_outputs=10)
    assert out[0].passed_gate is True and out[0].gate_failures == []


def test_a_passing_candidate_is_preferred_over_failing_ones():
    good = _c("GOOD", plddt=90, rmsd=1.0)
    bad = _c("BAD", plddt=10, rmsd=9.0)
    out = select_top_candidates([bad, good], num_outputs=10)
    # the gate still selects: only the clean one is returned when one exists
    assert [c.sequence for c in out] == ["GOOD"]


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


# ── Engine helpers added for #6 / Borrowed Bodies (FG1) ──────────────────────────

def test_mpnn_fixed_tokens_excludes_repack_and_dedups():
    from proteinredesign.worker import _mpnn_fixed_tokens
    # #6: no repack — all catalytic residues kept.
    assert _mpnn_fixed_tokens(["A57", "A102", "A195"]) == "A57 A102 A195"
    # BB2: interface residues in the repack set are dropped so MPNN redesigns them.
    assert _mpnn_fixed_tokens(["A57", "A102", "B10", "B11"], repack_refs=["B10", "B11"]) == "A57 A102"
    # De-dup, order-preserving.
    assert _mpnn_fixed_tokens(["A1", "A1", "A2"]) == "A1 A2"


def test_mpnn_checkpoint_selection_by_ligand():
    from proteinredesign.worker import _mpnn_checkpoint_for
    assert _mpnn_checkpoint_for({}) == "proteinmpnn"
    assert _mpnn_checkpoint_for({"ligand": None}) == "proteinmpnn"
    assert _mpnn_checkpoint_for({"ligand": {"resname": "NAI"}}) == "ligand_mpnn"


def test_motif_rmsd_gate():
    # motif gate off (inf) → NaN motif_rmsd is fine.
    c_nan = _c("AAA", plddt=90, rmsd=1.0)
    assert len(select_top_candidates([c_nan], num_outputs=10)) == 1
    # finite gate → NaN motif_rmsd fails (QC couldn't be computed), low passes, high fails.
    good = _c("GGG", plddt=90, rmsd=1.0); good.motif_rmsd = 0.8
    bad = _c("BBB", plddt=90, rmsd=1.0); bad.motif_rmsd = 3.0
    nan = _c("NNN", plddt=90, rmsd=1.0)  # motif_rmsd stays NaN
    out = select_top_candidates([good, bad, nan], num_outputs=10, motif_rmsd_gate=1.5)
    assert [c.sequence for c in out] == ["GGG"]
