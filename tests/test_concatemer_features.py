"""
Tests for the feature stage, the failure gate and ranking.

The gate carries most of the discrimination — it is the part that transfers out of distribution,
where a fitted ranking coefficient would not — so its flags are tested individually.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from concatemer import screens  # noqa: E402
from concatemer.digest import digest  # noqa: E402
from concatemer.features import FeatureRow, compute, failure_flags  # noqa: E402
from concatemer.pipeline import run, to_csv, to_fasta  # noqa: E402
from concatemer.rank import _average_ranks, rank  # noqa: E402
from concatemer.spec import PRESET_RULES, ConcatemerSpec, Peptide, Spacer  # noqa: E402

TRYPSIN = PRESET_RULES["trypsin"]
KEX2 = PRESET_RULES["kex2"]


def _spec(**kw):
    base = dict(peptides=[Peptide("GHK", "GHK", 1, 6)], spacers=[Spacer("GAR", "GAR")],
                rules=[TRYPSIN], length_min=6, length_max=200, max_units=20)
    base.update(kw)
    return ConcatemerSpec(**base)


def _flags(seq, spec=None):
    spec = spec or _spec()
    return failure_flags(seq, spec, digest(seq, spec))


# ── charge patterning ──────────────────────────────────────────────────────────

def test_sigma_is_defined_on_fractions_not_counts():
    """
    Counts make sigma scale-dependent: a blob is compared against the whole chain, so a long
    sequence reports huge spurious patterning. The first implementation gave kappa=0.97 for a
    chain with no negative residues at all to pattern against.
    """
    assert screens._sigma("KKKKK") == pytest.approx(1.0)
    assert screens._sigma("KKKKKKKKKKKKKKKKKKKK") == pytest.approx(1.0)   # scale-free
    assert screens._sigma("KEKEKEKEKE") == pytest.approx(0.0)


def test_kappa_separates_mixed_from_blocky_charge():
    assert screens.kappa("EKEKEKEKEKEKEKEK") < 0.05
    assert screens.kappa("KKKKKKKKEEEEEEEE") == pytest.approx(1.0)


def test_kappa_is_near_zero_for_a_single_signed_chain():
    """No patterning to measure; the liability there is net charge, carried separately."""
    assert screens.kappa("GHKGAR" * 10) < 0.1


# ── failure flags ──────────────────────────────────────────────────────────────

def test_free_cysteine_is_flagged():
    assert any("cysteine" in f for f in _flags("GHKCGARGHK"))


def test_er_retention_motif_is_flagged():
    assert any("retention" in f for f in _flags("GHKGARGHKHDEL"))
    assert not any("retention" in f for f in _flags("GHKGARGHKHDELA"))   # C-terminal only


def test_kr_is_flagged_when_the_design_does_not_nominate_a_kr_chemistry():
    """
    An in-vitro-digest design containing KR is cut by Kex2 in the Golgi regardless — processed
    before it ever leaves the cell. No in-vitro simulation would reveal that.
    """
    asp_n = PRESET_RULES["asp_n"]
    got = _flags("GDHKRGDHKR", _spec(rules=[asp_n]))
    assert any("Golgi" in f for f in got)


def test_kr_is_not_flagged_when_kex2_is_the_intended_chemistry():
    spec = _spec(rules=[KEX2], spacers=[Spacer("KR", "KR")])
    assert not any("Golgi" in f for f in _flags("GHAKRGHAKR", spec))


def test_a_globular_chain_violates_the_architecture():
    assert any("globular" in f for f in _flags("VLIVLIVLIVLIVLIVLIVLI"))


def test_no_cleavage_sites_is_flagged():
    spec = _spec(peptides=[Peptide("AAA", "AAA", 1, 5)], spacers=[], rules=[TRYPSIN])
    assert any("no cleavage sites" in f for f in _flags("AAAAAAAAA", spec))


def test_a_mandatory_peptide_that_never_releases_is_flagged():
    """GPKG's internal K means trypsin splits it; it can never be delivered by this chemistry."""
    spec = _spec(peptides=[Peptide("GPKG", "GPKG", min_copies=1, max_copies=4)], spacers=[])
    assert any("mandatory" in f for f in _flags("GPKGGPKGGPKG", spec))


def test_a_clean_design_raises_no_flags():
    assert _flags("GHKGARGHKGARGHKGAR" * 2) == []


# ── ranking ────────────────────────────────────────────────────────────────────

def test_average_ranks_share_ties():
    assert _average_ranks([5.0, 5.0, 1.0], higher_is_better=True) == [1.5, 1.5, 3.0]
    assert _average_ranks([1.0, 2.0, 3.0], higher_is_better=False) == [1.0, 2.0, 3.0]


def test_product_metrics_decide_the_order_not_one_sixth_of_it():
    """
    Equal group weighting made a 90%-release candidate outrank a 100% one. The product group is
    the objective; the rest is manufacturability.
    """
    spec = ConcatemerSpec(
        peptides=[Peptide("GHK", "GHK", 2, 6), Peptide("GQPR", "GQPR", 1, 5)],
        spacers=[Spacer("GAR", "GAR"), Spacer("GGA", "GGA")],
        rules=[TRYPSIN], length_min=40, length_max=90, max_units=20)
    res = run(spec, max_candidates=800)
    assert res.passed
    best = res.passed[0].values["pct_copies_exact"]
    assert best == max(r.values["pct_copies_exact"] for r in res.passed)
    assert best == pytest.approx(100.0)


def test_every_ranked_row_carries_its_reasons():
    res = run(_spec(), max_candidates=100)
    assert res.passed
    assert all(r.reasons and r.worst_group for r in res.passed)


def test_failed_rows_are_returned_unranked_with_their_failures():
    spec = _spec(peptides=[Peptide("GPKG", "GPKG", 1, 6)], spacers=[])
    res = run(spec, max_candidates=100)
    assert res.failed
    assert all(r.rank == 0 and r.failures for r in res.failed)


# ── pipeline plumbing ──────────────────────────────────────────────────────────

def test_funnel_accounts_for_every_candidate():
    res = run(_spec(), max_candidates=200)
    counts = dict(res.funnel)
    assert counts["passed failure gates"] + counts["below the gate"] == counts["assembled"]
    assert counts["assembled"] == len(res.rows)


def test_spec_errors_short_circuit_the_run():
    res = run(ConcatemerSpec(peptides=[Peptide("A", "GHK")], rules=[]))
    assert res.errors and not res.rows


def test_csv_carries_every_ranking_feature():
    from concatemer.features import FEATURES
    res = run(_spec(), max_candidates=50)
    header = to_csv(res).splitlines()[0].split(",")
    for f in FEATURES:
        assert f.key in header
    assert "failures" in header and "reasons" in header


def test_fasta_headers_are_self_describing():
    res = run(_spec(), max_candidates=20)
    fa = to_fasta(res, limit=2)
    head = fa.splitlines()[0]
    assert head.startswith(">") and "exact=" in head and "payload=" in head
    assert len(fa.splitlines()) == 4                       # 2 records, 2 lines each
