"""
Tests for concatemer assembly.

The search factorises because the liabilities are pairwise; these check that the factorisation
does not quietly lose valid designs, and that truncation — which is inevitable on a real spec —
is unbiased rather than an artefact of loop order.
"""

import itertools
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from concatemer.assemble import (  # noqa: E402
    Candidate, assemble, build, enumerate_counts, junction_cost,
    _order_grouped, _order_roundrobin,
)
from concatemer.spec import PRESET_RULES, ConcatemerSpec, Peptide, Spacer  # noqa: E402

TRYPSIN = PRESET_RULES["trypsin"]


def _spec(**kw):
    base = dict(peptides=[Peptide("A", "GHK", 1, 3), Peptide("B", "GQPR", 0, 3)],
                spacers=[Spacer("GAR", "GAR")], rules=[TRYPSIN],
                length_min=10, length_max=40, max_units=12)
    base.update(kw)
    return ConcatemerSpec(**base)


# ── multiset enumeration ───────────────────────────────────────────────────────

def test_pruning_loses_nothing_versus_brute_force():
    """The length lower bound is what keeps enumeration finite; it must not over-prune."""
    spec = _spec()
    got = {tuple(sorted(c.items())) for c in enumerate_counts(spec, spacer_len=3)}

    brute = set()
    for a in range(1, 4):
        for b in range(0, 4):
            units = a + b
            if not units or units > spec.max_units:
                continue
            total = a * 3 + b * 4 + (units - 1) * 3
            if spec.length_min <= total <= spec.length_max:
                brute.add((("A", a), ("B", b)))
    assert got == brute and got


def test_mandatory_copies_always_present():
    spec = _spec(peptides=[Peptide("A", "GHK", 2, 4), Peptide("B", "GQPR", 0, 3)])
    for counts in enumerate_counts(spec, spacer_len=3):
        assert counts["A"] >= 2


def test_copy_ceilings_respected():
    spec = _spec(peptides=[Peptide("A", "GHK", 0, 2), Peptide("B", "GQPR", 0, 2)],
                 length_min=3, length_max=200)
    for counts in enumerate_counts(spec, spacer_len=0):
        assert counts["A"] <= 2 and counts["B"] <= 2


def test_unit_cap_is_enforced():
    spec = _spec(peptides=[Peptide("A", "GHK", 0, 30)], spacers=[],
                 length_min=3, length_max=1000, max_units=5)
    for counts in enumerate_counts(spec, spacer_len=0):
        assert sum(counts.values()) <= 5


def test_every_candidate_lands_inside_the_length_envelope():
    spec = _spec()
    cands, _ = assemble(spec)
    assert cands
    for c in cands:
        assert spec.length_min <= c.length <= spec.length_max


# ── junction scoring ───────────────────────────────────────────────────────────

def test_junction_sequon_is_penalised():
    """Neither unit contains N-X-S/T; joining them creates one."""
    cost, flags = junction_cost("GHKN", "GTA", _spec())
    assert flags["junction_sequon"] == 1 and cost > 0


def test_clean_junction_costs_nothing():
    cost, flags = junction_cost("GHK", "GAR", _spec())
    assert cost == 0.0 and not any(flags.values())


def test_proline_blocking_a_junction_site_is_penalised():
    """A Pro-led unit after a K/R-terminated one blocks the release at that join."""
    cost, flags = junction_cost("GHK", "PAPA", _spec())
    assert flags["blocked_site"] == 1 and cost > 0


def test_hydrophobic_run_spanning_a_join_is_penalised():
    cost, flags = junction_cost("GGVLI", "VLIGG", _spec())
    assert flags["hydrophobic_run"] > 0 and cost > 0


# ── ordering architectures ─────────────────────────────────────────────────────

def test_grouped_and_roundrobin_differ_and_preserve_composition():
    units = [Peptide("A", "GHK")] * 2 + [Peptide("B", "GQPR")] * 2
    g = [u.name for u in _order_grouped(units)]
    r = [u.name for u in _order_roundrobin(units)]
    assert g == ["A", "A", "B", "B"]
    assert r != g
    assert Counter(g) == Counter(r) == Counter(["A", "A", "B", "B"])


def test_build_interleaves_the_spacer_and_totals_the_junctions():
    spec = _spec()
    cand = build([Peptide("A", "GHK")] * 3, Spacer("GAR", "GAR"), spec, "grouped")
    assert cand.sequence == "GHKGARGHKGARGHK"
    assert cand.counts == {"A": 3} and cand.n_units == 3
    assert cand.junction_cost == 0.0


def test_no_spacer_builds_a_bare_tandem():
    cand = build([Peptide("A", "GHK")] * 3, None, _spec(), "grouped")
    assert cand.sequence == "GHKGHKGHK" and cand.spacer == ""


# ── the search as a whole ──────────────────────────────────────────────────────

def test_identical_chains_are_deduplicated():
    """A single-peptide multiset orders identically under every architecture."""
    spec = _spec(peptides=[Peptide("A", "GHK", 3, 3)], spacers=[], length_min=9, length_max=9)
    cands, _ = assemble(spec)
    assert len({c.sequence for c in cands}) == len(cands) == 1


def test_truncation_is_unbiased_across_spacers():
    """
    Draining one spacer at a time makes the cap discard whole spacer families, which reads as a
    complete result and is not one.
    """
    spec = _spec(peptides=[Peptide("A", "GHK", 1, 6), Peptide("B", "GQPR", 1, 6)],
                 spacers=[Spacer("GAR", "GAR"), Spacer("GGAR", "GGAR"), Spacer("GGA", "GGA")],
                 length_min=30, length_max=120, max_units=20)
    cands, stats = assemble(spec, max_candidates=40)   # the spec yields 185 in full
    assert stats["truncated_candidates"]
    seen = Counter(c.spacer for c in cands)
    assert len([k for k in seen if k]) == 3, f"spacer families lost under truncation: {seen}"


def test_stats_report_truncation_rather_than_hiding_it():
    spec = _spec(peptides=[Peptide("A", "GHK", 1, 8)], length_min=10, length_max=200)
    _, stats = assemble(spec, max_candidates=3)
    assert stats["truncated_candidates"] is True and stats["generated"] == 3


def test_an_impossible_envelope_yields_nothing_without_erroring():
    spec = _spec(peptides=[Peptide("A", "GHK", 1, 2)], spacers=[],
                 length_min=500, length_max=600, max_units=4)
    cands, stats = assemble(spec)
    assert cands == [] and stats["generated"] == 0
