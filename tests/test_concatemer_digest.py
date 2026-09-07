"""
Tests for the concatemer digest simulator — the objective function.

Each case here is a design failure mode that no expression feature would catch, and several are
counter-intuitive enough that they would otherwise only surface at the bench.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from concatemer.digest import BLOCKED, CLEAN, IMPAIRED, cut, digest, find_sites  # noqa: E402
from concatemer.spec import (  # noqa: E402
    PRESET_RULES, CleavageRule, ConcatemerSpec, Peptide, Spacer, peptide_mass, residue_mass,
)

GHK = Peptide("GHK", "GHK")            # ATCUN copper tripeptide; ends in K
KTTKS = Peptide("KTTKS", "KTTKS")      # collagen I propeptide fragment; INTERNAL K
TRYPSIN = PRESET_RULES["trypsin"]
KEX2 = PRESET_RULES["kex2"]
KEX2_KEX1 = PRESET_RULES["kex2_kex1"]


def _spec(peptides, rules, spacers=()):
    return ConcatemerSpec(peptides=list(peptides), spacers=list(spacers), rules=list(rules))


# ── the happy path ─────────────────────────────────────────────────────────────

def test_peptides_ending_in_basic_need_no_spacer_at_all():
    """
    A peptide whose own C-terminus is K or R makes trypsin release it natively: the cut falls
    exactly at its terminus, and the next peptide starts clean. Zero-linker tandem, 100% payload.
    """
    rep = digest("GHK" * 4, _spec([GHK], [TRYPSIN]))
    assert [f.seq for f in rep.fragments] == ["GHK"] * 4
    assert rep.released_exact["GHK"] == 4
    assert rep.n_unintended == 0
    assert rep.payload_fraction_mass == pytest.approx(100.0)


def test_molar_yield_is_per_copy_and_gives_the_delivered_blend():
    seq = "GHK" * 4 + "AAR" + "GHK" * 2
    rep = digest(seq, _spec([GHK, Peptide("AAR", "AAR")], [TRYPSIN]))
    assert rep.molar_yield["GHK"] == 6.0
    assert rep.molar_yield["AAR"] == 1.0
    assert rep.delivered_ratio == {"GHK": 6.0, "AAR": 1.0}


# ── failure modes ──────────────────────────────────────────────────────────────

def test_kex1_processivity_eats_past_a_basic_c_terminus():
    """
    THE trap for GHK. Kex1 removes C-terminal K/R processively, so a peptide that itself ends in
    K is trimmed past its own terminus: GHK|KR -> GHKKR -> GHKK -> GHK -> GH. The construct looks
    textbook and delivers a different molecule.
    """
    rep = digest("GHKKR" * 3, _spec([GHK], [KEX2_KEX1], [Spacer("KR", "KR")]))
    assert [f.seq for f in rep.fragments] == ["GH"] * 3
    assert rep.released_exact["GHK"] == 0
    assert rep.n_unintended == 3


def test_kex2_without_kex1_leaves_the_spacer_attached():
    """Same construct, no carboxypeptidase: the KR stays on, so the C-terminus is still wrong."""
    rep = digest("GHKKR" * 3, _spec([GHK], [KEX2], [Spacer("KR", "KR")]))
    assert [f.seq for f in rep.fragments] == ["GHKKR"] * 3
    assert rep.released_exact["GHK"] == 0


def test_internal_basic_residue_shreds_the_peptide():
    """KTTKS carries an internal K, so trypsin cuts inside it. Zero yield, seven fragments."""
    rep = digest("KTTKS" * 3, _spec([KTTKS], [TRYPSIN]))
    assert rep.released_exact["KTTKS"] == 0
    assert rep.n_unintended == 7
    assert rep.payload_fraction_mass == 0.0


def test_proline_at_p1_prime_blocks_cleavage_entirely():
    """A Pro-rich spacer immediately after a site silently kills that release event."""
    rep = digest("GHK" + "PAPAP" + "GHK", _spec([GHK], [TRYPSIN], [Spacer("PAPAP", "PAPAP")]))
    assert rep.n_blocked == 1 and rep.n_clean == 0
    assert rep.fragments[0].seq == "GHKPAPAPGHK"      # never cut
    assert rep.released_exact["GHK"] == 0


def test_c_side_chemistry_puts_the_spacer_on_the_next_peptide():
    """
    Non-obvious and easy to design past: with C-side cleavage the spacer is carried into the
    DOWNSTREAM fragment's N-terminus, so only the first copy releases cleanly.
    """
    rep = digest("GHKGA" * 4, _spec([GHK], [TRYPSIN], [Spacer("GA", "GA")]))
    assert rep.released_exact["GHK"] == 1
    assert [f.seq for f in rep.fragments].count("GAGHK") == 3


def test_a_spacer_ending_in_basic_fixes_that():
    """Give the spacer its own C-side site and every copy comes back."""
    rep = digest("GHKGAR" * 4, _spec([GHK], [TRYPSIN], [Spacer("GAR", "GAR")]))
    assert rep.released_exact["GHK"] == 4
    assert rep.n_unintended == 0


# ── site classification ────────────────────────────────────────────────────────

def test_acidic_p1_prime_is_impaired_not_blocked():
    sites = find_sites("GHKEAAG", [KEX2 if False else TRYPSIN])
    assert [s.status for s in sites] == [IMPAIRED]
    assert "E" in sites[0].reason


def test_terminal_matches_are_not_cuts():
    """A basic residue at the C-terminus is not a cleavage site — there is nothing to release."""
    assert find_sites("GHK", [TRYPSIN]) == []


def test_overlapping_motifs_are_all_found():
    """In KRR trypsin sees successive basics; a naive finditer would miss the overlap."""
    assert [s.pos for s in find_sites("AKRRA", [TRYPSIN])] == [2, 3, 4]


def test_worst_status_wins_when_rules_collide():
    """One chemistry ignoring a proline block must not rescue a site another rule blocks."""
    permissive = CleavageRule("permissive", r"K", "C", blocked_by="")
    sites = find_sites("GKPA", [TRYPSIN, permissive])
    assert [s.status for s in sites] == [BLOCKED]


def test_blocked_sites_do_not_split_the_chain():
    seq = "GKPGKA"
    frags = cut(seq, find_sites(seq, [TRYPSIN]))
    assert [f.seq for f in frags] == ["GKPGK", "A"]      # cut after the 2nd K only


# ── mass accounting ────────────────────────────────────────────────────────────

def test_payload_fraction_uses_residue_mass_not_free_peptide_mass():
    """
    Hydrolysis consumes a water per bond, so summed free-peptide masses exceed the parent
    protein's. Measuring payload that way reports >100% for an all-payload chain.
    """
    seq = "GHK" * 4
    naive = 100.0 * 4 * peptide_mass("GHK") / peptide_mass(seq)
    assert naive > 100.0                                  # the bug this guards against
    assert digest(seq, _spec([GHK], [TRYPSIN])).payload_fraction_mass == pytest.approx(100.0)


def test_spacer_mass_dilutes_payload():
    rep = digest("GHKGAR" * 4, _spec([GHK], [TRYPSIN], [Spacer("GAR", "GAR")]))
    expected = 100.0 * 4 * residue_mass("GHK") / residue_mass("GHKGAR" * 4)
    assert rep.payload_fraction_mass == pytest.approx(expected)
    assert 0.0 < rep.payload_fraction_mass < 100.0


# ── spec validation ────────────────────────────────────────────────────────────

def test_spec_rejects_a_design_that_cannot_release_anything():
    spec = ConcatemerSpec(peptides=[GHK], rules=[])
    assert any("cleavage rule" in e for e in spec.errors())


def test_spec_rejects_mandatory_copies_that_cannot_fit():
    spec = ConcatemerSpec(peptides=[Peptide("P", "GHK", min_copies=10)],
                          rules=[TRYPSIN], length_max=20)
    assert any("above length_max" in e for e in spec.errors())


def test_spec_rejects_non_standard_residues():
    assert any("non-standard" in e for e in Peptide("bad", "GHX").errors())
