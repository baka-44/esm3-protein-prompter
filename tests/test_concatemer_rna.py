"""
Tests for start-codon accessibility.

This is the one RNA measurement that needs the whole transcript: a cargo can pair back into the
initiation window from hundreds of nucleotides away, and nothing computed on the cargo alone
would show it.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from concatemer.rna import (  # noqa: E402
    MIN_HELIX_BP, MissingContextError, TranscriptContext, accessibility, assemble_transcript,
    flags, longest_helix, pair_map,
)

RC = {"A": "U", "U": "A", "G": "C", "C": "G"}
UTR5 = "AAUAAUAAACAAAUACAAAUAAACAAUAACAAAUAACAA"
SIGNAL = "AUG" + "AAGUUCCCAUCAAUCUUCACAGCUGUUUUAUUCGCAGCAUCCUCCGCAUUA" * 2


def _revcomp(s):
    return "".join(RC[c] for c in reversed(s))


@pytest.fixture(scope="module")
def ctx():
    return TranscriptContext(utr5=UTR5, signal_cds=SIGNAL, name="synthetic")


# ── the context is required, never guessed ─────────────────────────────────────

def test_context_refuses_to_be_defaulted():
    """
    mRNA structure is exquisitely sequence-dependent, so a plausible substitute — a canonical
    AOX1 5'UTR, or a back-translated alpha-MF — returns a confident wrong answer. Better to
    refuse than to invent.
    """
    with pytest.raises(MissingContextError):
        TranscriptContext(utr5="", signal_cds=SIGNAL)
    with pytest.raises(MissingContextError):
        TranscriptContext(utr5=UTR5, signal_cds="")


def test_signal_must_begin_at_the_start_codon():
    with pytest.raises(MissingContextError):
        TranscriptContext(utr5=UTR5, signal_cds="GGGAUGAAA")


def test_dna_input_is_normalised_to_rna():
    c = TranscriptContext(utr5="AATAAT", signal_cds="ATGAAATTT")
    assert c.utr5 == "AAUAAU" and c.signal_cds.startswith("AUG")


# ── helpers ────────────────────────────────────────────────────────────────────

def test_pair_map_pairs_brackets():
    assert pair_map("((..))") == {0: 5, 5: 0, 1: 4, 4: 1}
    assert pair_map("....") == {}


def test_pair_map_survives_unbalanced_input():
    assert pair_map(")))") == {}


def test_longest_helix_requires_an_antiparallel_run():
    assert longest_helix([(1, 50), (2, 49), (3, 48)]) == 3     # a real duplex
    assert longest_helix([(1, 50), (2, 60), (3, 20)]) == 1     # scattered pairs
    assert longest_helix([]) == 0


def test_assemble_puts_the_atg_after_the_utr(ctx):
    t, atg = assemble_transcript(ctx, "AAACCC")
    assert atg == len(UTR5)
    assert t[atg:atg + 3] == "AUG"
    assert t.endswith("AAACCC")


# ── the measurement ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("complementary_nt,expect", [(8, False), (12, True), (24, True)])
def test_detection_floor_matches_the_calibration(ctx, complementary_nt, expect):
    """8 nt of complementarity does not form a duplex; 12 and above do."""
    transcript, atg = assemble_transcript(ctx, "")
    window = transcript[atg - 4:atg + 37]
    cargo = ("AAUCAAAUCAAU" * 3 + _revcomp(window[:complementary_nt]) + "AAUCAAAUCAAU" * 6)
    assert accessibility(ctx, cargo).cargo_sequesters_start is expect


def test_a_cargo_complementary_to_the_window_sequesters_the_start(ctx):
    transcript, atg = assemble_transcript(ctx, "")
    window = transcript[atg - 4:atg + 37]
    trap = "AAUCAAAUCAAU" * 3 + _revcomp(window) + "AAUCAAAUCAAU" * 3

    rep = accessibility(ctx, trap)
    assert rep.cargo_sequesters_start
    assert rep.longest_cargo_helix >= 20            # a designed duplex, not chance
    assert rep.frac_paired > 0.8
    assert not rep.accessible
    assert any("re-encode the cargo" in f for f in flags(rep))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_random_cargo_does_not_trip_the_flag(seed):
    """
    An MFE fold always finds scattered pairs. Over 60 random cargos the chance helix peaked at
    4-5 bp and never exceeded 8, while designed complementarity of 12 nt or more registers at
    full length — an 8-nt stretch does not form a duplex at all. The threshold sits in that gap.
    """
    c = TranscriptContext(utr5=UTR5, signal_cds=SIGNAL)
    random.seed(seed)
    cargo = "".join(random.choice("ACGU") for _ in range(250))
    rep = accessibility(c, cargo)
    assert rep.longest_cargo_helix < MIN_HELIX_BP
    assert not rep.cargo_sequesters_start


def test_window_positions_are_reported_relative_to_the_atg(ctx):
    transcript, atg = assemble_transcript(ctx, "")
    window = transcript[atg - 4:atg + 37]
    rep = accessibility(ctx, "AAUCAAAUCAAU" * 3 + _revcomp(window) + "AAUCAAAUCAAU" * 3)
    offsets = [i for i, _ in rep.partners_in_cargo]
    assert min(offsets) >= -4 and max(offsets) < 37


def test_long_transcripts_are_truncated_and_say_so(ctx):
    rep = accessibility(ctx, "AAUCAAAUCAAU" * 60, max_fold_nt=300)
    assert rep.truncated and rep.folded_nt == 300


def test_a_clean_result_raises_no_flags(ctx):
    random.seed(11)
    rep = accessibility(ctx, "".join(random.choice("ACGU") for _ in range(200)))
    assert not rep.cargo_sequesters_start
    assert not any("re-encode the cargo" in f for f in flags(rep))
