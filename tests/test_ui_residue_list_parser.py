"""Tests for the Composer's residue-list parser (the bulk-entry box for repack / fixed tiers)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from ui.composer_panel import _parse_residue_list  # noqa: E402


def test_bare_numbers_pass_through_as_ints():
    assert _parse_residue_list("280, 282 323") == [280, 282, 323]


def test_ranges_expand():
    assert _parse_residue_list("407-410") == [407, 408, 409, 410]


def test_chain_qualified_tokens_keep_their_chain():
    assert _parse_residue_list("B276-278 A42") == ["B276", "B277", "B278", "A42"]


def test_commas_and_whitespace_are_interchangeable():
    assert _parse_residue_list("1,2  3,  4") == [1, 2, 3, 4]


def test_empty_input_yields_nothing():
    assert _parse_residue_list("") == [] and _parse_residue_list("   ") == []


def test_the_real_kex2_redesign_list_parses():
    """The 23 exposed residues from the P-domain deletion, as a user would paste them."""
    got = _parse_residue_list("280 282 323 324 329 331 353 354 356 407-409 411 412 415 "
                              "430-434 438 440 444")
    assert len(got) == 23 and got[0] == 280 and got[-1] == 444


def test_malformed_token_raises_rather_than_silently_dropping():
    # a silently-ignored residue list would look like success while designing nothing
    with pytest.raises(ValueError):
        _parse_residue_list("28O")     # letter O, not a zero
