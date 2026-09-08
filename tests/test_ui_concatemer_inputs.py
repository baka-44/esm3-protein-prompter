"""
Tests for parsing the composer's editable tables.

The panel crashed in production for anyone who added a peptide row and left a copy-number cell
blank. `int(value or default)` reads as safe and is not: an empty data_editor cell arrives as
float('nan'), NaN is TRUTHY, so `nan or 0` evaluates to nan and int(nan) raises ValueError. The
whole panel died on a normal interaction.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from concatemer.spec import PRESET_RULES, ConcatemerSpec, Peptide  # noqa: E402
from ui.concatemer_panel import _build_spec, _cell_int, _cell_text  # noqa: E402

TRYPSIN = PRESET_RULES["trypsin"]


def test_nan_is_truthy_which_is_the_whole_bug():
    """Pinning the premise, so nobody reintroduces `value or default`."""
    assert bool(float("nan")) is True
    assert (float("nan") or 0) != 0
    with pytest.raises(ValueError):
        int(float("nan") or 0)


@pytest.mark.parametrize("blank", [np.nan, None, "", "  ", "nan", "NaN", "<NA>"])
def test_blank_cells_fall_back_instead_of_raising(blank):
    assert _cell_int(blank, 7) == 7
    assert _cell_text(blank, "fallback") == "fallback"


def test_numeric_cells_survive_the_float_round_trip():
    assert _cell_int(6, 1) == 6
    assert _cell_int(6.0, 1) == 6          # data_editor returns floats for integer columns
    assert _cell_int("4", 1) == 4
    assert _cell_int("not a number", 3) == 3


def test_a_row_with_blank_copy_numbers_no_longer_crashes():
    peptides = pd.DataFrame([
        {"name": "GHK", "sequence": "GHK", "min_copies": 2, "max_copies": 6},
        {"name": "KTTKS", "sequence": "KTTKS", "min_copies": np.nan, "max_copies": np.nan},
    ])
    spacers = pd.DataFrame([{"name": "GAR", "sequence": "GAR"}])
    spec = _build_spec(peptides, spacers, [TRYPSIN], 60, 160, 20)
    assert [p.name for p in spec.peptides] == ["GHK", "KTTKS"]
    assert (spec.peptides[1].min_copies, spec.peptides[1].max_copies) == (0, 1)


def test_a_blank_name_is_taken_from_the_sequence_not_called_nan():
    """str(nan) is the string "nan", so a blank name silently produced a peptide called "nan"."""
    peptides = pd.DataFrame([{"name": np.nan, "sequence": "ghk",
                              "min_copies": 1, "max_copies": 3}])
    spec = _build_spec(peptides, pd.DataFrame(columns=["name", "sequence"]), [TRYPSIN], 3, 60, 10)
    assert spec.peptides[0].name == "GHK"
    assert spec.peptides[0].sequence == "GHK"


def test_entirely_blank_rows_are_skipped():
    peptides = pd.DataFrame([
        {"name": "GHK", "sequence": "GHK", "min_copies": 1, "max_copies": 4},
        {"name": np.nan, "sequence": np.nan, "min_copies": np.nan, "max_copies": np.nan},
    ])
    spec = _build_spec(peptides, pd.DataFrame(columns=["name", "sequence"]), [TRYPSIN], 3, 60, 10)
    assert len(spec.peptides) == 1


def test_blank_spacer_rows_are_skipped_and_names_default_to_the_sequence():
    spacers = pd.DataFrame([
        {"name": np.nan, "sequence": "GAR"},
        {"name": np.nan, "sequence": np.nan},
    ])
    peptides = pd.DataFrame([{"name": "GHK", "sequence": "GHK", "min_copies": 1, "max_copies": 4}])
    spec = _build_spec(peptides, spacers, [TRYPSIN], 3, 60, 10)
    assert [s.name for s in spec.spacers] == ["GAR"]


def test_duplicate_peptide_names_are_rejected():
    """Copy counts and released tallies are keyed by name, so duplicates would silently merge."""
    spec = ConcatemerSpec(peptides=[Peptide("P", "GHK", 1, 3), Peptide("P", "GQPR", 1, 3)],
                          rules=[TRYPSIN], length_min=3, length_max=90)
    assert any("Duplicate peptide name" in e for e in spec.errors())


def test_the_reported_crash_case_end_to_end():
    """A realistic table straight out of the editor: some rows filled, some left blank."""
    peptides = pd.DataFrame([
        {"name": "GHK", "sequence": "GHK", "min_copies": 2, "max_copies": 6},
        {"name": "GQPR", "sequence": "GQPR", "min_copies": 1, "max_copies": 5},
        {"name": None, "sequence": None, "min_copies": None, "max_copies": None},
    ])
    spacers = pd.DataFrame([
        {"name": "GAR", "sequence": "GAR"},
        {"name": None, "sequence": None},
    ])
    spec = _build_spec(peptides, spacers, [TRYPSIN], 60, 160, 20)
    assert spec.errors() == []
    from concatemer.pipeline import run
    assert run(spec, max_candidates=200).passed
