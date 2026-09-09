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


def test_a_blank_name_is_auto_assigned_positionally():
    """
    str(nan) is the string "nan", so a blank name once produced a peptide called "nan".

    Names are now positional (P1, P2, …) rather than taken from the sequence: two rows holding
    the same sequence would otherwise collide on the duplicate-name check, and a positional id is
    what a user reading the results CSV can match back to a table row.
    """
    peptides = pd.DataFrame([
        {"name": np.nan, "sequence": "ghk", "min_copies": 1, "max_copies": 3},
        {"name": np.nan, "sequence": "ghk", "min_copies": 0, "max_copies": 2},
        {"name": "custom", "sequence": "GQPR", "min_copies": 0, "max_copies": 2},
    ])
    spec = _build_spec(peptides, pd.DataFrame(columns=["name", "sequence"]), [TRYPSIN], 3, 90, 10)
    assert [p.name for p in spec.peptides] == ["P1", "P2", "custom"]
    assert spec.peptides[0].sequence == "GHK"      # lowercase input is normalised
    assert spec.errors() == []                     # duplicate sequences do NOT collide on name


def test_entirely_blank_rows_are_skipped():
    peptides = pd.DataFrame([
        {"name": "GHK", "sequence": "GHK", "min_copies": 1, "max_copies": 4},
        {"name": np.nan, "sequence": np.nan, "min_copies": np.nan, "max_copies": np.nan},
    ])
    spec = _build_spec(peptides, pd.DataFrame(columns=["name", "sequence"]), [TRYPSIN], 3, 60, 10)
    assert len(spec.peptides) == 1


def test_blank_spacer_rows_are_skipped_and_names_auto_assigned():
    spacers = pd.DataFrame([
        {"name": np.nan, "sequence": "GAR"},
        {"name": np.nan, "sequence": np.nan},
        {"name": np.nan, "sequence": "GGAR"},
    ])
    peptides = pd.DataFrame([{"name": "GHK", "sequence": "GHK", "min_copies": 1, "max_copies": 4}])
    spec = _build_spec(peptides, spacers, [TRYPSIN], 3, 60, 10)
    assert [s.name for s in spec.spacers] == ["S1", "S2"]      # numbered over KEPT rows only
    assert [s.sequence for s in spec.spacers] == ["GAR", "GGAR"]


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


# ── source guards ──────────────────────────────────────────────────────────────
# Two bug classes here recurred after being fixed once, so they are pinned at source level. Both
# scans run over CODE ONLY: the comments and docstrings that explain these bugs necessarily
# contain the very patterns being banned, and matched themselves on the first attempt.

import io  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import tokenize  # noqa: E402


def _panel_code() -> str:
    """
    ui/concatemer_panel.py with comments and string literals blanked IN PLACE.

    Layout is preserved rather than re-joining tokens, so the result can still be searched
    structurally (`def _editable_table(`, `st.session_state[data_key] =`) — joining tokens with
    spaces splits those apart and the searches silently stop matching.
    """
    path = pathlib.Path(__file__).parent.parent / "ui" / "concatemer_panel.py"
    lines = path.read_text().splitlines(keepends=True)
    grid = [list(line) for line in lines]
    for tok in tokenize.generate_tokens(io.StringIO("".join(lines)).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        for row in range(r1 - 1, r2):
            lo = c1 if row == r1 - 1 else 0
            hi = c2 if row == r2 - 1 else len(grid[row])
            for col in range(lo, min(hi, len(grid[row]))):
                if grid[row][col] != "\n":
                    grid[row][col] = " "
    return "".join("".join(row) for row in grid)



def _int_calls_containing_or(src: str) -> list[str]:
    """
    Every `int(...)` call whose arguments contain a bare `or`, matching balanced parentheses.

    A regex cannot do this: the real offender was `int(r.get("max_copies") or 1)`, and any
    character-class approach stops at the inner `)` of `r.get(...)` before ever reaching the
    `or`. The first version of this guard passed while the bug was reintroduced, which is worse
    than having no guard at all.
    """
    found = []
    for m in re.finditer(r"\bint\(", src):
        depth, i = 1, m.end()
        while i < len(src) and depth:
            depth += (src[i] == "(") - (src[i] == ")")
            i += 1
        call = src[m.start():i]
        if re.search(r"\bor\b", call):
            found.append(" ".join(call.split()))
    return found


def test_the_int_or_default_pattern_is_banned_from_the_panel():
    """
    This bug appeared twice: once in _build_spec, and once in the codon head-room caption a few
    lines away, which ran on every rerun and so took the panel down while the table was still
    being filled in. `int(cell or default)` is silently wrong for ANY numeric data_editor cell,
    so ban the pattern rather than fix each site as it is discovered.
    """
    offenders = _int_calls_containing_or(_panel_code())
    assert not offenders, f"use _cell_int() instead: {offenders}"


def test_the_editor_is_never_fed_its_own_output():
    """
    The vanishing-entry bug. data_editor stores the user's edits as a DIFF against the base frame
    it was given, keyed by widget key. Writing the returned (already-edited) frame back as the
    next base re-applies that diff on top of itself, so indices shift and entries have to be
    typed two or three times before they stick.

    The base may only change on a deliberate mutation, which also bumps the version to reset the
    diff — so an assignment from `edited` is only legal alongside a version bump.
    """
    src = _panel_code()
    body = src[src.index("def _editable_table("):]
    assigns = re.findall(r"st\.session_state\[data_key\]\s*=\s*(.+)", body)
    for rhs in assigns:
        assert "edited.drop" in rhs, (
            f"base frame assigned from {rhs.strip()!r} — only a removal may rewrite the base"
        )
    assert "st.session_state[ver_key] += 1" in body, "a base rewrite must reset the widget diff"
