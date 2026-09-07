"""
concatemer/spec.py — the design IR for a peptide concatemer.

A spec is what the user supplies: the peptides to deliver, the spacers available to separate
them, the cleavage chemistry that will release them, and the size envelope of the finished
protein. `assemble.py` turns a spec into candidate chains; `digest.py` scores what each chain
would actually give back.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = "1.0"

# Average residue masses (Da); a peptide's mass is the sum plus one water.
RESIDUE_MASS: dict[str, float] = {
    "G": 57.0519, "A": 71.0788, "S": 87.0782, "P": 97.1167, "V": 99.1326,
    "T": 101.1051, "C": 103.1388, "L": 113.1594, "I": 113.1594, "N": 114.1038,
    "D": 115.0886, "Q": 128.1307, "K": 128.1741, "E": 129.1155, "M": 131.1926,
    "H": 137.1411, "F": 147.1766, "R": 156.1875, "Y": 163.1760, "W": 186.2132,
}
WATER = 18.0153
AA = set(RESIDUE_MASS)


def residue_mass(seq: str) -> float:
    """Summed residue masses, excluding the terminal water. Unknown residues contribute 0."""
    return sum(RESIDUE_MASS.get(c, 0.0) for c in seq)


def peptide_mass(seq: str) -> float:
    """Average mass of a free peptide in Da (residues + one water)."""
    return residue_mass(seq) + (WATER if seq else 0.0)


def _clean(seq: str) -> str:
    return "".join(seq.split()).upper()


@dataclass
class Peptide:
    """One bioactive peptide to be delivered by the hydrolysate."""

    name: str
    sequence: str
    min_copies: int = 0      # >=1 makes this peptide mandatory in every candidate
    max_copies: int = 1

    def __post_init__(self) -> None:
        self.sequence = _clean(self.sequence)

    def errors(self) -> list[str]:
        errs = []
        if not self.sequence:
            errs.append(f"{self.name}: empty sequence")
        bad = sorted(set(self.sequence) - AA)
        if bad:
            errs.append(f"{self.name}: non-standard residues {''.join(bad)}")
        if self.min_copies < 0 or self.max_copies < self.min_copies:
            errs.append(f"{self.name}: invalid copy range {self.min_copies}-{self.max_copies}")
        return errs

    @property
    def mass(self) -> float:
        return peptide_mass(self.sequence)


@dataclass
class Spacer:
    """
    A fixed sequence placed between peptides.

    Unlike a graft Linker (a length range for a generator to fill), a spacer here is an explicit
    sequence — nothing is being folded, so there is nothing to generate. Prefer spacers that are
    free of S and T: S/T is the +2 position of every N-glycosylation sequon AND the Pichia
    O-mannosylation target, so excluding it removes both liabilities at once. Note this rules out
    the reflexive (GGGGS)n.
    """

    name: str
    sequence: str

    def __post_init__(self) -> None:
        self.sequence = _clean(self.sequence)

    @property
    def mass(self) -> float:
        return peptide_mass(self.sequence)


@dataclass
class CleavageRule:
    """
    One cleavage chemistry.

    `motif` is a regex over the protein sequence; `side` says whether the chain is cut after the
    match ("C", e.g. trypsin after K/R, Kex2 after KR) or before it ("N", e.g. Asp-N).

    Context handling is deliberately three-valued rather than a fabricated efficiency number:
    a site is BLOCKED when P1' is in `blocked_by` (proline abolishes both trypsin and Kex2),
    IMPAIRED when P1' is in `impaired_by` (acidic P1' is known to slow Kex2), and otherwise CLEAN.
    Predicting "40% cleavage" from sequence is not defensible; reporting a clean/impaired/blocked
    census is.

    `trim_c_basic` models Kex1, the carboxypeptidase that follows Kex2 in vivo and PROCESSIVELY
    removes C-terminal K/R. That processivity is a trap worth simulating: a peptide that itself
    ends in K or R will be eaten past its own terminus (GHK|KR -> GHKKR -> ... -> GH).
    """

    name: str
    motif: str
    side: str = "C"               # "C" = cut after the match, "N" = cut before it
    blocked_by: str = "P"         # residues at P1' that abolish cleavage
    impaired_by: str = "DE"       # residues at P1' that reduce it
    trim_c_basic: bool = False    # Kex1-style processive C-terminal K/R removal

    def errors(self) -> list[str]:
        errs = []
        if self.side not in ("C", "N"):
            errs.append(f"{self.name}: side must be 'C' or 'N', got {self.side!r}")
        try:
            re.compile(self.motif)
        except re.error as exc:
            errs.append(f"{self.name}: bad motif regex ({exc})")
        return errs


# Chemistries worth having as defaults. Kex2 is the one Pichia already runs in the Golgi, so a
# construct containing KR is processed during secretion whether or not that was the intent.
PRESET_RULES: dict[str, CleavageRule] = {
    "trypsin": CleavageRule("trypsin", r"[KR]", "C", blocked_by="P"),
    "kex2": CleavageRule("kex2", r"KR", "C", blocked_by="P", impaired_by="DE"),
    "kex2_kex1": CleavageRule("kex2_kex1", r"KR", "C", blocked_by="P", impaired_by="DE",
                              trim_c_basic=True),
    "asp_n": CleavageRule("asp_n", r"D", "N", blocked_by=""),
    "glu_c": CleavageRule("glu_c", r"E", "C", blocked_by="P"),
}


@dataclass
class ConcatemerSpec:
    """A complete design brief. `assemble.py` searches the space this defines."""

    peptides: list[Peptide] = field(default_factory=list)
    spacers: list[Spacer] = field(default_factory=list)
    rules: list[CleavageRule] = field(default_factory=list)
    length_min: int = 60
    length_max: int = 300
    max_units: int = 40           # bound on total peptide copies, keeps the search finite

    def errors(self) -> list[str]:
        errs: list[str] = []
        if not self.peptides:
            errs.append("At least one peptide is required.")
        if len(self.peptides) > 10:
            errs.append("At most 10 peptides.")
        for p in self.peptides:
            errs.extend(p.errors())
        for r in self.rules:
            errs.extend(r.errors())
        if not self.rules:
            errs.append("At least one cleavage rule is required — without one nothing is released.")
        if self.length_max < self.length_min:
            errs.append(f"Invalid length range {self.length_min}-{self.length_max}.")
        # Reachability: the shortest possible chain that honours every min_copies must still fit.
        floor = sum(p.min_copies * len(p.sequence) for p in self.peptides)
        if floor > self.length_max:
            errs.append(f"Mandatory copies alone need {floor} residues, above length_max "
                        f"{self.length_max}.")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "length_min": self.length_min, "length_max": self.length_max,
                "max_units": self.max_units,
                "peptides": [asdict(p) for p in self.peptides],
                "spacers": [asdict(s) for s in self.spacers],
                "rules": [asdict(r) for r in self.rules]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConcatemerSpec":
        return cls(
            peptides=[Peptide(**p) for p in d.get("peptides", [])],
            spacers=[Spacer(**s) for s in d.get("spacers", [])],
            rules=[CleavageRule(**r) for r in d.get("rules", [])],
            length_min=d.get("length_min", 60), length_max=d.get("length_max", 300),
            max_units=d.get("max_units", 40),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
