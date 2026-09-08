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


# Codons per amino acid in the standard genetic code. This is a property of the CODE, not of any
# organism, so it needs no usage table — only Met and Trp are single-codon.
CODON_COUNTS: dict[str, int] = {
    "L": 6, "S": 6, "R": 6, "A": 4, "G": 4, "P": 4, "T": 4, "V": 4, "I": 3,
    "N": 2, "D": 2, "C": 2, "Q": 2, "E": 2, "H": 2, "K": 2, "F": 2, "Y": 2,
    "M": 1, "W": 1,
}


def encoding_capacity(seq: str) -> int:
    """
    How many distinct DNA sequences encode this peptide — the product of each residue's codon
    degeneracy (GHK = 4x2x2 = 16).

    It matters because a concatemer repeats peptides by design, and identical codons make long
    DIRECT REPEATS at the DNA level: synthesis vendors reject or surcharge them, and repeats in
    an integrated cassette can loop out by homologous recombination and delete copies. The fix is
    to encode each copy differently, which needs at least as many distinct encodings as copies.

    Capacity below the copy number is impossible in principle, not merely awkward. It multiplies,
    so it rarely binds — but when it does it is absolute, and catching it here beats a confusing
    failure inside a constraint solver later.
    """
    n = 1
    for c in seq.upper():
        n *= CODON_COUNTS.get(c, 1)
        if n > 10 ** 12:
            return 10 ** 12                      # saturate; the answer is "plenty"
    return n


@dataclass
class VectorContext:
    """
    The fixed construct context around the cargo. Every field is vector- or strain-specific and
    none is defaulted to a canonical sequence — see concatemer/rna.py for why substituting one
    produces a confident wrong answer.

    `signal_ste13` records whether the pre-pro retains the EA/EA spacer. Ste13 removes N-terminal
    X-Ala dipeptides processively, so with EAEA present it can trim INTO a cargo whose second
    residue is alanine. Many modern vectors delete EAEA precisely because that processing is
    often incomplete and gives heterogeneous N-termini.
    """

    utr5: str = ""          # transcription start site -> ATG (the PROMOTER is not transcribed)
    signal_cds: str = ""    # alpha-MF pre-pro NUCLEOTIDES, beginning at ATG
    utr3: str = ""          # stop codon -> poly-A site
    signal_ste13: bool = False
    name: str = ""

    @property
    def complete_for_folding(self) -> bool:
        return bool(self.utr5.strip() and self.signal_cds.strip())


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
    requires_p1prime: str = ""    # when set, P1' MUST be one of these or the site is dead
    trim_c_basic: bool = False    # Kex1-style processive C-terminal K/R removal
    consensus: str = ""           # human-readable, e.g. "ENLYFQ / G"
    note: str = ""                # what a user needs to know before choosing it

    @property
    def is_exopeptidase(self) -> bool:
        """
        No motif means no endopeptidase activity — the rule only post-processes fragment ends.

        Carboxypeptidase B is the case that forces this: it is not a site-specific cutter at all,
        it chews C-terminal Arg/Lys off whatever it is given. Modelling it with a motif would
        invent cut sites that do not exist.
        """
        return not self.motif.strip()

    @property
    def site_length(self) -> int:
        """
        Residues of recognition site, approximated from the motif's literal characters.

        This is the payload tax. In a concatemer every junction carries one site, so a 7-residue
        recogniser across ten junctions is 70 residues that are not product — the difference
        between TEV and Lys-C is tens of percent of payload fraction, not a detail.
        """
        return 0 if self.is_exopeptidase else len(re.sub(r"\[[^\]]*\]", "X", self.motif))

    def errors(self) -> list[str]:
        errs = []
        if self.requires_p1prime and set(self.requires_p1prime) & set(self.blocked_by):
            errs.append(f"{self.name}: P1' cannot be both required and blocking "
                        f"({sorted(set(self.requires_p1prime) & set(self.blocked_by))})")
        if self.is_exopeptidase and not self.trim_c_basic:
            errs.append(f"{self.name}: no motif and no trimming — the rule would do nothing")
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
    # ── broad specificity: no payload tax, but they cut anywhere the residue appears ──
    "trypsin": CleavageRule(
        "trypsin", r"[KR]", "C", blocked_by="P", consensus="K or R / X",
        note="Cuts after every K and R. No site to add, so no payload cost — but any peptide "
             "with an internal K or R is destroyed. Blocked by proline at P1'."),
    "lys_c": CleavageRule(
        "lys_c", r"K", "C", blocked_by="P", consensus="K / X",
        note="Cuts after K only, so more selective than trypsin and it spares peptides "
             "containing R. Industrially proven at scale in insulin manufacture."),
    "glu_c": CleavageRule(
        "glu_c", r"E", "C", blocked_by="P", consensus="E / X",
        note="Cuts after E (also after D in phosphate buffer). Useful when peptides carry K/R "
             "that must survive."),
    "asp_n": CleavageRule(
        "asp_n", r"D", "N", blocked_by="", consensus="X / D",
        note="Cuts BEFORE D, so the D belongs to the downstream fragment — the only preset here "
             "that leaves an N-terminal rather than C-terminal scar."),

    # ── the host's own machinery ──
    "kex2": CleavageRule(
        "kex2", r"KR", "C", blocked_by="P", impaired_by="DE", consensus="KR / X",
        note="What Pichia already runs in the Golgi, so a construct containing KR is processed "
             "during secretion whether or not that was the intent. Acidic P1' slows it."),
    "kex2_kex1": CleavageRule(
        "kex2_kex1", r"KR", "C", blocked_by="P", impaired_by="DE", trim_c_basic=True,
        consensus="KR / X, then C-terminal K/R trimmed",
        note="Kex2 followed by Kex1, the in vivo pairing. Kex1 removes C-terminal K/R "
             "PROCESSIVELY, so a peptide that itself ends in K or R is eaten past its own "
             "terminus (GHK becomes GH). Check the digest before choosing this."),

    # ── High-selectivity fusion proteases. Precise, but read `note` before choosing one: they
    # were designed for a SINGLE carrier->product junction, where the recognition site stays on
    # the carrier that then gets discarded. A tandem concatemer is the opposite geometry, and the
    # site ends up attached to the upstream peptide instead. ──
    "tev": CleavageRule(
        "tev", r"ENLYFQ", "C", blocked_by="", requires_p1prime="GS",
        consensus="ENLYFQ / G or S",
        note="The workhorse of single-junction fusion cleavage — a 6-residue site means it will "
             "not cut anywhere else, and P1' must be G or S. But it cuts AFTER its own site, so "
             "in a tandem layout that site stays on the UPSTREAM peptide: every copy but the "
             "last comes back carrying ENLYFQ on its C-terminus. Usable only if the payload "
             "tolerates that extension, or paired with a trimming step."),
    "enterokinase": CleavageRule(
        "enterokinase", r"DDDDK", "C", blocked_by="P", consensus="DDDDK / X",
        note="Enteropeptidase. Highly specific, and P1' is unconstrained, so the peptide "
             "DOWNSTREAM of a site gets a native N-terminus. Same tandem caveat as TEV: the "
             "DDDDK stays on the peptide upstream of it."),
    "thrombin": CleavageRule(
        "thrombin", r"LVPR", "C", blocked_by="P", consensus="LVPR / G-S",
        note="Canonical site LVPR/GS. Well established and cheap at scale, but less stringent "
             "than TEV or 3C — thrombin has documented secondary cleavage at related basic "
             "sites, so check the digest for unintended fragments. Same tandem caveat: the "
             "LVPR stays on the upstream peptide."),
    "factor_xa": CleavageRule(
        "factor_xa", r"I[ED]GR", "C", blocked_by="PR", consensus="IEGR or IDGR / X",
        note="Leaves no residue on the downstream peptide, so the released N-terminus is native "
             "— its main advantage over TEV and 3C, which both constrain P1'. Blocked by proline "
             "or arginine at P1'. Secondary cleavage is reported; check the digest."),
    "hrv_3c": CleavageRule(
        "hrv_3c", r"LEVLFQ", "C", blocked_by="", requires_p1prime="G",
        consensus="LEVLFQ / G",
        note="PreScission-type, active at 4 C, which helps when the payload is protease-labile. "
             "P1' must be G. Same tandem caveat as TEV — the site stays on the upstream "
             "peptide."),
    # ── proline-directed: the complement of everything above ──
    "anpep": CleavageRule(
        "anpep", r"P", "C", blocked_by="P", consensus="P / X",
        note="Aspergillus niger prolyl endopeptidase, used industrially to degrade gluten. Cuts "
             "AFTER proline — the residue that blocks nearly every other protease here — so it "
             "reaches junctions the others cannot. Directly relevant to collagen-derived "
             "peptides, which are proline-rich; equally, it will shred them if the proline sits "
             "inside a peptide you meant to keep. Does not cut Pro-Pro."),

    # ── exopeptidase: trims ends, creates no cut sites of its own ──
    "cpb": CleavageRule(
        "cpb", "", "C", blocked_by="", trim_c_basic=True,
        consensus="removes C-terminal K/R (no site)",
        note="Carboxypeptidase B. Not a site-specific cutter — it chews C-terminal Arg and Lys "
             "off whatever it is given, PROCESSIVELY, exactly as Kex1 does. Proven at scale in "
             "insulin manufacture. Pair it with an endopeptidase to tidy basic C-termini, but "
             "note the same trap as Kex1: a peptide that itself ends in K or R is eaten past its "
             "own terminus (GHK becomes GH)."),
}

# Grouping for the UI, so a user sees the trade-off rather than an undifferentiated list.
RULE_GROUPS: dict[str, list[str]] = {
    "Broad specificity — no payload cost": ["trypsin", "lys_c", "glu_c", "asp_n", "anpep"],
    "Host machinery (Pichia)": ["kex2", "kex2_kex1"],
    "High selectivity — costs payload per junction":
        ["tev", "hrv_3c", "enterokinase", "thrombin", "factor_xa"],
    "Exopeptidase — trims ends, adds no sites": ["cpb"],
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
    vector: VectorContext = field(default_factory=VectorContext)

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
        names = [p.name for p in self.peptides]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            # Copy counts and released tallies are keyed by name, so duplicates would silently
            # merge into one entry and misreport the delivered blend.
            errs.append(f"Duplicate peptide name(s): {', '.join(dupes)}. Names must be unique.")
        for p in self.peptides:
            cap = encoding_capacity(p.sequence)
            if p.max_copies > cap:
                errs.append(
                    f"{p.name}: {p.max_copies} copies requested but only {cap} distinct codon "
                    f"encodings exist, so some copies must share DNA and the repeat cannot be "
                    f"removed. Lower max_copies to {cap} or fewer."
                )
        # Reachability: the shortest possible chain that honours every min_copies must still fit.
        floor = sum(p.min_copies * len(p.sequence) for p in self.peptides)
        if floor > self.length_max:
            errs.append(f"Mandatory copies alone need {floor} residues, above length_max "
                        f"{self.length_max}.")
        return errs

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "length_min": self.length_min, "length_max": self.length_max,
                "max_units": self.max_units, "vector": asdict(self.vector),
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
            vector=VectorContext(**d.get("vector", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
