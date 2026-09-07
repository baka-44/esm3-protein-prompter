"""
concatemer/screens.py — sequence-level screens shared by the assembler and the feature stage.

Everything here is local and compositional, which is the point: under architecture A the chain is
engineered NOT to fold, so there is no structure to derive features from, and the properties that
govern secretion (exposed hydrophobicity, charge, aggregation-prone regions, glycosylation) are
all readable straight off the sequence.

Two of these — the Uversky charge/hydropathy position and the Das-Pappu charge patterning kappa —
replace the structural stage entirely. They ask "is this robustly disordered?", which is a
well-posed question, rather than "what shape is it?", which for a designed concatemer is not.
"""

from __future__ import annotations

import re

# Kyte-Doolittle hydropathy.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
POSITIVE, NEGATIVE = "KR", "DE"

# N-glycosylation sequon, with a lookahead so overlapping motifs are not missed.
SEQUON = re.compile(r"(?=(N[^P][ST]))")


# ── glycosylation ──────────────────────────────────────────────────────────────

def sequons(seq: str) -> set[int]:
    """Start positions of every N-X-S/T (X != P) motif."""
    return {m.start() for m in SEQUON.finditer(seq)}


def junction_sequons(units: list[str]) -> list[int]:
    """
    Sequons present in the concatenated chain but in NO unit on its own.

    These are the dangerous ones: a glycan is created by an ordering decision rather than by any
    peptide you chose. Beyond heterogeneity, a bulky high-mannose glycan sitting next to a
    cleavage site can sterically block the protease, so the peptide never releases at all — a
    failure that presents as partial, variable processing and is easily misattributed.
    """
    joined = "".join(units)
    own, offset = set(), 0
    for u in units:
        own |= {i + offset for i in sequons(u)}
        offset += len(u)
    return sorted(sequons(joined) - own)


def st_fraction(seq: str) -> float:
    """Ser+Thr fraction — the Pichia O-mannosylation substrate. No motif exists to key on."""
    return (sum(1 for c in seq if c in "ST") / len(seq)) if seq else 0.0


# ── hydrophobicity / aggregation ───────────────────────────────────────────────

def hydropathy(seq: str) -> float:
    """Mean Kyte-Doolittle hydropathy."""
    return (sum(KD.get(c, 0.0) for c in seq) / len(seq)) if seq else 0.0


def window_hydropathy(seq: str, window: int = 5) -> list[float]:
    if len(seq) < window:
        return [hydropathy(seq)] if seq else []
    return [hydropathy(seq[i:i + window]) for i in range(len(seq) - window + 1)]


def max_window_hydropathy(seq: str, window: int = 5) -> float:
    w = window_hydropathy(seq, window)
    return max(w) if w else 0.0


def max_hydrophobic_run(seq: str, threshold: float = 1.5) -> int:
    """Longest unbroken run of hydrophobic residues — the seed of an aggregation-prone region."""
    best = run = 0
    for c in seq:
        run = run + 1 if KD.get(c, 0.0) >= threshold else 0
        best = max(best, run)
    return best


# ── charge ─────────────────────────────────────────────────────────────────────

def fcr(seq: str) -> float:
    """Fraction of charged residues."""
    return (sum(1 for c in seq if c in POSITIVE + NEGATIVE) / len(seq)) if seq else 0.0


def ncpr(seq: str) -> float:
    """Net charge per residue."""
    if not seq:
        return 0.0
    return (sum(1 for c in seq if c in POSITIVE) - sum(1 for c in seq if c in NEGATIVE)) / len(seq)


def _sigma(seq: str) -> float:
    pos = sum(1 for c in seq if c in POSITIVE)
    neg = sum(1 for c in seq if c in NEGATIVE)
    return ((pos - neg) ** 2 / (pos + neg)) if (pos + neg) else 0.0


def _delta(seq: str, window: int) -> float:
    n = len(seq) - window + 1
    if n <= 0:
        return 0.0
    whole = _sigma(seq)
    return sum((_sigma(seq[i:i + window]) - whole) ** 2 for i in range(n)) / n


def kappa(seq: str) -> float:
    """
    Das-Pappu charge patterning, 0 (well-mixed) to 1 (fully segregated into +/- blocks).

    Blocky charge drives IDP compaction and phase separation. A concatemer that phase-separates
    in the ER is a plausible secretion failure with no other warning sign, and kappa is the only
    cheap read on it. Normalised against the maximally segregated permutation of the same
    composition (positives, then neutrals, then negatives).
    """
    if len(seq) < 6 or not any(c in POSITIVE + NEGATIVE for c in seq):
        return 0.0
    worst = ("".join(c for c in seq if c in POSITIVE)
             + "".join(c for c in seq if c not in POSITIVE + NEGATIVE)
             + "".join(c for c in seq if c in NEGATIVE))
    obs = (_delta(seq, 5) + _delta(seq, 6)) / 2
    mx = (_delta(worst, 5) + _delta(worst, 6)) / 2
    return round(obs / mx, 4) if mx > 0 else 0.0


# ── disorder ───────────────────────────────────────────────────────────────────

def uversky(seq: str) -> tuple[float, float, bool]:
    """
    Position on the Uversky charge-hydropathy plot.

    Returns (normalised mean hydropathy, |net charge per residue|, predicted_disordered). The
    empirical boundary is <H> = (<R> + 1.151) / 2.785; sequences below it are natively unfolded.
    This is the primary "is it robustly disordered" screen, standing in for the fold stage that
    architecture A removes.
    """
    h = (hydropathy(seq) + 4.5) / 9.0            # normalise KD onto 0-1
    r = abs(ncpr(seq))
    return round(h, 4), round(r, 4), h < (r + 1.151) / 2.785
