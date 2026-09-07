"""
concatemer/rna.py — start-codon accessibility on the assembled transcript.

This is the one RNA measurement that justifies assembling the whole transcript. mRNA folding is
not local: a repetitive, GC-skewed cargo can base-pair back to the 5'UTR from hundreds of
nucleotides away and sequester the start codon, and no windowed calculation over the cargo alone
will reveal it.

What is deliberately NOT here:

  * The elongation proxy (windowed dG along the CDS). In eukaryotes the ribosome has helicase
    activity and unwinds ordinary hairpins; it takes something like a pseudoknot to genuinely
    stall it. The two best-evidenced stalling mechanisms — polybasic and polyproline runs — are
    protein-level and already in `features.py` (F14, F15).

  * Initiation-window dG as a per-candidate feature. The window (-4..+37 around the AUG) sits
    entirely inside the fixed 5'UTR and the first codons of the fixed signal peptide, so it is
    CONSTANT across every candidate being compared. Only long-range pairing INTO that window
    varies, which is exactly what this module measures.

A liability found here is fixable by re-encoding — codon degeneracy gives ample freedom to break
a pairing without touching the protein — so the result belongs in the back-translator's constraint
set, not in the candidate ranker.

Cost: folding is O(n^3). A ~1.2 kb transcript takes seconds, so this runs on a SHORTLIST, late in
the cascade, never on the full candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Classic initiation window, relative to the A of the AUG.
DEFAULT_WINDOW = (-4, 37)
MAX_FOLD_NT = 1500
# Minimum contiguous helix that counts as sequestration. An MFE fold always produces scattered
# base pairs, so "any pairing into the cargo" fires on essentially every candidate and carries no
# information — a duplex only holds if it is a real helix.
#
# Calibrated against the null rather than guessed. Over 60 random 250-nt cargos the longest
# chance helix reaching the initiation window peaked at 4-5 bp and never exceeded 8. Designed
# complementarity behaves quite differently: an 8-nt complementary stretch does not form a stable
# duplex at all (3 bp observed), while 12 nt and above register at their full length. The
# threshold therefore sits in the gap between the chance ceiling (8) and the floor at which
# genuine sequestration appears (12).
MIN_HELIX_BP = 10


class MissingContextError(ValueError):
    """Raised when the vector context needed to assemble a transcript has not been supplied."""


@dataclass
class TranscriptContext:
    """
    The fixed vector context around a cargo CDS.

    Both fields are VECTOR-SPECIFIC and must come from the construct actually in use. They are
    deliberately not defaulted: mRNA structure is exquisitely sequence-dependent, so a plausible
    substitute (a canonical AOX1 5'UTR, or a back-translated alpha-MF) would return confident and
    wrong answers. pPICZalpha, pPIC9K and their relatives differ in 5'UTR length by cloning site.

      utr5       transcription start site -> ATG (transcribed; the PROMOTER is not, and is not
                 needed here — it sets how much mRNA is made, not how it folds)
      signal_cds alpha-MF pre-pro NUCLEOTIDES as they appear in the vector, not the protein
      utr3       stop codon -> poly-A site. Optional: it lies downstream of the stop codon so it
                 cannot affect elongation, and 5'-3' pairing is usually beyond the fold cap. It
                 is appended when supplied so the option can be tested rather than assumed away.
    """

    utr5: str
    signal_cds: str
    utr3: str = ""
    name: str = ""

    @classmethod
    def from_vector(cls, vector) -> "TranscriptContext":
        """Build from a spec.VectorContext, validating what folding actually requires."""
        return cls(utr5=vector.utr5, signal_cds=vector.signal_cds, utr3=vector.utr3,
                   name=vector.name)

    def __post_init__(self) -> None:
        self.utr5 = _rna(self.utr5)
        self.signal_cds = _rna(self.signal_cds)
        self.utr3 = _rna(self.utr3)
        if not self.utr5 or not self.signal_cds:
            raise MissingContextError(
                "Both utr5 and signal_cds are required, from the vector actually in use. "
                "A substituted sequence would fold differently and give a confidently wrong answer."
            )
        if not self.signal_cds.startswith("AUG"):
            raise MissingContextError("signal_cds must begin at the start codon (ATG/AUG).")


@dataclass
class AccessibilityReport:
    dg: float = 0.0
    folded_nt: int = 0
    atg_index: int = 0
    window: tuple[int, int] = DEFAULT_WINDOW

    n_window_nt: int = 0
    n_paired: int = 0
    frac_paired: float = 0.0
    start_codon_paired: bool = False

    partners_in_cargo: list[tuple[int, int]] = field(default_factory=list)
    partners_elsewhere: list[tuple[int, int]] = field(default_factory=list)
    longest_cargo_helix: int = 0
    cargo_sequesters_start: bool = False

    dot_bracket: str = ""
    truncated: bool = False

    @property
    def accessible(self) -> bool:
        """Start codon unpaired and no cargo reaching back into the initiation window."""
        return not self.start_codon_paired and not self.cargo_sequesters_start


def _rna(seq: str) -> str:
    return "".join(seq.split()).upper().replace("T", "U")


def longest_helix(pairs: list[tuple[int, int]]) -> int:
    """
    Longest antiparallel run in a set of (position, partner) pairs.

    A helix advances one base on each strand in opposite directions, so consecutive members
    satisfy i+1 and j-1. Scattered pairs that happen to share a partner region are not a duplex
    and do not hold the window closed.
    """
    if not pairs:
        return 0
    ordered = sorted(pairs)
    best = run = 1
    for (i0, j0), (i1, j1) in zip(ordered, ordered[1:]):
        run = run + 1 if (i1 == i0 + 1 and j1 == j0 - 1) else 1
        best = max(best, run)
    return best


def pair_map(dot_bracket: str) -> dict[int, int]:
    """Dot-bracket -> {index: partner index}. Unpaired positions are absent."""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for i, c in enumerate(dot_bracket):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if not stack:
                continue                      # unbalanced input; ignore rather than crash
            j = stack.pop()
            pairs[i], pairs[j] = j, i
    return pairs


def assemble_transcript(ctx: TranscriptContext, cargo_cds: str) -> tuple[str, int]:
    """Return (transcript, index of the A in the AUG)."""
    return ctx.utr5 + ctx.signal_cds + _rna(cargo_cds) + ctx.utr3, len(ctx.utr5)


def accessibility(ctx: TranscriptContext, cargo_cds: str,
                  window: tuple[int, int] = DEFAULT_WINDOW,
                  max_fold_nt: int = MAX_FOLD_NT, temp: float = 30.0) -> AccessibilityReport:
    """
    Fold the assembled transcript and report whether the start codon stays open.

    Folding is from the 5' end and capped at `max_fold_nt`: the cost is cubic, and pairing that
    reaches the AUG from beyond ~1.5 kb is not something an MFE fold resolves meaningfully
    anyway. `temp` defaults to 30 C, the usual Pichia induction temperature.
    """
    import seqfold

    transcript, atg = assemble_transcript(ctx, cargo_cds)
    cargo_start = len(ctx.utr5) + len(ctx.signal_cds)
    cargo_end = cargo_start + len(_rna(cargo_cds))
    truncated = len(transcript) > max_fold_nt
    folded = transcript[:max_fold_nt]

    structs = seqfold.fold(folded, temp=temp)
    db = seqfold.dot_bracket(folded, structs)
    pairs = pair_map(db)
    dg = sum(s.e for s in structs if s.e == s.e)

    lo, hi = atg + window[0], atg + window[1]
    lo, hi = max(0, lo), min(len(folded), hi)
    idxs = range(lo, hi)

    paired = [i for i in idxs if i in pairs]
    in_cargo, elsewhere = [], []
    for i in paired:
        j = pairs[i]
        if lo <= j < hi:
            continue                          # the window paired with itself — local hairpin
        # Only the CARGO is a design variable; pairing into the fixed 5'UTR, signal or 3'UTR is
        # constant across candidates and cannot be fixed by re-encoding the cargo.
        (in_cargo if cargo_start <= j < cargo_end else elsewhere).append((i - atg, j))

    rep = AccessibilityReport(
        dg=round(dg, 2), folded_nt=len(folded), atg_index=atg, window=window,
        n_window_nt=hi - lo, n_paired=len(paired),
        frac_paired=round(len(paired) / (hi - lo), 4) if hi > lo else 0.0,
        start_codon_paired=any(i in pairs for i in range(atg, min(atg + 3, len(folded)))),
        partners_in_cargo=in_cargo, partners_elsewhere=elsewhere,
        longest_cargo_helix=longest_helix(in_cargo),
        cargo_sequesters_start=longest_helix(in_cargo) >= MIN_HELIX_BP,
        dot_bracket=db, truncated=truncated,
    )
    return rep


def flags(rep: AccessibilityReport) -> list[str]:
    """Failure flags. Each is a re-encoding instruction, not a reason to drop the design."""
    out: list[str] = []
    if rep.start_codon_paired:
        out.append("start codon is base-paired — re-encode to open the initiation window")
    if rep.cargo_sequesters_start:
        out.append(f"cargo forms a {rep.longest_cargo_helix} bp helix with the initiation window "
                   f"({len(rep.partners_in_cargo)} paired positions) — re-encode the cargo")
    if rep.frac_paired > 0.5 and not out:
        out.append(f"{rep.frac_paired:.0%} of the initiation window is paired")
    return out


# ── back-translation (required to have a cargo CDS to fold at all) ─────────────

def parse_codon_table(text: str) -> dict[str, str]:
    """
    Parse a one-codon-per-residue table from lines like "A GCT" or "A=GCT" or "A,GCT".

    No default table is shipped. Codon usage is organism- and strain-specific, and accessibility
    is entirely an artefact of the encoding — a made-up table would fold differently from the
    gene actually ordered, which is the one thing this check exists to avoid.
    """
    table: dict[str, str] = {}
    for line in text.splitlines():
        parts = [p for p in line.replace("=", " ").replace(",", " ").replace("\t", " ").split() if p]
        if len(parts) != 2:
            continue
        aa, codon = parts[0].strip().upper(), parts[1].strip().upper().replace("U", "T")
        if len(aa) == 1 and len(codon) == 3 and set(codon) <= set("ACGT"):
            table[aa] = codon
    return table


def back_translate(protein: str, table: dict[str, str]) -> str:
    """
    Protein -> CDS using one codon per residue.

    Deliberately naive: this exists so the accessibility check has a sequence to fold, not to
    produce an orderable gene. A real construct needs constraint-based optimisation with repeat
    de-duplication, which belongs in the (unbuilt) DNA stage.
    """
    missing = sorted(set(protein.upper()) - set(table))
    if missing:
        raise MissingContextError(f"codon table has no entry for: {', '.join(missing)}")
    return "".join(table[c] for c in protein.upper())


def screen_shortlist(sequences: list[tuple[str, str]], ctx: TranscriptContext,
                     table: dict[str, str], **kw) -> list[tuple[str, AccessibilityReport, list[str]]]:
    """
    Run accessibility over a SHORTLIST of (id, protein) pairs.

    Folding is O(n^3) and takes seconds per transcript, so this is a late-cascade filter — it
    cannot run over a full candidate set, in the same way ESMFold cannot.
    """
    out = []
    for cid, protein in sequences:
        rep = accessibility(ctx, back_translate(protein, table), **kw)
        out.append((cid, rep, flags(rep)))
    return out
