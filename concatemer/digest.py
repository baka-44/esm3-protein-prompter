"""
concatemer/digest.py — the objective function.

Everything else in the pipeline scores whether the protein can be MADE. This scores whether it
gives back the PRODUCT: simulate the cleavage, enumerate the fragments, and compare them against
the peptides the design was supposed to deliver.

A candidate that expresses beautifully and returns four of eight peptides with ragged termini is
a failure, and no expression feature would catch it. On a pentapeptide one extra residue is 20%
wrong; for an ATCUN metal-binder (Xaa-Xaa-His, e.g. GHK) a single extra N-terminal residue
abolishes copper coordination outright, because the free alpha-amino group is one of the ligands.
Terminus fidelity is therefore pass/fail, not a similarity score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from concatemer.spec import ConcatemerSpec, Peptide, Spacer, peptide_mass, residue_mass

CLEAN, IMPAIRED, BLOCKED = "clean", "impaired", "blocked"
BASIC = "KR"


@dataclass
class Site:
    """One cleavage position. `pos` is the bond index: the chain breaks BEFORE seq[pos]."""

    pos: int
    rule: str
    status: str
    reason: str = ""


@dataclass
class Fragment:
    seq: str
    start: int
    kind: str = "unintended"   # "peptide" | "spacer" | "unintended"
    name: str = ""


@dataclass
class DigestReport:
    sites: list[Site] = field(default_factory=list)
    fragments: list[Fragment] = field(default_factory=list)

    designed_copies: dict[str, int] = field(default_factory=dict)
    released_exact: dict[str, int] = field(default_factory=dict)
    molar_yield: dict[str, float] = field(default_factory=dict)
    impaired_bounding: dict[str, int] = field(default_factory=dict)

    pct_copies_exact: float = 0.0
    n_unintended: int = 0
    payload_fraction_mass: float = 0.0
    protein_mass: float = 0.0

    n_clean: int = 0
    n_impaired: int = 0
    n_blocked: int = 0

    @property
    def delivered_ratio(self) -> dict[str, float]:
        """Molar yields normalised to the smallest non-zero — the blend as actually delivered."""
        vals = [v for v in self.molar_yield.values() if v > 0]
        if not vals:
            return {k: 0.0 for k in self.molar_yield}
        lo = min(vals)
        return {k: round(v / lo, 3) for k, v in self.molar_yield.items()}


def find_sites(seq: str, rules) -> list[Site]:
    """
    Every cleavage position a rule set produces, classified clean / impaired / blocked.

    Matches are found with a lookahead so overlapping motifs are not missed (in "KRR", trypsin
    sees three basic residues, not one). Where two rules land on the same bond the worse status
    wins: a site blocked by one chemistry is not rescued by another that ignores the block.
    """
    found: dict[int, Site] = {}
    for rule in rules:
        for m in re.finditer(f"(?=({rule.motif}))", seq):
            span = m.group(1)
            if not span:
                continue
            start = m.start()
            pos = start + len(span) if rule.side == "C" else start
            if pos <= 0 or pos >= len(seq):
                continue                       # a cut at either terminus is not a cut
            p1prime = seq[pos]
            if p1prime in rule.blocked_by:
                status, why = BLOCKED, f"P1' is {p1prime}"
            elif p1prime in rule.impaired_by:
                status, why = IMPAIRED, f"P1' is {p1prime}"
            else:
                status, why = CLEAN, ""
            prev = found.get(pos)
            rank = {CLEAN: 0, IMPAIRED: 1, BLOCKED: 2}
            if prev is None or rank[status] > rank[prev.status]:
                found[pos] = Site(pos, rule.name, status, why)
    return [found[p] for p in sorted(found)]


def _trim_c_basic(seq: str) -> str:
    """
    Kex1: processive removal of C-terminal K/R.

    Processive is the operative word. A peptide whose own C-terminus is K or R gets eaten past
    its terminus, so GHK followed by a KR spacer returns GH, not GHK.
    """
    i = len(seq)
    while i > 0 and seq[i - 1] in BASIC:
        i -= 1
    return seq[:i]


def cut(seq: str, sites: list[Site], trim: bool = False) -> list[Fragment]:
    """Split at every non-blocked site. Blocked sites stay joined — that is the failure mode."""
    cuts = [s.pos for s in sites if s.status != BLOCKED]
    frags: list[Fragment] = []
    prev = 0
    for pos in cuts + [len(seq)]:
        piece = seq[prev:pos]
        if piece:
            frags.append(Fragment(_trim_c_basic(piece) if trim else piece, prev))
        prev = pos
    return [f for f in frags if f.seq]


def digest(seq: str, spec: ConcatemerSpec, layout: list | None = None) -> DigestReport:
    """
    Simulate the digest of `seq` and score it against the peptides `spec` wanted delivered.

    `layout` is the ordered list of Peptide/Spacer units the chain was built from; it supplies
    the designed copy numbers. Without it, designed copies are counted from the sequence.
    """
    rules = spec.rules
    sites = find_sites(seq, rules)
    trim = any(getattr(r, "trim_c_basic", False) for r in rules)
    frags = cut(seq, sites, trim=trim)

    by_seq = {p.sequence: p.name for p in spec.peptides}
    spacer_seqs = {s.sequence for s in spec.spacers}

    rep = DigestReport(sites=sites, fragments=frags)
    rep.designed_copies = {p.name: 0 for p in spec.peptides}
    rep.released_exact = {p.name: 0 for p in spec.peptides}
    rep.impaired_bounding = {p.name: 0 for p in spec.peptides}

    if layout is not None:
        for unit in layout:
            if isinstance(unit, Peptide):
                rep.designed_copies[unit.name] = rep.designed_copies.get(unit.name, 0) + 1
    else:
        for p in spec.peptides:
            rep.designed_copies[p.name] = seq.count(p.sequence)

    for f in frags:
        if f.seq in by_seq:
            f.kind, f.name = "peptide", by_seq[f.seq]
            rep.released_exact[f.name] += 1
        elif f.seq in spacer_seqs:
            f.kind = "spacer"
        else:
            rep.n_unintended += 1

    # Impaired sites bounding each intended peptide occurrence — the honest downside case,
    # rather than inventing a partial-cleavage percentage.
    impaired_pos = {s.pos for s in sites if s.status == IMPAIRED}
    for p in spec.peptides:
        for m in re.finditer(f"(?={re.escape(p.sequence)})", seq):
            a, b = m.start(), m.start() + len(p.sequence)
            if a in impaired_pos or b in impaired_pos:
                rep.impaired_bounding[p.name] += 1

    rep.molar_yield = {n: float(v) for n, v in rep.released_exact.items()}
    designed = sum(rep.designed_copies.values())
    rep.pct_copies_exact = (100.0 * sum(rep.released_exact.values()) / designed) if designed else 0.0

    # Payload fraction is measured on RESIDUE mass, not free-peptide mass. Hydrolysis consumes a
    # water per bond broken, so summed fragment masses legitimately exceed the parent protein —
    # dividing free-peptide masses by the protein mass yields >100% for an all-payload chain.
    # Residue mass on both sides asks the economic question: what share of the chain is product.
    rep.protein_mass = peptide_mass(seq)
    payload = sum(rep.released_exact[p.name] * residue_mass(p.sequence) for p in spec.peptides)
    total = residue_mass(seq)
    rep.payload_fraction_mass = (100.0 * payload / total) if total else 0.0

    rep.n_clean = sum(1 for s in sites if s.status == CLEAN)
    rep.n_impaired = len(impaired_pos)
    rep.n_blocked = sum(1 for s in sites if s.status == BLOCKED)
    return rep


def theoretical_payload_fraction(layout: list) -> float:
    """Payload by mass if every site cleaved perfectly — the design's ceiling, before chemistry."""
    total = sum(residue_mass(u.sequence) for u in layout)
    pep = sum(residue_mass(u.sequence) for u in layout if isinstance(u, Peptide))
    return (100.0 * pep / total) if total else 0.0
