"""
concatemer/assemble.py — turn a spec into candidate chains.

Naive enumeration does not survive the arithmetic: ordered subsets of 10 peptides is ~9.9M before
a single spacer is placed, and allowing repeats makes it far worse. But almost every liability is
PAIRWISE — junction sequons, junction-window aggregation, cleavage-site integrity all depend only
on which two units are adjacent — and the global ones (length, payload fraction) are additive.

So the search factorises:
  1. enumerate the multiset of peptide copies, pruned hard by the length envelope
  2. order each multiset, scoring only adjacent pairs

Ordering offers the two canonical architectures alongside the cost-optimised one. Blocked and
round-robin layouts are not merely baselines: they are regular, easy to reason about, and far
easier to interpret when a construct fails at the bench than an arbitrary permutation is.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from concatemer.screens import junction_sequons, max_hydrophobic_run
from concatemer.spec import ConcatemerSpec, Peptide, Spacer, residue_mass

# Junction penalty weights. Deliberately coarse: these order candidates within a search, they are
# not a fitted model of anything. A sequon or a blocked site is near-absolute, hydrophobic run-on
# is a matter of degree.
W_SEQUON = 10.0
W_BLOCKED = 10.0
W_HYDROPHOBIC = 1.0
FLANK = 6                      # residues either side of a join that a junction feature can span


@dataclass
class Candidate:
    candidate_id: str
    sequence: str
    layout: list = field(default_factory=list)          # [Peptide | Spacer, ...]
    counts: dict[str, int] = field(default_factory=dict)
    spacer: str = ""
    architecture: str = ""
    junction_cost: float = 0.0
    junction_flags: dict[str, int] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def n_units(self) -> int:
        return sum(self.counts.values())


def junction_cost(a: str, b: str, spec: ConcatemerSpec) -> tuple[float, dict[str, int]]:
    """
    Cost of placing unit `b` immediately after unit `a`, scored on the join window only.

    Three liabilities, all invisible when the units are inspected separately:
      * a sequon created by the join (glycan heterogeneity, and steric blocking of a nearby site)
      * a cleavage site created at the join whose P1' is a blocker, so the release never fires
      * a hydrophobic run spanning the join, seeding aggregation
    """
    flags = {"junction_sequon": 0, "blocked_site": 0, "hydrophobic_run": 0}
    cost = 0.0

    flags["junction_sequon"] = len(junction_sequons([a, b]))
    cost += W_SEQUON * flags["junction_sequon"]

    # Cleavage integrity across the join, on the local window only — the authoritative,
    # whole-chain answer comes from digest.simulate.
    from concatemer.digest import BLOCKED, find_sites
    left, right = a[-FLANK:], b[:FLANK]
    win = left + right
    for s in find_sites(win, spec.rules):
        if s.status == BLOCKED and abs(s.pos - len(left)) <= 1:
            flags["blocked_site"] += 1
    cost += W_BLOCKED * flags["blocked_site"]

    run_join = max_hydrophobic_run(a[-FLANK:] + b[:FLANK])
    run_alone = max(max_hydrophobic_run(a), max_hydrophobic_run(b))
    flags["hydrophobic_run"] = max(0, run_join - run_alone)
    cost += W_HYDROPHOBIC * flags["hydrophobic_run"]
    return cost, flags


def enumerate_counts(spec: ConcatemerSpec, spacer_len: int, cap: int = 20000) -> list[dict[str, int]]:
    """
    Every peptide-copy multiset that fits the length envelope and the per-peptide copy bounds.

    Depth-first with a length lower bound at each node, which is what keeps this finite: the
    unpruned space is the product of the copy ranges. Returns at most `cap` multisets; the caller
    reports truncation rather than silently sampling.
    """
    peps = spec.peptides
    tail_len = [0] * (len(peps) + 1)
    tail_units = [0] * (len(peps) + 1)
    for i in range(len(peps) - 1, -1, -1):
        tail_len[i] = tail_len[i + 1] + peps[i].min_copies * len(peps[i].sequence)
        tail_units[i] = tail_units[i + 1] + peps[i].min_copies

    out: list[dict[str, int]] = []

    def rec(i: int, counts: dict[str, int], length: int, units: int) -> None:
        if len(out) >= cap:
            return
        if i == len(peps):
            total = length + max(0, units - 1) * spacer_len
            if units and spec.length_min <= total <= spec.length_max:
                out.append(dict(counts))
            return
        p = peps[i]
        for n in range(p.min_copies, p.max_copies + 1):
            nl, nu = length + n * len(p.sequence), units + n
            if nu + tail_units[i + 1] > spec.max_units:
                break
            floor = nl + tail_len[i + 1] + max(0, nu + tail_units[i + 1] - 1) * spacer_len
            if floor > spec.length_max:
                break                       # only grows with n — prune the rest of the range
            counts[p.name] = n
            rec(i + 1, counts, nl, nu)
            if len(out) >= cap:
                return
        counts.pop(p.name, None)

    rec(0, {}, 0, 0)
    return out


def _expand(counts: dict[str, int], by_name: dict[str, Peptide]) -> list[Peptide]:
    return [by_name[n] for n, c in counts.items() for _ in range(c)]


def _order_grouped(units: list[Peptide]) -> list[Peptide]:
    """All copies of each peptide together — AAABBBCCC."""
    return sorted(units, key=lambda p: p.name)


def _order_roundrobin(units: list[Peptide]) -> list[Peptide]:
    """One of each in turn — ABCABCABC. Spreads any single peptide's liability along the chain."""
    buckets: dict[str, list[Peptide]] = {}
    for u in units:
        buckets.setdefault(u.name, []).append(u)
    out: list[Peptide] = []
    while any(buckets.values()):
        for name in sorted(buckets):
            if buckets[name]:
                out.append(buckets[name].pop())
    return out


def _order_optimised(units: list[Peptide], spacer: Spacer | None,
                     spec: ConcatemerSpec) -> list[Peptide]:
    """
    Greedy nearest-neighbour on junction cost, then 2-opt.

    Cost depends only on adjacent TYPES, so the pairwise table is at most 10x10 however many
    copies are involved. Exact minimisation over a multiset is NP-hard in general; this is a
    heuristic and is labelled as one.
    """
    if len(units) <= 2:
        return list(units)
    mid = spacer.sequence if spacer else ""

    cache: dict[tuple[str, str], float] = {}
    def cost(a: Peptide, b: Peptide) -> float:
        key = (a.name, b.name)
        if key not in cache:
            c1, _ = junction_cost(a.sequence, mid, spec) if mid else (0.0, {})
            c2, _ = junction_cost(mid, b.sequence, spec) if mid else (0.0, {})
            c0, _ = junction_cost(a.sequence, b.sequence, spec) if not mid else (0.0, {})
            cache[key] = c1 + c2 + c0
        return cache[key]

    remaining = list(units)
    path = [remaining.pop(0)]
    while remaining:
        nxt = min(range(len(remaining)), key=lambda i: cost(path[-1], remaining[i]))
        path.append(remaining.pop(nxt))

    improved = True
    while improved:                                   # 2-opt on segment reversal
        improved = False
        for i in range(len(path) - 1):
            for j in range(i + 2, len(path)):
                a, b = path[i], path[i + 1]
                c, d = path[j], path[j + 1] if j + 1 < len(path) else None
                before = cost(a, b) + (cost(c, d) if d else 0.0)
                after = cost(a, c) + (cost(b, d) if d else 0.0)
                if after < before - 1e-9:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    improved = True
    return path


def build(units: list[Peptide], spacer: Spacer | None, spec: ConcatemerSpec,
          architecture: str) -> Candidate:
    """Interleave the ordered peptides with the spacer and score every junction."""
    layout: list = []
    for i, u in enumerate(units):
        if i and spacer:
            layout.append(spacer)
        layout.append(u)
    seq = "".join(u.sequence for u in layout)

    total, flags = 0.0, {"junction_sequon": 0, "blocked_site": 0, "hydrophobic_run": 0}
    for a, b in zip(layout, layout[1:]):
        c, f = junction_cost(a.sequence, b.sequence, spec)
        total += c
        for k, v in f.items():
            flags[k] += v

    counts: dict[str, int] = {}
    for u in units:
        counts[u.name] = counts.get(u.name, 0) + 1
    cid = hashlib.sha1(f"{seq}|{architecture}".encode()).hexdigest()[:10]
    return Candidate(candidate_id=cid, sequence=seq, layout=layout, counts=counts,
                     spacer=spacer.name if spacer else "", architecture=architecture,
                     junction_cost=round(total, 3), junction_flags=flags)


def assemble(spec: ConcatemerSpec, max_candidates: int = 5000,
             multiset_cap: int = 20000) -> tuple[list[Candidate], dict]:
    """
    Enumerate candidate chains. Returns (candidates, stats) — stats records truncation so an
    empty or short result is never confused with a bug.
    """
    by_name = {p.name: p for p in spec.peptides}
    spacer_options: list[Spacer | None] = [None] + list(spec.spacers)
    stats = {"multisets": 0, "generated": 0, "truncated_multisets": False,
             "truncated_candidates": False}

    # Work items are interleaved ACROSS spacers before generation. Draining one spacer at a time
    # would make the candidate cap truncate along spacer order, silently discarding whole spacer
    # families — the first run of this dropped every GGA candidate for that reason, which is a
    # biased result presented as a complete one.
    per_spacer: list[list[tuple]] = []
    for spacer in spacer_options:
        slen = len(spacer.sequence) if spacer else 0
        multisets = enumerate_counts(spec, slen, cap=multiset_cap)
        if len(multisets) >= multiset_cap:
            stats["truncated_multisets"] = True
        stats["multisets"] += len(multisets)
        per_spacer.append([(spacer, c) for c in multisets])

    work: list[tuple] = []
    for i in range(max(len(w) for w in per_spacer) if per_spacer else 0):
        for w in per_spacer:
            if i < len(w):
                work.append(w[i])

    out: list[Candidate] = []
    seen: set[str] = set()
    for spacer, counts in work:
        units = _expand(counts, by_name)
        orders = {"grouped": _order_grouped(units),
                  "roundrobin": _order_roundrobin(units),
                  "optimised": _order_optimised(units, spacer, spec)}
        for arch, ordered in orders.items():
            cand = build(ordered, spacer, spec, arch)
            if cand.sequence in seen:
                continue                          # identical chains from different architectures
            seen.add(cand.sequence)
            out.append(cand)
            if len(out) >= max_candidates:
                stats["truncated_candidates"] = True
                stats["generated"] = len(out)
                return out, stats
    stats["generated"] = len(out)
    return out, stats
