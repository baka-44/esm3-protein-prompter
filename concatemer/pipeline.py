"""
concatemer/pipeline.py — assemble → digest → screen → gate → rank, plus the exports.

The funnel is a first-class output, not a diagnostic. Without it an empty result is
indistinguishable from a bug, and there is no way to tell an over-aggressive gate from a design
space that genuinely has nothing in it.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

from concatemer.assemble import Candidate, assemble
from concatemer.digest import digest
from concatemer.features import FEATURES, compute
from concatemer.rank import RankedRow, rank
from concatemer.spec import ConcatemerSpec


@dataclass
class PipelineResult:
    rows: list[RankedRow] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    funnel: list[tuple[str, int]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> list[RankedRow]:
        return [r for r in self.rows if not r.failures]

    @property
    def failed(self) -> list[RankedRow]:
        return [r for r in self.rows if r.failures]

    def failure_census(self) -> dict[str, int]:
        """How many candidates each gate removed — reveals a gate that is doing all the work."""
        census: dict[str, int] = {}
        for r in self.failed:
            for f in r.failures:
                key = f.split("—")[0].split("(")[0].strip()
                key = " ".join(w for w in key.split() if not w.isdigit())
                census[key] = census.get(key, 0) + 1
        return dict(sorted(census.items(), key=lambda kv: -kv[1]))


def run(spec: ConcatemerSpec, max_candidates: int = 5000) -> PipelineResult:
    errs = spec.errors()
    if errs:
        return PipelineResult(errors=errs)

    cands, stats = assemble(spec, max_candidates=max_candidates)
    by_id = {c.candidate_id: c for c in cands}

    rows = []
    for c in cands:
        rep = digest(c.sequence, spec, layout=c.layout)
        rows.append(compute(c.sequence, spec, rep, [u.sequence for u in c.layout],
                            candidate_id=c.candidate_id))

    ranked = rank(rows)
    n_pass = sum(1 for r in ranked if not r.failures)
    funnel = [("assembled", len(cands)),
              ("passed failure gates", n_pass),
              ("below the gate", len(cands) - n_pass)]
    return PipelineResult(rows=ranked, candidates=by_id, funnel=funnel, stats=stats)


def to_csv(result: PipelineResult) -> str:
    """Full export: every feature that fed the ranking, plus the reasons and failures."""
    cols = (["rank", "candidate_id", "sequence", "length", "architecture", "spacer", "counts",
             "ranksum", "worst_rank", "worst_group"]
            + [f"group_{g}" for g in sorted({f.group for f in FEATURES})]
            + [f.key for f in FEATURES]
            + ["reasons", "failures"])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in result.rows:
        c = result.candidates.get(r.candidate_id)
        row = {"rank": r.rank or "", "candidate_id": r.candidate_id,
               "sequence": c.sequence if c else "", "length": c.length if c else "",
               "architecture": c.architecture if c else "", "spacer": c.spacer if c else "",
               "counts": ";".join(f"{k}x{v}" for k, v in (c.counts if c else {}).items()),
               "product_rank": r.product_rank, "ranksum_mfg": r.ranksum_mfg,
               "ranksum": r.ranksum, "worst_rank": r.worst_rank, "worst_group": r.worst_group,
               "reasons": " | ".join(r.reasons), "failures": " | ".join(r.failures)}
        row.update({f"group_{g}": v for g, v in r.group_ranks.items()})
        row.update(r.values)
        w.writerow(row)
    return buf.getvalue()


def to_fasta(result: PipelineResult, limit: int = 0) -> str:
    """Ranked candidates as FASTA. Headers carry the metrics so the file is self-describing."""
    lines = []
    rows = [r for r in result.rows if r.rank] or result.rows
    rows = sorted(rows, key=lambda r: (r.rank == 0, r.rank))
    if limit:
        rows = rows[:limit]
    for r in rows:
        c = result.candidates.get(r.candidate_id)
        if not c:
            continue
        v = r.values
        head = (f"{c.candidate_id} rank={r.rank or 'FAILED'} len={c.length} "
                f"arch={c.architecture} spacer={c.spacer or 'none'} "
                f"exact={v.get('pct_copies_exact', 0):.1f}% "
                f"payload={v.get('payload_fraction_mass', 0):.1f}%")
        if r.failures:
            head += f" QC=FAIL[{'; '.join(r.failures)}]"
        lines.append(f">{head}\n{c.sequence}")
    return "\n".join(lines) + ("\n" if lines else "")
