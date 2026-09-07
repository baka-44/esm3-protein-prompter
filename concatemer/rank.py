"""
concatemer/rank.py — gate, group, then rank.

There is deliberately NO composite score. Its weights would be invented, and a natural-protein
expression corpus cannot supply them either: a synthetic repeat concatemer of short peptides is
far outside the distribution any such model was fitted on. An arbitrary weighting that looks
authoritative is worse than none.

What is defensible is an ordering that needs no weights:

  1. GATE on the failure flags. They are mechanistic and near-absolute, so they transfer where
     fitted coefficients do not. Most of the discrimination should happen here.
  2. GROUP correlated features before ranking. Summing raw ranks over correlated columns silently
     upweights whatever is measured most often, while presenting as neutral.
  3. Report BOTH the rank-sum and the WORST group rank. Rank-sum rewards mediocrity — a candidate
     median everywhere beats one excellent on five axes and catastrophic on one — and in biology
     the single severe liability usually decides the outcome. Worst-rank surfaces "no severe
     weakness", which is closer to how the failure behaves.

  4. Sort on the PRODUCT group first, then on the rest. Equal group weighting makes "does this
     deliver the peptides" one sixth of the ordering, so a candidate releasing 90% of its payload
     outranks one releasing 100% — which is what the first run of this actually did. The product
     metrics are the objective; everything else is whether the thing can be manufactured. That is
     a structural distinction, not an invented coefficient, and ties in the product rank are
     common enough that the manufacturability sum still decides most orderings.

Every row carries the features that dragged it down. A rank without a reason is not actionable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from concatemer.features import BY_KEY, FEATURES, GROUPS, HIGHER, FeatureRow


PRODUCT = "product"


@dataclass
class RankedRow:
    candidate_id: str
    rank: int = 0
    ranksum: float = 0.0          # over every group, for reference
    ranksum_mfg: float = 0.0      # over the manufacturability groups only — the tie-break
    product_rank: float = 0.0     # the objective
    worst_rank: float = 0.0
    worst_group: str = ""
    group_ranks: dict[str, float] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _average_ranks(values: list[float], higher_is_better: bool) -> list[float]:
    """Competition-free average ranking, 1 = best. Ties share the mean of their positions."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i], reverse=higher_is_better)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def rank(rows: list[FeatureRow]) -> list[RankedRow]:
    """Rank the rows that passed the gate. Failed rows come back unranked, with their reasons."""
    passed = [r for r in rows if r.passed]
    failed = [r for r in rows if not r.passed]

    out: list[RankedRow] = []
    if passed:
        per_feature: dict[str, list[float]] = {}
        for f in FEATURES:
            vals = [r.values.get(f.key, 0.0) for r in passed]
            per_feature[f.key] = _average_ranks(vals, f.direction == HIGHER)

        # Within a group, average the member ranks, then re-rank those averages so every group
        # contributes on the same 1..N scale regardless of how many features it holds.
        group_scores: dict[str, list[float]] = {}
        for g in GROUPS:
            keys = [f.key for f in FEATURES if f.group == g]
            group_scores[g] = [sum(per_feature[k][i] for k in keys) / len(keys)
                               for i in range(len(passed))]
        group_ranks = {g: _average_ranks(group_scores[g], higher_is_better=False)
                       for g in GROUPS}

        # A feature on which EVERY candidate ties at the optimum carries no information, but
        # average ranking still gives it a high shared rank — so a group where nothing is wrong
        # gets reported as the weakest axis, listing three zeroes as "liabilities". Only features
        # where this candidate is actually worse than the best observed value can be a reason.
        best: dict[str, float] = {}
        for f in FEATURES:
            vals = [r.values.get(f.key, 0.0) for r in passed]
            best[f.key] = max(vals) if f.direction == HIGHER else min(vals)

        for i, r in enumerate(passed):
            gr = {g: group_ranks[g][i] for g in GROUPS}
            off = {f.key for f in FEATURES if r.values.get(f.key, 0.0) != best[f.key]}
            live = [g for g in GROUPS if any(f.key in off for f in FEATURES if f.group == g)]
            worst_g = max(live, key=lambda g: gr[g]) if live else ""
            keys = [f.key for f in FEATURES if f.group == worst_g and f.key in off]
            reasons = sorted(keys, key=lambda k: per_feature[k][i], reverse=True)[:3]
            out.append(RankedRow(
                candidate_id=r.candidate_id, ranksum=round(sum(gr.values()), 2),
                ranksum_mfg=round(sum(v for g, v in gr.items() if g != PRODUCT), 2),
                product_rank=gr.get(PRODUCT, 0.0),
                # worst_rank stays the true maximum so sorting is unaffected; worst_group is
                # blank when the candidate is at the optimum on every feature that varies.
                worst_rank=max(gr.values()), worst_group=worst_g,
                group_ranks={k: round(v, 2) for k, v in gr.items()},
                values=dict(r.values),
                reasons=([f"{k} ({r.values.get(k)})" for k in reasons]
                         or ["at the best observed value on every feature"]),
            ))
        out.sort(key=lambda x: (x.product_rank, x.ranksum_mfg, x.worst_rank))
        for n, row in enumerate(out, start=1):
            row.rank = n

    for r in failed:
        out.append(RankedRow(candidate_id=r.candidate_id, rank=0, values=dict(r.values),
                             failures=list(r.failures)))
    return out
