"""
proteinredesign/graft_metrics.py — live metrics for a graft composition (CC12).

Cheap CPU geometry over the composite + spec. Each metric carries its value plus the tooltip
triple the UI shows (CC9/CC12): what it measures · biological meaning · desired direction. The
`critical` flag drives the export-readiness gate (CC13): any critical failure disables export.

Phase-1 set (the essential, cheap ones): linker gaps, steric clash, compactness (Rg), and a
closure-feasibility check. Shape-complementarity / void / SASA / buried-charge are Phase-2 adds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from proteinredesign.graft import Fragment, GraftPackage, Linker
from utils.pdb_utils import get_residues

CA_PER_RESIDUE = 3.8   # Å reach of one extended residue (linker feasibility)
CLASH_DIST = 2.5       # Å heavy-atom overlap threshold


@dataclass
class Metric:
    key: str
    label: str
    value: float
    unit: str
    desired: str            # e.g. "Ideal = 0", "lower is better", "green = ok"
    what: str               # what it measures
    meaning: str            # biological interpretation
    critical: bool = False  # a failing critical metric disables export (CC13)
    ok: bool = True         # False = this metric is in a bad state


def _ca_index(composite_pdb: bytes) -> dict[tuple[str, int], np.ndarray]:
    idx: dict[tuple[str, int], np.ndarray] = {}
    for r in get_residues(composite_pdb, chain_id=None):
        if "CA" in r:
            idx[(r.get_parent().id, r.id[1])] = np.array(r["CA"].coord, dtype=float)
    return idx


def _heavy_atoms_by_chain(composite_pdb: bytes, exclude: set[tuple[str, int]] | None = None) -> dict[str, np.ndarray]:
    exclude = exclude or set()
    by_chain: dict[str, list] = {}
    for r in get_residues(composite_pdb, chain_id=None):
        ch = r.get_parent().id
        if (ch, r.id[1]) in exclude:
            continue
        for a in r:
            if a.element != "H":
                by_chain.setdefault(ch, []).append(a.coord)
    return {c: np.array(v, dtype=float) for c, v in by_chain.items()}


def compute_metrics(package: GraftPackage) -> list[Metric]:
    spec = package.spec
    composite = package.composite_pdb
    ca = _ca_index(composite)
    metrics: list[Metric] = []

    # ── Linker gaps + closure feasibility (per junction) ──────────────────────
    frags = [s for s in spec.chain_order if isinstance(s, Fragment)]
    order = spec.chain_order
    gaps: list[float] = []
    infeasible = 0
    for i, seg in enumerate(order):
        if not isinstance(seg, Linker):
            continue
        prev_frag = order[i - 1]
        next_frag = order[i + 1]
        a = ca.get((prev_frag.chain, prev_frag.end))
        b = ca.get((next_frag.chain, next_frag.start))
        if a is None or b is None:
            continue
        d = float(np.linalg.norm(a - b))
        gaps.append(d)
        # feasible if the max linker can physically span the gap (with a little slack).
        if seg.length_max * CA_PER_RESIDUE < d - CA_PER_RESIDUE:
            infeasible += 1

    max_gap = max(gaps) if gaps else 0.0
    metrics.append(Metric(
        "max_linker_gap", "Max linker gap", round(max_gap, 1), "Å",
        desired="shorter is better (the linker must span it)",
        what="The largest end-to-end distance a generated linker must bridge.",
        meaning="Long gaps force RF3 to invent lots of backbone → low, unpredictable foldability.",
        critical=False, ok=(max_gap <= 25.0),
    ))
    metrics.append(Metric(
        "closure", "Closure feasibility", float(infeasible), "junctions",
        desired="Ideal = 0 infeasible junctions",
        what="How many junctions have a gap too long for even the maximum linker length.",
        meaning="An infeasible junction cannot be physically connected — the graft can't close.",
        critical=True, ok=(infeasible == 0),
    ))

    # ── Steric clash (CORE–CORE heavy-atom overlaps) ──────────────────────────
    # Exclude the repack shell: interface contact there is EXPECTED (MPNN redesigns it, BB2/CC9).
    # Only overlaps between the fixed CORES are true, unresolvable clashes.
    repack = {(r["chain"], r["author_num"]) for r in spec.repack_residues}
    heavy = _heavy_atoms_by_chain(composite, exclude=repack)
    chains = list(heavy)
    clashes = 0
    if len(chains) >= 2:
        A, B = heavy[chains[0]], heavy[chains[1]]
        if len(A) and len(B):
            dmat = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            clashes = int(np.count_nonzero(dmat < CLASH_DIST))
    metrics.append(Metric(
        "clash", "Core clashes", float(clashes), "atom pairs",
        desired="Ideal = 0",
        what="Heavy-atom overlaps between the two bodies' cores (excluding the repack interface).",
        meaning="Core clashes between the fixed bodies can't be relaxed and doom the fold. "
                "(Contact at the repack interface is expected and gets redesigned.)",
        critical=True, ok=(clashes == 0),
    ))

    # ── Compactness (radius of gyration over all CA) ──────────────────────────
    coords = np.array(list(ca.values())) if ca else np.zeros((1, 3))
    rg = float(np.sqrt(((coords - coords.mean(0)) ** 2).sum(1).mean())) if len(coords) > 1 else 0.0
    metrics.append(Metric(
        "rg", "Compactness (Rg)", round(rg, 1), "Å",
        desired="lower is better (compact, globular)",
        what="Radius of gyration of the composed backbone.",
        meaning="Compact/globular arrangements fold and express better than elongated dumbbells.",
        critical=False, ok=True,
    ))

    _ = frags  # (reserved for per-fragment metrics in Phase 2)
    return metrics


def critical_failures(metrics: list[Metric]) -> list[str]:
    """Labels of critical metrics in a bad state — non-empty ⇒ disable export (CC13)."""
    return [m.label for m in metrics if m.critical and not m.ok]
