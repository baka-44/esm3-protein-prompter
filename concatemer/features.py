"""
concatemer/features.py — per-candidate features and failure flags.

Cross-referenced to the MoComMoSec secretion spec (F-numbers in `spec_id`), with two deliberate
departures from it:

  * STRUCTURE FEATURES ARE ABSENT. F23 (relative contact order), F55 (loop exposure) and the
    SASA-weighted forms of F41/F43 all require a fold. Under architecture A the chain is
    engineered not to have one, and ESMFold does not return "disordered" — it returns a
    confident-looking arbitrary structure whose SASA is fiction. Their sequence-level
    counterparts are here instead, and F35's pLDDT INVERTS in meaning: low is expected, and an
    unexpectedly high-confidence region is a warning that something is folding where it should
    not, potentially burying a cleavage site.

  * DNA/mRNA FEATURES ARE ABSENT (F04, F05, F06, F12, part of F07). These are properties of the
    codon choice, not of the design — a free variable. Ranking a candidate down for a GC extreme
    its optimiser should have removed is a category error. They belong in back-translation, which
    flags only what cannot be fixed.

Not implemented for want of a defensible resource, rather than by choice: F36 (BiP site density,
needs the LIMBO DnaK matrix) and F49 (degrons, needs the DEGRONOPEDIA motif set). Inventing motif
sets for either would produce confident numbers with no basis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from concatemer import screens
from concatemer.digest import DigestReport
from concatemer.spec import ConcatemerSpec

LOWER, HIGHER = "lower", "higher"


@dataclass
class FeatureDef:
    key: str
    group: str
    direction: str            # which way is better
    spec_id: str = ""         # MoComMoSec cross-reference
    doc: str = ""


# Groups exist so ranking happens over ~6 roughly orthogonal axes rather than over dozens of
# correlated columns. Summing correlated ranks silently upweights whatever is measured most
# often — six flavours of hydrophobicity would outvote everything else while looking neutral.
FEATURES: list[FeatureDef] = [
    # ── product: what the digest actually delivers ────────────────────────────
    FeatureDef("pct_copies_exact", "product", HIGHER, "", "Designed copies released with exact termini."),
    FeatureDef("payload_fraction_mass", "product", HIGHER, "", "Share of chain residue mass that is product."),
    FeatureDef("n_unintended", "product", LOWER, "", "Fragments that are neither a target peptide nor a spacer."),
    FeatureDef("n_impaired_sites", "product", LOWER, "F18", "Sites with a context that slows cleavage."),
    FeatureDef("ratio_deviation", "product", LOWER, "", "Max fold-deviation of delivered blend from designed."),
    # ── aggregation ───────────────────────────────────────────────────────────
    FeatureDef("max_hydrophobic_run", "aggregation", LOWER, "F38", "Longest unbroken hydrophobic run."),
    FeatureDef("max_window_hydropathy", "aggregation", LOWER, "F41", "Peak 5-residue Kyte-Doolittle window."),
    FeatureDef("mean_hydropathy", "aggregation", LOWER, "F22", "Mean hydropathy of the whole chain."),
    # ── disorder: the architecture-A conformance axis ─────────────────────────
    FeatureDef("uversky_margin", "disorder", HIGHER, "", "Distance below the Uversky boundary; >0 = disordered."),
    FeatureDef("kappa", "disorder", LOWER, "", "Charge patterning; blocky charge drives ER phase separation."),
    FeatureDef("fcr", "disorder", HIGHER, "", "Fraction of charged residues."),
    # ── glycosylation / PTM burden ────────────────────────────────────────────
    FeatureDef("n_sequons", "glycosylation", LOWER, "F43", "N-X-S/T motifs; all exposed on a disordered chain."),
    FeatureDef("n_junction_sequons", "glycosylation", LOWER, "F43", "Sequons created by the ordering alone."),
    FeatureDef("st_fraction", "glycosylation", LOWER, "F45", "Ser+Thr fraction — O-mannosylation substrate."),
    # ── translation ───────────────────────────────────────────────────────────
    FeatureDef("max_polybasic_run", "translation", LOWER, "F14", "Consecutive K/R; stalls the ribosome."),
    FeatureDef("max_polypro_run", "translation", LOWER, "F15", "Consecutive prolines; stalls the ribosome."),
    # ── trafficking ───────────────────────────────────────────────────────────
    FeatureDef("net_charge_ph7", "trafficking", LOWER, "F61", "Net charge; strongly cationic chains stick to the cell wall."),
    FeatureDef("length", "trafficking", LOWER, "F57", "Chain length, standing in for hydrodynamic size."),
]
BY_KEY = {f.key: f for f in FEATURES}
GROUPS = sorted({f.group for f in FEATURES})


@dataclass
class FeatureRow:
    candidate_id: str
    values: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def _max_run(seq: str, chars: str) -> int:
    best = run = 0
    for c in seq:
        run = run + 1 if c in chars else 0
        best = max(best, run)
    return best


def failure_flags(seq: str, spec: ConcatemerSpec, report: DigestReport) -> list[str]:
    """
    Near-absolute, mechanistic disqualifiers.

    These are the part of the screen that transfers out of distribution. A fitted ranking weight
    learned on natural Pichia proteins says little about a synthetic repeat concatemer, but a
    construct with no cleavage site cannot release anything regardless of what any model thinks.
    """
    out: list[str] = []

    if not report.sites:
        out.append("no cleavage sites — nothing can be released")
    elif report.n_clean == 0:
        out.append(f"no clean cleavage sites ({report.n_blocked} blocked, {report.n_impaired} impaired)")
    if sum(report.released_exact.values()) == 0:
        out.append("zero peptides released with exact termini")

    missing = [p.name for p in spec.peptides
               if p.min_copies > 0 and report.released_exact.get(p.name, 0) == 0]
    if missing:
        out.append(f"mandatory peptide(s) never released: {', '.join(missing)}")

    n_cys = seq.count("C")
    if n_cys:
        # F29. On a disordered secreted chain there is no partner to pair with, so free thiols
        # keep PDI engaged and drive covalent aggregation.
        out.append(f"{n_cys} free cysteine(s)")

    if re.search(r"(HDEL|KDEL)$", seq):
        out.append("C-terminal ER retention motif (F60)")

    # If the design does NOT nominate a KR chemistry, any KR is still cut by Kex2 in the Golgi
    # during secretion. An in-vitro-digest design containing KR is therefore processed before it
    # ever leaves the cell — a catastrophic surprise that no in-vitro simulation would show.
    kr_intended = any("KR" in r.motif for r in spec.rules)
    if not kr_intended and "KR" in seq:
        out.append(f"{seq.count('KR')} internal KR — Kex2 will cut these in the Golgi (F54)")

    _, _, disordered = screens.uversky(seq)
    if not disordered:
        out.append("predicted globular — violates the disordered-by-design architecture")
    return out


def compute(seq: str, spec: ConcatemerSpec, report: DigestReport,
            unit_seqs: list[str], candidate_id: str = "") -> FeatureRow:
    """Every ranking feature plus the failure census for one candidate."""
    h, r, _ = screens.uversky(seq)
    boundary = (r + 1.151) / 2.785

    designed = {k: v for k, v in report.designed_copies.items() if v}
    ratios = []
    for name, n in designed.items():
        got = report.released_exact.get(name, 0)
        ratios.append(abs((got / n) if n else 0.0))
    # Spread of per-peptide recovery: 0 when every peptide is recovered at the same rate, which
    # is what keeps the delivered blend equal to the designed one.
    ratio_dev = (max(ratios) - min(ratios)) if len(ratios) > 1 else 0.0

    vals = {
        "pct_copies_exact": round(report.pct_copies_exact, 3),
        "payload_fraction_mass": round(report.payload_fraction_mass, 3),
        "n_unintended": float(report.n_unintended),
        "n_impaired_sites": float(report.n_impaired),
        "ratio_deviation": round(ratio_dev, 4),

        "max_hydrophobic_run": float(screens.max_hydrophobic_run(seq)),
        "max_window_hydropathy": round(screens.max_window_hydropathy(seq), 4),
        "mean_hydropathy": round(screens.hydropathy(seq), 4),

        "uversky_margin": round(boundary - h, 4),
        "kappa": screens.kappa(seq),
        "fcr": round(screens.fcr(seq), 4),

        "n_sequons": float(len(screens.sequons(seq))),
        "n_junction_sequons": float(len(screens.junction_sequons(unit_seqs))),
        "st_fraction": round(screens.st_fraction(seq), 4),

        "max_polybasic_run": float(_max_run(seq, "KR")),
        "max_polypro_run": float(_max_run(seq, "P")),

        "net_charge_ph7": round(screens.ncpr(seq) * len(seq), 3),
        "length": float(len(seq)),
    }
    return FeatureRow(candidate_id=candidate_id, values=vals,
                      failures=failure_flags(seq, spec, report))
