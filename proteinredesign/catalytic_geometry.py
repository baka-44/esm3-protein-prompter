"""
proteinredesign/catalytic_geometry.py — deterministic catalytic-site geometry QC (tier-1 filter).

Why this exists: neither a docking score nor a global pLDDT can tell you whether a designed or
re-bodied enzyme still has a *competent* active site. Docking scores a pocket's shape and will
happily reward a catalytically dead protein; global pLDDT averages away a handful of residues
that are the entire point. This module measures the geometry that actually has to be right —
charge-relay H-bond distances, the relay angle, oxyanion-hole placement, and metal-site
integrity — and reports the model's own per-residue confidence at exactly those positions.

It is deterministic, dependency-light (Biopython + numpy) and fast, so it runs as the FIRST
filter over a large generated pool: kill the junk here, spend GPU on co-folding only on
survivors.

Reference bands for a Ser/Cys-His-Asp charge relay are drawn from canonical serine-protease
geometry; they are deliberately permissive, because the purpose is to reject the clearly broken,
not to score the excellent. Use `reference_pdb` to compare against a known-good structure of the
same enzyme when you have one — that is always stronger than absolute bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

# ── canonical geometry (Angstrom / degrees) ───────────────────────────────────
# Nucleophile OG/SG -> His NE2 hydrogen bond.
NUC_BASE_OK = (2.4, 3.6)
# His ND1 -> Asp/Glu carboxylate hydrogen bond.
BASE_ACID_OK = (2.4, 3.5)
# Angle nucleophile--NE2--ND1: the relay is not linear; the imidazole geometry puts this
# in a broad band. Outside it, the histidine is not oriented to shuttle the proton.
RELAY_ANGLE_OK = (80.0, 150.0)
# Oxyanion-hole amide (Asn ND2 in subtilases) to the nucleophile.
OXYANION_OK = (3.5, 7.5)
# Metal-ligand cluster: max pairwise spread of coordinating atoms around their centroid.
METAL_CLUSTER_OK = 6.5

_SIDECHAIN_O = {
    "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"], "ASN": ["OD1"], "GLN": ["OE1"],
    "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"],
}


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _angle(a, b, c) -> float:
    """Angle a-b-c in degrees (b is the vertex)."""
    v1, v2 = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@dataclass
class CatalyticSite:
    """Where the chemistry happens. Residue numbers are author numbering in the PDB."""

    name: str
    nucleophile: tuple            # (chain, resi) — Ser/Cys/Thr
    base: tuple                   # (chain, resi) — His
    acid: Optional[tuple] = None  # (chain, resi) — Asp/Glu
    oxyanion: list = field(default_factory=list)          # [(chain, resi), ...]
    metal_sites: dict = field(default_factory=dict)       # {"Ca1": [(chain, resi), ...]}


@dataclass
class Finding:
    label: str
    value: Optional[float]
    unit: str
    ok: bool
    desired: str
    note: str = ""


class Structure:
    """Thin wrapper: residue lookup + per-residue confidence from the B-factor column."""

    def __init__(self, path: str):
        from Bio.PDB import PDBParser
        self.path = path
        self._s = PDBParser(QUIET=True).get_structure("s", path)
        self._res = {}
        for model in self._s:
            for chain in model:
                for r in chain:
                    self._res[(chain.id, r.id[1])] = r
            break  # first model only

    def residue(self, key):
        return self._res.get(key)

    def atom(self, key, names) -> Optional[np.ndarray]:
        r = self.residue(key)
        if r is None:
            return None
        for n in ([names] if isinstance(names, str) else names):
            if n in r:
                return np.asarray(r[n].coord, dtype=float)
        return None

    def resname(self, key) -> Optional[str]:
        r = self.residue(key)
        return r.get_resname() if r is not None else None

    def confidence(self, key) -> Optional[float]:
        """Mean B-factor over the residue = pLDDT for ESMFold / AlphaFold output."""
        r = self.residue(key)
        if r is None:
            return None
        b = [a.get_bfactor() for a in r]
        return float(np.mean(b)) if b else None

    def mean_confidence(self, lo: int = None, hi: int = None, chain: str = None) -> Optional[float]:
        vals = []
        for (c, i), r in self._res.items():
            if chain and c != chain:
                continue
            if lo is not None and i < lo:
                continue
            if hi is not None and i > hi:
                continue
            vals.extend(a.get_bfactor() for a in r)
        return float(np.mean(vals)) if vals else None

    def coordinating_atom(self, key) -> Optional[np.ndarray]:
        """Best guess at the metal-coordinating atom for a residue."""
        rn = self.resname(key)
        if rn in _SIDECHAIN_O:
            a = self.atom(key, _SIDECHAIN_O[rn])
            if a is not None:
                return a
        return self.atom(key, "O")   # backbone carbonyl is a common Ca ligand


def measure(struct: Structure, site: CatalyticSite) -> list[Finding]:
    """Measure the catalytic geometry. Returns findings, most diagnostic first."""
    out: list[Finding] = []
    nuc = struct.atom(site.nucleophile, ["OG", "SG", "OG1"])
    ne2 = struct.atom(site.base, "NE2")
    nd1 = struct.atom(site.base, "ND1")

    # 1. nucleophile -> general base
    d = _dist(nuc, ne2) if nuc is not None and ne2 is not None else None
    out.append(Finding("nucleophile–base", d, "Å", d is not None and NUC_BASE_OK[0] <= d <= NUC_BASE_OK[1],
                       f"{NUC_BASE_OK[0]}–{NUC_BASE_OK[1]} Å (H-bond)",
                       "Ser/Cys must H-bond the His to be deprotonated; the reaction cannot start otherwise."))

    # 2. general base -> acid
    d2 = None
    if site.acid and nd1 is not None:
        ac = struct.atom(site.acid, _SIDECHAIN_O.get(struct.resname(site.acid) or "ASP", ["OD1", "OD2"]))
        # take the closer carboxylate oxygen
        cands = []
        for nm in ["OD1", "OD2", "OE1", "OE2"]:
            a = struct.atom(site.acid, nm)
            if a is not None:
                cands.append(_dist(nd1, a))
        d2 = min(cands) if cands else (_dist(nd1, ac) if ac is not None else None)
    out.append(Finding("base–acid", d2, "Å", d2 is not None and BASE_ACID_OK[0] <= d2 <= BASE_ACID_OK[1],
                       f"{BASE_ACID_OK[0]}–{BASE_ACID_OK[1]} Å (H-bond)",
                       "Asp/Glu orients and polarises the His; without it the relay loses most of its rate."))

    # 3. relay angle
    ang = _angle(nuc, ne2, nd1) if all(x is not None for x in (nuc, ne2, nd1)) else None
    out.append(Finding("relay angle", ang, "°", ang is not None and RELAY_ANGLE_OK[0] <= ang <= RELAY_ANGLE_OK[1],
                       f"{RELAY_ANGLE_OK[0]:.0f}–{RELAY_ANGLE_OK[1]:.0f}°",
                       "His must present the right imidazole face to the nucleophile."))

    # 4. oxyanion hole
    for ox in site.oxyanion:
        a = struct.atom(ox, ["ND2", "NE2", "N"])
        d3 = _dist(nuc, a) if (nuc is not None and a is not None) else None
        out.append(Finding(f"oxyanion {ox[0]}{ox[1]}", d3, "Å",
                           d3 is not None and OXYANION_OK[0] <= d3 <= OXYANION_OK[1],
                           f"{OXYANION_OK[0]}–{OXYANION_OK[1]} Å",
                           "Stabilises the tetrahedral intermediate; the main source of rate enhancement."))

    # 5. metal sites — in an apo prediction there is no ion, so test whether the ligands
    #    still form a tight cluster. A splayed cluster means the site has collapsed.
    for mname, ligs in site.metal_sites.items():
        pts = [struct.coordinating_atom(l) for l in ligs]
        pts = [p for p in pts if p is not None]
        spread = None
        if len(pts) >= 2:
            cen = np.mean(pts, axis=0)
            spread = float(max(np.linalg.norm(p - cen) for p in pts))
        out.append(Finding(f"metal site {mname} spread", spread, "Å",
                           spread is not None and spread <= METAL_CLUSTER_OK,
                           f"<= {METAL_CLUSTER_OK} Å from centroid",
                           f"{len(pts)}/{len(ligs)} ligands resolved; apo models have no ion, so this "
                           "measures whether the coordinating residues still converge."))
    return out


def site_confidence(struct: Structure, site: CatalyticSite) -> dict:
    """Per-residue model confidence (pLDDT) at the catalytic positions themselves."""
    keys = {"nucleophile": site.nucleophile, "base": site.base}
    if site.acid:
        keys["acid"] = site.acid
    for i, ox in enumerate(site.oxyanion, 1):
        keys[f"oxyanion{i}"] = ox
    for mname, ligs in site.metal_sites.items():
        for j, l in enumerate(ligs, 1):
            keys[f"{mname}_lig{j}"] = l
    return {k: struct.confidence(v) for k, v in keys.items()}


def triad_rmsd(struct: Structure, ref: Structure, site: CatalyticSite, ref_site: CatalyticSite) -> Optional[float]:
    """RMSD of the catalytic functional atoms against a known-good reference (no superposition
    of the whole chain — we superpose on the triad itself, so this measures internal geometry)."""
    from Bio.SVDSuperimposer import SVDSuperimposer
    def pts(s, si):
        p = [s.atom(si.nucleophile, ["OG", "SG", "OG1"]), s.atom(si.base, "NE2"), s.atom(si.base, "ND1")]
        if si.acid:
            p.append(s.atom(si.acid, ["OD1", "OE1"]))
        return None if any(x is None for x in p) else np.asarray(p)
    a, b = pts(struct, site), pts(ref, ref_site)
    if a is None or b is None or len(a) != len(b):
        return None
    sup = SVDSuperimposer()
    sup.set(b, a)
    sup.run()
    return float(sup.get_rms())


def verdict(findings: Iterable[Finding]) -> tuple[bool, list[str]]:
    """Overall pass/fail plus the reasons for failure."""
    fails = [f.label for f in findings if not f.ok]
    return (not fails), fails


def report(struct: Structure, site: CatalyticSite, label: str = "") -> str:
    """Human-readable one-site report."""
    fs = measure(struct, site)
    conf = site_confidence(struct, site)
    ok, fails = verdict(fs)
    lines = [f"── {label or site.name} ──"]
    for f in fs:
        v = "n/a" if f.value is None else f"{f.value:.2f}"
        lines.append(f"  {'PASS' if f.ok else 'FAIL'}  {f.label:<28} {v:>7} {f.unit:<2}  (want {f.desired})")
    cs = {k: v for k, v in conf.items() if v is not None}
    if cs:
        lines.append(f"  confidence at catalytic residues: "
                     + ", ".join(f"{k}={v:.1f}" for k, v in cs.items()))
        lines.append(f"  min catalytic confidence: {min(cs.values()):.1f}")
    lines.append(f"  VERDICT: {'PASS' if ok else 'FAIL — ' + ', '.join(fails)}")
    return "\n".join(lines)
