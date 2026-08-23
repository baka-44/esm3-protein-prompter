#!/usr/bin/env python3
"""
Submit the Kex2 P-domain-deletion fold experiment to the rf3-worker.

The question: Kex2's P/Homo B domain (462-596) is described as structurally required, but the
same S8 catalytic fold is autonomously stable at 275 aa in subtilisin BPN', which has no
P-domain at all. So — does the Kex2 catalytic domain still fold on its own?

Four constructs, in the order they answer that:
  Kex2_catP_114_599      486 aa  CONTROL   — catalytic + P domain (the crystallised unit)
  Kex2_cat_114_453       340 aa  TEST      — P-domain deleted, all six Ca ligands retained
  Kex2_S8_141_453        313 aa  TEST tight— loses Ca ligand 135, so Ca site 1 is incomplete
  SubBPN_mature_108_382  275 aa  REFERENCE — an S8 protease that is natively P-domain-free

Read the result as a comparison, not an absolute: the AlphaFold reference values are
catalytic domain 96.5 / P-domain 98.2 / subtilisin mature 98.0 mean pLDDT. What matters is
how far Kex2_cat_114_453 falls below its own control, and whether the geometry filter still
passes on the triad, the oxyanion hole and the two calcium sites.

Usage (needs GCP credentials + the proteinredesign env):
    python3 scripts/submit_kex2_fold.py you@phyx44.com
"""

import sys

import requests

UNIPROT = "https://rest.uniprot.org/uniprotkb/{}.fasta"

# (accession, construct name, first residue, last residue) — 1-based inclusive, parent numbering
CONSTRUCTS = [
    ("P13134", "Kex2_catP_114_599", 114, 599),
    ("P13134", "Kex2_cat_114_453", 114, 453),
    ("P13134", "Kex2_S8_141_453", 141, 453),
    ("P00782", "SubBPN_mature_108_382", 108, 382),
]

# Catalytic machinery in PARENT numbering. The worker renumbers each model into this frame,
# so these positions are valid regardless of where the construct was sliced.
SITES = {
    "P13134": {  # Kex2: triad Asp175-His213-Ser385, oxyanion Asn314
        "nucleophile": ["A", 385], "base": ["A", 213], "acid": ["A", 175],
        "oxyanion": [["A", 314]],
        "metal_sites": {"Ca1": [["A", 135], ["A", 184], ["A", 227]],
                        "Ca2": [["A", 277], ["A", 320], ["A", 350]]},
    },
    "P00782": {  # Subtilisin BPN' precursor numbering (mature +107): Asp32/His64/Ser221/Asn155
        "nucleophile": ["A", 328], "base": ["A", 171], "acid": ["A", 139],
        "oxyanion": [["A", 262]],
        "metal_sites": {"CaA": [["A", 109], ["A", 148], ["A", 182],
                                ["A", 184], ["A", 186], ["A", 188]],
                        "CaB": [["A", 276], ["A", 278], ["A", 281]]},
    },
}


def fetch(acc: str) -> str:
    r = requests.get(UNIPROT.format(acc), timeout=30)
    r.raise_for_status()
    return "".join(l.strip() for l in r.text.splitlines() if not l.startswith(">"))


def clip_site(site: dict, lo: int, hi: int) -> dict:
    """Drop catalytic/metal positions that fall outside the construct.

    Silently scoring a site whose ligands were sliced away would report a false failure, so
    incomplete metal sites are removed rather than measured. (Kex2_S8_141_453 loses Ca1's
    residue 135 this way — which is exactly why 114-453 is the construct to trust.)"""
    inside = lambda p: lo <= int(p[1]) <= hi          # noqa: E731
    out = {k: v for k, v in site.items() if k not in ("oxyanion", "metal_sites")}
    if not all(inside(out[k]) for k in ("nucleophile", "base", "acid") if out.get(k)):
        raise ValueError("catalytic triad falls outside the construct")
    out["oxyanion"] = [p for p in site.get("oxyanion", []) if inside(p)]
    out["metal_sites"] = {m: ligs for m, ligs in site.get("metal_sites", {}).items()
                          if all(inside(p) for p in ligs)}
    dropped = [m for m in site.get("metal_sites", {}) if m not in out["metal_sites"]]
    if dropped:
        print(f"      note: metal site(s) {', '.join(dropped)} incomplete in this construct — not scored")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    user_email = sys.argv[1]

    cache, sequences, offsets, sites = {}, {}, {}, {}
    for acc, name, lo, hi in CONSTRUCTS:
        cache.setdefault(acc, fetch(acc))
        sequences[name] = cache[acc][lo - 1:hi]
        offsets[name] = lo - 1                      # local residue 1 -> parent residue `lo`
        print(f"  {name:24s} {len(sequences[name]):>4} aa  (offset {offsets[name]})")
        sites[name] = clip_site(SITES[acc], lo, hi)

    from proteinredesign.submit import submit_fold
    rec = submit_fold(sequences=sequences, user_email=user_email, offsets=offsets,
                      catalytic_sites=sites,
                      title="Kex2 P-domain deletion · fold + active-site QC")
    print(f"\nsubmitted job {getattr(rec, 'job_id', rec)}")
    print("results.json will carry mean_plddt, min_catalytic_plddt and the geometry verdict "
          "per construct; each fold is written as <name>.pdb in parent numbering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
