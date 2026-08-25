#!/usr/bin/env python3
"""
Resize the linkers in an existing .graft package from the gaps its composite actually has.

Packages exported before gap-derived linker sizing (CC18) carry whatever fixed range the
composer defaulted to — `3-8` — regardless of geometry, and the contig is built from that
stored range at submit time. Re-uploading such a package therefore re-runs with the old
linkers even on a fixed backend.

This rewrites only the `linker` entries in graft_spec.json. composite.pdb is copied through
byte-for-byte, so the pose, coordinates, repack set and catalytic site are all preserved and
there is no need to re-pose the graft by hand.

    python scripts/resize_graft_linkers.py in.graft [-o out.graft] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinredesign.composer import _ca_gap, linker_span  # noqa: E402

SPEC = "graft_spec.json"
COMPOSITE = "composite.pdb"


def resize(spec: dict, composite: bytes) -> tuple[dict, list[tuple]]:
    """Return (updated spec, [(index, gap, old, new), ...]) — pure, for testing."""
    order = spec["chain_order"]
    changes = []
    for i, seg in enumerate(order):
        if seg.get("kind") != "linker":
            continue
        prev_, next_ = order[i - 1], order[i + 1]
        if prev_.get("kind") != "fragment" or next_.get("kind") != "fragment":
            continue
        gap = _ca_gap(composite, (prev_["chain"], prev_["end"]), (next_["chain"], next_["start"]))
        lo, hi = linker_span(gap)
        old = (seg["length_min"], seg["length_max"])
        if old != (lo, hi):
            seg["length_min"], seg["length_max"] = lo, hi
            changes.append((i, gap, old, (lo, hi), prev_["label"], next_["label"]))
    if changes:
        spec.setdefault("provenance", {})["linker_sizing"] = "auto (resized post-export)"
    return spec, changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("-o", "--out", help="default: <name>.resized.graft")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args()

    with zipfile.ZipFile(a.package) as z:
        names = z.namelist()
        for required in (SPEC, COMPOSITE):
            if required not in names:
                print(f"error: {a.package} has no {required}", file=sys.stderr)
                return 2
        spec = json.loads(z.read(SPEC))
        composite = z.read(COMPOSITE)
        others = {n: z.read(n) for n in names if n not in (SPEC, COMPOSITE)}

    spec, changes = resize(spec, composite)

    if not changes:
        print("linkers already match the composite's gaps — nothing to do")
        return 0
    for _, gap, old, new, pl, nl in changes:
        g = "unmeasurable" if gap != gap or gap == float("inf") else f"{gap:.1f} Å"
        ext_old = f"{gap / ((old[0] + 1) * 3.8) * 100:.0f}%" if g != "unmeasurable" else "—"
        ext_new = f"{gap / ((new[0] + 1) * 3.8) * 100:.0f}%" if g != "unmeasurable" else "—"
        print(f"  {pl} -> {nl}   gap {g:>12}   {old[0]}-{old[1]} ({ext_old} extended)"
              f"  ->  {new[0]}-{new[1]} ({ext_new})")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out = a.out or a.package.rsplit(".graft", 1)[0] + ".resized.graft"
    tmp = out + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(SPEC, json.dumps(spec, indent=2))
        z.writestr(COMPOSITE, composite)          # byte-for-byte, pose preserved
        for n, b in others.items():
            z.writestr(n, b)
    shutil.move(tmp, out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
