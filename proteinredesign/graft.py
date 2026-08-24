"""
proteinredesign/graft.py — the model-agnostic **graft package** (Borrowed Bodies, BB3/CC14).

A graft package is a self-contained, portable description of a domain-insertion graft: an ordered
N→C chain of FIXED fragments (from a mount + a torso, at their composed coordinates) interleaved
with GENERATED linkers, plus the repack set (BB2/CC9). It is NOT tied to any generator — per-model
adapters translate it. `to_engine_params()` is the RFdiffusion3 adapter (→ the indexed
multi-segment + all-atom + repack engine we already run for #3/#6).

Package layout (a zip):
    graft_package/
      composite.pdb      # fixed fragments at their posed coords (mount + torso, one frame)
      graft_spec.json    # the neutral spec below

The package also doubles as the Composer's session-save format (CC16).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Any

SPEC_NAME = "graft_spec.json"
COMPOSITE_NAME = "composite.pdb"
VERSION = "1.0"


@dataclass
class Fragment:
    """One contiguous FIXED block in the final chain (a kept mount/torso piece)."""

    label: str                 # e.g. "TORSO-1", "MOUNT-1" (CC7)
    source: str                # "mount" | "torso"
    chain: str                 # chain id in the composite PDB
    start: int                 # author residue number (inclusive)
    end: int                   # author residue number (inclusive)
    fixed_atoms: str = "BKBN"  # "BKBN" (backbone) | "ALL" (all-atom, e.g. mount catalytic — CC3)

    def residues(self) -> list[tuple[str, int]]:
        return [(self.chain, n) for n in range(self.start, self.end + 1)]

    def contig_token(self) -> str:
        return f"{self.chain}{self.start}-{self.end}"


@dataclass
class Linker:
    """A GENERATED bridge between two fixed fragments."""

    length_min: int
    length_max: int

    def contig_token(self) -> str:
        return str(self.length_min) if self.length_min == self.length_max else \
            f"{self.length_min}-{self.length_max}"


@dataclass
class GraftSpec:
    """
    The neutral graft specification (role-annotated, in N→C order).

    chain_order alternates Fragment / Linker / Fragment / … (CC7). repack_residues (CC9) are
    fixed-fragment residues MPNN may re-identify (backbone stays fixed). provenance + metrics are
    informational.
    """

    chain_order: list  # list[Fragment | Linker], N→C
    repack_residues: list[dict] = field(default_factory=list)   # [{chain, author_num}]
    k: int = 5
    m: int = 3
    provenance: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    # OPTIONAL active-site spec (CatalyticSite JSON). When present the worker reports
    # catalytic geometry per generated candidate; when absent that reporting is simply
    # skipped. It is never required — a graft without one designs exactly as before.
    catalytic_site: dict = field(default_factory=dict)
    version: str = VERSION

    # ── serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        order = []
        for seg in self.chain_order:
            if isinstance(seg, Fragment):
                order.append({"kind": "fragment", **asdict(seg)})
            else:
                order.append({"kind": "linker", **asdict(seg)})
        return {
            "version": self.version,
            "chain_order": order,
            "repack_residues": self.repack_residues,
            "k": self.k, "m": self.m,
            "provenance": self.provenance,
            "metrics": self.metrics,
            "catalytic_site": self.catalytic_site,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GraftSpec":
        order: list = []
        for seg in d.get("chain_order", []):
            kind = seg.get("kind")
            if kind == "fragment":
                order.append(Fragment(
                    label=seg["label"], source=seg["source"], chain=seg["chain"],
                    start=int(seg["start"]), end=int(seg["end"]),
                    fixed_atoms=seg.get("fixed_atoms", "BKBN"),
                ))
            elif kind == "linker":
                order.append(Linker(int(seg["length_min"]), int(seg["length_max"])))
            else:
                raise ValueError(f"Unknown chain_order segment kind: {kind!r}")
        return cls(
            chain_order=order,
            repack_residues=d.get("repack_residues", []),
            k=int(d.get("k", 5)), m=int(d.get("m", 3)),
            provenance=d.get("provenance", {}), metrics=d.get("metrics", {}),
            catalytic_site=d.get("catalytic_site") or {},
            version=d.get("version", VERSION),
        )

    # ── validity (CC13 export gate uses this) ─────────────────────────────────
    def fragments(self) -> list[Fragment]:
        return [s for s in self.chain_order if isinstance(s, Fragment)]

    def linkers(self) -> list[Linker]:
        return [s for s in self.chain_order if isinstance(s, Linker)]

    def validate(self) -> list[str]:
        """Return a list of CRITICAL problems (empty = ok to export/generate)."""
        errs: list[str] = []
        frags = self.fragments()
        if len(frags) < 2:
            errs.append("A graft needs at least 2 fixed fragments with a linker between them.")
        # chain_order must alternate fragment/linker/fragment, starting and ending on a fragment.
        for i, seg in enumerate(self.chain_order):
            expect_fragment = (i % 2 == 0)
            if expect_fragment and not isinstance(seg, Fragment):
                errs.append(f"Segment {i} should be a fragment (chain must start/alternate on fragments).")
            if not expect_fragment and not isinstance(seg, Linker):
                errs.append(f"Segment {i} should be a linker between fragments.")
        if self.chain_order and not isinstance(self.chain_order[-1], Fragment):
            errs.append("The chain must end on a fixed fragment, not a linker.")
        for lk in self.linkers():
            if lk.length_min < 1 or lk.length_max < lk.length_min:
                errs.append(f"Invalid linker length {lk.length_min}-{lk.length_max}.")
        return errs


@dataclass
class GraftPackage:
    """A GraftSpec + the composite structure it references."""

    spec: GraftSpec
    composite_pdb: bytes

    def to_bytes(self) -> bytes:
        """Serialise to a .graft zip (composite.pdb + graft_spec.json)."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(COMPOSITE_NAME, self.composite_pdb)
            zf.writestr(SPEC_NAME, json.dumps(self.spec.to_dict(), indent=2).encode())
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "GraftPackage":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if SPEC_NAME not in names or COMPOSITE_NAME not in names:
                raise ValueError(f"Not a graft package: missing {SPEC_NAME} or {COMPOSITE_NAME}.")
            spec = GraftSpec.from_dict(json.loads(zf.read(SPEC_NAME).decode()))
            composite = zf.read(COMPOSITE_NAME)
        return cls(spec=spec, composite_pdb=composite)


# ── RFdiffusion3 adapter: graft package → engine params (CC15) ─────────────────

def to_engine_params(package: GraftPackage) -> dict:
    """
    Translate a graft package into params for the (validated) RF3 indexed-multi-segment engine
    (`_generate_motif_candidates`, generalised for select_fixed_atoms). Returns the params dict;
    the composite PDB (package.composite_pdb) is uploaded as the job's input.

      contig             — indexed multi-segment, N→C: "A1-15,14,B5-90,10,A30-46" (chain-labelled
                           fragments fixed from the composite; unlabelled numbers = generated linkers).
      select_fixed_atoms — per-residue atom fixing for ALL-atom fragments (mount catalytic — CC3).
      motif_residues     — every fixed-fragment residue (MPNN fixes these, minus repack; motif-RMSD).
      repack_residues    — fixed-fragment residues MPNN may re-identify (BB2/CC9) as "<chain><num>".
    """
    spec = package.spec
    errs = spec.validate()
    if errs:
        raise ValueError("Invalid graft package: " + "; ".join(errs))

    contig = ",".join(seg.contig_token() for seg in spec.chain_order)

    select_fixed_atoms: dict[str, str] = {}
    motif_residues: list[dict] = []
    for frag in spec.fragments():
        for chain, num in frag.residues():
            motif_residues.append({"chain_id": chain, "author_num": num})
            if frag.fixed_atoms and frag.fixed_atoms != "BKBN":
                select_fixed_atoms[f"{chain}{num}"] = frag.fixed_atoms

    repack_tokens = [f"{r['chain']}{r['author_num']}" for r in spec.repack_residues]

    params = {
        "contig": contig,
        "select_fixed_atoms": select_fixed_atoms or None,
        "motif_residues": motif_residues,
        "repack_residues": repack_tokens,
        "k": spec.k, "m": spec.m,
    }
    # Optional: carried through only when the package defines one. Absent => the worker
    # skips geometry reporting entirely; the key is not even present in params.
    if spec.catalytic_site:
        params["catalytic_site"] = spec.catalytic_site
    return params
