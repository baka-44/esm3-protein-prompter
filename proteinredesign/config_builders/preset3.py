"""
proteinredesign/config_builders/preset3.py — Motif scaffolding / inpainting (RF3, indexed
multi-segment "fill-in-the-middle"). See docs/plans/rfdiffusion_mpnn_backend.md D12.

Keep N discontiguous blocks of an input structure fixed (at their absolute coordinates and
register — *indexed*) and generate the bridges between them, as a single chain. General case:
N kept blocks → N-1 generated internal bridges.

Example: keep "1-20, 50-80, 130-160, 190-200" → contig
    A1-20,29,A50-80,49,A130-160,29,A190-200
where each unlabeled number is a generated bridge of that length (default = the original gap
length, so total length + register are preserved). Chain-labelled ranges are taken from the
input (indexed — coordinates preserved); unlabelled numbers are designed.

This is the generation engine Borrowed Bodies drives (the mount + torso halves become the kept
blocks); here the kept blocks all come from one input protein, so the bridges rebuild geometry
that was originally there (well-posed).

Decision B5 — code, not an LLM, builds the contig. Terminal regions outside the first/last kept
block are dropped (internal bridges only — terminal extension deferred, D12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from proteinredesign.config_builders.common import ConfigError, index_chain_residues

__all__ = ["ConfigError", "Preset3Config", "build_preset3_config", "K_MAX", "M_MAX", "MAX_BLOCKS"]

K_MAX = 10        # RF3 designs (diffusion_batch_size)
M_MAX = 10        # MPNN sequences per design
MAX_BLOCKS = 12   # sanity cap on the number of kept blocks

_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def _parse_keep_ranges(s: str) -> list[tuple[int, int]]:
    """'1-20, 50-80, 190-200' → [(1,20),(50,80),(190,200)]. Single '50' → (50,50)."""
    ranges: list[tuple[int, int]] = []
    for tok in (t for t in s.split(",") if t.strip()):
        m = _RANGE.match(tok)
        if not m:
            raise ConfigError(f"Could not parse keep-range '{tok.strip()}'. Use forms like '50-80' or '90'.")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            raise ConfigError(f"Keep-range '{tok.strip()}' is reversed (start > end).")
        ranges.append((a, b))
    return ranges


@dataclass
class Preset3Config:
    """Validated RF3 indexed multi-segment (inpainting) spec."""

    chain_id: str
    keep_ranges: list[tuple[int, int]]          # author-numbered kept blocks (sorted)
    gaps: list[int]                             # generated bridge lengths, between consecutive blocks
    contig: str                                 # RF3 contig (code-built — B5)
    motif_residues: list[dict]                  # every kept residue {chain_id, author_num} (MPNN fix + motif-RMSD)
    k: int
    m: int
    warnings: list[str] = field(default_factory=list)

    @property
    def total_designs(self) -> int:
        return self.k * self.m

    def to_params(self) -> dict:
        return {
            "chain_id": self.chain_id,
            "keep_ranges": [list(r) for r in self.keep_ranges],
            "gaps": self.gaps,
            "contig": self.contig,
            "motif_residues": self.motif_residues,
            "k": self.k,
            "m": self.m,
        }


def build_preset3_config(
    pdb_source,
    keep_ranges_str: str,
    k: int,
    m: int,
    chain_id: str | None = None,
) -> Preset3Config:
    """
    Build a validated inpainting config: keep the listed blocks, generate the bridges between
    them (each bridge = the original gap length, preserving register).

    Raises:
        ConfigError: <2 kept blocks, overlapping/out-of-order ranges, a residue not in the chain,
                     or K/M out of range.
    """
    if not (1 <= int(k) <= K_MAX):
        raise ConfigError(f"K (designs) must be between 1 and {K_MAX}; got {k}.")
    if not (1 <= int(m) <= M_MAX):
        raise ConfigError(f"M (sequences per design) must be between 1 and {M_MAX}; got {m}.")

    ranges = _parse_keep_ranges(keep_ranges_str)
    if len(ranges) < 2:
        raise ConfigError(
            "Motif scaffolding needs at least 2 kept blocks with a gap to fill between them "
            "(e.g. '1-20, 50-80'). To vary a whole backbone use Scaffold diversification."
        )
    if len(ranges) > MAX_BLOCKS:
        raise ConfigError(f"Too many kept blocks ({len(ranges)}); max {MAX_BLOCKS}.")

    ranges.sort()
    for (a1, b1), (a2, b2) in zip(ranges, ranges[1:]):
        if a2 <= b1:
            raise ConfigError(
                f"Kept blocks {a1}-{b1} and {a2}-{b2} overlap or touch — they must be separated "
                f"by a gap to fill."
            )

    chain_id, by_author, warnings = index_chain_residues(pdb_source, chain_id)
    for a, b in ranges:
        for n in (a, b):
            if n not in by_author:
                raise ConfigError(
                    f"Residue {n} is not present in chain '{chain_id}' (author numbering). "
                    f"Check the keep-ranges."
                )

    # Gaps between consecutive blocks (original lengths → preserve register). Build contig.
    gaps: list[int] = []
    contig_parts: list[str] = [f"{chain_id}{ranges[0][0]}-{ranges[0][1]}"]
    for (a1, b1), (a2, b2) in zip(ranges, ranges[1:]):
        gap = a2 - b1 - 1
        if gap < 1:
            raise ConfigError(f"No gap to fill between blocks ending {b1} and starting {a2}.")
        gaps.append(gap)
        contig_parts.append(str(gap))
        contig_parts.append(f"{chain_id}{a2}-{b2}")
    contig = ",".join(contig_parts)

    motif_residues = [
        {"chain_id": chain_id, "author_num": n}
        for a, b in ranges for n in range(a, b + 1)
    ]

    return Preset3Config(
        chain_id=chain_id,
        keep_ranges=ranges,
        gaps=gaps,
        contig=contig,
        motif_residues=motif_residues,
        k=int(k),
        m=int(m),
        warnings=warnings,
    )
