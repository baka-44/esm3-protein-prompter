"""
cofold/config_builders/preset1.py — Preset #1 (fixed-backbone redesign).

Turns scientist-level inputs — an uploaded PDB plus a list of residues to keep
fixed, given in the PDB's own (author) numbering, e.g. "K67, R82" or "67, 82" —
into a validated ProteinMPNN `fixed_positions` spec.

Two things this builder is responsible for (decision B5):
  1. **Author-numbering → sequential-position mapping.** Scientists type residue
     numbers as they appear in the PDB/paper (author numbering, `residue.id[1]`),
     which is often NOT 1-based-sequential (gaps, non-1 starts, insertions).
     ProteinMPNN wants 1-based positions within the chain's residue order. We map
     between them and surface the mapping (e.g. "K67 → A#12").
  2. **residue# ↔ AA validation.** If the scientist writes a letter (K67), we
     verify residue 67 really is Lys in the PDB. A mismatch almost always means
     they used the wrong numbering — we raise rather than silently fix the wrong
     residue.

Code, not an LLM, builds this config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.pdb_utils import get_residues

# One token: optional leading letter, number, optional trailing letter.
# Accepts K67, 67K, 67, 1M — matching the existing nl_parser residue grammar.
_RESIDUE_TOKEN = re.compile(r"([A-Za-z]?)(\d+)([A-Za-z]?)")


class ConfigError(ValueError):
    """Raised when the scientist's input can't be turned into a valid config."""


@dataclass
class FixedResidue:
    author_num: int   # PDB author residue number (what the scientist typed)
    chain_id: str
    seq_pos: int      # 1-based position within the chain (ProteinMPNN indexing)
    aa: str           # single-letter AA at that residue in the PDB


@dataclass
class Preset1Config:
    """Validated ProteinMPNN fixed-positions spec for preset #1."""

    chain_id: str
    fixed_positions: dict[str, list[int]]   # {chain: [seq_pos, ...]} — ProteinMPNN format
    fixed_residues: list[FixedResidue]
    mapping_summary: str                    # e.g. "K67→A#12, R82→A#27"
    warnings: list[str] = field(default_factory=list)

    def to_params(self) -> dict:
        """Serialisable params for the JobManifest."""
        return {
            "chain_id": self.chain_id,
            "fixed_positions": self.fixed_positions,
            "fixed_residues": [
                {"author_num": r.author_num, "chain_id": r.chain_id,
                 "seq_pos": r.seq_pos, "aa": r.aa}
                for r in self.fixed_residues
            ],
            "mapping_summary": self.mapping_summary,
        }


def _aa_one_letter(resname: str) -> str:
    """3-letter residue name → 1-letter code ('X' if unknown). Lazy import of Bio."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    return protein_letters_3to1.get(resname.strip().upper(), "X")


def _index_chain_residues(
    pdb_source, chain_id: str | None
) -> tuple[str, dict[int, tuple[int, str]], list[str]]:
    """
    Build a lookup for the chosen chain.

    Returns:
        chain_id:    the chain actually used
        by_author:   {author_num: (seq_pos_1based, aa_one_letter)}
        warnings:    list of non-fatal warnings
    """
    residues = get_residues(pdb_source, chain_id=None)  # all chains, ATOM only
    if not residues:
        raise ConfigError("No protein residues found in the PDB (ATOM records).")

    # Group residues by chain, preserving order.
    chains: dict[str, list] = {}
    for res in residues:
        cid = res.get_parent().id
        chains.setdefault(cid, []).append(res)

    warnings: list[str] = []
    if chain_id is None:
        chain_id = next(iter(chains))
        if len(chains) > 1:
            warnings.append(
                f"PDB has {len(chains)} chains ({', '.join(chains)}); using chain "
                f"'{chain_id}'. Specify a chain to redesign a different one."
            )
    if chain_id not in chains:
        raise ConfigError(
            f"Chain '{chain_id}' not found. Available chains: {', '.join(chains) or '(none)'}."
        )

    by_author: dict[int, tuple[int, str]] = {}
    for seq_pos, res in enumerate(chains[chain_id], start=1):
        author_num = res.id[1]
        by_author[author_num] = (seq_pos, _aa_one_letter(res.get_resname()))

    return chain_id, by_author, warnings


def parse_fixed_residue_tokens(fixed_residues_str: str) -> list[tuple[str | None, int]]:
    """
    Parse "K67, R82, 90" → [("K", 67), ("R", 82), (None, 90)].

    Raises ConfigError on a token that isn't a residue spec.
    """
    out: list[tuple[str | None, int]] = []
    for raw in re.split(r"[,\s]+", fixed_residues_str.strip()):
        token = raw.strip()
        if not token:
            continue
        m = _RESIDUE_TOKEN.fullmatch(token)
        if not m:
            raise ConfigError(
                f"Could not parse residue '{token}'. Use forms like K67, 67K, or 67."
            )
        letter = (m.group(1) or m.group(3)).upper() or None
        out.append((letter, int(m.group(2))))
    return out


def build_preset1_config(
    pdb_source,
    fixed_residues_str: str,
    chain_id: str | None = None,
) -> Preset1Config:
    """
    Build a validated ProteinMPNN fixed-positions config for preset #1.

    Args:
        pdb_source:         PDB path or raw bytes/string (parsed via pdb_utils).
        fixed_residues_str: Residues to keep fixed, PDB author numbering (e.g. "K67, R82").
        chain_id:           Chain to redesign. None → first chain (warns if multiple).

    Raises:
        ConfigError: no fixed residues given, unknown residue number, or a stated
                     AA letter that doesn't match the PDB (residue#↔AA mismatch).
    """
    tokens = parse_fixed_residue_tokens(fixed_residues_str)
    if not tokens:
        raise ConfigError(
            "No fixed residues provided. List at least one residue to keep fixed "
            "(the rest of the sequence is redesigned)."
        )

    chain_id, by_author, warnings = _index_chain_residues(pdb_source, chain_id)

    fixed_residues: list[FixedResidue] = []
    seen: set[int] = set()
    for letter, author_num in tokens:
        if author_num in seen:
            continue
        seen.add(author_num)

        if author_num not in by_author:
            raise ConfigError(
                f"Residue {author_num} is not present in chain '{chain_id}'. "
                f"Check the numbering (this tool uses the PDB's author numbering)."
            )
        seq_pos, aa = by_author[author_num]

        if letter and letter != aa:
            raise ConfigError(
                f"'{letter}{author_num}' does not match the PDB: residue {author_num} "
                f"in chain '{chain_id}' is {aa}, not {letter}. This usually means the "
                f"residue numbering is off (author vs sequential)."
            )

        fixed_residues.append(
            FixedResidue(author_num=author_num, chain_id=chain_id, seq_pos=seq_pos, aa=aa)
        )

    # ProteinMPNN fixed_positions: {chain: sorted list of 1-based sequential positions}
    positions = sorted(r.seq_pos for r in fixed_residues)
    fixed_positions = {chain_id: positions}

    mapping_summary = ", ".join(
        f"{r.aa}{r.author_num}→{r.chain_id}#{r.seq_pos}" for r in fixed_residues
    )

    return Preset1Config(
        chain_id=chain_id,
        fixed_positions=fixed_positions,
        fixed_residues=fixed_residues,
        mapping_summary=mapping_summary,
        warnings=warnings,
    )
