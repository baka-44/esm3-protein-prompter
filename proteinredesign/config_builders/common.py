"""
proteinredesign/config_builders/common.py — shared helpers for preset config builders.

Residue-token parsing, PDB-chain indexing, and author→sequential mapping are
identical across presets that accept "fixed residues" input (preset #1 and #2).
Factored out here so both builders stay in sync — a fix to the numbering logic
(e.g. the author-vs-sequential mismatch class of bug) only needs to happen once.

Code, not an LLM, builds these configs (decision B5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


def aa_one_letter(resname: str) -> str:
    """3-letter residue name → 1-letter code ('X' if unknown). Lazy import of Bio."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    return protein_letters_3to1.get(resname.strip().upper(), "X")


def index_chain_residues(
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
        by_author[author_num] = (seq_pos, aa_one_letter(res.get_resname()))

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


def resolve_fixed_residues(
    pdb_source,
    fixed_residues_str: str,
    chain_id: str | None,
) -> tuple[str, list[FixedResidue], list[str]]:
    """
    Parse + validate fixed-residue tokens against the PDB (author-numbering →
    sequential mapping, residue#↔AA validation). `fixed_residues_str` may be
    empty — callers decide whether that's an error (preset #1 requires at least
    one; preset #2 allows none, relying on ligand context alone).

    Returns:
        chain_id:        the chain actually used
        fixed_residues:  validated FixedResidue list (empty if none given)
        warnings:        non-fatal warnings (e.g. multi-chain PDB defaulted)

    Raises:
        ConfigError: unknown residue number, or a stated AA letter that doesn't
                     match the PDB (residue#↔AA mismatch).
    """
    tokens = parse_fixed_residue_tokens(fixed_residues_str)
    chain_id, by_author, warnings = index_chain_residues(pdb_source, chain_id)

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

    return chain_id, fixed_residues, warnings


def fixed_residues_to_params(fixed_residues: list[FixedResidue]) -> list[dict]:
    return [
        {"author_num": r.author_num, "chain_id": r.chain_id, "seq_pos": r.seq_pos, "aa": r.aa}
        for r in fixed_residues
    ]


def mapping_summary(fixed_residues: list[FixedResidue]) -> str:
    return ", ".join(f"{r.aa}{r.author_num}→{r.chain_id}#{r.seq_pos}" for r in fixed_residues)
