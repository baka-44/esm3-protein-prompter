"""
sequence_utils.py — Sequence formatting, comparison, and diversity utilities.
"""

from __future__ import annotations

import io


def to_fasta(sequences: list[tuple[str, str]], label_prefix: str = "candidate") -> str:
    """
    Convert a list of (header, sequence) tuples to a FASTA-formatted string.

    Args:
        sequences:    List of (header, sequence) tuples.
        label_prefix: Fallback prefix if header is empty.

    Returns:
        Multi-sequence FASTA string.
    """
    lines = []
    for i, (header, seq) in enumerate(sequences, start=1):
        name = header.strip() if header.strip() else f"{label_prefix}_{i}"
        lines.append(f">{name}")
        # Wrap sequence at 80 chars
        for j in range(0, len(seq), 80):
            lines.append(seq[j:j + 80])
    return "\n".join(lines) + "\n"


def sequence_identity(seq_a: str, seq_b: str) -> float:
    """
    Compute pairwise sequence identity between two equal-length sequences.

    Returns a value in [0.0, 1.0].
    If sequences differ in length, aligns by the shorter one (truncated comparison).
    """
    if not seq_a or not seq_b:
        return 0.0
    length = min(len(seq_a), len(seq_b))
    matches = sum(a == b for a, b in zip(seq_a[:length], seq_b[:length]))
    return matches / length


def sequence_identity_to_reference(candidate: str, reference: str) -> float:
    """
    Compute the sequence identity of a candidate relative to a reference.

    Returns a percentage (0–100).
    """
    return sequence_identity(candidate, reference) * 100.0


def mean_pairwise_diversity(sequences: list[str]) -> float:
    """
    Compute mean pairwise sequence diversity (1 - identity) for a list of sequences.

    Returns a value in [0.0, 1.0].
    Higher = more diverse set of candidates.
    """
    if len(sequences) < 2:
        return 0.0

    total, count = 0.0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            total += 1.0 - sequence_identity(sequences[i], sequences[j])
            count += 1

    return total / count if count > 0 else 0.0


def build_masked_sequence(
    protein_length: int,
    fixed_residues: dict[int, str],
) -> str:
    """
    Build an ESM3-compatible masked sequence string.

    Args:
        protein_length:  Total length of the sequence.
        fixed_residues:  Mapping of {0-based position: single-letter AA}.
                         Positions not in this dict are masked with '_'.

    Returns:
        String of length protein_length with known AAs at fixed positions
        and '_' elsewhere.

    Example:
        build_masked_sequence(10, {2: "G", 5: "K"})
        → "__G__K____"
    """
    seq = ["_"] * protein_length
    for pos, aa in fixed_residues.items():
        if 0 <= pos < protein_length:
            seq[pos] = aa.upper()
    return "".join(seq)


def count_masked(sequence: str) -> int:
    """Return the number of masked ('_') positions in a sequence."""
    return sequence.count("_")


def novelty_percent(candidate: str, reference: str) -> float:
    """
    Return the percentage of positions that differ from the reference.
    Higher = more novel / distant from the reference.
    """
    return (1.0 - sequence_identity(candidate, reference)) * 100.0
