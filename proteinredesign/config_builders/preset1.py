"""
proteinredesign/config_builders/preset1.py — Preset #1 (fixed-backbone redesign).

Turns scientist-level inputs — an uploaded PDB plus a list of residues to keep
fixed, given in the PDB's own (author) numbering, e.g. "K67, R82" or "67, 82" —
into a validated ProteinMPNN `fixed_positions` spec.

Two things this builder is responsible for (decision B5), implemented in
config_builders/common.py and shared with preset #2:
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

from dataclasses import dataclass, field

from proteinredesign.config_builders.common import (
    ConfigError,
    FixedResidue,
    fixed_residues_to_params,
    mapping_summary,
    parse_fixed_residue_tokens,
    resolve_fixed_residues,
)

__all__ = ["ConfigError", "FixedResidue", "Preset1Config", "build_preset1_config",
           "parse_fixed_residue_tokens"]


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
            "fixed_residues": fixed_residues_to_params(self.fixed_residues),
            "mapping_summary": self.mapping_summary,
        }


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
    # Check parsed tokens, not just raw-string emptiness — a string of only
    # commas/whitespace (e.g. " , , ") parses to zero tokens too.
    if not parse_fixed_residue_tokens(fixed_residues_str):
        raise ConfigError(
            "No fixed residues provided. List at least one residue to keep fixed "
            "(the rest of the sequence is redesigned)."
        )

    chain_id, fixed_residues, warnings = resolve_fixed_residues(
        pdb_source, fixed_residues_str, chain_id
    )

    # ProteinMPNN fixed_positions: {chain: sorted list of 1-based sequential positions}
    positions = sorted(r.seq_pos for r in fixed_residues)
    fixed_positions = {chain_id: positions}

    return Preset1Config(
        chain_id=chain_id,
        fixed_positions=fixed_positions,
        fixed_residues=fixed_residues,
        mapping_summary=mapping_summary(fixed_residues),
        warnings=warnings,
    )
