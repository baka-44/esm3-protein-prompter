"""
proteinredesign/config_builders/preset2.py — Preset #2 (ligand-aware redesign).

Turns a scientist-level input — an uploaded protein+ligand complex PDB (from
UniProt/X-ray) and a chosen HETATM group — into a validated LigandMPNN config.

Unlike preset #1, fixed residues are OPTIONAL here: the primary conditioning
signal is the ligand's atomic context (LigandMPNN sees it automatically once
present in the PDB), not a list of pinned residues. Scientists may still pin
specific residues (e.g. known catalytic positions) on top of that.

Per decision D8 (B6): the ligand arrives as HETATM records already in the
uploaded complex PDB (real coordinates, zero extra scientist effort — no
SMILES/separate-file input). The config builder must strip every *other*
HETATM group (waters, ions, crystallization buffer) before the job runs, so
LigandMPNN's automatic ligand-context parsing only sees the intended ligand —
see utils.pdb_utils.filter_pdb_keep_ligand().

Code, not an LLM, builds this config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from proteinredesign.config_builders.common import (
    ConfigError,
    FixedResidue,
    fixed_residues_to_params,
    mapping_summary,
    resolve_fixed_residues,
)
from utils.pdb_utils import HetGroup, filter_pdb_keep_ligand, get_hetatm_groups

__all__ = ["ConfigError", "Preset2Config", "build_preset2_config"]


@dataclass
class Preset2Config:
    """Validated LigandMPNN config for preset #2."""

    chain_id: str
    ligand: HetGroup
    fixed_positions: dict[str, list[int]]      # {chain: [seq_pos, ...]} — may be empty
    fixed_residues: list[FixedResidue]
    mapping_summary: str                        # "" if no fixed residues given
    filtered_pdb_bytes: bytes                    # protein + ONLY the chosen ligand's HETATM
    warnings: list[str] = field(default_factory=list)

    def to_params(self) -> dict:
        """Serialisable params for the JobManifest."""
        return {
            "chain_id": self.chain_id,
            "ligand": {
                "resname": self.ligand.resname,
                "chain_id": self.ligand.chain_id,
                "res_seq": self.ligand.res_seq,
            },
            "fixed_positions": self.fixed_positions,
            "fixed_residues": fixed_residues_to_params(self.fixed_residues),
            "mapping_summary": self.mapping_summary,
        }


def build_preset2_config(
    pdb_source,
    ligand_key: tuple[str, str, int],
    fixed_residues_str: str = "",
    chain_id: str | None = None,
) -> Preset2Config:
    """
    Build a validated LigandMPNN config for preset #2.

    Args:
        pdb_source:          PDB path or raw bytes/string (the protein+ligand complex).
        ligand_key:          (resname, chain_id, res_seq) identifying the ligand HETATM
                              group to condition on — from get_hetatm_groups().
        fixed_residues_str:  Optional residues to keep fixed, PDB author numbering.
                              Empty is valid — ligand context alone drives design.
        chain_id:            Protein chain to redesign. None → first chain (warns if multiple).

    Raises:
        ConfigError: the chosen ligand isn't found among the PDB's HETATM groups, or
                     (if fixed residues given) unknown residue number / residue#↔AA mismatch.
    """
    candidates = get_hetatm_groups(pdb_source)
    match = next((g for g in candidates if g.key == ligand_key), None)
    if match is None:
        if not candidates:
            raise ConfigError(
                "No ligand-like HETATM groups found in this PDB (after excluding waters, "
                "ions, and common crystallization additives). Use Fixed-backbone redesign "
                "if there's no ligand to condition on."
            )
        raise ConfigError(
            f"Selected ligand {ligand_key} was not found in the PDB. "
            f"Available: {', '.join(g.label() for g in candidates)}."
        )

    if fixed_residues_str.strip():
        chain_id, fixed_residues, warnings = resolve_fixed_residues(
            pdb_source, fixed_residues_str, chain_id
        )
        fixed_positions = {chain_id: sorted(r.seq_pos for r in fixed_residues)}
    else:
        # No fixed residues — still need to resolve which protein chain to redesign.
        from proteinredesign.config_builders.common import index_chain_residues
        chain_id, _, warnings = index_chain_residues(pdb_source, chain_id)
        fixed_residues = []
        fixed_positions = {}

    filtered_pdb = filter_pdb_keep_ligand(pdb_source, match.key)

    return Preset2Config(
        chain_id=chain_id,
        ligand=match,
        fixed_positions=fixed_positions,
        fixed_residues=fixed_residues,
        mapping_summary=mapping_summary(fixed_residues),
        filtered_pdb_bytes=filtered_pdb,
        warnings=warnings,
    )
