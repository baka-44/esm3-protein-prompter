"""
proteinredesign/config_builders/preset6.py — Enzyme active-site scaffolding (RF3 all-atom).

Turns scientist-level inputs — a parent enzyme PDB, its catalytic residues (author
numbering), an optional bound ligand/cofactor, and a target scaffold length — into a
validated RF3 all-atom input spec that scaffolds a NEW protein around the catalytic
site while holding the catalytic geometry.

Mechanism (see docs/plans/rfdiffusion_mpnn_backend.md D6 + borrowed_bodies_composer.md §13):
- The catalytic residues are **unindexed** (`unindex`): RF3 preserves their fixed-atom
  geometry (`select_fixed_atoms`) but is free to place the constellation and choose where
  it lands in the new sequence — the classic de-novo enzyme-scaffolding mode. (This is the
  distinction from the Borrowed-Bodies *indexed* multi-segment case, which preserves an
  absolute composed pose.)
- The ligand (if present) arrives as HETATM in the PDB (decision D8, reused from preset #2):
  we filter to keep only the chosen ligand and expose it to RF3 via `ligand`.

Decision B5 — **code, not an LLM, builds the RF3 syntax** (`unindex`, `select_fixed_atoms`,
`length`). The scientist never writes RF3 config.
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
from utils.pdb_utils import HetGroup, filter_pdb_keep_ligand, get_hetatm_groups

__all__ = ["ConfigError", "Preset6Config", "build_preset6_config",
           "FIXED_ATOM_MODES", "LENGTH_MIN", "LENGTH_MAX", "K_MAX", "M_MAX"]

# RF3 select_fixed_atoms shorthands (docs/rfd3 input.md): TIP = catalytic tip atoms,
# BKBN = N,CA,C,O, ALL = every atom. Default TIP — tightest hold on the chemistry with
# the most scaffolding freedom.
FIXED_ATOM_MODES = ("TIP", "ALL", "BKBN")

LENGTH_MIN = 40      # smallest sensible scaffold
LENGTH_MAX = 400     # L4 VRAM / runtime budget guardrail
K_MAX = 10           # RF3 designs (diffusion_batch_size)
M_MAX = 10           # MPNN sequences per design


@dataclass
class Preset6Config:
    """Validated RF3 all-atom enzyme-scaffolding spec (unindexed catalytic motif)."""

    chain_id: str
    catalytic_residues: list[FixedResidue]
    fixed_atoms_mode: str                       # one of FIXED_ATOM_MODES
    ligand: HetGroup | None
    length_min: int
    length_max: int
    k: int
    m: int
    filtered_pdb_bytes: bytes | None            # protein + ONLY the chosen ligand (None if no ligand)
    mapping_summary: str
    warnings: list[str] = field(default_factory=list)

    @property
    def total_designs(self) -> int:
        return self.k * self.m

    def rf3_unindex(self) -> str:
        """Comma-joined chain+author residue refs, e.g. 'A57,A102,A195' (RF3 `unindex`)."""
        return ",".join(f"{r.chain_id}{r.author_num}" for r in self.catalytic_residues)

    def rf3_select_fixed_atoms(self) -> dict[str, str]:
        """RF3 `select_fixed_atoms`: catalytic residues at `fixed_atoms_mode`; ligand fully fixed."""
        d: dict[str, str] = {
            f"{r.chain_id}{r.author_num}": self.fixed_atoms_mode
            for r in self.catalytic_residues
        }
        if self.ligand is not None:
            d[self.ligand.resname] = "ALL"   # cofactor geometry is part of the active site
        return d

    def to_params(self) -> dict:
        """Serialisable params for the JobManifest (RF3 spec + MPNN/QC handles)."""
        return {
            "chain_id": self.chain_id,
            "catalytic_residues": fixed_residues_to_params(self.catalytic_residues),
            "fixed_atoms_mode": self.fixed_atoms_mode,
            "ligand": (
                {"resname": self.ligand.resname, "chain_id": self.ligand.chain_id,
                 "res_seq": self.ligand.res_seq}
                if self.ligand is not None else None
            ),
            "length": f"{self.length_min}-{self.length_max}",
            "unindex": self.rf3_unindex(),
            "select_fixed_atoms": self.rf3_select_fixed_atoms(),
            "k": self.k,
            "m": self.m,
            "mapping_summary": self.mapping_summary,
        }


def build_preset6_config(
    pdb_source,
    catalytic_residues_str: str,
    *,
    fixed_atoms_mode: str = "TIP",
    ligand_key: tuple[str, str, int] | None = None,
    length_min: int,
    length_max: int,
    k: int,
    m: int,
    chain_id: str | None = None,
) -> Preset6Config:
    """
    Build a validated RF3 all-atom enzyme-scaffolding config.

    Args:
        pdb_source:            Parent enzyme PDB (path / bytes / string).
        catalytic_residues_str: Catalytic residues, PDB author numbering (e.g. "H57, D102, S195").
        fixed_atoms_mode:      Which atoms of each catalytic residue to fix (FIXED_ATOM_MODES).
        ligand_key:            Optional (resname, chain, res_seq) of a bound cofactor to condition on.
        length_min/length_max: Target scaffold length range (RF3 `length`).
        k, m:                  RF3 designs (diffusion_batch_size) and MPNN sequences per design.
        chain_id:              Chain the catalytic residues live on. None → first chain (warns).

    Raises:
        ConfigError: no catalytic residues, bad fixed-atoms mode, residue#↔AA mismatch,
                     unknown ligand, or out-of-range length / K / M.
    """
    if fixed_atoms_mode not in FIXED_ATOM_MODES:
        raise ConfigError(
            f"fixed_atoms_mode must be one of {FIXED_ATOM_MODES}; got '{fixed_atoms_mode}'."
        )
    if not parse_fixed_residue_tokens(catalytic_residues_str):
        raise ConfigError(
            "No catalytic residues provided. List the active-site residues to preserve "
            "(e.g. 'H57, D102, S195')."
        )
    if not (LENGTH_MIN <= int(length_min) <= int(length_max) <= LENGTH_MAX):
        raise ConfigError(
            f"Scaffold length must satisfy {LENGTH_MIN} ≤ min ≤ max ≤ {LENGTH_MAX} "
            f"(got {length_min}-{length_max})."
        )
    if not (1 <= int(k) <= K_MAX):
        raise ConfigError(f"K (designs) must be between 1 and {K_MAX}; got {k}.")
    if not (1 <= int(m) <= M_MAX):
        raise ConfigError(f"M (sequences per design) must be between 1 and {M_MAX}; got {m}.")

    # Validate catalytic residues against the PDB (author→sequential + residue#↔AA), reusing
    # the same guard as presets #1/#2.
    chain_id, catalytic_residues, warnings = resolve_fixed_residues(
        pdb_source, catalytic_residues_str, chain_id
    )
    if length_min < len(catalytic_residues) + 2:
        raise ConfigError(
            f"Scaffold length min ({length_min}) is too small to hold "
            f"{len(catalytic_residues)} catalytic residues plus connecting structure."
        )

    # Optional ligand — same detect/confirm/filter flow as preset #2.
    ligand: HetGroup | None = None
    filtered_pdb: bytes | None = None
    if ligand_key is not None:
        candidates = get_hetatm_groups(pdb_source)
        ligand = next((g for g in candidates if g.key == ligand_key), None)
        if ligand is None:
            if not candidates:
                raise ConfigError(
                    "No ligand-like HETATM groups found in this PDB (waters/ions/additives "
                    "excluded). Run without a ligand if the site has no cofactor."
                )
            raise ConfigError(
                f"Selected ligand {ligand_key} not found. "
                f"Available: {', '.join(g.label() for g in candidates)}."
            )
        filtered_pdb = filter_pdb_keep_ligand(pdb_source, ligand.key)

    return Preset6Config(
        chain_id=chain_id,
        catalytic_residues=catalytic_residues,
        fixed_atoms_mode=fixed_atoms_mode,
        ligand=ligand,
        length_min=int(length_min),
        length_max=int(length_max),
        k=int(k),
        m=int(m),
        filtered_pdb_bytes=filtered_pdb,
        mapping_summary=mapping_summary(catalytic_residues),
        warnings=warnings,
    )
