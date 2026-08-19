"""
proteinredesign/config_builders/preset5.py — Preset "Scaffold diversification"
(RF3 partial diffusion; taxonomy #8).

Turns scientist-level inputs — an uploaded PDB, a chain, and three dials
(K backbones · M sequences-per-backbone · diversity) — into a validated RF3
partial-diffusion config plus the downstream MPNN fan-out shape.

Unlike presets #1/#2 this preset **generates new backbones**: RF3 partially
noises the whole chain to a chosen noise level (`partial_t`, in Å) and re-denoises,
producing structurally-diverse-but-related variants of the *same length and fold*.
There are **no fixed residues** here (that is motif scaffolding #3 territory) — the
only structural anchor is the noised input backbone as the diffusion starting point
(see docs/plans/rfdiffusion_mpnn_backend.md, D10).

Decision B5 — **code, not an LLM, builds the RF3 contig syntax.** The scientist never
sees a contig string; this builder emits `"<chain>1-<L>"` from the validated chain +
its residue count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from proteinredesign.config_builders.common import ConfigError, index_chain_residues

__all__ = ["ConfigError", "Preset5Config", "build_preset5_config",
           "PARTIAL_T_MIN", "PARTIAL_T_MAX", "K_MAX", "M_MAX"]

# RF3 partial-diffusion noise magnitude is specified in Å (NOT a timestep count).
# The foundry rfd3 docs recommend 5.0–15.0 Å; we allow a slightly wider floor so
# users can request very tight variants (start small, increase for more diversity).
PARTIAL_T_MIN = 2.0
PARTIAL_T_MAX = 15.0

# Fan-out ceilings (D10.3). K = RF3 backbones, M = MPNN sequences per backbone.
K_MAX = 10
M_MAX = 10


@dataclass
class Preset5Config:
    """Validated RF3 partial-diffusion + MPNN-fan-out spec for scaffold diversification."""

    chain_id: str
    length: int                 # residue count of the chosen chain (contig span)
    contig: str                 # RF3 contig string, e.g. "A1-107" (code-built — B5)
    partial_t: float            # RF3 noise magnitude, Å
    k: int                      # number of RF3 backbones to generate
    m: int                      # MPNN sequences designed per backbone
    warnings: list[str] = field(default_factory=list)

    @property
    def total_designs(self) -> int:
        return self.k * self.m

    def to_params(self) -> dict:
        """Serialisable params for the JobManifest."""
        return {
            "chain_id": self.chain_id,
            "length": self.length,
            "contig": self.contig,
            "partial_t": self.partial_t,
            "k": self.k,
            "m": self.m,
        }


def build_preset5_config(
    pdb_source,
    partial_t: float,
    k: int,
    m: int,
    chain_id: str | None = None,
) -> Preset5Config:
    """
    Build a validated scaffold-diversification config.

    Args:
        pdb_source:  PDB path or raw bytes/string (the backbone to diversify).
        partial_t:   RF3 partial-diffusion noise magnitude in Å (more → more diverse).
        k:           Number of RF3 backbones to generate (1..K_MAX).
        m:           MPNN sequences designed per backbone (1..M_MAX).
        chain_id:    Chain to diversify. None → first chain (warns if multiple).

    Raises:
        ConfigError: no protein chain / unknown chain, or K/M/partial_t out of range.
    """
    if not (PARTIAL_T_MIN <= partial_t <= PARTIAL_T_MAX):
        raise ConfigError(
            f"Diversity (partial_t) must be between {PARTIAL_T_MIN} and {PARTIAL_T_MAX} Å; got {partial_t}."
        )
    if not (1 <= int(k) <= K_MAX):
        raise ConfigError(f"K (backbones) must be between 1 and {K_MAX}; got {k}.")
    if not (1 <= int(m) <= M_MAX):
        raise ConfigError(f"M (sequences per backbone) must be between 1 and {M_MAX}; got {m}.")

    # Resolve the chain and its length (reuses the same chain-indexing as #1/#2 so a
    # numbering/parse fix stays in one place). by_author maps author_num→(seq_pos, aa);
    # the chain's residue count is its number of entries.
    chain_id, by_author, warnings = index_chain_residues(pdb_source, chain_id)
    length = len(by_author)
    if length < 2:
        raise ConfigError(
            f"Chain '{chain_id}' has too few residues ({length}) to diversify."
        )

    # RF3 contig spec — sequential 1-based over the whole chain (B5: code builds this).
    contig = f"{chain_id}1-{length}"

    return Preset5Config(
        chain_id=chain_id,
        length=length,
        contig=contig,
        partial_t=float(partial_t),
        k=int(k),
        m=int(m),
        warnings=warnings,
    )
