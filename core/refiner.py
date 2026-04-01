"""
refiner.py — Iterative chain-of-thought refinement of protein candidates.

Takes an existing CandidateResult (from a previous ESM3 generation round)
and constructs a new, tighter PromptSpec for the next generation round by:

  1. Fixing high-confidence residues (pLDDT > threshold) from the candidate.
  2. Keeping the original user constraints (active site, function keywords).
  3. Layering in optional new constraints: SS8 hints, SASA bias, extra keywords.
  4. Optionally compressing the protein (condensing scaffold while retaining key sites).

The resulting PromptSpec is passed to prompt_builder → esm_backend for the
next generation round, implementing the chain-of-thought protocol from the
ESM3 paper (Section "Generating a new fluorescent protein").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.nl_parser import PromptSpec
    from core.result_processor import CandidateResult


# ── RefineOptions ──────────────────────────────────────────────────────────────

@dataclass
class RefineOptions:
    """
    User-specified options controlling how a candidate is refined.
    All fields are optional — omitting them keeps the previous round's settings.
    """

    plddt_fix_threshold: float = 70.0
    """
    Residues with mean pLDDT >= this threshold are fixed in the new template.
    Lower = more positions masked (more exploratory).
    Higher = fewer positions masked (more conservative refinement).
    Range: 0–100. Default 70 = fix confidently-predicted residues.
    """

    extra_keywords: list[str] = field(default_factory=list)
    """
    Additional InterPro-style function keywords to add to this round's prompt.
    E.g. ["high fluorescence intensity", "thermostable"] to push toward better variants.
    """

    ss8_hint: str | None = None
    """
    Free-text secondary structure instruction.
    E.g. "replace the loop around position 50 with an alpha helix"
         "increase beta-strand content in the N-terminal domain"
    Passed to Claude for interpretation → translated to SS8 tokens if possible.
    """

    sasa_bias: str | None = None
    """
    Desired surface accessibility bias.
    One of: "more buried core", "more exposed surface", "amphipathic", None (no change).
    Translated to SASA-related function keywords.
    """

    condense: bool = False
    """Whether to reduce the protein length (scaffold compression)."""

    condense_target_length: int | None = None
    """
    Absolute target length (in residues) after condensation.
    If None and condense=True, defaults to 80% of current length.
    """

    free_text: str = ""
    """
    Any additional free-text instruction from the scientist.
    Interpreted by Claude as part of the refinement prompt.
    """


# ── SASA bias → keyword mapping ────────────────────────────────────────────────

_SASA_KEYWORDS: dict[str, list[str]] = {
    "more buried core":    ["hydrophobic core packing", "buried hydrophobic residues"],
    "more exposed surface":["solvent-exposed surface", "increased hydrophilicity"],
    "amphipathic":         ["amphipathic helix", "amphipathic structure"],
}


# ── Main refinement function ───────────────────────────────────────────────────

def build_refinement_spec(
    candidate: "CandidateResult",
    original_spec: "PromptSpec",
    options: RefineOptions,
    anthropic_client=None,
) -> "PromptSpec":
    """
    Build a new PromptSpec for the next refinement round.

    Strategy:
      - Fix residues where pLDDT >= threshold (keep what ESM3 was confident about).
      - Mask residues where pLDDT < threshold (regenerate uncertain regions).
      - Preserve all original fixed positions (active site, motifs).
      - Apply condensation if requested (shorter scaffold, same key sites).
      - Layer in new SS8/SASA/keyword constraints.

    Args:
        candidate:        The CandidateResult to refine from.
        original_spec:    The PromptSpec used to generate this candidate.
        options:          User-specified refinement controls.
        anthropic_client: Optional Anthropic client (used if SS8 hint needs parsing).

    Returns:
        A new PromptSpec for the next generation round.
    """
    from core.nl_parser import PromptSpec

    seq = candidate.sequence
    if not seq:
        raise ValueError("Candidate has no sequence to refine from.")

    # ── 1. Determine protein length ────────────────────────────────────────────
    if options.condense and options.condense_target_length:
        target_length = max(20, options.condense_target_length)
    elif options.condense:
        target_length = max(20, int(len(seq) * 0.80))
    else:
        target_length = len(seq)

    condensing = (target_length < len(seq))

    # ── 2. Build fixed positions from pLDDT confidence ─────────────────────────
    fixed_from_plddt = _fixed_positions_from_plddt(
        sequence=seq,
        plddt=candidate.plddt_per_residue,
        threshold=options.plddt_fix_threshold,
        target_length=target_length,
        original_length=len(seq),
    )

    # ── 3. Always preserve original fixed positions (active site etc.) ─────────
    original_fixed = _remap_positions(
        positions=original_spec.fixed_positions,
        source_length=original_spec.protein_length,
        target_length=target_length,
    )

    # Merge: original fixed positions take precedence
    merged_fixed: dict[int, str] = {**fixed_from_plddt, **original_fixed}

    # ── 4. Build sequence template ─────────────────────────────────────────────
    if condensing:
        # For condensed proteins, start from a fully masked template of target length
        # and place only the preserved key positions
        seq_template = _build_condensed_template(
            source_sequence=seq,
            source_length=len(seq),
            target_length=target_length,
            fixed_positions=merged_fixed,
        )
    else:
        # Same length: build template from candidate sequence, masking low-pLDDT regions
        template_list = list(seq[:target_length].ljust(target_length, "_"))
        for pos in range(target_length):
            if pos not in merged_fixed:
                template_list[pos] = "_"
            else:
                template_list[pos] = merged_fixed[pos]
        seq_template = "".join(template_list)

    # ── 5. Accumulate function keywords ────────────────────────────────────────
    keywords = list(original_spec.function_keywords)

    # Add SASA bias keywords
    if options.sasa_bias and options.sasa_bias in _SASA_KEYWORDS:
        keywords.extend(_SASA_KEYWORDS[options.sasa_bias])

    # Add compression keywords
    if condensing:
        keywords.extend(["compact protein", "minimal scaffold"])

    # Add user extra keywords
    keywords.extend(options.extra_keywords)

    # Deduplicate while preserving order
    seen: set[str] = set()
    keywords = [k for k in keywords if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]
    keywords = keywords[:8]  # ESM3 function track limit

    # ── 6. Remap structure motif indices if condensing ─────────────────────────
    motif_indices = original_spec.motif_residue_indices
    use_motif = original_spec.use_structure_motif
    if condensing and motif_indices:
        motif_indices = _remap_list(
            positions=motif_indices,
            source_length=original_spec.protein_length,
            target_length=target_length,
        )

    # ── 7. Parse SS8 hint via Claude (if provided) ─────────────────────────────
    ss8_keywords: list[str] = []
    if options.ss8_hint:
        ss8_keywords = _parse_ss8_hint_to_keywords(options.ss8_hint)
        keywords = list(dict.fromkeys(keywords + ss8_keywords))[:8]

    # ── 8. Compose notes_to_user ───────────────────────────────────────────────
    fixed_pct = len(merged_fixed) / target_length * 100 if target_length else 0
    masked_count = seq_template.count("_")

    notes = (
        f"Refining from candidate #{candidate.rank} "
        f"(pTM={candidate.ptm:.3f}, pLDDT={candidate.mean_plddt:.1f}). "
        f"Fixed {len(merged_fixed)} residues ({fixed_pct:.0f}%) with pLDDT ≥ {options.plddt_fix_threshold:.0f}. "
        f"Regenerating {masked_count} masked positions."
    )
    if condensing:
        notes += f" Condensing from {len(seq)} → {target_length} residues."
    if options.ss8_hint:
        notes += f" SS8 hint: '{options.ss8_hint}'."
    if options.sasa_bias:
        notes += f" SASA bias: {options.sasa_bias}."
    if options.free_text:
        notes += f" Additional instruction: {options.free_text}."

    # ── 9. Assemble new PromptSpec ─────────────────────────────────────────────
    from core.nl_parser import PromptSpec

    return PromptSpec(
        protein_length=target_length,
        sequence_template=seq_template,
        fixed_positions=merged_fixed,
        function_keywords=keywords,
        use_structure_motif=use_motif,
        motif_residue_indices=motif_indices,
        motif_chain_id=original_spec.motif_chain_id,
        num_candidates=original_spec.num_candidates,
        generation_temperature=max(0.1, original_spec.generation_temperature - 0.1),
        # Slightly lower temperature each round — more focused search
        num_steps=original_spec.num_steps,
        notes_to_user=notes,
    )


# ── Helper functions ───────────────────────────────────────────────────────────

def _fixed_positions_from_plddt(
    sequence: str,
    plddt: list[float],
    threshold: float,
    target_length: int,
    original_length: int,
) -> dict[int, str]:
    """
    Return {position: aa} for residues where pLDDT >= threshold.
    If pLDDT is empty, fixes nothing (fully re-generates).
    Positions are mapped to target_length space if condensing.
    """
    if not plddt or not sequence:
        return {}

    # Normalise pLDDT to [0, 100] if in [0, 1]
    plddt_arr = list(plddt)
    if plddt_arr and max(plddt_arr) <= 1.0:
        plddt_arr = [v * 100.0 for v in plddt_arr]

    # Pad or trim to sequence length
    L = len(sequence)
    if len(plddt_arr) < L:
        plddt_arr = plddt_arr + [0.0] * (L - len(plddt_arr))

    fixed: dict[int, str] = {}
    for i, (aa, score) in enumerate(zip(sequence, plddt_arr)):
        if score >= threshold:
            mapped_pos = _map_position(i, original_length, target_length)
            if mapped_pos is not None and mapped_pos < target_length:
                fixed[mapped_pos] = aa

    return fixed


def _remap_positions(
    positions: dict[int, str],
    source_length: int,
    target_length: int,
) -> dict[int, str]:
    """Remap a dict of {position: aa} from source_length space to target_length space."""
    if source_length == target_length:
        return dict(positions)
    remapped: dict[int, str] = {}
    for pos, aa in positions.items():
        new_pos = _map_position(pos, source_length, target_length)
        if new_pos is not None and new_pos < target_length:
            remapped[new_pos] = aa
    return remapped


def _remap_list(
    positions: list[int],
    source_length: int,
    target_length: int,
) -> list[int]:
    """Remap a list of positions from source_length space to target_length space."""
    if source_length == target_length:
        return list(positions)
    result = []
    for p in positions:
        new_p = _map_position(p, source_length, target_length)
        if new_p is not None and new_p < target_length:
            result.append(new_p)
    return sorted(set(result))


def _map_position(
    pos: int,
    source_length: int,
    target_length: int,
) -> int | None:
    """Proportionally map a position from source_length to target_length space."""
    if source_length <= 0 or target_length <= 0:
        return None
    if source_length == target_length:
        return pos
    mapped = round(pos * (target_length - 1) / (source_length - 1)) if source_length > 1 else 0
    return min(mapped, target_length - 1)


def _build_condensed_template(
    source_sequence: str,
    source_length: int,
    target_length: int,
    fixed_positions: dict[int, str],
) -> str:
    """
    Build a masked template of target_length, placing fixed AAs at their
    mapped positions and '_' everywhere else.
    """
    template = ["_"] * target_length
    for pos, aa in fixed_positions.items():
        if 0 <= pos < target_length:
            template[pos] = aa
    return "".join(template)


def _parse_ss8_hint_to_keywords(hint: str) -> list[str]:
    """
    Convert a free-text SS8 hint to function keywords ESM3 understands.
    Simple heuristic mapping — Claude could be used for richer parsing.
    """
    hint_lower = hint.lower()
    keywords: list[str] = []

    if any(w in hint_lower for w in ["helix", "helical", "alpha-helix", "alpha helix"]):
        keywords.append("alpha-helical")
    if any(w in hint_lower for w in ["strand", "beta", "sheet", "barrel"]):
        keywords.append("beta sheet")
    if any(w in hint_lower for w in ["loop", "flexible", "disordered"]):
        keywords.append("flexible loop region")
    if any(w in hint_lower for w in ["coil", "random coil"]):
        keywords.append("random coil")
    if any(w in hint_lower for w in ["turn", "hairpin"]):
        keywords.append("beta turn")
    if any(w in hint_lower for w in ["rigid", "ordered", "structured"]):
        keywords.append("well-ordered structure")

    return keywords


def describe_refinement(options: RefineOptions, round_num: int) -> str:
    """Return a human-readable description of what this refinement round will do."""
    parts = [f"**Round {round_num} refinement**"]

    parts.append(
        f"Fixing residues with pLDDT ≥ {options.plddt_fix_threshold:.0f} "
        f"(confident positions preserved, uncertain regions regenerated)"
    )

    if options.extra_keywords:
        parts.append(f"New keywords: {', '.join(options.extra_keywords)}")
    if options.ss8_hint:
        parts.append(f"SS8 hint: _{options.ss8_hint}_")
    if options.sasa_bias:
        parts.append(f"SASA bias: {options.sasa_bias}")
    if options.condense:
        if options.condense_target_length:
            parts.append(f"Condensing to {options.condense_target_length} residues")
        else:
            parts.append("Condensing scaffold by ~20%")
    if options.free_text:
        parts.append(f"Additional: _{options.free_text}_")

    return "\n".join(f"- {p}" if i > 0 else p for i, p in enumerate(parts))
