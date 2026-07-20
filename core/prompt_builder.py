"""
prompt_builder.py — PromptSpec → ESMProtein.

Translates the structured PromptSpec (output of nl_parser) into an ESMProtein
object that ESM3 can use for conditional generation.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

import numpy as np

from core.nl_parser import PromptSpec
from utils.pdb_utils import extract_backbone_coordinates
from utils.sequence_utils import build_masked_sequence


def build_esm_protein(
    spec: PromptSpec,
    pdb_source: str | bytes | Path | None = None,
) -> "ESMProtein":
    """
    Build an ESMProtein from a PromptSpec, optionally incorporating backbone
    coordinates from an uploaded PDB file for structural motif constraints.

    Args:
        spec:       Parsed PromptSpec from the NL parser.
        pdb_source: Path to a PDB file, or raw PDB bytes/string content.
                    Required if spec.use_structure_motif is True.

    Returns:
        ESMProtein object ready for ESM3 generation.

    Raises:
        ImportError:  If the esm package is not installed.
        ValueError:   If structure motif is requested but no PDB is provided,
                      or if PDB extraction fails.
    """
    try:
        from esm.sdk.api import ESMProtein as _ESMProtein
    except ImportError as e:
        raise ImportError(
            "ESM SDK not found. Install with: pip install esm"
        ) from e

    # ── 1. Build masked sequence ───────────────────────────────────────────────
    if spec.sequence_template:
        sequence = spec.sequence_template
        # Ensure correct length
        if len(sequence) != spec.protein_length:
            sequence = sequence.ljust(spec.protein_length, "_")[:spec.protein_length]

        # If the template has no mask characters (e.g. haiku provided the full
        # reference sequence), derive the mask from fixed_positions: keep only
        # those positions and mask everything else so ESM3 has something to generate.
        if "_" not in sequence:
            if spec.fixed_positions:
                seq_list = ["_"] * len(sequence)
                for pos, aa in spec.fixed_positions.items():
                    if 0 <= pos < len(seq_list):
                        seq_list[pos] = aa.upper()
                sequence = "".join(seq_list)
                _log(f"INFO: sequence_template had no masks — derived mask from "
                      f"{len(spec.fixed_positions)} fixed_positions.")
            else:
                # No fixed positions either → fully de novo at the given length
                sequence = "_" * len(sequence)
                _log("INFO: sequence_template had no masks and no fixed_positions "
                      "— using fully masked (de novo) generation.")
    else:
        # Fully masked — de novo generation
        sequence = build_masked_sequence(spec.protein_length, spec.fixed_positions)

    # Replace any non-standard characters with underscore (ESM3 mask token)
    sequence = "".join(
        c if c.isalpha() or c == "_" else "_"
        for c in sequence.upper()
    )

    # ── Minimum mask enforcement ────────────────────────────────────────────
    # When the user provides a sequence for redesign but Haiku was too
    # conservative (e.g. only 1 masked position out of 207), ESM3 can't
    # generate meaningful diversity. Expand the mask to at least 15% of
    # positions, choosing non-fixed positions randomly.
    MIN_MASK_FRACTION = 0.15
    mask_count = sequence.count("_")
    total_len = len(sequence)
    if 0 < mask_count < int(total_len * MIN_MASK_FRACTION) and total_len > 30:
        import random
        fixed_set = set(spec.fixed_positions.keys()) if spec.fixed_positions else set()
        maskable = [i for i, c in enumerate(sequence) if c != "_" and i not in fixed_set]
        needed = int(total_len * MIN_MASK_FRACTION) - mask_count
        if needed > 0 and maskable:
            to_mask = random.sample(maskable, min(needed, len(maskable)))
            seq_list = list(sequence)
            for idx in to_mask:
                seq_list[idx] = "_"
            sequence = "".join(seq_list)
            _log(f"INFO: expanded mask from {mask_count} to {sequence.count('_')}/{total_len} "
                 f"(min {MIN_MASK_FRACTION:.0%} threshold)")

    # Final safety check: if still no masks, ESM3 has nothing to generate.
    if "_" not in sequence:
        _log("WARNING: sequence has no mask tokens after processing — "
              "masking all positions to allow generation.")
        sequence = "_" * len(sequence)

    # ── 2a. Full structure conditioning (PDB provided, no explicit motif pinning) ──
    # When a PDB is uploaded and the user hasn't requested specific motif-index
    # pinning, pass backbone coordinates for ALL residues. This gives ESM3
    # complete structural context at every masked position (inverse-folding mode),
    # preventing the K/R bias that arises from sequence-only generation when many
    # K/R residues are fixed.
    import torch
    full_coords: "torch.Tensor | None" = None
    if pdb_source is not None and not spec.use_structure_motif:
        try:
            from utils.pdb_utils import extract_full_backbone
            pdb_seq, full_coords_np = extract_full_backbone(
                pdb_source, chain_id=spec.motif_chain_id
            )
            if len(pdb_seq) != spec.protein_length:
                _log(
                    f"INFO: PDB length ({len(pdb_seq)}) ≠ protein_length "
                    f"({spec.protein_length}) — skipping full structure conditioning"
                )
                full_coords = None
            else:
                full_coords = torch.tensor(full_coords_np, dtype=torch.float32)
                _log(
                    f"INFO: full structure conditioning enabled "
                    f"({spec.protein_length} residues)"
                )
        except Exception as e:
            _log(f"WARNING: full backbone extraction failed ({e}) — no structure track")
            full_coords = None

    # ── 2b. Partial motif pinning (explicit motif_residue_indices requested) ───
    coordinates: np.ndarray | None = None

    if spec.use_structure_motif:
        if pdb_source is None:
            _log("INFO: use_structure_motif=True but no PDB provided — skipping structure track.")
            spec.use_structure_motif = False
        elif not spec.motif_residue_indices:
            _log("INFO: use_structure_motif=True but motif_residue_indices is empty — skipping structure track.")
            spec.use_structure_motif = False
        elif spec.motif_source_indices:
            # Scaffold condensation path: source positions in PDB ≠ target positions in new protein.
            # Extract backbone coords from original PDB positions, place at remapped target positions.
            try:
                from utils.pdb_utils import extract_motif_by_source_indices
                coordinates = extract_motif_by_source_indices(
                    pdb_source=pdb_source,
                    target_length=spec.protein_length,
                    source_indices=spec.motif_source_indices,
                    target_indices=spec.motif_residue_indices,
                    chain_id=spec.motif_chain_id,
                )
                _log(
                    f"INFO: condensation structure motif — extracted {len(spec.motif_source_indices)} "
                    f"residues from PDB at {spec.motif_source_indices}, "
                    f"placed at {spec.motif_residue_indices} in {spec.protein_length}-aa target"
                )
            except Exception as e:
                _log(f"WARNING: condensation coordinate extraction failed ({e}) — no structure track")
                coordinates = None
        else:
            coordinates = extract_backbone_coordinates(
                pdb_source=pdb_source,
                protein_length=spec.protein_length,
                motif_residue_indices=spec.motif_residue_indices,
                chain_id=spec.motif_chain_id,
            )

    # ── 3. Build function annotations ─────────────────────────────────────────
    function_annotations = _build_function_annotations(
        spec.function_keywords, protein_length=spec.protein_length
    )

    # ── 4. Construct ESMProtein ────────────────────────────────────────────────
    protein_kwargs: dict = {"sequence": sequence}

    # Full backbone takes priority over partial motif coordinates.
    # Both are converted to torch.Tensor (ESMProtein requires Tensor, not ndarray).
    if full_coords is not None:
        protein_kwargs["coordinates"] = full_coords
    elif coordinates is not None:
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=torch.float32)
        protein_kwargs["coordinates"] = coordinates

    if function_annotations:
        protein_kwargs["function_annotations"] = function_annotations

    protein = _ESMProtein(**protein_kwargs)
    return protein


def _build_function_annotations(keywords: list[str], protein_length: int = 1) -> list | None:
    """
    Convert a list of keyword strings into ESMProtein function_annotations format.

    ESM3 function annotations are InterPro-derived. The SDK accepts them as a
    list of FunctionAnnotation objects with integer start/end positions (1-based).
    Passing start=None or end=None causes a TypeError inside the SDK's bounds check,
    so we always supply concrete positions spanning the full protein.
    """
    if not keywords:
        return None

    # Clamp to valid range — annotations span the entire protein
    start = 1
    end = max(1, protein_length)

    try:
        from esm.sdk.api import FunctionAnnotation
    except (ImportError, AttributeError):
        return keywords  # Fallback: some SDK versions accept raw strings

    # Try each keyword individually — skip ones not in ESM3's InterPro vocabulary
    valid = []
    for kw in keywords:
        try:
            ann = FunctionAnnotation(label=kw, start=start, end=end)
            # Validate by attempting a dry-run encode if possible
            valid.append(ann)
        except Exception as e:
            _log(f"INFO: Skipping function keyword '{kw}' (not in ESM3 vocabulary: {e})")

    return valid if valid else None


def describe_prompt(spec: PromptSpec, pdb_provided: bool = False) -> str:
    """
    Return a human-readable summary of the ESM3 prompt that will be constructed.
    Shown to the scientist before generation begins.
    """
    parts = [f"**Protein length:** {spec.protein_length} residues"]

    masked_count = spec.sequence_template.count("_") if spec.sequence_template else spec.protein_length
    fixed_count = len(spec.fixed_positions)
    parts.append(f"**Fixed residues:** {fixed_count} | **Masked (to generate):** {masked_count}")

    if spec.function_keywords:
        parts.append(f"**Function keywords:** {', '.join(spec.function_keywords)}")

    if spec.motif_source_indices and pdb_provided:
        parts.append(
            f"**Scaffold condensation:** {len(spec.motif_source_indices)} key residues anchored "
            f"from PDB at remapped positions in {spec.protein_length}-residue target"
        )
    elif spec.use_structure_motif and pdb_provided:
        parts.append(
            f"**Structure motif:** {len(spec.motif_residue_indices)} backbone positions "
            f"pinned from uploaded PDB"
        )
    elif spec.use_structure_motif and not pdb_provided:
        parts.append("**Structure motif:** requested but no PDB uploaded — will be skipped")

    parts.append(f"**Candidates:** {spec.num_candidates} | **Temperature:** {spec.generation_temperature}")

    return "\n".join(parts)
