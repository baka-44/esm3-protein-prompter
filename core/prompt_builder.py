"""
prompt_builder.py — PromptSpec → ESMProtein.

Translates the structured PromptSpec (output of nl_parser) into an ESMProtein
object that ESM3 can use for conditional generation.
"""

from __future__ import annotations

from pathlib import Path

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
    else:
        # Fully masked — de novo generation
        sequence = build_masked_sequence(spec.protein_length, spec.fixed_positions)

    # Replace any non-standard characters with underscore (ESM3 mask token)
    sequence = "".join(
        c if c.isalpha() or c == "_" else "_"
        for c in sequence.upper()
    )

    # ── 2. Extract backbone coordinates (structure motif) ─────────────────────
    coordinates: np.ndarray | None = None

    if spec.use_structure_motif:
        if pdb_source is None:
            raise ValueError(
                "Structure motif was requested (use_structure_motif=True) but no PDB "
                "file was provided. Please upload a PDB file."
            )
        if not spec.motif_residue_indices:
            raise ValueError(
                "use_structure_motif is True but motif_residue_indices is empty. "
                "Specify which residue positions should have their coordinates fixed."
            )

        coordinates = extract_backbone_coordinates(
            pdb_source=pdb_source,
            protein_length=spec.protein_length,
            motif_residue_indices=spec.motif_residue_indices,
            chain_id=spec.motif_chain_id,
        )

    # ── 3. Build function annotations ─────────────────────────────────────────
    function_annotations = _build_function_annotations(spec.function_keywords)

    # ── 4. Construct ESMProtein ────────────────────────────────────────────────
    protein_kwargs: dict = {"sequence": sequence}

    if coordinates is not None:
        protein_kwargs["coordinates"] = coordinates

    if function_annotations:
        protein_kwargs["function_annotations"] = function_annotations

    protein = _ESMProtein(**protein_kwargs)
    return protein


def _build_function_annotations(keywords: list[str]) -> list | None:
    """
    Convert a list of keyword strings into ESMProtein function_annotations format.

    ESM3 function annotations are InterPro-derived. The SDK accepts them as a
    list of annotation objects. We use the SDK's own annotation type if available,
    otherwise fall back to a plain list of strings which later versions of the
    SDK also accept.
    """
    if not keywords:
        return None

    try:
        from esm.sdk.api import FunctionAnnotation
        return [FunctionAnnotation(label=kw, start=None, end=None) for kw in keywords]
    except (ImportError, AttributeError):
        pass

    # Fallback: some SDK versions accept raw strings
    return keywords


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

    if spec.use_structure_motif and pdb_provided:
        parts.append(
            f"**Structure motif:** {len(spec.motif_residue_indices)} backbone positions "
            f"pinned from uploaded PDB"
        )
    elif spec.use_structure_motif and not pdb_provided:
        parts.append("**Structure motif:** requested but no PDB uploaded — will be skipped")

    parts.append(f"**Candidates:** {spec.num_candidates} | **Temperature:** {spec.generation_temperature}")

    return "\n".join(parts)
