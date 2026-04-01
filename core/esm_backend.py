"""
esm_backend.py — ESM3 inference wrapper.

Provides a unified interface for generating protein candidates using either:
  - EvolutionaryScale Forge API (hosted, higher-quality models up to 98B)
  - Local ESM3-open (1.4B, runs on GPU; Colab Pro A100 recommended)

Both backends expose the same `.generate()` interface via the ESM SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.nl_parser import PromptSpec


@dataclass
class GenerationResult:
    """Raw output from a single ESM3 generation call, before post-processing."""
    esm_protein: object  # ESMProtein instance
    index: int           # 0-based candidate index


def generate_candidates(
    esm_protein: object,
    spec: "PromptSpec",
    client: object | None = None,
    progress_callback=None,
) -> list[GenerationResult]:
    """
    Generate N candidate proteins by running ESM3 inference N times.

    Args:
        esm_protein:        The ESMProtein prompt object (from prompt_builder).
        spec:               The PromptSpec (used for num_candidates, temperature, steps).
        client:             ESM3 client (Forge or local). If None, loads from config.
        progress_callback:  Optional callable(current: int, total: int) for progress updates.

    Returns:
        List of GenerationResult objects, one per candidate.

    Raises:
        RuntimeError: If ESM SDK is not installed or inference fails.
    """
    try:
        from esm.sdk.api import GenerationConfig
    except ImportError as e:
        raise RuntimeError(
            "ESM SDK not found. Install with: pip install esm"
        ) from e

    if client is None:
        from config import get_esm_client
        client = get_esm_client()

    gen_config = GenerationConfig(
        track="sequence",
        num_steps=spec.num_steps,
        temperature=spec.generation_temperature,
    )

    results: list[GenerationResult] = []

    for i in range(spec.num_candidates):
        if progress_callback:
            progress_callback(i, spec.num_candidates)

        try:
            generated = client.generate(esm_protein, gen_config)
            results.append(GenerationResult(esm_protein=generated, index=i))
        except Exception as e:
            # If the error is due to invalid FunctionAnnotation labels,
            # retry without function annotations — sequence/structure constraints
            # are preserved, only keyword guidance is dropped.
            if "FunctionAnnotation" in str(e) or "Unknown label" in str(e):
                print(f"INFO: Retrying candidate {i + 1} without function annotations...")
                try:
                    import copy
                    ep_no_func = copy.copy(esm_protein)
                    ep_no_func.function_annotations = None
                    generated = client.generate(ep_no_func, gen_config)
                    results.append(GenerationResult(esm_protein=generated, index=i))
                except Exception as e2:
                    print(f"WARNING: Candidate {i + 1} failed on retry: {e2}")
            else:
                print(f"WARNING: Candidate {i + 1} failed: {e}")

    if progress_callback:
        progress_callback(spec.num_candidates, spec.num_candidates)

    return results


def generate_with_structure(
    esm_protein: object,
    spec: "PromptSpec",
    client: object | None = None,
    progress_callback=None,
) -> list[GenerationResult]:
    """
    Two-stage generation: first generate structure tokens, then sequence.

    Used when structure motifs are provided — generates a backbone first that
    respects the coordinate constraints, then generates the sequence on top.

    Args: same as generate_candidates.
    Returns: List of GenerationResult objects.
    """
    try:
        from esm.sdk.api import GenerationConfig
    except ImportError as e:
        raise RuntimeError("ESM SDK not found. Install with: pip install esm") from e

    if client is None:
        from config import get_esm_client
        client = get_esm_client()

    structure_config = GenerationConfig(
        track="structure",
        num_steps=spec.num_steps,
        temperature=max(0.3, spec.generation_temperature - 0.2),  # slightly more conservative for structure
    )
    sequence_config = GenerationConfig(
        track="sequence",
        num_steps=spec.num_steps,
        temperature=spec.generation_temperature,
    )

    results: list[GenerationResult] = []
    total_steps = spec.num_candidates * 2  # structure + sequence per candidate

    for i in range(spec.num_candidates):
        if progress_callback:
            progress_callback(i * 2, total_steps)

        try:
            # Stage 1: generate structure tokens respecting coordinate constraints
            with_structure = client.generate(esm_protein, structure_config)

            if progress_callback:
                progress_callback(i * 2 + 1, total_steps)

            # Stage 2: generate sequence conditioned on the new structure
            with_sequence = client.generate(with_structure, sequence_config)
            results.append(GenerationResult(esm_protein=with_sequence, index=i))

        except Exception as e:
            print(f"WARNING: Candidate {i + 1} (structure+sequence) failed: {e}")
            continue

    if progress_callback:
        progress_callback(total_steps, total_steps)

    return results


def choose_generation_strategy(spec: "PromptSpec"):
    """
    Return the appropriate generation function based on the PromptSpec.

    If structure motif is used → two-stage generation.
    Otherwise → sequence-only generation.
    """
    if spec.use_structure_motif and spec.motif_residue_indices:
        return generate_with_structure
    return generate_candidates
