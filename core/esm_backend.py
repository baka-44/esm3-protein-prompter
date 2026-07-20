"""
esm_backend.py — ESM3 inference wrapper.

Provides a unified interface for generating protein candidates using either:
  - EvolutionaryScale Forge API (hosted, higher-quality models up to 98B)
  - Local ESM3-open (1.4B, runs on GPU; Colab Pro A100 recommended)

Both backends expose the same `.generate()` interface via the ESM SDK.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

if TYPE_CHECKING:
    from core.nl_parser import PromptSpec


@dataclass
class GenerationResult:
    """Raw output from a single ESM3 generation call, before post-processing."""
    esm_protein: object  # ESMProtein instance
    index: int           # 0-based candidate index


def _is_protein_error(obj) -> bool:
    """Return True if the ESM SDK returned an ESMProteinError instead of ESMProtein."""
    return type(obj).__name__ == "ESMProteinError"


def _extract_error_msg(obj) -> str:
    """Pull the human-readable message out of an ESMProteinError object."""
    # Try common attribute names used across ESM SDK versions
    for attr in ("error_msg", "message", "msg", "detail", "description"):
        val = getattr(obj, attr, None)
        if val:
            return str(val)
    # Fallback: dump all non-dunder attrs
    try:
        parts = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return str(parts)
    except Exception:
        return str(obj)


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

    import copy

    # Strip function annotations from a clean copy for retry use
    def _make_no_func_protein(ep):
        ep2 = copy.copy(ep)
        ep2.function_annotations = None
        return ep2

    def _seq_is_empty(protein_obj) -> bool:
        """Return True if the generated sequence is None, empty, or all mask tokens."""
        seq = getattr(protein_obj, "sequence", None)
        if seq is None:
            return True
        if isinstance(seq, str):
            return not seq.replace("_", "").strip()
        # tensor or other type — treat as non-empty (let _get_sequence handle it)
        return False

    for i in range(spec.num_candidates):
        if progress_callback:
            progress_callback(i, spec.num_candidates)

        try:
            generated = client.generate(esm_protein, gen_config)

            # The ESM SDK returns ESMProteinError (instead of raising) when the
            # Forge API rejects the request (bad token, model not found, etc.).
            if _is_protein_error(generated):
                err_msg = _extract_error_msg(generated)
                _log(f"ERROR: Forge API returned ESMProteinError for candidate {i+1}: {err_msg}")
                # If the error is about invalid function annotations, strip them and retry.
                if ("FunctionAnnotation" in err_msg or "Unknown label" in err_msg
                        or "function" in err_msg.lower()) and esm_protein.function_annotations:
                    _log(f"INFO: Candidate {i+1} — bad FunctionAnnotation label, retrying without…")
                    ep_no_func = _make_no_func_protein(esm_protein)
                    generated = client.generate(ep_no_func, gen_config)
                    if _is_protein_error(generated):
                        err_msg2 = _extract_error_msg(generated)
                        raise RuntimeError(
                            f"Forge API error (after stripping annotations): {err_msg2}. "
                            "Check your Forge API token and model selection."
                        )
                else:
                    # Not an annotation issue — surface the real error immediately.
                    raise RuntimeError(
                        f"Forge API error: {err_msg}. "
                        "Check your Forge API token and model selection."
                    )

            seq_dbg = getattr(generated, "sequence", None)
            _log(f"DEBUG candidate {i+1}: type={type(generated).__name__}, "
                  f"seq_type={type(seq_dbg).__name__}, "
                  f"seq_preview={repr(str(seq_dbg)[:60]) if seq_dbg is not None else None}")

            # If the Forge API returned an empty/null sequence (can happen when
            # function_annotations contain terms outside ESM3's InterPro vocabulary),
            # retry without function annotations.
            if _seq_is_empty(generated) and esm_protein.function_annotations:
                _log(f"INFO: Candidate {i+1} returned empty sequence — "
                      f"retrying without function annotations…")
                ep_no_func = _make_no_func_protein(esm_protein)
                generated = client.generate(ep_no_func, gen_config)
                if _is_protein_error(generated):
                    err_msg = _extract_error_msg(generated)
                    raise RuntimeError(f"Forge API error on retry: {err_msg}")
                seq_dbg2 = getattr(generated, "sequence", None)
                _log(f"DEBUG candidate {i+1} retry: seq_preview={repr(str(seq_dbg2)[:60]) if seq_dbg2 is not None else None}")

            results.append(GenerationResult(esm_protein=generated, index=i))

        except RuntimeError:
            raise  # propagate Forge API errors to the caller
        except Exception as e:
            # If the error is due to invalid FunctionAnnotation labels,
            # retry without function annotations — sequence/structure constraints
            # are preserved, only keyword guidance is dropped.
            if "FunctionAnnotation" in str(e) or "Unknown label" in str(e) or "function" in str(e).lower():
                _log(f"INFO: Retrying candidate {i + 1} without function annotations (error: {e})…")
                try:
                    ep_no_func = _make_no_func_protein(esm_protein)
                    generated = client.generate(ep_no_func, gen_config)
                    results.append(GenerationResult(esm_protein=generated, index=i))
                except Exception as e2:
                    _log(f"WARNING: Candidate {i + 1} failed on retry: {e2}")
            else:
                _log(f"WARNING: Candidate {i + 1} failed: {e}")

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
            _log(f"WARNING: Candidate {i + 1} (structure+sequence) failed: {e}")
            continue

    if progress_callback:
        progress_callback(total_steps, total_steps)

    return results


def protein_to_pdb_string(protein_obj) -> str | None:
    """
    Extract a PDB-format string from an ESMProtein object.

    Handles both the old API (to_pdb() returns string) and the new API
    (to_pdb(path) writes to file). Falls back to a temp-file round-trip.

    Returns:
        PDB string, or None if extraction fails.
    """
    import tempfile, os

    # 1. Try the old no-arg form
    try:
        result = protein_obj.to_pdb()
        if isinstance(result, str) and result.strip():
            return result
    except TypeError:
        pass  # new API requires a path argument
    except AttributeError:
        pass

    # 2. New API: to_pdb(path) writes to disk — read it back
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        protein_obj.to_pdb(tmp_path)
        with open(tmp_path) as fh:
            pdb_str = fh.read()
        os.unlink(tmp_path)
        if pdb_str.strip():
            return pdb_str
    except Exception as e:
        _log(f"DEBUG protein_to_pdb_string: to_pdb(path) failed: {e}")

    # 3. Alternative method names
    for attr in ("to_pdb_string", "pdb_string"):
        method = getattr(protein_obj, attr, None)
        if callable(method):
            try:
                result = method()
                if isinstance(result, str) and result.strip():
                    return result
            except Exception:
                pass

    return None


def fold_sequence(sequence: str, client=None) -> str | None:
    """
    Fold a protein sequence using ESM3 structure generation (ESMFold via Forge).

    Args:
        sequence: Amino acid sequence string.
        client:   ESM3 client (Forge or local). If None, loads from config.

    Returns:
        PDB-format string, or None if folding fails.
    """
    try:
        from esm.sdk.api import ESMProtein, GenerationConfig
    except ImportError as e:
        raise RuntimeError("ESM SDK not found. Install with: pip install esm") from e

    if client is None:
        from config import get_esm_client
        client = get_esm_client()

    protein = ESMProtein(sequence=sequence)
    config = GenerationConfig(track="structure", num_steps=8, temperature=0.0)

    try:
        folded = client.generate(protein, config)
    except Exception as e:
        raise RuntimeError(f"ESMFold generation failed: {e}") from e

    if _is_protein_error(folded):
        raise RuntimeError(f"ESMFold Forge API error: {_extract_error_msg(folded)}")

    pdb = protein_to_pdb_string(folded)
    if pdb is None:
        raise RuntimeError("ESMFold succeeded but could not extract PDB coordinates.")
    return pdb


def choose_generation_strategy(spec: "PromptSpec"):
    """
    Return the appropriate generation function based on the PromptSpec.

    If structure motif is used → two-stage generation.
    Otherwise → sequence-only generation.
    """
    if spec.use_structure_motif and spec.motif_residue_indices:
        return generate_with_structure
    return generate_candidates
