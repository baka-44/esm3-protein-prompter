"""
result_processor.py — Rank, score, and package ESM3 generation outputs.

Takes a list of raw GenerationResult objects from the ESM backend and returns
a list of CandidateResult dataclasses, sorted by composite quality score.

Scoring components:
  - pTM:        ESM3's predicted TM-score (structural fold quality, 0–1)
  - pLDDT:      Mean per-residue confidence (0–100)
  - ESM2 score: Masked marginal log-likelihood (zero-shot fitness proxy)

Composite = 0.5 * pTM + 0.3 * (pLDDT/100) + 0.2 * esm2_norm
            where esm2_norm is the ESM2 score min-max normalised across candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sys

import numpy as np


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

from core.esm_backend import GenerationResult
from utils.sequence_utils import novelty_percent, mean_pairwise_diversity, to_fasta


@dataclass
class CandidateResult:
    """Processed and scored protein design candidate."""

    rank: int
    """1-based rank (1 = best composite score)."""

    sequence: str
    """Full amino acid sequence."""

    mean_plddt: float
    """Mean per-residue pLDDT confidence score (0–100). Higher = more confident structure."""

    ptm: float
    """Predicted TM-score (0–1). Higher = better predicted fold quality."""

    esm2_score: float
    """
    ESM2 masked-marginal log-likelihood per residue (typically -3 to 0).
    Higher = more consistent with natural protein biology = better fitness proxy.
    """

    esm2_score_norm: float
    """ESM2 score normalised to [0, 1] across candidates in this batch."""

    composite_score: float
    """
    Weighted composite:
      0.5 * pTM + 0.3 * (pLDDT/100) + 0.2 * esm2_norm
    Range [0, 1]. Primary sort key.
    """

    novelty_pct: float
    """% positions that differ from the reference/template sequence. Higher = more novel."""

    pdb_string: str | None = None
    """PDB-format structure string, if available from ESM3."""

    plddt_per_residue: list[float] = field(default_factory=list)
    """Per-residue pLDDT scores for visualisation and refinement threshold decisions."""

    index: int = 0
    """Original generation index (0-based)."""

    has_structure_scores: bool = False
    """
    True when pTM and pLDDT are populated (i.e. structure track was run).
    False for sequence-only generation — fold the candidate to get real scores.
    """

    has_novelty_ref: bool = False
    """
    True when novelty_pct was computed against a valid reference sequence.
    False when no reference was available (e.g. de novo generation or masked template).
    When False, novelty_pct is 0.0 and should be displayed as '—'.
    """

    def fasta_header(self) -> str:
        return (
            f"candidate_{self.rank} "
            f"pTM={self.ptm:.3f} "
            f"pLDDT={self.mean_plddt:.1f} "
            f"ESM2={self.esm2_score:.3f} "
            f"novelty={self.novelty_pct:.1f}%"
        )


def process_results(
    raw_results: list[GenerationResult],
    reference_sequence: str = "",
    spec=None,
    run_esm2_scoring: bool = True,
    esm2_mode: str = "pseudo",
    progress_callback=None,
) -> list[CandidateResult]:
    """
    Convert raw ESM3 generation outputs into ranked CandidateResult objects.

    Args:
        raw_results:         List of GenerationResult from esm_backend.
        reference_sequence:  Optional reference sequence for novelty calculation.
        spec:                Optional PromptSpec (used to extract reference from template).
        run_esm2_scoring:    Whether to compute ESM2 log-likelihood scores.
        esm2_mode:           "pseudo" (fast) or "masked" (accurate, slower).
        progress_callback:   Optional callable(current, total) for ESM2 scoring progress.

    Returns:
        List of CandidateResult sorted by composite_score descending (best first).
    """
    if not raw_results:
        return []

    # Determine the original input sequence for dedup filtering
    original_input = ""
    if spec and hasattr(spec, "original_sequence") and spec.original_sequence:
        original_input = spec.original_sequence

    if not reference_sequence and spec and spec.sequence_template:
        # Only use the template as a novelty reference when it contains no mask
        # characters — stripping '_' would produce a shorter string than the
        # generated sequence, causing a positional offset that inflates novelty
        # for sequences that are actually very similar to the template.
        template = spec.sequence_template
        if "_" not in template:
            reference_sequence = template

    # Use original_input as novelty reference if we still don't have one
    if not reference_sequence and original_input:
        reference_sequence = original_input

    # ── Extract raw metrics from ESMProtein objects ────────────────────────────
    candidates: list[CandidateResult] = []
    for result in raw_results:
        candidate = _extract_candidate(result.esm_protein, result.index, reference_sequence)
        candidates.append(candidate)

    # ── Deduplicate: keep only unique sequences ──────────────────────────────
    seen_seqs: set[str] = set()
    unique_candidates: list[CandidateResult] = []
    for c in candidates:
        if c.sequence not in seen_seqs:
            seen_seqs.add(c.sequence)
            unique_candidates.append(c)
        else:
            _log(f"INFO: dropping duplicate candidate (index={c.index})")
    if len(unique_candidates) < len(candidates):
        _log(f"INFO: dedup reduced {len(candidates)} → {len(unique_candidates)} candidates")
    candidates = unique_candidates

    # ── Filter out candidates identical to the input sequence ─────────────────
    if original_input:
        before_count = len(candidates)
        candidates = [c for c in candidates if c.sequence != original_input]
        dropped = before_count - len(candidates)
        if dropped:
            _log(f"INFO: filtered {dropped} candidate(s) identical to input sequence")

    if not candidates:
        _log("WARNING: all candidates were duplicates or identical to input — "
             "returning empty list")
        return []

    # ── ESM2 log-likelihood scoring ────────────────────────────────────────────
    if run_esm2_scoring:
        _add_esm2_scores(candidates, mode=esm2_mode, progress_callback=progress_callback)
    else:
        # Set neutral placeholders
        for c in candidates:
            c.esm2_score = 0.0
            c.esm2_score_norm = 0.5

    # ── Compute composite score and sort ──────────────────────────────────────
    for c in candidates:
        if c.has_structure_scores:
            # Full composite when structure scores are available
            c.composite_score = (
                0.5 * c.ptm
                + 0.3 * (c.mean_plddt / 100.0)
                + 0.2 * c.esm2_score_norm
            )
        else:
            # Sequence-only: composite is purely ESM2 fitness proxy
            c.composite_score = c.esm2_score_norm

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank

    return candidates


def _add_esm2_scores(
    candidates: list[CandidateResult],
    mode: str = "pseudo",
    progress_callback=None,
):
    """Compute ESM2 scores for all candidates and normalise them in-place."""
    from core.esm2_scorer import score_sequences, normalise_scores

    sequences = [c.sequence for c in candidates]
    try:
        raw_scores = score_sequences(
            sequences,
            mode=mode,
            progress_callback=progress_callback,
        )
    except Exception as e:
        import traceback
        _log(f"WARNING: ESM2 scoring failed ({type(e).__name__}: {e}). Using neutral scores.")
        _log(traceback.format_exc())
        raw_scores = [0.0] * len(candidates)

    normalised = normalise_scores(raw_scores)

    for c, raw, norm in zip(candidates, raw_scores, normalised):
        c.esm2_score = raw
        c.esm2_score_norm = norm


def _extract_candidate(
    protein: object,
    index: int,
    reference_sequence: str,
) -> CandidateResult:
    """Extract metrics from an ESMProtein object into a CandidateResult (pre-ESM2)."""

    sequence = _get_sequence(protein)
    plddt_per_residue = _get_plddt(protein)
    ptm = _get_ptm(protein)
    pdb_string = _get_pdb_string(protein)

    mean_plddt = float(np.mean(plddt_per_residue)) if plddt_per_residue else 0.0
    if mean_plddt <= 1.0:
        mean_plddt *= 100.0
        plddt_per_residue = [v * 100.0 for v in plddt_per_residue]

    # Structure scores are available only when the ESMProtein has a real ptm value
    # (sequence-only generation returns ptm=None → 0.0, which we can detect here
    # by checking the raw attribute before our 0.0 fallback was applied).
    raw_ptm = getattr(protein, "ptm", None)
    has_structure_scores = raw_ptm is not None and float(raw_ptm) > 0.0

    novelty_pct = 0.0
    has_novelty_ref = False
    if reference_sequence and sequence:
        novelty_pct = novelty_percent(sequence, reference_sequence)
        has_novelty_ref = True

    # ESM2 scores filled in later by _add_esm2_scores
    return CandidateResult(
        rank=0,
        sequence=sequence,
        mean_plddt=mean_plddt,
        ptm=ptm,
        esm2_score=0.0,
        esm2_score_norm=0.5,
        composite_score=0.0,  # recomputed after ESM2 scoring
        novelty_pct=novelty_pct,
        pdb_string=pdb_string,
        plddt_per_residue=plddt_per_residue,
        index=index,
        has_structure_scores=has_structure_scores,
        has_novelty_ref=has_novelty_ref,
    )


def _get_sequence(protein: object) -> str:
    seq = getattr(protein, "sequence", None)
    _log(f"DEBUG _get_sequence: type={type(seq).__name__}, "
          f"repr={repr(str(seq)[:80]) if seq is not None else None}")
    if seq is None:
        return ""
    seq_str = str(seq)
    # Replace mask token '_' with nothing — keep valid amino acids
    cleaned = seq_str.replace("_", "")
    if not cleaned and seq_str:
        # Sequence exists but is all mask tokens — generation may have failed silently.
        # Log and return empty so the candidate is flagged as invalid.
        _log(f"WARNING _get_sequence: sequence is entirely mask tokens "
              f"(len={len(seq_str)}) — generation produced no amino acids.")
    return cleaned


def _get_plddt(protein: object) -> list[float]:
    plddt = getattr(protein, "plddt", None)
    if plddt is None:
        return []
    try:
        arr = np.asarray(plddt).flatten()
        return arr.tolist()
    except Exception:
        return list(plddt) if hasattr(plddt, "__iter__") else []


def _get_ptm(protein: object) -> float:
    ptm = getattr(protein, "ptm", None)
    if ptm is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(ptm)))
    except Exception:
        return 0.0


def _get_pdb_string(protein: object) -> str | None:
    coords = getattr(protein, "coordinates", None)
    if coords is None:
        return None
    try:
        arr = np.asarray(coords)
        if np.all(np.isnan(arr)):
            return None
    except Exception:
        pass
    from core.esm_backend import protein_to_pdb_string
    return protein_to_pdb_string(protein)


def candidates_to_fasta(candidates: list[CandidateResult]) -> str:
    """Export all candidates as a multi-sequence FASTA string."""
    return to_fasta([(c.fasta_header(), c.sequence) for c in candidates])


def diversity_summary(candidates: list[CandidateResult]) -> float:
    """Mean pairwise diversity across all candidates (0–1)."""
    seqs = [c.sequence for c in candidates if c.sequence]
    return mean_pairwise_diversity(seqs)
