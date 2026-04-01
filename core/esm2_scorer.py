"""
esm2_scorer.py — ESM2 log-likelihood scoring for protein fitness estimation.

Uses ESM2's masked marginal log-likelihood as a zero-shot fitness proxy.
Higher scores correlate with better stability, expression, and function.

Theoretical basis:
  Proteins that survived billions of years of evolution have sequences
  where every residue is "expected" given its context. ESM2 has learned
  this distribution. A sequence with high masked marginal log-likelihood
  is more consistent with natural protein biology — and therefore more
  likely to fold stably and function correctly.

Two scoring modes:
  - "pseudo" (default, fast): single forward pass — log P(x_i | x) for all i.
    O(1) forward passes per sequence. Good for ranking large candidate pools.
  - "masked" (accurate): one forward pass per residue — log P(x_i | x_{-i}).
    O(L) passes. Theoretically correct but slow for long proteins (>150 aa).
    Batched internally to amortise cost.

Reference: Meier et al. 2021 (ESM-1v); Notin et al. 2022 (EVE comparison).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

# Default model — tiny and fast; good enough for ranking.
# Upgrade to "facebook/esm2_t30_150M_UR50D" or "facebook/esm2_t33_650M_UR50D"
# for better score quality at the cost of speed/memory.
DEFAULT_ESM2_MODEL = "facebook/esm2_t6_8M_UR50D"

# Module-level cache so we load once per session.
_model_cache: dict[str, tuple] = {}


def _load_model(model_name: str = DEFAULT_ESM2_MODEL):
    """Load (and cache) ESM2 tokeniser + model. Returns (tokenizer, model, device)."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from transformers import AutoTokenizer, EsmForMaskedLM
    except ImportError as e:
        raise ImportError(
            "transformers is required for ESM2 scoring. "
            "Install with: pip install transformers"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()

    _model_cache[model_name] = (tokenizer, model, device)
    return tokenizer, model, device


# ── Public API ─────────────────────────────────────────────────────────────────

def score_sequence(
    sequence: str,
    mode: Literal["pseudo", "masked"] = "pseudo",
    model_name: str = DEFAULT_ESM2_MODEL,
) -> float:
    """
    Compute the normalised ESM2 log-likelihood of a single sequence.

    Args:
        sequence:   Amino acid sequence (single-letter codes, no gaps).
        mode:       "pseudo" = fast single-pass; "masked" = accurate masked marginal.
        model_name: HuggingFace ESM2 model identifier.

    Returns:
        Mean log-likelihood per residue (typically in range [-3, 0]).
        Higher is better. Comparable across sequences of different lengths.
    """
    sequence = _clean_sequence(sequence)
    if not sequence:
        return float("-inf")

    tokenizer, model, device = _load_model(model_name)

    if mode == "pseudo":
        return _pseudo_log_likelihood(sequence, tokenizer, model, device)
    else:
        return _masked_marginal_log_likelihood(sequence, tokenizer, model, device)


def score_sequences(
    sequences: list[str],
    mode: Literal["pseudo", "masked"] = "pseudo",
    model_name: str = DEFAULT_ESM2_MODEL,
    progress_callback=None,
) -> list[float]:
    """
    Score a list of sequences. Returns scores in the same order.

    Args:
        sequences:         List of amino acid sequences.
        mode:              Scoring mode ("pseudo" or "masked").
        model_name:        ESM2 HuggingFace model name.
        progress_callback: Optional callable(current, total).

    Returns:
        List of mean log-likelihood scores (one per sequence).
    """
    tokenizer, model, device = _load_model(model_name)
    scores = []

    for i, seq in enumerate(sequences):
        if progress_callback:
            progress_callback(i, len(sequences))

        seq = _clean_sequence(seq)
        if not seq:
            scores.append(float("-inf"))
            continue

        if mode == "pseudo":
            score = _pseudo_log_likelihood(seq, tokenizer, model, device)
        else:
            score = _masked_marginal_log_likelihood(seq, tokenizer, model, device)
        scores.append(score)

    if progress_callback:
        progress_callback(len(sequences), len(sequences))

    return scores


def normalise_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalise a list of scores to [0, 1].
    Handles -inf values by treating them as the minimum.
    """
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return [0.0] * len(scores)

    lo, hi = min(finite), max(finite)
    if lo == hi:
        return [1.0 if math.isfinite(s) else 0.0 for s in scores]

    result = []
    for s in scores:
        if not math.isfinite(s):
            result.append(0.0)
        else:
            result.append((s - lo) / (hi - lo))
    return result


# ── Internal scoring functions ─────────────────────────────────────────────────

def _pseudo_log_likelihood(
    sequence: str,
    tokenizer,
    model,
    device: str,
) -> float:
    """
    Single-pass pseudo log-likelihood: sum_i log P(x_i | x).

    Fast approximation — does NOT mask each position individually.
    The model sees the full sequence at all positions, which slightly
    over-estimates likelihood (the model can "cheat" by looking at x_i
    when predicting x_i), but is well-correlated with masked marginal
    for ranking purposes.
    """
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits  # (1, L+2, vocab_size)  — +2 for BOS/EOS

        log_probs = torch.log_softmax(logits, dim=-1)  # (1, L+2, vocab)
        input_ids = inputs["input_ids"]  # (1, L+2)

        # Skip BOS (index 0) and EOS (index -1)
        seq_log_probs = log_probs[0, 1:-1, :]   # (L, vocab)
        seq_ids = input_ids[0, 1:-1]             # (L,)

        # Gather log P of the actual amino acid at each position
        token_log_probs = seq_log_probs.gather(
            1, seq_ids.unsqueeze(-1)
        ).squeeze(-1)                            # (L,)

    return token_log_probs.mean().item()


def _masked_marginal_log_likelihood(
    sequence: str,
    tokenizer,
    model,
    device: str,
    batch_size: int = 16,
) -> float:
    """
    Masked marginal log-likelihood: sum_i log P(x_i | x_{-i}).

    For each position i: mask it, run a forward pass, record log P(true_aa).
    Batched: `batch_size` positions are masked simultaneously per forward pass.
    For long sequences (>200 aa) this can take a while even on GPU.
    """
    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)  # (1, L+2)
    mask_token_id = tokenizer.mask_token_id

    L = input_ids.shape[1] - 2  # sequence length without BOS/EOS
    all_log_probs = []

    with torch.no_grad():
        # Process in batches of `batch_size` masked positions at a time
        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            batch_ids = input_ids.repeat(end - start, 1)  # (batch, L+2)

            for j, pos in enumerate(range(start, end)):
                batch_ids[j, pos + 1] = mask_token_id  # +1 to skip BOS

            outputs = model(input_ids=batch_ids)
            logits = outputs.logits  # (batch, L+2, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)

            for j, pos in enumerate(range(start, end)):
                true_id = input_ids[0, pos + 1]
                lp = log_probs[j, pos + 1, true_id].item()
                all_log_probs.append(lp)

    return float(np.mean(all_log_probs)) if all_log_probs else float("-inf")


def _clean_sequence(seq: str) -> str:
    """Remove non-amino-acid characters and upper-case."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(c for c in seq.upper() if c in valid)
