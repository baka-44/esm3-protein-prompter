"""
proteinredesign/worker.py — generation pipeline entrypoint (runs on the GPU Cloud Run Job).

Flow (increment 1, preset #1 fixed-backbone redesign — MPNN-only):
    manifest (GCS) → download PDB
      → ProteinMPNN design (fixed_positions) — over-generate (B4)
      → multi-checkpoint MPNN scoring (D1: ProteinMPNN + SolubleMPNN, metadata)
      → ESMFold QC (D2 hard gate: pLDDT + RMSD-to-input-backbone)
      → ESM2 soft floor (D2: drop clearly-unnatural tail) + rank (B3)
      → keep top `num_outputs` (=10) QC-passed, ranked
      → write results JSON + per-candidate FASTA/PDB to GCS; update Firestore

The ML tool calls (ProteinMPNN, ESMFold) are isolated in adapter functions that
run inside the worker container (weights from GCS — A6). The QC-gate + ranking
logic (`select_top_candidates`) is pure and unit-tested.

Entrypoint: `python -m proteinredesign.worker` with env PROTEINREDESIGN_MANIFEST_URI set by the job.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field

from proteinredesign.manifest import Preset


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

# ── Tunable gates / budget (D2, B4) ───────────────────────────────────────────
PLDDT_GATE = float(os.getenv("PROTEINREDESIGN_PLDDT_GATE", "70.0"))     # hard structural gate
RMSD_GATE = float(os.getenv("PROTEINREDESIGN_RMSD_GATE", "2.0"))         # Å, self-consistency
ESM2_DROP_FRACTION = float(os.getenv("PROTEINREDESIGN_ESM2_DROP", "0.10"))  # soft floor: drop bottom 10%
OVERGEN_FACTOR = int(os.getenv("PROTEINREDESIGN_OVERGEN_FACTOR", "3"))  # generate 3× to survive QC


@dataclass
class Candidate:
    """One designed sequence and its scores/QC results."""

    sequence: str
    mpnn_scores: dict[str, float] = field(default_factory=dict)  # {checkpoint: score}
    esm2_score: float = 0.0
    plddt: float = 0.0
    rmsd_to_design: float = float("inf")
    pdb: str = ""              # ESMFold-predicted structure (PDB text)
    composite_score: float = 0.0
    rank: int = 0

    @property
    def passes_structural_gate(self) -> bool:
        return self.plddt >= PLDDT_GATE and self.rmsd_to_design <= RMSD_GATE


# ── Pure QC-gate + ranking (unit-tested) ──────────────────────────────────────

def _minmax(values: list[float]) -> list[float]:
    finite = [v for v in values if v not in (float("inf"), float("-inf")) and v == v]
    if not finite:
        return [0.0] * len(values)
    lo, hi = min(finite), max(finite)
    if lo == hi:
        return [1.0 if (v == v and v not in (float("inf"), float("-inf"))) else 0.0 for v in values]
    return [((v - lo) / (hi - lo)) if (v == v and v not in (float("inf"), float("-inf"))) else 0.0
            for v in values]


def _percentile(values: list[float], pct: float) -> float:
    """Simple linear-interpolated percentile (pct in [0,100])."""
    xs = sorted(values)
    if not xs:
        return float("-inf")
    k = (len(xs) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def select_top_candidates(
    candidates: list[Candidate],
    num_outputs: int,
    plddt_gate: float = PLDDT_GATE,
    rmsd_gate: float = RMSD_GATE,
    esm2_drop_fraction: float = ESM2_DROP_FRACTION,
) -> list[Candidate]:
    """
    Apply the QC gate (D2) and return the top `num_outputs` ranked candidates.

    1. Hard structural gate: pLDDT ≥ plddt_gate AND RMSD-to-design ≤ rmsd_gate.
    2. ESM2 soft floor (D2): drop the clearly-unnatural bottom `esm2_drop_fraction`
       (only when there are more survivors than requested — never starves output,
       never hard-ranks by naturalness).
    3. Rank by a composite of the metadata scores (MPNN checkpoints + ESM2),
       min-max normalised across the surviving set (B3: metadata used for ranking).
    """
    passed = [
        c for c in candidates
        if c.plddt >= plddt_gate and c.rmsd_to_design <= rmsd_gate
    ]
    if not passed:
        return []

    # Soft floor — only prune the tail when we have headroom over the request.
    if esm2_drop_fraction > 0 and len(passed) > num_outputs:
        thr = _percentile([c.esm2_score for c in passed], esm2_drop_fraction * 100.0)
        floored = [c for c in passed if c.esm2_score >= thr]
        if len(floored) >= num_outputs:
            passed = floored

    # Composite ranking over the surviving set.
    checkpoints = sorted({k for c in passed for k in c.mpnn_scores})
    norm_esm2 = _minmax([c.esm2_score for c in passed])
    norm_ckpt = {
        ck: _minmax([c.mpnn_scores.get(ck, float("nan")) for c in passed])
        for ck in checkpoints
    }
    for i, c in enumerate(passed):
        # ESM2 log-likelihood: higher = better → use as-is.
        # MPNN score is a mean negative log-likelihood: lower = better → invert.
        # (Assumes every candidate carries every checkpoint score, which the
        #  pipeline guarantees by scoring all sequences under all checkpoints.)
        goodness = [norm_esm2[i]] + [1.0 - norm_ckpt[ck][i] for ck in checkpoints]
        c.composite_score = sum(goodness) / len(goodness) if goodness else 0.0

    ranked = sorted(passed, key=lambda c: c.composite_score, reverse=True)[:num_outputs]
    for i, c in enumerate(ranked, start=1):
        c.rank = i
    return ranked


# ── ML adapters (run inside the worker container; weights from GCS — A6) ───────
# These are pinned against the LigandMPNN repo (ProteinMPNN/SolubleMPNN checkpoints)
# and ESMFold during container build. Kept isolated so the orchestration + gating
# above stay pure and testable.

def _mpnn_weights_dir() -> str:
    from proteinredesign.storage import ensure_weights
    return ensure_weights("mpnn")


def _esmfold_weights_dir() -> str:
    from proteinredesign.storage import ensure_weights
    return ensure_weights("esmfold")


# NOTE — these adapters are a first draft to be validated at container-build time.
# Interface assumptions to VERIFY against the pinned tool versions:
#   [MPNN-1] LigandMPNN `run.py` flags + weight filenames under mpnn/<checkpoint>.pt
#   [MPNN-2] --fixed_residues token format is "<chain><author_resnum>" (PDB numbering)
#   [MPNN-3] output FASTA path (out/seqs/*.fa), first record = native (skipped),
#            and the header score field name/direction ("score" = neg-LL, lower better)
#   [ESM-1]  transformers EsmForProteinFolding: output_to_pdb() + pLDDT in B-factors (0–100)

_LIGANDMPNN_DIR = os.getenv("LIGANDMPNN_DIR", "/opt/ligandmpnn")

# checkpoint id -> (run.py --model_type, weight-flag)
_CHECKPOINT_FLAG = {
    "proteinmpnn": ("protein_mpnn", "--checkpoint_protein_mpnn"),
    "soluble_mpnn": ("soluble_mpnn", "--checkpoint_soluble_mpnn"),
    "ligand_mpnn": ("ligand_mpnn", "--checkpoint_ligand_mpnn"),
}

_HEADER_SCORE = re.compile(r"(?:^|,\s*)(?:score|overall_confidence|global_score)=([-\d.]+)")

# Which MPNN checkpoint performs the actual sequence design, per preset. #1 keeps
# the backbone with no ligand context (ProteinMPNN); #2 conditions on the ligand
# HETATM already filtered into the PDB by the config builder (LigandMPNN).
_DESIGN_CHECKPOINT = {
    Preset.FIXED_BACKBONE_REDESIGN: "proteinmpnn",
    Preset.LIGAND_AWARE_REDESIGN: "ligand_mpnn",
}


def _fixed_residue_tokens(params: dict) -> str:
    """Author-numbered residue ids for LigandMPNN --fixed_residues, e.g. 'A67 A82' [MPNN-2]."""
    return " ".join(f"{r['chain_id']}{r['author_num']}" for r in params.get("fixed_residues", []))


def _mpnn_checkpoint_path(checkpoint: str) -> str:
    return os.path.join(_mpnn_weights_dir(), f"{checkpoint}.pt")


def _run_ligandmpnn(pdb_path, out_folder, checkpoint, fixed_tokens,
                    n_batches, batch_size, temperature, extra=None):
    model_type, ckpt_flag = _CHECKPOINT_FLAG[checkpoint]
    cmd = [
        "python", os.path.join(_LIGANDMPNN_DIR, "run.py"),
        "--model_type", model_type,
        ckpt_flag, _mpnn_checkpoint_path(checkpoint),
        "--pdb_path", pdb_path,
        "--out_folder", out_folder,
        "--number_of_batches", str(n_batches),
        "--batch_size", str(batch_size),
        "--temperature", str(temperature),
        "--save_stats", "1",
    ]
    if fixed_tokens:
        cmd += ["--fixed_residues", fixed_tokens]
    if extra:
        cmd += extra
    subprocess.run(cmd, check=True, cwd=_LIGANDMPNN_DIR)


def _parse_seqs_fasta(out_folder: str) -> list[tuple[str, float]]:
    """Parse LigandMPNN's seqs/*.fa: drop the native (first) record, return (seq, score) [MPNN-3]."""
    files = sorted(glob.glob(os.path.join(out_folder, "seqs", "*.fa")) +
                   glob.glob(os.path.join(out_folder, "seqs", "*.fasta")))
    if not files:
        return []
    records: list[tuple[str, str]] = []
    header, chunks = None, []
    with open(files[0]) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line, []
            elif line:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    designs = records[1:] if len(records) > 1 else records  # first = native input
    out = []
    for h, seq in designs:
        m = _HEADER_SCORE.search(h)
        score = float(m.group(1)) if m else float("nan")
        out.append((seq.replace("/", ""), score))  # strip multi-chain separators
    return out


def run_proteinmpnn(
    pdb_path: str,
    fixed_residue_tokens: str,
    n_seqs: int,
    checkpoint: str = "proteinmpnn",
    temperature: float = 0.1,
) -> list[tuple[str, float]]:
    """
    Design ~`n_seqs` sequences on the fixed backbone (keeping the fixed residues).
    Returns [(sequence, mpnn_score), ...] (score = mean neg-LL, lower = better [MPNN-3]).

    For checkpoint="ligand_mpnn", LigandMPNN's run.py automatically parses and
    conditions on any HETATM ligand present in the PDB — the caller is responsible
    for the PDB containing ONLY the intended ligand's HETATM records (see
    utils.pdb_utils.filter_pdb_keep_ligand, used by the preset #2 config builder).
    """
    out_folder = tempfile.mkdtemp(prefix="mpnn_")
    batch_size = min(max(1, n_seqs), 8)
    n_batches = max(1, -(-n_seqs // batch_size))  # ceil
    extra = ["--ligand_mpnn_use_atom_context", "1"] if checkpoint == "ligand_mpnn" else None
    _run_ligandmpnn(pdb_path, out_folder, checkpoint, fixed_residue_tokens,
                    n_batches, batch_size, temperature, extra=extra)
    return _parse_seqs_fasta(out_folder)[:n_seqs]


def score_with_checkpoint(sequences: list[str], pdb_path: str, checkpoint: str) -> list[float]:
    """
    Score existing sequences under `checkpoint` (D1 metadata / ranking signal only —
    never a hard filter). Currently always returns NaN: LigandMPNN's score.py has no
    way to score an externally-supplied sequence — it evaluates only the sequence
    already present in the input PDB (no --fasta_path / --sequence flag exists). A
    real implementation would need to write a copy of the PDB with each candidate
    sequence's residue identities substituted in, then score that per-sequence PDB
    with score.py's --single_aa_score / --autoregressive_score mode — non-trivial
    and not required for MVP (D1 metadata gracefully NaNs; composite ranking already
    handles NaN in _minmax). Parked as future work, not a preset #2 blocker.
    """
    return [float("nan")] * len(sequences)


_esmfold_model = None


def run_esmfold(sequence: str) -> tuple[str, float]:
    """
    Fold `sequence` with ESMFold (transformers EsmForProteinFolding). Returns
    (pdb_text, mean_pLDDT). Weights: local esmfold dir if present, else HF id [ESM-1].
    """
    global _esmfold_model
    import torch
    if _esmfold_model is None:
        from transformers import AutoTokenizer, EsmForProteinFolding
        local = _esmfold_weights_dir()
        src = local if (os.path.isdir(local) and os.listdir(local)) else "facebook/esmfold_v1"
        tok = AutoTokenizer.from_pretrained(src)
        model = EsmForProteinFolding.from_pretrained(src).eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _esmfold_model = (tok, model)

    tok, model = _esmfold_model
    with torch.no_grad():
        ids = tok([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"]
        if torch.cuda.is_available():
            ids = ids.cuda()
        outputs = model(ids)
        pdb_text = model.output_to_pdb(outputs)[0]
    return pdb_text, _mean_ca_bfactor(pdb_text)


def _mean_ca_bfactor(pdb_text: str) -> float:
    """
    Mean CA B-factor = mean pLDDT. transformers ESMFold writes pLDDT into the B-factor
    column on a **0–1 scale**; normalise to the conventional 0–100 scale so the QC gate
    and UI read correctly [ESM-1 verified against a real run].
    """
    vals = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return mean * 100.0 if mean <= 1.5 else mean  # 0–1 → 0–100


def compute_ca_rmsd(pred_pdb: str, ref_pdb_source, chain_id: str | None) -> float:
    """
    CA-RMSD between the ESMFold prediction and the input (reference) backbone —
    the self-consistency metric for preset #1 (the backbone is held fixed).
    """
    from Bio.PDB import Superimposer
    from utils.pdb_utils import get_residues

    pred = get_residues(pred_pdb)  # ESMFold output is single chain
    ref = get_residues(ref_pdb_source, chain_id=chain_id)
    n = min(len(pred), len(ref))
    if n == 0:
        return float("inf")
    pred_ca = [r["CA"] for r in pred[:n] if "CA" in r]
    ref_ca = [r["CA"] for r in ref[:n] if "CA" in r]
    m = min(len(pred_ca), len(ref_ca))
    if m == 0:
        return float("inf")
    sup = Superimposer()
    sup.set_atoms(ref_ca[:m], pred_ca[:m])
    return float(sup.rms)


def esm2_scores(sequences: list[str]) -> list[float]:
    """ESM2 pseudo-log-likelihood per sequence (soft-floor + ranking signal)."""
    from core.esm2_scorer import score_sequences
    return score_sequences(sequences, mode="pseudo")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_pipeline(manifest, workdir: str) -> dict:
    """
    Execute preset #1 end-to-end and return a results dict (also written to GCS).

    `manifest` is a proteinredesign.manifest.JobManifest. Firestore status is updated at
    each stage so the dashboard can show live progress (B7/B8).
    """
    from proteinredesign import jobstore
    from proteinredesign import storage
    from proteinredesign.manifest import JobStatus

    design_checkpoint = _DESIGN_CHECKPOINT.get(manifest.preset)
    if design_checkpoint is None:
        raise NotImplementedError(
            f"Preset {manifest.preset.value} not yet implemented in the worker."
        )

    job_id = manifest.job_id
    params = manifest.params
    chain_id = params.get("chain_id")
    fixed_tokens = _fixed_residue_tokens(params)
    n_target = manifest.num_outputs
    n_generate = max(n_target * OVERGEN_FACTOR, n_target)

    # 0. Fetch the input PDB.
    jobstore.update_job(job_id, status=JobStatus.RUNNING.value, stage="Preparing inputs", progress=0.05)
    pdb_path = os.path.join(workdir, "input.pdb")
    storage.download_to_path(manifest.pdb_uri, pdb_path)

    # 1. MPNN design (over-generate). Checkpoint depends on the preset (D6): plain
    # ProteinMPNN for #1, LigandMPNN for #2 (conditions on the ligand already
    # filtered into the PDB by the preset #2 config builder).
    jobstore.update_job(job_id, stage=f"{design_checkpoint} design", progress=0.2)
    designed = run_proteinmpnn(pdb_path, fixed_tokens, n_seqs=n_generate, checkpoint=design_checkpoint)
    candidates = [Candidate(sequence=seq, mpnn_scores={design_checkpoint: score})
                  for seq, score in designed]

    # 2. Multi-checkpoint scoring (D1) — SolubleMPNN as additional metadata.
    jobstore.update_job(job_id, stage="MPNN scoring", progress=0.35)
    seqs = [c.sequence for c in candidates]
    for ck in ("soluble_mpnn",):
        for c, s in zip(candidates, score_with_checkpoint(seqs, pdb_path, ck)):
            c.mpnn_scores[ck] = s

    # 3. ESM2 soft-floor / ranking signal (D2).
    jobstore.update_job(job_id, stage="ESM2 scoring", progress=0.5)
    for c, s in zip(candidates, esm2_scores(seqs)):
        c.esm2_score = s

    # 4. ESMFold QC (D2 hard gate).
    jobstore.update_job(job_id, stage="ESMFold QC", progress=0.65)
    for c in candidates:
        c.pdb, c.plddt = run_esmfold(c.sequence)
        c.rmsd_to_design = compute_ca_rmsd(c.pdb, pdb_path, chain_id)

    # 5. Gate + rank + trim to N.
    jobstore.update_job(job_id, stage="Ranking", progress=0.85)
    top = select_top_candidates(candidates, num_outputs=n_target)

    # 6. Write artifacts to GCS.
    jobstore.update_job(job_id, stage="Writing results", progress=0.95)
    results = _write_results(job_id, manifest, top, storage)

    jobstore.update_job(
        job_id, status=JobStatus.DONE.value, stage="Done", progress=1.0,
        result_uri=results["results_uri"],
    )
    return results


def _write_results(job_id, manifest, top: list[Candidate], storage) -> dict:
    fasta_lines, records = [], []
    for c in top:
        header = (f"candidate_{c.rank} score={c.composite_score:.3f} "
                  f"esm2={c.esm2_score:.3f} pLDDT={c.plddt:.1f} rmsd={c.rmsd_to_design:.2f}")
        fasta_lines.append(f">{header}\n{c.sequence}")
        pdb_name = f"candidate_{c.rank}.pdb"
        if c.pdb:
            storage.write_output(job_id, pdb_name, c.pdb.encode(), content_type="chemical/x-pdb")
        records.append({
            "rank": c.rank, "sequence": c.sequence, "composite_score": c.composite_score,
            "esm2_score": c.esm2_score, "plddt": c.plddt, "rmsd_to_design": c.rmsd_to_design,
            "mpnn_scores": c.mpnn_scores, "pdb": pdb_name if c.pdb else None,
        })
    storage.write_output(job_id, "candidates.fasta", ("\n".join(fasta_lines) + "\n").encode())
    results = {"job_id": job_id, "preset": manifest.preset.value, "count": len(top),
               "candidates": records, "params": manifest.params}
    results_uri = storage.write_output(job_id, "results.json", json.dumps(results, indent=2).encode(),
                                       content_type="application/json")
    results["results_uri"] = results_uri
    return results


def main() -> int:
    from proteinredesign import jobstore, storage
    from proteinredesign.manifest import JobManifest, JobStatus

    manifest_uri = os.getenv("PROTEINREDESIGN_MANIFEST_URI") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not manifest_uri:
        print("ERROR: set PROTEINREDESIGN_MANIFEST_URI (or pass the manifest gs:// URI as argv[1]).",
              file=sys.stderr)
        return 2

    # Best-effort job id from the URI so we can mark FAILED even if manifest load fails.
    job_id = _job_id_from_uri(manifest_uri)
    try:
        manifest = JobManifest.from_json(storage.read_manifest(manifest_uri))
        job_id = manifest.job_id
        with tempfile.TemporaryDirectory() as workdir:
            run_pipeline(manifest, workdir)
        return 0
    except Exception as exc:  # noqa: BLE001 — surface any failure to the dashboard
        traceback.print_exc()
        if job_id:
            try:
                jobstore.update_job(job_id, status=JobStatus.FAILED.value,
                                    error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
        return 1


def _job_id_from_uri(uri: str) -> str:
    """Extract <job_id> from gs://bucket/jobs/<job_id>/manifest.json (best effort)."""
    parts = uri.rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else ""


if __name__ == "__main__":
    raise SystemExit(main())
