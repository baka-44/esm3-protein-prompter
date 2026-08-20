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
# Enzyme scaffolding (#6) catalytic-fidelity gate (Å): reject designs whose catalytic
# residues drift more than this from the parent. [RF3-enz-1] mapping validated at the first
# GPU run (motif_rmsd ~0.7 Å on a clean scaffold), so this is ON by default. Env-overridable;
# set to "inf" to disable (report-only).
MOTIF_RMSD_GATE = float(os.getenv("PROTEINREDESIGN_MOTIF_RMSD_GATE", "1.5"))


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
    # RF3 presets (e.g. scaffold diversification): the generated backbone this
    # sequence was designed on — used as the self-consistency RMSD *reference*
    # (NOT the input; partial diffusion moves the backbone by design — D10.2).
    parent_backbone_path: str = ""
    # RF3 presets only: CA-RMSD of the final fold vs the ORIGINAL input backbone —
    # reported as a "diversity / drift-from-input" metric (NaN = not applicable).
    diversity_from_input: float = float("nan")
    # Enzyme scaffolding (#6) / Borrowed Bodies: all-atom RMSD of the CATALYTIC residues
    # between the fold and the parent — the catalytic-fidelity gate (NaN = not applicable).
    motif_rmsd: float = float("nan")
    # 1-based sequence positions of the catalytic residues in this candidate (where RF3
    # placed the unindexed motif) — used to compute motif_rmsd against the parent [RF3-enz-1].
    catalytic_pred_positions: list[int] = field(default_factory=list)

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
    motif_rmsd_gate: float = float("inf"),
) -> list[Candidate]:
    """
    Apply the QC gate (D2) and return the top `num_outputs` ranked candidates.

    1. Hard structural gate: pLDDT ≥ plddt_gate AND RMSD-to-design ≤ rmsd_gate. For enzyme
       scaffolding / Borrowed Bodies, additionally motif-RMSD ≤ motif_rmsd_gate — the
       catalytic-fidelity check (default inf = not applied). Candidates whose motif_rmsd is
       NaN are treated as failing when the gate is finite (motif QC could not be computed).
    2. ESM2 soft floor (D2): drop the clearly-unnatural bottom `esm2_drop_fraction`
       (only when there are more survivors than requested — never starves output,
       never hard-ranks by naturalness).
    3. Rank by a composite of the metadata scores (MPNN checkpoints + ESM2),
       min-max normalised across the surviving set (B3: metadata used for ranking).
    """
    motif_ok = (
        (lambda c: True) if motif_rmsd_gate == float("inf")
        else (lambda c: c.motif_rmsd == c.motif_rmsd and c.motif_rmsd <= motif_rmsd_gate)
    )
    passed = [
        c for c in candidates
        if c.plddt >= plddt_gate and c.rmsd_to_design <= rmsd_gate and motif_ok(c)
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
    Preset.SCAFFOLD_DIVERSIFICATION: "proteinmpnn",  # RF3 backbones → ProteinMPNN (D6)
}


def _fixed_residue_tokens(params: dict) -> str:
    """Author-numbered residue ids for LigandMPNN --fixed_residues, e.g. 'A67 A82' [MPNN-2]."""
    return " ".join(f"{r['chain_id']}{r['author_num']}" for r in params.get("fixed_residues", []))


def _mpnn_fixed_tokens(fixed_refs: list[str], repack_refs: list[str] | None = None) -> str:
    """
    LigandMPNN --fixed_residues string = the residues to KEEP, minus any REPACK set.

    General for every preset (FG1): #6 fixes the catalytic residues (repack empty); the
    Borrowed-Bodies path fixes the torso/mount but excludes the interface `repack_residues`
    so MPNN redesigns the junction chemistry (BB2). Order-preserving, de-duplicated.
    """
    drop = set(repack_refs or [])
    seen: set[str] = set()
    out: list[str] = []
    for r in fixed_refs:
        if r in drop or r in seen:
            continue
        seen.add(r)
        out.append(r)
    return " ".join(out)


def _mpnn_checkpoint_for(params: dict, default: str = "proteinmpnn") -> str:
    """Use LigandMPNN whenever a ligand/cofactor is present (#2, #6, BB); else the default."""
    return "ligand_mpnn" if params.get("ligand") else default


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


def compute_motif_rmsd(
    pred_pdb: str,
    ref_pdb_source,
    pred_seq_positions: list[int],
    ref_residue_keys: list[tuple[str, int]],
    atom_names: tuple[str, ...] = ("CA",),
    ref_chain: str | None = None,
) -> float:
    """
    All-atom RMSD over the CATALYTIC residues between the ESMFold prediction and the parent
    enzyme — the catalytic-fidelity metric for #6 / Borrowed Bodies.

    `pred_seq_positions` are 1-based positions of the catalytic residues in the (single-chain)
    prediction; `ref_residue_keys` are the (chain, author_num) of the same residues in the
    parent PDB; `atom_names` selects which atoms to superpose (default CA — [RF3-enz-2]:
    upgrade to the `select_fixed_atoms` tip atoms once the RF3 output atom naming is confirmed
    at the GPU run). Returns inf if no atoms could be matched.
    """
    from Bio.PDB import Superimposer
    from utils.pdb_utils import get_residues

    pred = get_residues(pred_pdb)  # single chain, in sequence order
    ref_all = get_residues(ref_pdb_source, chain_id=ref_chain)
    ref_by_key = {(r.get_parent().id, r.id[1]): r for r in ref_all}

    pred_atoms, ref_atoms = [], []
    for pos, key in zip(pred_seq_positions, ref_residue_keys):
        if not (1 <= pos <= len(pred)):
            continue
        pr = pred[pos - 1]
        rr = ref_by_key.get(tuple(key))
        if rr is None:
            continue
        for a in atom_names:
            if a in pr and a in rr:
                pred_atoms.append(pr[a])
                ref_atoms.append(rr[a])
    if not pred_atoms:
        return float("inf")
    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    return float(sup.rms)


def esm2_scores(sequences: list[str]) -> list[float]:
    """ESM2 pseudo-log-likelihood per sequence (soft-floor + ranking signal)."""
    from core.esm2_scorer import score_sequences
    return score_sequences(sequences, mode="pseudo")


# ── RF3 adapter (rf3-worker image only — foundry base, Python 3.12) ────────────
# RF3 outputs BACKBONES ONLY (no sequence); the downstream MPNN→ESMFold stages
# (above) design + QC the sequence. This adapter runs ONLY in the rf3-worker image
# (D11) — the mpnn-worker image (#1/#2) never imports/executes it.
#
# Interface VERIFIED 2026-08-19 against foundry 0.2.0 (`rfd3` / models/rfd3 docs):
#   [RF3-1] CLI: `rfd3 design out_dir=<dir> inputs=<cfg.json> ckpt_path=<file> \
#           diffusion_batch_size=<K> skip_existing=False` (hydra key=value overrides).
#   [RF3-2] Input JSON is {<run_name>: <InputSpecification>}. For whole-backbone
#           diversification the minimal spec is {"input": <pdb>, "partial_t": <Å>} —
#           NO contig needed (docs' minimal partial-diffusion example), which also
#           sidesteps RF3 contig author-numbering. partial_t is a NOISE MAGNITUDE in Å
#           (recommended 5–15), not a timestep count. (Multi-chain inputs diffuse all
#           chains; the config builder warns when the PDB has >1 chain — MVP scope.)
#   [RF3-3] Number of designs (K) = `diffusion_batch_size` (default 8).
#   [RF3-4] Checkpoint: single file `rfd3_latest.ckpt`, passed as `ckpt_path=`. Synced
#           from the GCS weights bucket subdir "rfdiffusion" via ensure_weights (A6).
#   [RF3-5] Output: one `*.cif.gz` (+ `.json`) per design under out_dir — converted to
#           PDB here for the downstream ProteinMPNN stage (which reads PDB).

_RFD3_CMD = os.getenv("RFD3_CMD", "rfd3")
_RFD3_CKPT_NAME = os.getenv("RFD3_CKPT_NAME", "rfd3_latest.ckpt")


def _rf3_weights_dir() -> str:
    from proteinredesign.storage import ensure_weights
    return ensure_weights("rfdiffusion")


def _cif_gz_to_pdb(cif_gz_path: str) -> str:
    """Decompress an RF3 `.cif.gz` design and write it back out as a `.pdb` [RF3-5]."""
    import gzip

    from Bio.PDB import MMCIFParser, PDBIO

    cif_path = cif_gz_path[:-3] if cif_gz_path.endswith(".gz") else cif_gz_path + ".cif"
    with gzip.open(cif_gz_path, "rt") as fi, open(cif_path, "w") as fo:
        fo.write(fi.read())
    structure = MMCIFParser(QUIET=True).get_structure("rf3", cif_path)
    pdb_path = cif_path.rsplit(".", 1)[0] + ".pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    return pdb_path


def run_rf3_design(
    input_spec: dict,
    num_designs: int,
    out_dir: str,
    run_name: str = "design",
) -> list[str]:
    """
    General RF3 all-atom design (the engine every RF3 preset drives — FG1).

    `input_spec` is the inner RF3 InputSpecification dict — the config builder assembles
    it, so the *same* adapter runs every topology:
      - partial diffusion (#8):  {"input": <pdb>, "partial_t": <Å>}
      - enzyme scaffold  (#6):   {"input": <pdb>, "unindex": ..., "select_fixed_atoms": ...,
                                   "ligand": ..., "length": "min-max"}
      - Borrowed Bodies (later):  {"input": <composite>, "contig": ..., "select_fixed_atoms": ...,
                                   "ligand": ...}   (indexed multi-segment; see borrowed_bodies_composer.md)

    Returns the generated design PDB paths (one per design), converted from RF3's native
    `.cif.gz`. Keys whose value is None/"" are dropped so a builder can pass a uniform dict.
    See the [RF3-*] verification notes above.
    """
    os.makedirs(out_dir, exist_ok=True)
    spec_inner = {k: v for k, v in input_spec.items() if v not in (None, "")}
    cfg_path = os.path.join(out_dir, "rfd3_input.json")
    with open(cfg_path, "w") as fh:
        json.dump({run_name: spec_inner}, fh)

    ckpt_path = os.path.join(_rf3_weights_dir(), _RFD3_CKPT_NAME)
    cmd = [
        _RFD3_CMD, "design",
        f"out_dir={out_dir}",
        f"inputs={cfg_path}",
        f"ckpt_path={ckpt_path}",
        f"diffusion_batch_size={int(num_designs)}",
        "skip_existing=False",
    ]
    subprocess.run(cmd, check=True)

    cifs = sorted(glob.glob(os.path.join(out_dir, "**", "*.cif.gz"), recursive=True))
    if not cifs:
        raise RuntimeError(f"RF3 produced no design outputs (*.cif.gz) in {out_dir} (spec={spec_inner}).")
    return [_cif_gz_to_pdb(c) for c in cifs]


def run_rf3_partial(input_pdb_path: str, partial_t: float, num_designs: int, out_dir: str) -> list[str]:
    """Partial-diffusion (#8) — thin wrapper over the general run_rf3_design."""
    return run_rf3_design(
        {"input": input_pdb_path, "partial_t": float(partial_t)},
        num_designs, out_dir, run_name="diversify",
    )


# ── Orchestration ─────────────────────────────────────────────────────────────

def _generate_rf3_candidates(
    manifest, input_pdb_path: str, workdir: str, design_checkpoint: str, jobstore, job_id: str
) -> list["Candidate"]:
    """
    RF3 partial diffusion (K backbones) → ProteinMPNN (M sequences each) → K×M
    candidates, each tagged with its parent backbone for dual-QC (D10.2/D10.3).
    """
    params = manifest.params
    partial_t = float(params["partial_t"])
    k = int(params["k"])
    m = int(params["m"])

    jobstore.update_job(job_id, stage="RF3 partial diffusion", progress=0.15)
    rf3_dir = os.path.join(workdir, "rf3")
    backbones = run_rf3_partial(
        input_pdb_path, partial_t=partial_t, num_designs=k, out_dir=rf3_dir
    )

    candidates: list[Candidate] = []
    for i, bb_path in enumerate(backbones, start=1):
        jobstore.update_job(
            job_id, stage=f"{design_checkpoint} design (backbone {i}/{len(backbones)})",
            progress=0.2 + 0.15 * (i / max(len(backbones), 1)),
        )
        # No fixed residues in this preset — the whole chain is redesigned on the
        # generated backbone (empty fixed-tokens string).
        designed = run_proteinmpnn(bb_path, "", n_seqs=m, checkpoint=design_checkpoint)
        for seq, score in designed:
            candidates.append(Candidate(
                sequence=seq, mpnn_scores={design_checkpoint: score},
                parent_backbone_path=bb_path,
            ))
    return candidates


_RESIDUE_REF = re.compile(r"([A-Za-z]+)(\d+)")


def _parse_residue_ref(ref: str) -> tuple[str, int] | None:
    """'A19' → ('A', 19). Returns None if unparseable."""
    m = _RESIDUE_REF.fullmatch(str(ref).strip())
    return (m.group(1), int(m.group(2))) if m else None


def _rf3_diffused_index_map(design_pdb_path: str) -> dict[str, str]:
    """
    RF3's per-design `diffused_index_map`: {input_residue_ref: output_residue_ref}, e.g.
    {"A6": "A19", "A17": "A61", "A32": "A48"} — where RF3 placed each (unindexed) motif
    residue in the output. Read from the sibling `.json` [RF3-enz-1, verified 2026-08-20].
    """
    json_path = design_pdb_path[:-4] + ".json" if design_pdb_path.endswith(".pdb") else ""
    try:
        with open(json_path) as fh:
            return json.load(fh).get("diffused_index_map", {}) or {}
    except Exception:  # noqa: BLE001
        return {}


def _rf3_enzyme_output_mapping(
    design_pdb_path: str,
    motif_keys: list[tuple[str, int]],
    identity_fallback: bool = False,
) -> tuple[list[str], list[int]]:
    """
    Map the parent's fixed-motif residues to their RF3 *output* placement via the design's
    `diffused_index_map`. Used by enzyme scaffolding (catalytic residues, unindexed) and motif
    scaffolding / inpainting (kept-block residues, indexed).

    Returns (output_residue_refs, seq_positions):
      - output_residue_refs — the motif residues in the DESIGN (e.g. ["A19",...]), for MPNN
        --fixed_residues (fix them at their real output positions, NOT the input ones).
      - seq_positions — 1-based sequence index of each motif residue in the design, ALIGNED to
        `motif_keys` (0 for any that didn't map), for motif-RMSD against the parent.

    `identity_fallback=True` (indexed motifs, e.g. inpainting): if a residue isn't in the map,
    assume it kept its input (chain, author_num) in the output. Unindexed motifs (enzyme) leave
    it False — identity would fix the wrong residue since RF3 renumbers them.
    """
    idx_map = _rf3_diffused_index_map(design_pdb_path)
    from utils.pdb_utils import get_residues

    try:
        residues = get_residues(design_pdb_path)
    except Exception:  # noqa: BLE001
        residues = []
    order = {(r.get_parent().id, r.id[1]): i for i, r in enumerate(residues, start=1)}

    out_refs: list[str] = []
    positions: list[int] = []
    for chain, num in motif_keys:
        out = idx_map.get(f"{chain}{num}")
        parsed = _parse_residue_ref(out) if out else None
        if parsed is None and identity_fallback and (chain, num) in order:
            parsed, out = (chain, num), f"{chain}{num}"
        if parsed is not None:
            out_refs.append(out)
            positions.append(order.get(parsed, 0))
        else:
            positions.append(0)
    return out_refs, positions


def _persist_rf3_outputs(job_id: str, out_dir: str, storage) -> None:
    """
    Save the raw RF3 design outputs (`*.cif.gz` all-atom structures + `*.json` metadata) to
    GCS under the job prefix. These carry the ligand + the motif conditioning annotations the
    ESMFold-refold PDBs lose, and are the true generative artefact. Best-effort (never fails a job).
    """
    try:
        for path in sorted(glob.glob(os.path.join(out_dir, "**", "*.cif.gz"), recursive=True) +
                           glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True)):
            name = os.path.basename(path)
            ct = "application/gzip" if name.endswith(".cif.gz") else "application/json"
            storage.write_output(job_id, f"rf3/{name}", open(path, "rb").read(), content_type=ct)
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN: could not persist RF3 outputs: {exc}")


def _generate_enzyme_candidates(
    manifest, input_pdb_path: str, workdir: str, jobstore, job_id: str
) -> list["Candidate"]:
    """
    Enzyme active-site scaffolding (#6): RF3 all-atom scaffolds K new bodies around the
    unindexed catalytic motif; ProteinMPNN/LigandMPNN designs M sequences on each, KEEPING
    the catalytic residues fixed. K×M candidates, each tagged with its parent backbone and
    the catalytic residues' output positions (for motif-RMSD QC).
    """
    params = manifest.params
    k, m = int(params["k"]), int(params["m"])
    checkpoint = _mpnn_checkpoint_for(params)  # LigandMPNN if a cofactor is present
    catalytic_keys = [(r["chain_id"], r["author_num"]) for r in params.get("catalytic_residues", [])]
    repack = list(params.get("repack_residues", []))  # empty for #6; the BB path (BB2) fills it

    # RF3 all-atom scaffold spec (assembled by the config builder — code, not the LLM, B5).
    spec = {
        "input": input_pdb_path,
        "unindex": params.get("unindex"),
        "select_fixed_atoms": params.get("select_fixed_atoms"),
        "ligand": (params["ligand"]["resname"] if params.get("ligand") else None),
        "length": params.get("length"),
    }
    jobstore.update_job(job_id, stage="RF3 all-atom scaffolding", progress=0.15)
    rf3_dir = os.path.join(workdir, "rf3")
    designs = run_rf3_design(spec, num_designs=k, out_dir=rf3_dir, run_name="enzyme")
    from proteinredesign import storage as _storage
    _persist_rf3_outputs(job_id, rf3_dir, _storage)  # save all-atom scaffolds + motif annotations

    candidates: list[Candidate] = []
    for i, design_pdb in enumerate(designs, start=1):
        jobstore.update_job(
            job_id, stage=f"{checkpoint} design (scaffold {i}/{len(designs)})",
            progress=0.2 + 0.15 * (i / max(len(designs), 1)),
        )
        # Fix the catalytic residues at their REAL output positions (RF3 renumbers the
        # unindexed motif) — from the design's diffused_index_map. Same map gives the
        # sequence positions used for motif-RMSD.
        out_refs, cat_positions = _rf3_enzyme_output_mapping(design_pdb, catalytic_keys)
        fixed_tokens = _mpnn_fixed_tokens(out_refs, repack)
        designed = run_proteinmpnn(design_pdb, fixed_tokens, n_seqs=m, checkpoint=checkpoint)
        for seq, score in designed:
            candidates.append(Candidate(
                sequence=seq, mpnn_scores={checkpoint: score},
                parent_backbone_path=design_pdb, catalytic_pred_positions=cat_positions,
            ))
    return candidates


def _generate_motif_candidates(
    manifest, input_pdb_path: str, workdir: str, jobstore, job_id: str
) -> list["Candidate"]:
    """
    Motif scaffolding / inpainting (#3): RF3 generates the bridges between the indexed kept
    blocks (multi-segment contig), producing K structures; ProteinMPNN designs M sequences on
    each, KEEPING the kept-block residues fixed and designing only the bridges. K×M candidates.

    Unlike enzyme scaffolding the motif is *indexed* (kept blocks keep their coordinates and
    register), so the output→input residue map is usually identity — with the diffused_index_map
    consulted first and identity as the fallback.
    """
    params = manifest.params
    k, m = int(params["k"]), int(params["m"])
    checkpoint = _mpnn_checkpoint_for(params)  # ProteinMPNN (no ligand for #3)
    motif_keys = [(r["chain_id"], r["author_num"]) for r in params.get("motif_residues", [])]
    repack = list(params.get("repack_residues", []))  # empty for #3; Borrowed Bodies (BB2) fills it

    # select_fixed_atoms is None for #3 (backbone-only, contig fixes it) and set for Borrowed
    # Bodies (the mount's all-atom pins) — run_rf3_design drops None keys.
    spec = {"input": input_pdb_path, "contig": params.get("contig"),
            "select_fixed_atoms": params.get("select_fixed_atoms")}
    jobstore.update_job(job_id, stage="RF3 inpainting (bridges)", progress=0.15)
    rf3_dir = os.path.join(workdir, "rf3")
    designs = run_rf3_design(spec, num_designs=k, out_dir=rf3_dir, run_name="motif")
    from proteinredesign import storage as _storage
    _persist_rf3_outputs(job_id, rf3_dir, _storage)

    candidates: list[Candidate] = []
    for i, design_pdb in enumerate(designs, start=1):
        jobstore.update_job(
            job_id, stage=f"{checkpoint} design (scaffold {i}/{len(designs)})",
            progress=0.2 + 0.15 * (i / max(len(designs), 1)),
        )
        out_refs, motif_positions = _rf3_enzyme_output_mapping(
            design_pdb, motif_keys, identity_fallback=True
        )
        fixed_tokens = _mpnn_fixed_tokens(out_refs, repack)
        designed = run_proteinmpnn(design_pdb, fixed_tokens, n_seqs=m, checkpoint=checkpoint)
        for seq, score in designed:
            candidates.append(Candidate(
                sequence=seq, mpnn_scores={checkpoint: score},
                parent_backbone_path=design_pdb, catalytic_pred_positions=motif_positions,
            ))
    return candidates


def run_pipeline(manifest, workdir: str) -> dict:
    """
    Execute preset #1 end-to-end and return a results dict (also written to GCS).

    `manifest` is a proteinredesign.manifest.JobManifest. Firestore status is updated at
    each stage so the dashboard can show live progress (B7/B8).
    """
    from proteinredesign import jobstore
    from proteinredesign import storage
    from proteinredesign.manifest import JobStatus, Preset

    job_id = manifest.job_id
    params = manifest.params
    chain_id = params.get("chain_id")
    fixed_tokens = _fixed_residue_tokens(params)
    n_target = manifest.num_outputs
    n_generate = max(n_target * OVERGEN_FACTOR, n_target)
    is_rf3 = manifest.requires_rfdiffusion()
    is_enzyme = manifest.preset == Preset.ENZYME_ACTIVE_SITE
    # Motif scaffolding (#3) and Borrowed Bodies both run the indexed multi-segment path.
    is_motif = manifest.preset in (Preset.MOTIF_SCAFFOLDING, Preset.BORROWED_BODIES)
    has_motif_qc = is_enzyme or is_motif  # motif-RMSD fidelity check applies

    # 0. Fetch the input PDB.
    jobstore.update_job(job_id, status=JobStatus.RUNNING.value, stage="Preparing inputs", progress=0.05)
    pdb_path = os.path.join(workdir, "input.pdb")
    storage.download_to_path(manifest.pdb_uri, pdb_path)

    # 1. Generate candidates. Three shapes (D11 / FG1):
    #    - MPNN-only presets (#1/#2): design directly on the fixed input backbone.
    #    - Scaffold diversification (#8): RF3 partial diffusion → K backbones → MPNN(M).
    #    - Enzyme active-site scaffolding (#6): RF3 all-atom scaffolds K bodies around the
    #      catalytic motif → MPNN(M) keeping the catalytic residues fixed.
    if is_enzyme:
        candidates = _generate_enzyme_candidates(manifest, pdb_path, workdir, jobstore, job_id)
    elif is_motif:
        candidates = _generate_motif_candidates(manifest, pdb_path, workdir, jobstore, job_id)
    elif manifest.preset == Preset.SCAFFOLD_DIVERSIFICATION:
        candidates = _generate_rf3_candidates(
            manifest, pdb_path, workdir, _DESIGN_CHECKPOINT[manifest.preset], jobstore, job_id
        )
    elif manifest.preset in _DESIGN_CHECKPOINT:
        design_checkpoint = _DESIGN_CHECKPOINT[manifest.preset]
        jobstore.update_job(job_id, stage=f"{design_checkpoint} design", progress=0.2)
        designed = run_proteinmpnn(pdb_path, fixed_tokens, n_seqs=n_generate,
                                   checkpoint=design_checkpoint)
        candidates = [
            Candidate(sequence=seq, mpnn_scores={design_checkpoint: score},
                      parent_backbone_path=pdb_path)
            for seq, score in designed
        ]
    else:
        raise NotImplementedError(
            f"Preset {manifest.preset.value} not yet implemented in the worker."
        )

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

    # 4. ESMFold QC (D2 hard gate). The self-consistency RMSD reference depends on
    # the preset (D10.2): for MPNN-only presets the input backbone is held fixed, so
    # RMSD-to-input IS the self-consistency metric. For RF3 presets the backbone was
    # regenerated, so we measure against each candidate's OWN generated backbone, and
    # separately report RMSD-to-input as the "diversity / drift-from-input" metric.
    jobstore.update_job(job_id, stage="ESMFold QC", progress=0.65)
    # Motif residues for the fidelity check: catalytic residues (#6) or kept-block residues (#3).
    motif_keys = [
        (r["chain_id"], r["author_num"])
        for r in (params.get("catalytic_residues") or params.get("motif_residues") or [])
    ]
    for c in candidates:
        c.pdb, c.plddt = run_esmfold(c.sequence)
        ref_path = c.parent_backbone_path or pdb_path
        # Generated RF3 backbones are single-chain (chain from the contig); compare
        # against all chains there. Input-referenced presets use the chosen chain.
        ref_chain = None if is_rf3 else chain_id
        c.rmsd_to_design = compute_ca_rmsd(c.pdb, ref_path, ref_chain)
        # Diversification-only: drift of the fold from the ORIGINAL input backbone.
        if manifest.preset == Preset.SCAFFOLD_DIVERSIFICATION:
            c.diversity_from_input = compute_ca_rmsd(c.pdb, pdb_path, chain_id)
        # Enzyme / motif scaffolding: fidelity of the fixed motif vs the parent (motif-RMSD).
        if has_motif_qc and c.catalytic_pred_positions:
            c.motif_rmsd = compute_motif_rmsd(
                c.pdb, pdb_path, c.catalytic_pred_positions, motif_keys, ref_chain=chain_id
            )

    # 5. Gate + rank + trim to N. Enzyme/motif scaffolding add the motif-RMSD fidelity gate.
    jobstore.update_job(job_id, stage="Ranking", progress=0.85)
    top = select_top_candidates(
        candidates, num_outputs=n_target,
        motif_rmsd_gate=(MOTIF_RMSD_GATE if has_motif_qc else float("inf")),
    )

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
        if c.diversity_from_input == c.diversity_from_input:  # not NaN → diversification
            header += f" diversity={c.diversity_from_input:.2f}"
        if c.motif_rmsd == c.motif_rmsd:  # not NaN → enzyme scaffolding
            header += f" motif_rmsd={c.motif_rmsd:.2f}"
        fasta_lines.append(f">{header}\n{c.sequence}")
        pdb_name = f"candidate_{c.rank}.pdb"
        if c.pdb:
            storage.write_output(job_id, pdb_name, c.pdb.encode(), content_type="chemical/x-pdb")
        records.append({
            "rank": c.rank, "sequence": c.sequence, "composite_score": c.composite_score,
            "esm2_score": c.esm2_score, "plddt": c.plddt, "rmsd_to_design": c.rmsd_to_design,
            "diversity_from_input": c.diversity_from_input, "motif_rmsd": c.motif_rmsd,
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
