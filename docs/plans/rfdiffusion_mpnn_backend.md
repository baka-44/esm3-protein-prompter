# RFdiffusion + MPNN Generation Backend — Design Notes

> **Status:** Design-notes / decisions log — **not a build plan yet.** The full
> tool-integration plan (adding RFdiffusion as a user-toggleable generation backend) is
> deferred by request. This doc captures established facts and design decisions as they
> emerge, for later recall. Started 2026-07-18.

## Context / motivation

The team is skeptical about relying on **ESM3** (served via the EvolutionaryScale **Forge
API**, which requires a **paid commercial license**) for commercially relevant protein and
enzyme engineering. We are evaluating the **RFdiffusion + ProteinMPNN family** as a
commercial-clean, **self-hostable** alternative/complement for the generation step — and
eventually exposing it as a backend the user can toggle against ESM3 in the tool.

## Established facts (reference)

**Pipeline shape.** RFdiffusion outputs **backbones only** (no sequence), so the pipeline is
always three stages:
```
RFdiffusion (backbone) → ProteinMPNN/LigandMPNN (sequence) → ESMFold/AF2 (QC self-consistency)
```

**When RFdiffusion is NOT needed.** For **fixed-backbone redesign** — keep the existing fold,
pin known functional residues, novelize the rest (e.g. the **sweet-protein redesign** task:
retain receptor-binding/sweetness residues, redesign the scaffold) — RFdiffusion is
unnecessary. **ProteinMPNN alone** (whole PDB as fixed backbone + `fixed_positions`) is the
right tool; QC fold still applies. RFdiffusion is for *new geometry*: de novo, motif
scaffolding, binders, and enzyme active-site scaffolding (RFdiffusion2 / RFdiffusionAA).

**Caveat for fixed-residue redesign:** fixing residue *identity* ≠ preserving its
*microenvironment*; redesigned neighbors can shift local packing/electrostatics. Consider
pinning second-shell residues and treating functional surfaces as patches, not point residues.

**Licensing — all commercial-clean and self-hostable:**
| Component | License | Commercial |
|---|---|---|
| RFdiffusion v1 | BSD | ✅ |
| **RFdiffusion3 (MVP choice)** | CC BY 4.0 (attribution) | ✅ |
| RFdiffusion2 / AA | verify repo LICENSE | ✅ |
| ProteinMPNN / SolubleMPNN / LigandMPNN | MIT (code + weights) | ✅ |
| ESMFold (local) | MIT | ✅ |
| AlphaFold2 params (local) | CC BY 4.0 (attribution) | ✅ |
| ESM3 / ESMFold via Forge API | Paid commercial license | ⚠️ |

**MPNN family = one codebase.** The **LigandMPNN repo** is the unified codebase; ProteinMPNN,
SolubleMPNN, and LigandMPNN are selectable **model checkpoints** (`--model_type` / checkpoint),
not three separate tools.

---

## Task taxonomy

The task the scientist picks determines **(a)** whether RFdiffusion is even needed, **(b)** which
MPNN checkpoint, and **(c)** which QC protocol. The tool should offer these as **presets** that
compile to the correct contig map / MPNN config / QC protocol — never expose raw contig syntax.

### Routing (engineering view)
| # | Task | RFdiffusion? | Sequence design | QC |
|---|---|---|---|---|
| 1 | Fixed-backbone redesign | **No — MPNN only** | ProteinMPNN / SolubleMPNN (`fixed_positions`) | ESMFold/AF2 |
| 2 | Ligand/cofactor-aware redesign | **No — MPNN only** | LigandMPNN (ligand atoms) | ESMFold/AF2 |
| 3 | Motif scaffolding | Yes (contigs, full-noise) | ProteinMPNN | ESMFold/AF2 |
| 4 | De novo generation | Yes (unconditional) | ProteinMPNN | ESMFold/AF2 |
| 5 | Binder design | Yes (binder mode + hotspots) | ProteinMPNN | AF2 "initial guess" |
| 6 | Enzyme active-site scaffolding | Yes (RFdiffusion2 / AA) | LigandMPNN | ESMFold/AF2 + Rosetta ΔΔG |
| 7 | Scaffold condensation | Yes (motif-scaffolding, shorter contig) | ProteinMPNN | ESMFold/AF2 |
| 8 | Scaffold diversification | Yes (partial diffusion) | ProteinMPNN | ESMFold/AF2 |
| 9 | Symmetric oligomers | Yes (symmetry) | ProteinMPNN (tied positions) | AF2-multimer |

> **#7 vs #8 are distinct mechanisms** (previously conflated). **Condensation** shortens a
> protein by pinning key residues as a motif and generating a *shorter* scaffold around them
> (length changes) — mechanically a motif-scaffolding run. **Diversification** partially noises
> the *whole* existing backbone to a chosen timestep and re-denoises (same length, same fold,
> controllably perturbed) — mechanically partial diffusion. Partial diffusion cannot change length.

### Scientist-facing descriptions (UI copy)
- **Fixed-backbone redesign** — Keep the input backbone exactly as-is and design new sequences
  that fold into it, optionally pinning specified residues. No new geometry is generated;
  ProteinMPNN (or SolubleMPNN) redesigns the sequence on the fixed backbone.
- **Ligand/cofactor-aware redesign** — Redesign the sequence on a fixed backbone that carries a
  bound small molecule, metal, or nucleic acid. LigandMPNN conditions on the ligand's atomic
  context so pocket-lining residues respect the binding chemistry.
- **Motif scaffolding** — Hold a functional motif's backbone geometry fixed and generate an
  entirely new protein around it from full noise. Scaffold length and the motif's position are
  set by the contig spec.
- **De novo generation** — Generate novel backbones of a target length with no template,
  optionally under symmetry or secondary-structure / fold conditioning. Unconditional diffusion
  followed by sequence design.
- **Binder design** — Design a de novo binder against a target protein, steering the interface
  toward specified hotspot residues. RFdiffusion builds the binder backbone against the target;
  validated with the AF2 "initial guess" protocol.
- **Enzyme active-site scaffolding** — Build a new enzyme scaffold around a catalytic-site
  geometry / substrate description using RFdiffusion2 (atom-level active sites), then design the
  sequence with LigandMPNN. The route for de novo catalytic function.
- **Scaffold condensation** — Shorten a protein while preserving the backbone geometry of
  specified key residues. Mechanically a motif-scaffolding run: the key residues are pinned as
  the motif and a shorter scaffold is generated around them (total length changes).
- **Scaffold diversification** — Generate controlled variants of an existing structure by
  partially noising the whole backbone to a chosen timestep and re-denoising. Same length and
  overall fold; the partial-noise level (`partial_T`) sets how divergent the variants are.
- **Symmetric oligomers** — Design symmetric assemblies (cyclic / dihedral / etc.) where
  identical subunits are generated under a symmetry constraint; ProteinMPNN designs the sequence
  with positions tied across subunits.

---

## Decisions

### D1 — MPNN multi-model scoring of every candidate
Score **every generated sequence** under all applicable MPNN checkpoints, and optionally ESM2:
- **ProteinMPNN score** → foldability / fit to the backbone.
- **SolubleMPNN score** → solubility bias (surface hydrophobicity).
- **LigandMPNN score** → ligand-context fit — **only when a ligand/cofactor/metal/nucleic acid
  is present in the input PDB** (adds nothing for ligand-free redesign like sweet proteins).
- *(optional)* **ESM2 pseudo-likelihood** → sequence "naturalness" (usage governed by **D2**).

**Mechanism — score-based consensus, NOT identical-sequence intersection.** The models are
stochastic samplers; identical full sequences essentially never overlap. Consensus is expressed
as agreement across **scores/positions**, not duplicate sequences.

**Usage — keep configurable (do not hard-wire one):**
- (a) **Filter** candidates on the scores *before* the QC fold step, or
- (b) **Metadata** — carry the scores through and **display them alongside the sequences that
  pass QC**.
Default TBD; expose as a toggle.

**Rationale.** A cheap in-silico prefilter / information signal. SolubleMPNN score is a
legitimate **partial** solubility/expressibility proxy; combined with AF2/ESMFold
self-consistency (foldability) and ESM2 (naturalness) it gives a multi-axis triage.
**Caveat:** these are blind to secretion-specific drivers (signal-peptide processing,
glycosylation, chaperones, proteases, codon usage) — **not** a true expressibility predictor.
It complements, and does not replace, the Sequential Interaction Model
([docs/plans/sequential_interaction_model.md](sequential_interaction_model.md)) and wet-lab
validation.

### D2 — QC filtering roles (ESM2 naturalness = soft floor + metadata)
Defines how the QC step decides which candidates to keep.
- **Primary hard gate: AF2/ESMFold self-consistency** (RMSD-to-design + pLDDT/pAE) — the
  trusted foldability criterion; this is what makes the keep/discard call.
- **ESM2 pseudo-log-likelihood (naturalness): soft floor + metadata, NOT a hard top-filter.**
  - Reject only the **clearly-unnatural tail** (guards against degenerate/pathological MPNN
    outputs — low-complexity stretches, unfoldable drift).
  - Otherwise carry as **metadata / ranking signal** alongside the SolubleMPNN scores; never a
    strict "keep top N% by naturalness" cut.
  - **Scale aggressiveness with how de novo the task is:** firmer for fixed-backbone redesign
    near a natural fold (e.g. sweet protein), softer for de novo / novel folds.

**Rationale.** ESM2 PLL is *sequence-based* and orthogonal to the *structure-based* fold
check, so it adds independent signal and cheaply catches pathological sequences. But
naturalness rewards *resembling known proteins*, so **hard-filtering biases against the novelty
that is the design goal** — a genuinely good novel design can legitimately score low. Precedent:
the existing tool already uses ESM2 as a *ranking* component in `composite_score`, not a gate.

### D3 — Frontend: engine-select first
The UI asks the scientist to **choose the generation engine up front** (ESM3 vs
RFdiffusion/MPNN), then shows an engine-tailored input panel. Within the RFdiffusion engine, the
**task-taxonomy presets** (above) drive the specific inputs.
- **Rationale:** the two engines have genuinely different capabilities, inputs, outputs, and
  execution models (ESM3 = synchronous sequence generation; RFdiffusion = async multi-stage
  backbone→sequence→QC). An explicit up-front choice makes each tool's capabilities legible to
  the scientist and keeps backend routing clean.
- Chosen over "task-first shared core" and "fully seamless NL" (higher mis-routing risk given
  how differently the engines behave).

### D4 — Cost governance / auto-teardown (self-host path)
Guardrails so a forgotten self-hosted backend can't quietly leak money — chosen conservatively
(no data loss, no automatic billing kill):
- **Scale-to-zero execution** — jobs on GCP Batch / Cloud Run Jobs / Vertex custom jobs; **no**
  standing GPU VM and **no** standing GKE control plane. Idle GPU cost = $0. (See *Compute &
  cost* below.)
- **Keep GCS artifacts — no lifecycle auto-deletion.** Buckets retain fold/output data and model
  weights. The (small) recurring storage cost is accepted in exchange for not losing data.
- **No auto-disable billing** (explicitly declined). Instead, keep the whole backend in a
  **dedicated GCP folder/project** so the billing/invoice shows as a **single auditable line
  item**.
- **6-month review timer** — Cloud Scheduler → Cloud Function fires at 6 months and **sends an
  email reminder** to review the backend (renew or manually tear down). **Reminder only — no
  automatic teardown and no data deletion.** Any teardown is a manual decision (script it via
  Terraform/gcloud so re-provisioning is one command).

### D5 — Backend architecture (self-host)
Resolutions from the pending-questions round:
- **A1 — Compute: self-host** (not third-party API). Resolves the earlier self-host-vs-API question.
- **A2 — Execution engine: Cloud Run Jobs (GPU/L4) for MVP** ✓ **confirmed**;
  add **GCP Batch** later for large parallel campaigns (array jobs, A100/Spot). Vertex custom
  jobs only if we later adopt training/fine-tuning.
- **A3 — Job state: Firestore** — durable, survives frontend restarts, supports concurrent users;
  records **tagged by user email**. (Not GCS status files / in-memory.)
- **A4 — Container: single monolith worker** (RFdiffusion + MPNN + ESMFold) for MVP ✓ **confirmed**;
  refactor to per-stage containers later for GPU
  right-sizing / parallel QC fan-out at scale. Pairs with A2 as two tiers: monolith+Cloud Run
  Jobs (MVP) → per-stage+Batch (scale-out).
- **A5 — QC folder: ESMFold** (local, no MSA infra) — confirmed over AF2.
- **A6 — Weights: mounted from GCS** (not baked into image). *Revises setup step 8.* Trade-off:
  smaller image + update weights without rebuild, at the cost of slower cold-start (network load);
  mitigate with a persistent-disk / gcsfuse cache.
- **A7 — Auth/multi-user: existing `@phyx44.com` OAuth; jobs user-linked** (tagged by email in
  Firestore). No per-user quotas or cost attribution.

### D6 — MVP task scope + RFdiffusion version (B1, B2)
**MVP presets:** #1 Fixed-backbone redesign · #2 Ligand-aware redesign · #3 Motif scaffolding ·
#6 Enzyme active-site scaffolding · #8 Scaffold diversification.

**RFdiffusion version: RFdiffusion3 (RFD3).** Of the diffusion-requiring presets (#3, #6, #8),
**only RF3 supports all three in one model** — v1 is protein-only (no ligands → can't do #6);
RF2 is enzyme-specialized and lacks unconditional/binder/symmetry and general protein partial
diffusion (#8). RF3 also is **best at enzyme scaffolding** (90% of the 41-site benchmark, beating
RF2), **~10× faster than RF2** (major cost lever), and **CC BY 4.0 — commercial use with
attribution**. Keeps the A4 monolith simple (one diffusion model).
- **Caveat:** RF3 is new (Dec 2025, via Rosetta Commons *foundry*) — evolving API, less
  battle-testing than v1; **verify the foundry repo LICENSE before commercial deploy.**

**Per-preset model stack (MVP):**
| Preset | RFdiffusion | Sequence design | QC |
|---|---|---|---|
| #1 Fixed-backbone redesign | — | ProteinMPNN / SolubleMPNN | ESMFold |
| #2 Ligand-aware redesign | — | LigandMPNN | ESMFold |
| #3 Motif scaffolding | RF3 | ProteinMPNN | ESMFold |
| #6 Enzyme active-site | RF3 | LigandMPNN | ESMFold |
| #8 Scaffold diversification | RF3 (partial diffusion) | ProteinMPNN | ESMFold |

### D7 — MVP feature decisions (B3, B4, B5, B7, B8)
- **B3 — MPNN scoring: metadata by default**, used for **ranking** (same role ESM2
  log-likelihood plays in the existing tool's `composite_score`) — not a filter by default.
  (Consistent with D1's "keep configurable"; the shipped default is metadata/ranking.)
- **B4 — Output count: 10 final QC-passed, ranked outputs per task** (MVP). The pipeline must
  **over-generate upstream** so ≥10 survive the ESMFold QC gate, then rank by MPNN/ESM2 scores.
- **B5 — Config building: deterministic per-preset builder is the core.** Structured-input UI
  (PDB + residue picker validated against the PDB + length / diversify controls) → **code**
  compiles the contig map / atomic-motif spec / `partial_T`. Optional Claude layer parses
  free-text/messy input into the structured spec **only**. **Rule: code, never the LLM, generates
  RFdiffusion syntax** (per the ESM3 Haiku-template bug). Validate each catalytic residue# ↔ AA
  against the PDB (author-vs-sequential numbering). Mirrors existing
  `build_template_from_structured` + `nl_parser` split.
- **B7 — Persistent job history** surviving frontend refresh / re-login (Firestore-backed per D5,
  user-tagged per A7).
- **B8 — No push notifications.** The history/dashboard shows live job status with a clear,
  visible **"done / completed"** indicator; users check back (matches the async poll model).

### D8 — Ligand input + #1 default checkpoint (B6)
- **B6 — Ligand input: HETATM records within the uploaded complex PDB.** Scientists upload the
  protein+ligand complex they already have (UniProt / X-ray); the ligand arrives with **real
  coordinates in the protein's frame** — ideal for LigandMPNN (#2) and RF3 (#6). Easiest (zero
  extra effort) and best data quality. **SMILES / separate-ligand-file deferred** (they lack
  aligned coordinates → would need docking/placement; not needed for MVP).
  - Builder nuance: experimental PDBs carry other HETATMs (waters `HOH`, ions, buffer). The
    config builder **auto-detects HETATM groups, filters waters/ions/additives by default, and
    asks the scientist to confirm which group is the ligand** (folds into the B5 input flow).
- **Default MPNN checkpoint for #1 fixed-backbone redesign: ProteinMPNN** (SolubleMPNN available
  as an option / for scoring per D1).

### D10 — Scaffold diversification preset (#8 / UI "Scaffold diversification") — build decisions
First RFdiffusion-requiring preset to ship; chosen deliberately as the **lowest-risk way to stand
up RF3 in the worker** (partial diffusion = one input backbone + one noise dial, no motif-placement
or residue-index mapping). Decisions locked with the user (2026-08-19):

- **D10.1 — RF3 partial diffusion** (not classic RFdiffusion). Consistent with D6 (one diffusion
  model across #3/#6/#8). **✅ VERIFIED 2026-08-19** (foundry `production` branch, model `rfd3`):
  - **Partial diffusion IS supported.** Parameter is **`partial_t`** — note it is a *noise
    magnitude in Å* (recommended **5.0–15.0 Å**; start ~2 Å and increase), **not** a timestep
    count, and it does *not* change `num_timesteps`. More Å → more diversity. Input is JSON/YAML.
  - **Minimal whole-protein-diversification config confirmed** — single chain, no motif, no
    binder/hotspots, nothing fixed (everything unfixes by default):
    `{"input": "prot.pdb", "contig": "A1-<L>", "partial_t": <Å>}`; run `rfd3 design inputs=cfg.json`.
    This is the *simplest* RF3 mode — validates the "lowest-risk RF3 bring-up" rationale for #8.
  - **⚠️ Known caveat (foundry issue #153, closed, no maintainer reply):** RF3 partial diffusion
    lost binding-site constraints on a **binder** (spurious transmembrane helices). That failure
    mode is **hotspot/binder-specific** — our no-hotspot whole-protein diversification avoids it —
    but confirm empirically on the first real run (part of deploy task).
  - **License: BSD 3-Clause** (repo `LICENSE.md`, SPDX `BSD-3-Clause`; no separate weights license
    or non-commercial/academic carve-out found in `models/rfd3/`). **This supersedes the CC BY 4.0
    assumption in D6/D9** — RF3 is *more* permissive than assumed (BSD needs no attribution beyond
    retaining the copyright notice in redistributed source). Residual: weights download via
    `foundry install rfd3` from a separate checkpoint host — confirm no click-through terms at
    install; keep the credits footnote regardless (good practice).
  - **🔴 Integration finding — RF3 does NOT drop into the current worker container.** `rc-foundry`
    requires **Python ≥ 3.12** + `torch ≥ 2.2` + NVIDIA `cuequivariance_*_cu12` ops; our worker
    image is **Python 3.10** (Ubuntu 22.04 default). **Recommended container-strategy change:** base
    `Dockerfile.worker` on the official **`rosettacommons/foundry`** image (bundles RF3 + Py3.12 +
    weights; "slim" tag can fetch weights) and layer LigandMPNN + ESMFold + our pipeline on top —
    rather than pip-installing RF3 into the existing 3.10 image. Keeps the A4 monolith. **Must
    re-verify** the MPNN/ESMFold/ESM2 stack (currently 3.10) runs under 3.12 on the new base.
    Weights → pre-download with `foundry install rfd3` and push to `gs://…-weights/rfdiffusion/`
    (same GCS pattern as `esmfold/`); exact checkpoint size TBD empirically (docs don't state).
- **D10.2 — Dual QC reference.** The self-consistency hard gate (pLDDT + RMSD) is measured against
  **each candidate's own RF3-generated backbone** — NOT the input (partial diffusion intentionally
  moves the backbone, so RMSD-to-input is large by design and would reject everything). Separately,
  **RMSD-to-input is reported as a "diversity / drift-from-input" metric** — a first-class results
  column, since controllable drift *is* this preset's scientific signal. Requires the pipeline to
  track candidate→parent-backbone parentage and a per-preset choice of RMSD reference (contained
  change to `run_pipeline`; all RF3 presets will need this branch).
- **D10.3 — Two-level fan-out via user sliders (replaces the fixed num_outputs + OVERGEN_FACTOR
  for this preset).** `K` = number of RF3 backbones (slider **1–N**, N=10 max); `M` = MPNN
  sequences per backbone (slider **1–10**); total raw designs = K×M, all folded through ESMFold
  QC, final results = all QC-passed, ranked (cap K×M). Plus a **diversity slider** → RF3
  `partial_T` (low/med/high noise level).
- **D10.4 — UI surface:** one input PDB + K slider + M slider + diversity slider + **optional**
  chain selector. **No fixed residues** in the base version (that's motif scaffolding #3 territory).
- **D10.5 — Compute guardrail (open, proposed).** Worst case K=10×M=10 = **100 ESMFold folds** +
  10 RF3 partial runs; on L4 that risks approaching/exceeding the **3600 s** job timeout for
  larger proteins. **Proposed:** surface a UI time-estimate warning above a K×M threshold (mirror
  the ESM3 app's `n_candidates>20` warning) and/or a soft cap on K×M; bump the Cloud Run Job
  timeout only if needed. Confirm approach at implementation.

### D11 — Container architecture: TWO images / TWO jobs (supersedes A4 monolith for the RF3 tier)
**Decided 2026-08-19.** A4's single-monolith choice is **revised** in light of D10.1's finding that
RF3 requires **Python 3.12** while the working #1/#2 stack is **Python 3.10**. Rather than migrate
the proven prod presets into a shared 3.12 image, split by preset family:

- **`mpnn-worker`** (existing image, Python 3.10 — MPNN + ESMFold + ESM2): serves #1, #2.
  **Never rebuilt for RF3 work** → zero regression risk to working prod presets.
- **`rf3-worker`** (new image, Python 3.12 on the official `rosettacommons/foundry` base —
  RF3 + MPNN + ESMFold + ESM2): serves #8 (and later #3/#6). **Self-contained**: runs the whole
  `RF3 → MPNN → ESMFold` pipeline in **one job execution** (no cross-container chaining — the clean
  `submit→job→done` model is preserved).

**Why two Cloud Run Jobs (not one job routing by a preset var):** a Cloud Run Job's **image is
fixed in its template** — per-execution `Overrides` can set env/args/resources/timeout but **not the
image**. So one job ⟺ one image; two images necessarily means two jobs. The preset→image routing
therefore lives at the **frontend** (`submit.py` maps preset → job name, like `_DESIGN_CHECKPOINT`
maps preset → checkpoint today). The second job is cheap: **scales to zero ($0 idle)**, shares the
**same buckets / Firestore / worker SA**; only the image + Python differ. Terraform = a near-copy of
the existing job block.

**Weights are NOT duplicated (A6 unchanged):** both images fetch weights at runtime from the **same
GCS weights bucket** via `ensure_weights(subdir)` — `mpnn/`, `esmfold/`, plus a new `rfdiffusion/`
subdir for RF3. The only thing duplicated between images is the **runtime dependency environment**
(torch + libs), because the two images sit on different Python/base layers (Artifact Registry can't
dedupe them). The multi-GB model weights live once in GCS. The tiny ESM2 model (~30 MB) stays baked
in both (negligible).

**Trade-off accepted:** MPNN/ESMFold deps are installed in both images, and MPNN/ESMFold must be
re-verified on Python 3.12 in the new image (isolated — if they break there, #1/#2 are untouched).
Stage-splitting (RF3-only container + shared MPNN/QC, chained across two executions) was rejected
for MVP — its tighter image isolation isn't worth the two-execution orchestration; revisit only if
per-stage GPU right-sizing is needed (deferred with A2/A4 scale-out).

### D9 — Ops / compliance (C)
- **Budget alerts: not required.** Cost governance is already covered by D4 (scale-to-zero,
  dedicated project/folder = single billing line, 6-month review reminder); auto-disable-billing
  was declined.
- **RF3 attribution: surface CC BY 4.0 attribution as a UI footnote** (credits line). Crediting
  the MIT tools (ProteinMPNN / LigandMPNN, ESMFold) alongside is good practice, though only RF3's
  CC BY 4.0 strictly requires it.

---

## Compute & cost (self-host path)

> Analysis to support the open self-host-vs-API compute decision. Planning estimates
> (GCP us-central1, 2026) — **not quotes**; real numbers swing with protein/campaign size,
> region, and how many candidates you fold.

**Recommendation: scale-to-zero GPU workers** (GCP Batch / Vertex custom jobs / GKE
autoscaling GPU node pool). Pay GPU rates **only while a job runs**; idle → $0. **Never run an
always-on GPU VM** — an L4 24/7 ≈ **$620/mo**, an A100 ≈ **$2,680/mo**, regardless of use. Use
**Spot GPUs** for batch generation (jobs are idempotent/restartable → ~60–80% off).

**GPU choice:** **L4** (24 GB; ~$0.85/hr on-demand, ~$0.25–0.35 Spot) is the workhorse — runs
MPNN + ESMFold + moderate RFdiffusion. Reserve **A100** (40 GB ~$3.67/hr; 80 GB ~$5/hr) for
large proteins or speed.

**Runtime (planning):** RFdiffusion ~20–60 s/backbone (100–200 res); ProteinMPNN ~1 s/seq;
ESMFold ~5–15 s/fold. ESMFold often **dominates** redesign campaigns → fold a filtered subset,
not every sequence.

**Per-campaign variable cost (on L4):**
| Workflow | GPU-hr | On-demand | Spot |
|---|---|---|---|
| Fixed-backbone redesign (MPNN+QC; sweet protein) | ~0.6 | ~$0.50 | ~$0.20 |
| RFdiffusion motif scaffolding (500 backbones) | ~12 | ~$10–12 | ~$3–4 |
| De novo binder campaign (10k backbones) | ~300+ | ~$280–330 | ~$85–115 |

**Fixed costs ≈ $10–15/mo** (GCS weights + I/O, Artifact Registry, idle job queue; Streamlit
frontend already deployed).

**Realistic monthly** (moderate use — ~20 redesigns + ~10 scaffolding campaigns): **~$130
on-demand / ~$55–65 with Spot**, plus ~$85–330 per heavy binder campaign if run.

**Cost levers (biggest first):** scale-to-zero → Spot → right-size to L4 → fold only top
candidates → batch designs per VM spin-up (amortize ~2–5 min cold-start).

**Gotchas:** GPU quota needs GCP approval (request early); cold-start (VM boot + weight load)
is billed; egress minor. The real cost of self-hosting is **engineering/ops** to build the
async job system — that, not the cloud bill, is the true trade-off vs a third-party API.

---

## Open questions / deferred

**Resolved** (see Decisions): compute = **self-host** (A1/D5) · execution = **Cloud Run Jobs**
MVP (A2/D5) · job state = **Firestore** (A3/D5) · container = **monolith** MVP (A4/D5) · QC =
**ESMFold local** (A5/D5) · weights = **GCS mount** (A6/D5) · auth = **`@phyx44.com`,
user-linked** (A7/D5) · engine toggle → **engine-select-first** sidesteps common-input
reconciliation (D3).

**Features / UX (B): all resolved** (D1, D6, D7, D8).

**Ops / compliance (C): all resolved** (D9) — RF3 CC BY 4.0 → UI footnote; budget alerts not required.

**→ All A/B/C design questions resolved. MVP fully specified (D1–D9 + build breakdown below).**

---

## Build task breakdown (MVP)

**Strategy — walking skeleton.** Build **preset #1 first as a full end-to-end vertical** (MPNN-only,
the simplest ML path) to prove the entire spine — engine-select frontend → async job submission →
Firestore state → GCS I/O → Cloud Run Job worker → MPNN + ESMFold → results & history. Then add
**one axis of complexity per milestone**: ligand handling → RF3/diffusion → RF3+ligand → hardening.

### M0 — GCP foundation & scaffolding
1. Create dedicated GCP **project inside its own folder** (D4 single-line billing).
2. Enable APIs (Cloud Run, Artifact Registry, Cloud Storage, Firestore, Cloud Scheduler, Cloud
   Functions, IAM); **request L4 GPU quota** (lead time).
3. Terraform skeleton for all resources (D4 resurrect-in-one-command).
4. Create **GCS buckets** (weights / inputs / outputs; no lifecycle deletion — D4); upload
   ProteinMPNN + SolubleMPNN + ESMFold weights.
5. Provision **Firestore** (job store) + **Artifact Registry** repo.
6. Repo: add `proteinredesign/` package skeleton on a feature branch. ESM3 code untouched.

### M1 — First vertical: preset #1 (fixed-backbone redesign) end-to-end
**Backend**
7. Worker container (monolith shell): ProteinMPNN + SolubleMPNN + ESMFold; **weights mounted from
   GCS** (A6). RF3 added in M3.
8. **Config builder for #1**: PDB + fixed residues → MPNN `fixed_positions`. Reuse residue parsing
   from `core/nl_parser.py::build_template_from_structured` and PDB parsing in `utils/pdb_utils.py`;
   **validate residue# ↔ AA** against the PDB (B5).
9. Pipeline entrypoint: job manifest (GCS) → ProteinMPNN → **multi-checkpoint scoring** (D1) →
   **ESMFold QC** (D2 hard gate + ESM2 soft floor via `core/esm2_scorer.py`) → **over-generate to
   yield 10 QC-passed** ranked outputs (B4) → write PDBs/sequences/scores JSON to GCS.
10. Deploy worker as a **Cloud Run Job (GPU/L4)**; thin submission path.
11. **Firestore job store** module: user-tagged records (A7), status lifecycle
    (queued/running/done/failed + progress); survives restarts (B7).
12. Wire **audit logging** (reuse `utils/audit_log.py`).

**Frontend**
13. **Engine selector** at entry (ESM3 | RFdiffusion/MPNN) — D3; extend `app.py`, ESM3 path unchanged.
14. **#1 input panel**: PDB upload + fixed-residues picker (reuse "structured inputs" patterns in
    `ui/chat.py`) + output count (default 10).
15. **Submit → async**: create Firestore job + trigger Cloud Run Job; persist job id in session.
16. **Job dashboard / history**: list the user's jobs from Firestore with live status + a clear
    **"done/completed"** indicator (B7/B8).
17. **Results view**: on done, fetch from GCS; render sequences, MPNN scores (metadata ranking), QC
    metrics, 3D viewer, downloads — reuse `ui/results_panel.py`.
18. Add **RF3 CC BY 4.0 footnote** infra (D9).

**Validate**
19. End-to-end #1 run (sweet-protein redesign case): submit → 10 QC-passed ranked sequences in
    history → downloads work.
20. Governance checks: job/GPU scales to zero after; Firestore state survives a frontend refresh;
    billing shows a single line item.

### M2 — Preset #2 (ligand-aware redesign) — adds ligand handling
21. Add **LigandMPNN** to the worker container (weights from GCS).
22. **HETATM handling**: parse HETATM from the complex PDB, **filter waters/ions/buffer, surface
    groups for the scientist to confirm the ligand** (D8/B6). Extend `utils/pdb_utils.py`.
23. Config builder for #2 (fixed backbone + ligand atoms → LigandMPNN).
24. Frontend: enable #2 preset + ligand-confirm UI. Validate end-to-end.

### M3 — RFdiffusion3 + diffusion presets (#3, #8)
25. Add **RF3** to the worker container (weights from GCS); verify L4 VRAM headroom (bump GPU if needed).
26. Config builders: **#3 motif scaffolding** (motif residues + length → contig map); **#8
    diversification** (diversify slider → `partial_T` + optional fixed regions).
27. Frontend: enable #3 and #8 presets + input panels + taxonomy UI copy; **per-stage progress**
    (RFdiff → MPNN → QC). Validate both end-to-end.

### M4 — Preset #6 (enzyme active-site) — RF3 all-atom + ligand
28. **Atomic-motif config builder**: catalytic residues (residue# + AA + functional-group atoms) +
    confirmed ligand → RF3 all-atom atomic motif + scaffold length.
29. Wire RF3 all-atom mode + LigandMPNN sequence design. (Rosetta ΔΔG refinement optional — defer past MVP.)
30. Frontend: enable #6 preset. Validate on an enzyme test case.

### M5 — Cost governance & hardening
31. **Cloud Scheduler (6-month) → Cloud Function → email reminder** (D4); no auto-teardown.
32. Confirm scale-to-zero, artifact retention (no lifecycle delete), single-line billing.
33. Finalize **Terraform** (one-command provision/teardown).
34. Concurrency/multi-user test (Firestore), error handling + retries, per-task design caps.

---

## Implementation status

**Increment 1 (M0 + M1) — code landed** on branch `feat/rfdiffusion-mpnn-backend`:
- `proteinredesign/` package: `manifest.py`, `config_builders/preset1.py` (author→sequential mapping +
  residue#↔AA validation), `storage.py` (GCS), `jobstore.py` (Firestore), `worker.py` (pipeline
  + pure `select_top_candidates` QC-gate/ranking), `submit.py` (job trigger), `Dockerfile.worker`.
- `config.py`: GCS + Firestore client getters.
- Frontend: engine selector in `app.py` (D3); `ui/rfd_panel.py` (#1 panel + live validation);
  `ui/job_dashboard.py` (persistent history + results).
- `terraform/`: buckets (no lifecycle delete), Firestore, Artifact Registry, GPU Cloud Run Job, IAM.
- Tests: `tests/test_proteinredesign_preset1.py`, `tests/test_proteinredesign_worker.py` — **21 passing** (config
  builder + manifest + QC-gate/ranking).

**Pending (needs GPU/GCP, done at deploy):** the ML adapters in `worker.py`
(`run_proteinmpnn`, `score_with_checkpoint`, `run_esmfold`) are wired to the container tools and
raise `NotImplementedError` outside it — validate during container build; `terraform apply` +
weights upload + worker image build; end-to-end #1 run.
