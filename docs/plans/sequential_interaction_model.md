# Sequential Interaction Model

> **Status:** Planned — not yet started. Saved 2026-07-18 for later execution.
> Strategic roadmap for a future R&D program; review with the team before building.

## Context

Goal: predict sequence modifications (to a protein-of-interest and/or native host proteins)
that maximize **secreted yield in *Pichia pastoris* (Komagataella phaffii)**, by modeling
expression as a "relay race" of protein–protein interactions (PPIs) between the POI and
~15–20 host secretory-machinery proteins. Desirable interactions (chaperones, correct
protease processing) should be strengthened; undesirable ones (degradative proteases)
weakened.

Constraints / starting point:
- **Commercial use required** → AlphaFold3 is out (weights non-commercial; <50 institutions
  have local access as of 2026).
- **Starting cold** → no experimental yield/interaction/structure data. Phase 1 must
  bootstrap interaction labels from the co-folding oracle itself, then seed a wet-lab
  active-learning loop.
- An existing ESM3 Prompter repo already does ESM3 variant generation + ESM2 scoring +
  Streamlit review UI + Forge API + audit logging — reuse it as the "propose variants" and
  "review" front-end.

Key design insight: **co-folding is an expensive oracle, not the inner search loop.** You
cannot co-fold millions of variants. The architecture is oracle → cheap surrogate →
million-scale search → co-fold top candidates → wet lab → retrain (active learning).

---

## Recommended stack

| Layer | Tool | Why |
|---|---|---|
| Co-folding oracle | **Boltz-2** (MIT license) | AF3-class complex prediction + **binding-affinity module** (~FEP accuracy, 1000× faster). Commercial-cleared. |
| Oracle fallback / cross-check | **Chai-1** (Apache-2.0, weights) | Independent AF3-class model; use for consensus scoring on borderline pairs. |
| Interaction confidence | **PAE-based LIS** (Local Interaction Score, PAE cutoff ~12) + ipTM + interface-PAE | LIS outperforms ipTM alone for transient/small interfaces — critical for chaperone/secretion partners. |
| Physics ΔΔG (Phase 4) | **Rosetta Flex ddG** / **FoldX** | Rank interface point mutations; still competitive with DL ΔΔG predictors. |
| Variant proposal | **ESM3** (existing repo) | Generate POI / host-protein variants under constraints. |
| Surrogate features | **ESM2 embeddings** (existing scorer) | Cheap per-variant features for the million-scale surrogate. |
| MSA generation | ColabFold MMseqs2 server or Boltz MSA server | Required input for co-folding; Pichia host proteins have buildable MSAs. |

Compute: Boltz-2/Chai-1 need a modern GPU (A100/H100 ideal; large complexes need more VRAM).
Budget the oracle carefully — it is the cost bottleneck.

---

## Architecture

```
ESM3 (propose POI / host variants)  ──►  SURROGATE (ESM2 emb → interaction score)
                                              │  scores millions of variants (cheap)
                                              ▼
                                     select top-K candidates
                                              │
                       Boltz-2 / Chai-1 co-fold + affinity + LIS   (expensive ORACLE)
                                              │  top few
                                     Rosetta/FoldX ΔΔG refine  (Phase 4)
                                              │
                                          WET LAB  ──►  retrain surrogate (active learning)
```

**Yield objective (learned, not hand-assigned):**
`yield_score = Σ wᵢ · interaction(POI, partnerᵢ)` where weights `wᵢ` are *fit from wet-lab
data*, positive for chaperones/folding partners, negative for proteases/anti-targets.
Until wet-lab data exists (cold start), initialize weights from prior biology and treat the
whole equation as a hypothesis to be calibrated.

---

## Pichia host-partner panel (the "relay race")

Curate sequences (UniProt / *K. phaffii* GS115 genome) + MSAs for:

**Desirable (maximize interaction / correct processing):**
- **Kar2** (BiP/Hsp70 ER chaperone) — high tractability
- **Pdi1** (protein disulfide isomerase), **Ero1**
- **Cne1** (calnexin)
- **Kex2**, **Ste13** — correct α-MF prepro leader processing (mis-processing is a common bottleneck)
- Sec62/Sec63 post-translational translocon partners

**Anti-targets (minimize interaction):**
- **Pep4** (proteinase A), **Prb1** (proteinase B) — vacuolar proteases that degrade product
- **Yps1** (yapsin) — membrane aspartic protease implicated in product clipping
- Off-target Kex2 sites *within* the POI

**Low-tractability (flag, handle later / coarse-grain):**
- Sec61 translocon complex, OST (glycosylation), ribosome/SRP, COPII — large and/or
  transient; co-folding confidence is least reliable here. Do **not** over-trust scores;
  consider domain-level or sub-complex modeling.

---

## Phased roadmap

### Phase 0 — Infra + data
- New package (e.g. `cofold/`): oracle wrappers, panel data, scoring, surrogate.
- Curate the Pichia partner panel (sequences + MSAs) under `cofold/panel/`.
- Stand up Boltz-2 (primary) and Chai-1 (fallback) with a common `predict_complex()` interface
  returning structure + ipTM + interface-PAE + affinity.

### Phase 1 — Co-folding oracle + WT interaction map (PROOF OF CONCEPT, de-risk)
- Co-fold **wild-type POI × each panel partner**; extract LIS/ipTM/PAE/affinity.
- Sanity checks that must pass before scaling:
  - Known secreted-friendly proteins score higher chaperone-interaction than aggregation-prone controls.
  - Protease anti-targets and off-target Kex2 sites behave sensibly.
- Deliverable: a WT interaction fingerprint + calibrated confidence thresholds.

### Phase 2 — Variant proposal + oracle labeling + surrogate
- Use existing ESM3 pipeline to propose POI variants (and, separately, host-protein variants).
- Co-fold a *sampled* subset against the panel → labeled dataset.
- Train **surrogate**: ESM2 embeddings (+ variant/partner features) → predicted interaction
  score per partner. Validate on held-out oracle labels.

### Phase 3 — Optimization + active-learning loop
- Run surrogate over large variant libraries; optimize the multi-objective yield equation
  (maximize desirable, minimize protease interactions).
- **Acquisition function** (e.g. expected improvement / uncertainty sampling) selects the next
  co-fold batch and the next **wet-lab batch** (expression + protease-degradation assays).
- Feed measured yields back → refit weights `wᵢ` and retrain surrogate. Iterate.

### Phase 4 (optional) — Physics ΔΔG refinement tier
- For top candidates, relax complexes and score interface ΔΔG with Rosetta Flex ddG / FoldX
  to rank point mutations more precisely before committing to synthesis.

---

## Reuse from existing repo
- `core/esm_backend.py`, `core/prompt_builder.py` — ESM3 variant generation (propose step).
- `core/esm2_scorer.py`, `core/result_processor.py` — ESM2 embeddings/scoring (surrogate features).
- `config.py` — API client init pattern (extend for Boltz/Chai endpoints).
- `ui/`, `app.py` — Streamlit review pattern for inspecting interaction maps / candidates.
- `utils/audit_log.py` — log oracle runs and experiment-design decisions.

## Key scientific caveats (must hold in review with the team)
1. **Confidence ≠ binding energy.** ipTM/LIS = geometric confidence; affinity comes from
   Boltz-2's affinity module or Rosetta/FoldX. Keep the two axes separate in all scoring.
2. **"Better co-folding → better expression" is a hypothesis.** Weights must be *learned* from
   wet-lab data, not assumed from confidence.
3. **Secretion machinery is co-folding's weak spot** (large/transient/IDR complexes). Treat
   low-tractability partners' scores with skepticism.
4. **Boltz-2 affinity is best-validated on protein–ligand**; protein–protein affinity needs
   calibration against your own data.

## Verification
- Phase 1: WT interaction map reproduces known biology (chaperone > control; protease
  anti-targets flagged); confidence thresholds separate a small positive/negative control set.
- Phase 2: surrogate predicts held-out oracle interaction scores with acceptable
  correlation (e.g. Spearman ρ target agreed with team); ablate ESM2 features.
- Phase 3: closed-loop dry run on synthetic labels before first real wet-lab batch; confirm the
  acquisition function proposes diverse, high-value experiments.
- Phase 4: Flex ddG/FoldX rankings agree with co-folding affinity trend on a benchmark pair.

---

## Reference tools & licensing (verified 2026-07)
- Boltz-2 — MIT license, structure + affinity: https://boltz.bio/boltz2 · https://github.com/jwohlwend/boltz
- Chai-1 — Apache-2.0 (code + weights): https://github.com/chaidiscovery/chai-lab
- AlphaFold3 — weights non-commercial only: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
- PAE-based Local Interaction Score (LIS) for PPI screening: https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1
- Rosetta Flex ddG: https://pubs.acs.org/doi/10.1021/acs.jpcb.7b11367
