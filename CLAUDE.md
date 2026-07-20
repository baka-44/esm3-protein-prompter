# CLAUDE.md — ESM3 Protein Engineering Prompter

Developer context for Claude Code. Keep this file up to date as patterns change.

---

## Project overview

A Streamlit chat app deployed on Google Cloud Run. Scientists describe protein engineering goals in plain English. The pipeline:

```
User prompt
  → Claude Haiku (nl_parser.py)  → PromptSpec (mask regions, keywords, motifs)
  → ESM3 Forge API (esm_backend.py) → raw ESMProtein candidates
  → ESM2 HuggingFace (result_processor.py) → ranked CandidateResult list
  → Streamlit UI (ui/) → table, 3D viewer, FASTA/PDB download, refinement
```

**Live URL:** `https://prot-prompt-131717488078.us-central1.run.app`
**GCP project:** `phyx44-pp-codonlm-v1`
**Cloud Run service:** `prot-prompt` (region: `us-central1`)
**Auth:** Google OAuth — only `@phyx44.com` accounts (or addresses listed in `ALLOWED_EMAILS` secret).

---

## Saved plans / roadmap

| Plan | File | Summary |
|---|---|---|
| **Sequential Interaction Model** | [`docs/plans/sequential_interaction_model.md`](docs/plans/sequential_interaction_model.md) | Planned (not started). Future R&D program: predict sequence modifications that maximize secreted yield in *Pichia pastoris* by modeling expression as a relay of protein–protein interactions with host secretory machinery. Commercially-licensed co-folding oracles (**Boltz-2** / Chai-1) + an **ESM2-embedding surrogate** + active-learning wet-lab loop. Read the doc to resume. |
| **RFdiffusion + MPNN backend** | [`docs/plans/rfdiffusion_mpnn_backend.md`](docs/plans/rfdiffusion_mpnn_backend.md) | Design-notes / decisions log (full build deferred). Evaluating **RFdiffusion + ProteinMPNN/LigandMPNN/SolubleMPNN** (all commercial-clean, self-hostable) as a toggleable generation backend vs ESM3. Records design decisions — e.g. **D1: multi-model MPNN scoring** of every candidate as filter or metadata. |

---

## Key files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit entry point, session state, generation loop |
| `auth.py` | Google OAuth2 gate (bypassed when `GOOGLE_CLIENT_ID` not set) |
| `core/nl_parser.py` | Claude Haiku → `PromptSpec` JSON parsing |
| `core/prompt_builder.py` | `PromptSpec` → `ESMProtein` object for ESM3 |
| `core/esm_backend.py` | ESM3 Forge API calls, error handling, retry logic |
| `core/result_processor.py` | ESM2 scoring, composite score, novelty, `CandidateResult` |
| `ui/chat.py` | Chat input panel, function keyword selector |
| `ui/results_panel.py` | Results table, 3D viewer, per-candidate detail |
| `config.py` | API client initialisation |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Cloud Run container definition |

---

## Deployment

### Standard deploy + IAM

```bash
gcloud run deploy prot-prompt \
  --source . \
  --project phyx44-pp-codonlm-v1 \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600 \
  --set-secrets="ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,FORGE_API_TOKEN=FORGE_API_TOKEN:latest,GOOGLE_CLIENT_ID=GOOGLE_CLIENT_ID:latest,GOOGLE_CLIENT_SECRET=GOOGLE_CLIENT_SECRET:latest,OAUTH_REDIRECT_URI=OAUTH_REDIRECT_URI:latest,ALLOWED_EMAILS=ALLOWED_EMAILS:latest"

# ALWAYS re-apply IAM after every deploy (Cloud Run resets it per revision)
gcloud run services add-iam-policy-binding prot-prompt \
  --project phyx44-pp-codonlm-v1 \
  --region us-central1 \
  --member="allUsers" \
  --role="roles/run.invoker"
```

### Staging / pre-prod validation via Cloud Run revision tags (Option A)

Validate a new revision at its own URL **before** it takes production traffic — no second
environment needed. Helper script: `scripts/staging.sh` (`deploy` | `url` | `promote` | `rollback`).

```bash
# 1. Build + deploy a new revision tagged "staging", with NO production traffic.
#    Same flags/secrets as the standard deploy, plus: --no-traffic --tag staging
gcloud run deploy prot-prompt \
  --source . --project phyx44-pp-codonlm-v1 --region us-central1 \
  --no-traffic --tag staging \
  --memory 4Gi --cpu 2 --timeout 600 \
  --set-secrets="ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,FORGE_API_TOKEN=FORGE_API_TOKEN:latest,GOOGLE_CLIENT_ID=GOOGLE_CLIENT_ID:latest,GOOGLE_CLIENT_SECRET=GOOGLE_CLIENT_SECRET:latest,OAUTH_REDIRECT_URI=OAUTH_REDIRECT_URI:latest,ALLOWED_EMAILS=ALLOWED_EMAILS:latest"

# 2. Get the staging URL (prod traffic still on the old revision):
gcloud run services describe prot-prompt --project phyx44-pp-codonlm-v1 --region us-central1 \
  --format='value(status.traffic[].url)' | tr ';' '\n' | grep staging
#   → https://staging---prot-prompt-<hash>-uc.a.run.app   ← validate here

# 3. Promote the SAME validated revision to 100% prod traffic (build-once-promote):
gcloud run services update-traffic prot-prompt --project phyx44-pp-codonlm-v1 --region us-central1 \
  --to-tags staging=100

# Rollback (if a promoted revision misbehaves) — shift traffic back:
gcloud run revisions list --service prot-prompt --project phyx44-pp-codonlm-v1 --region us-central1
gcloud run services update-traffic prot-prompt --project phyx44-pp-codonlm-v1 --region us-central1 \
  --to-revisions <PREVIOUS_REVISION>=100
```

**Why this is safe:** `--no-traffic --tag staging` builds and deploys the revision but sends it
**zero** production traffic; you exercise it only via the tagged URL. Promotion shifts traffic to
that *exact* revision — no rebuild — so prod runs the identical artifact you validated. If you
never promote, prod is untouched (the "rollback" is just not promoting).

**Caveats (this is single-project, shared config — not full isolation):**
- The tagged revision shares the service's **secrets/env**, including `OAUTH_REDIRECT_URI` (= the
  prod URL). Ideal for validating startup, UI, and generation behaviour. To test the **OAuth
  login flow** on the tagged URL, add that URL to the OAuth client's *Authorized redirect URIs*.
- proteinredesign jobs submitted from the tagged revision hit the **prod proteinredesign backend** (same env) — so
  avoid running heavy validation jobs from staging until a data-isolated staging proteinredesign project
  exists. For real data/secret isolation, graduate to **Option B** (separate staging project +
  build-once-promote CI/CD). See `docs/plans/` notes.

### Rotating the Forge API token

```bash
# Add a new secret version
echo -n "NEW_TOKEN_VALUE" | gcloud secrets versions add FORGE_API_TOKEN \
  --project phyx44-pp-codonlm-v1 \
  --data-file=-

# Redeploy to pick up the new version (`:latest` resolves at deploy time)
# ... run the deploy command above
```

### Secret names in GCP Secret Manager

| Secret name | Env var consumed by |
|---|---|
| `ANTHROPIC_API_KEY` | `config.py` → Claude Haiku (nl_parser) |
| `FORGE_API_TOKEN` | `config.py` → ESM3 Forge API (esm_backend) |
| `GOOGLE_CLIENT_ID` | `auth.py` |
| `GOOGLE_CLIENT_SECRET` | `auth.py` |
| `OAUTH_REDIRECT_URI` | `auth.py` |
| `ALLOWED_EMAILS` | `auth.py` |

> The secrets are named `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — **not** `OAUTH_CLIENT_ID` / `OAUTH_CLIENT_SECRET`.

---

## Architecture decisions and gotchas

### ESM SDK error handling — no exceptions
ESM3 Forge returns `ESMProteinError` objects (not raised exceptions) when API calls fail.
Must detect with `type(obj).__name__ == "ESMProteinError"`.

```python
def _is_protein_error(obj) -> bool:
    return type(obj).__name__ == "ESMProteinError"
```

### FunctionAnnotation validation
Forge validates function labels server-side. Invalid labels (e.g. "sweetness", "sweet taste", "antimicrobial") cause `ESMProteinError` with "Unknown label" in the message.
Fix: detect this error, retry without function_annotations.

### Invalid keyword blocklist in nl_parser.py
Haiku frequently hallucinates non-InterPro terms like `sweetness`, `sweet taste`, `antimicrobial`, `stability`, `solubility`, `thermostability`, `expression`. These are stripped at parse time in `_dict_to_spec()` via `_INVALID_KEYWORDS` set before the spec is returned. The extended blocklist also includes: `sweet`, `bitter`, `taste`, `flavor`, `flavour`, `homodimer`, `homodimerization`, `dimerization`, `dimer`, `heterodimerization`, `heterodimer`, `oligomerization`, `ligand binding`, `ligand-binding`, `ligand`.

### ESM2 import conflict — `esm` SDK shadows `transformers.models.esm`
The EvolutionaryScale `esm` SDK installs a top-level `esm` module that shadows `transformers.models.esm.modeling_esm`. This caused `Failed to import transformers.models.esm.modeling_esm` and silent fallback to all-zero ESM2 scores.
Fix in `esm2_scorer.py` `_load_model()`: temporarily pop both `esm` and all `esm.*` submodules from `sys.modules`, import transformers, then restore. Belt-and-suspenders: Dockerfile also installs `transformers` before the `esm` SDK.

### ESM2 scoring — torchvision CPU compatibility
The `esm` SDK pulls in `torchvision` as a transitive dependency. The default PyPI `torchvision` expects CUDA torch; on CPU-only Cloud Run it raises `RuntimeError: operator torchvision::nms does not exist`, causing all ESM2 scores to silently fall back to 0.0.
Fix in `Dockerfile`: install `torchvision` from the same CPU wheel index as `torch` in the same `pip install` invocation:
```dockerfile
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```
**Do not install torchvision separately or from PyPI default index.**

### Candidate diversity — minimum mask enforcement
When Haiku specifies only 1–2 masked positions out of 200+, ESM3 has no freedom to generate diverse sequences.
Fix in `prompt_builder.py`: after the template is built, if `0 < mask_count < 15% of total_len` and `total_len > 30`, randomly expand the mask to 15% of positions, skipping `fixed_positions`. Logs "expanded mask from N to M/L".

### Candidate diversity — deduplication and input filtering
ESM3 sometimes returns identical sequences across candidates. Also, the input sequence itself is not a useful "design" candidate.
Fix in `result_processor.py` `process_results()`:
1. Deduplicate by exact sequence (keep first occurrence).
2. Remove candidates whose sequence equals `spec.original_sequence` (the input pasted by the user).
If all candidates are filtered, returns empty list with a warning log.

### Candidate diversity — 2× over-generation
To compensate for dedup/input-filtering reducing the pool, `app.py` `_build_and_generate()` temporarily doubles `spec.num_candidates` (capped at 50) before calling generation. After `process_results()`, results are trimmed back to the originally requested N. The spinner text surfaces this: "ESM3 generating 30 candidates (→ top 15 after dedup)".

### Generation time budget — 50-candidate runs
At ~4.5s/candidate via Forge API, 50 user-requested candidates = 100 API calls ≈ 7.5 min. Cloud Run timeout is set to 600s. Sidebar warns users when `n_candidates > 20` with an estimated duration and a reminder to keep the tab active. The progress bar updates incrementally via callback — users should not refresh mid-run.

### Sequence mask requirement
ESM3 requires at least one `_` mask token. Sending a fully-specified sequence causes "Cannot sample sequence when input has no masks".
Fix in `prompt_builder.py`: if template has no `_`, derive mask from `fixed_positions`, or fully mask.

### `to_pdb()` API change
Newer ESM SDK changed `to_pdb()` to require a file-path argument.
Fix: `protein_to_pdb_string()` in `esm_backend.py` tries both the no-arg and path-arg forms.

### 3D viewer — no stmol
`stmol` is archived and broken with Streamlit ≥ 1.35.
Fix: use `py3Dmol._make_html()` + `st.components.v1.html()` directly. **Do not re-add stmol.**

### pTM / pLDDT only available with structure-track generation
Sequence-only generation returns 0.0 for both. Show `"—"` in UI, not `0.000`.
Tracked by `CandidateResult.has_structure_scores`.

### Novelty score — do not use masked template as reference
`sequence_template.replace("_", "")` produces a reference *shorter* than the generated sequence (by the number of masked positions), causing a positional cascade of false mismatches that inflates novelty to ~40%+ even for nearly-identical sequences.
Fix in `result_processor.py`: only use the template as reference when it contains no `_` characters.
Tracked by `CandidateResult.has_novelty_ref`; show `"—"` when `False`.

### Haiku sequence_template concatenation bug
When asked to write a `sequence_template` for a long protein, Haiku incorrectly concatenates: it adds N underscores then re-appends a downstream anchor subsequence. This makes the template *longer* than the original (corrupting all downstream positions) and the tail gets silently trimmed.

Example:
- Original: `...WAAASKGDAALDAGGR...` (KGDAALDAGGR = 11 chars)
- Haiku writes: `...WAAAS________DAALDAGGR...` (17 chars instead of 11 — duplicates DAALDAGGR)

**Fix (nl_parser.py):**
- Added `mask_regions: list[{start, end}]` to `PromptSpec` and the JSON schema.
- Haiku now specifies integer ranges instead of writing the template string.
- `_apply_mask_regions()` reconstructs the template programmatically from the extracted original sequence + ranges.
- Fallback: if Haiku still writes a template manually, it's validated against the original sequence (mismatch rate check). Templates with >5% positional mismatches are discarded.

### JSON control characters in Haiku output
Haiku sometimes writes literal `\n` inside JSON string values (e.g. in `notes_to_user`), making the JSON invalid.
Fix: `_escape_json_control_chars()` in `nl_parser.py` — string-aware scanner that only escapes control chars *inside* quoted strings, leaving structural JSON whitespace intact.
**Do not use a naïve `str.replace('\n', '\\n')` — it corrupts structural newlines.**

### Structured protein inputs — deterministic template building
Haiku consistently inverts masking logic: it masks residues listed as "fixed" and keeps everything else. Cannot be fixed by prompting alone.
Fix: `ui/chat.py` expander `"🔬 Structured protein inputs"` exposes three fields: sequence, fixed residues (e.g. `K67, R82`), mask regions (e.g. `18-25, 66-72`). When any are filled, `nl_parser.py` calls `build_template_from_structured()` **after** Haiku runs and overwrites the template.
- Fixed-residues mode: start all-`_`, pin each listed residue from the sequence.
- Mask-regions mode: start full sequence, apply `_` at listed ranges.
- 1-based input, 0-based storage. Accepts `K67` (letter+number), `67K` (number+letter), `1M` (number-first), or `67` (number only) — regex: `r"([A-Za-z]?)(\d+)([A-Za-z]?)"`.
Haiku is still called for `function_keywords`, `num_candidates`, `notes_to_user` etc. — only masking logic is bypassed.

### Scaffold condensation — residue numbering
Users enter key residue positions using **original protein numbering** (1-based, as in the input PDB/sequence). The parser applies proportional remapping automatically:
```
new_pos = round(old_pos × (target_len - 1) / (orig_len - 1))
```
Backbone coordinates are extracted from the PDB at the original positions and anchored at the remapped target positions. The ESM3 Prompt Summary card shows the full mapping (e.g. `K67→pos49, R82→pos61`) before generation begins. Do not ask users to pre-calculate new positions.

### Full structure conditioning (inverse-folding mode)
When a PDB is uploaded and `use_structure_motif=False`, `prompt_builder.py` now calls `extract_full_backbone()` from `pdb_utils.py` to pass backbone N/CA/C/O for **all** residues as `ESMProtein.coordinates`. This gives ESM3 structural context at every masked position (buried hydrophobic core → L/I/V/F, loops constrained by phi/psi), preventing K/R bias from fixed anchor residues.
- Only activates if PDB length matches `spec.protein_length`; logs `INFO: full structure conditioning enabled (N residues)`.
- Full backbone takes priority over partial motif coordinates in `protein_kwargs` assembly.
- Partial motif path (`use_structure_motif=True` + `motif_residue_indices`) is unchanged.
- `extract_full_backbone()` is in `pdb_utils.py`; returns `(sequence: str, coords: np.ndarray shape (L, 37, 3))`.
- When PDB is uploaded, `nl_parser.py` also extracts the PDB sequence and injects it into Haiku's message so mask_regions are computed against the correct sequence.

### ESMProtein.coordinates must be torch.Tensor, not numpy
`ESMProtein.coordinates` requires a `torch.float32` Tensor. Passing a numpy ndarray causes `isnan(): argument 'input' must be Tensor, not numpy.ndarray` inside `generate_with_structure()`.
Fix in `prompt_builder.py`: both full-backbone and partial-motif coordinate arrays are explicitly converted with `torch.tensor(coords, dtype=torch.float32)` before assignment to `protein_kwargs["coordinates"]`.

### Haiku protein_length=100 / fixed_residues=0 — cascading root causes
When structured inputs are active but user didn't paste a sequence in the form:
1. `_si_sequence` is empty → override condition was `if _has_structured and _si_sequence:` → False → Haiku's `protein_length=100` used.
2. `_dict_to_spec` applied Haiku's `mask_regions` by slicing to 100 chars, truncating the 207-aa PDB sequence.
3. Full structure conditioning skipped: `PDB length (207) ≠ protein_length (100)`.

Fixes:
- **Fix A** (`parse()`): override reference now uses `_si_sequence or _pdb_sequence or original_sequence` — works even when user didn't paste sequence in the form.
- **Fix B** (`_dict_to_spec()`): after parsing Haiku's JSON, if `original_sequence` is available and its length ≠ Haiku's `protein_length`, correct `protein_length` to match. Logged as `WARN _dict_to_spec: correcting protein_length from X to Y`.
- **Fix C** (`_build_messages()`): when `known_protein_length > 0` (computed from `_si_sequence` or `_pdb_sequence`), inject it into the `[STRUCTURED INPUTS PROVIDED]` instruction so Haiku outputs the correct length directly.

---

## Scoring

| Score | Description | Range |
|---|---|---|
| `pTM` | Predicted TM-score (fold quality) | 0–1 |
| `pLDDT` | Mean per-residue confidence | 0–100 |
| `ESM2 LL` | Masked marginal log-likelihood (fitness proxy) | ~−3 to 0 |
| `composite_score` | `0.5×pTM + 0.3×(pLDDT/100) + 0.2×ESM2_norm` when structure scores available; `ESM2_norm` only otherwise | 0–1 |
| `novelty_pct` | % positions differing from reference (shown as `"—"` when no valid reference) | 0–100 |

---

## Function keyword rules

InterPro/UniProt annotation terms only. Invalid terms cause Forge API to return `ESMProteinError`.

**Invalid (will fail):** `sweetness`, `sweet taste`, `sweet protein`, `antimicrobial`, `stability`, `solubility`
**Valid examples:** `fluorescence`, `beta barrel`, `serine protease activity`, `kinase activity`, `DNA binding`, `zinc finger`, `thaumatin family`, `pathogenesis-related protein`

The UI exposes a curated `multiselect` dropdown (`ui/chat.py`) so users can pick valid terms directly. Selected keywords override any keywords Haiku infers from the prompt.

---

## Iterative refinement

After each generation round the user can refine any top-5 candidate:
- Fix confident residues (pLDDT threshold)
- Add/change function keywords
- Add SS8 / SASA hints (free text)
- Scaffold condensation (shorten while preserving active site)
- Free-text instruction to Claude

Each round's candidates are preserved in session state; the UI shows a breadcrumb for navigation.

---

## Local development

Auth is bypassed when `GOOGLE_CLIENT_ID` is not set. Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
FORGE_API_TOKEN=...
```

Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+. ESM2 scoring model (`facebook/esm2_t6_8M_UR50D`, ~30 MB) is pre-downloaded during `docker build` and baked into the image — no HuggingFace network call at runtime.
