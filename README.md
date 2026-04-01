# ESM3 Protein Engineering Prompter

A chat-based interface for protein engineering using [ESM3](https://www.evolutionaryscale.ai/papers/esm3-simulating-500-million-years-of-evolution-with-a-language-model) — the generative protein language model from EvolutionaryScale that simulates 500 million years of evolution.

Scientists describe what they want in plain English. Claude interprets the request → ESM3 generates candidate proteins → ESM2 scores fitness → results are ranked and shown with 3D structure views, FASTA/PDB downloads, and an iterative chain-of-thought refinement system.

---

## Features

- **Natural language prompting** — Describe active sites, conserved residues, desired characteristics in plain English
- **ESM3 multimodal constraints** — Sequence masking, structure motifs (PDB upload), function keywords, SS8 and SASA hints
- **ESM2 fitness scoring** — Zero-shot log-likelihood proxy for sequence naturalness and stability
- **Iterative refinement** — Chain-of-thought protocol: refine from any top-5 candidate with configurable pLDDT threshold, new keywords, SS8/SASA hints, or scaffold condensation
- **Scaffold condensation** — Reduce protein length while preserving active sites (inspired by the paper's trypsin 223→150 residue compression)
- **Multi-round history** — Navigate all previous generation rounds; each round's candidates are preserved

---

## Quickstart (Google Colab Pro + A100)

The recommended way to run this tool is via the included Colab notebook, which handles GPU access, Drive persistence, and public URL tunnelling automatically.

1. Open `colab_launcher.ipynb` in Google Colab
2. Set runtime to **GPU → A100** (`Runtime → Change runtime type`)
3. Run **Cell 1** (mounts Drive and syncs latest code from GitHub)
4. Run **Cell 2** (verifies GPU)
5. Run **Cell 3** (installs dependencies)
6. Run **Cell 4** (sets API keys — or enter them in the app sidebar)
7. Run **Cell 5** (launches Streamlit + localtunnel → public URL)

All generated proteins (FASTA + PDB files) are saved to `Drive/ProteinPrompter/outputs/` automatically.

---

## Local Setup

```bash
git clone https://github.com/<your-username>/esm3-protein-prompter.git
cd esm3-protein-prompter

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys

streamlit run app.py
```

Requires Python 3.10+ and (for local ESM3) a CUDA GPU with ≥16GB VRAM.

---

## API Keys

| Key | Where to get it | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Yes (NL parsing) |
| `FORGE_API_TOKEN` | [forge.evolutionaryscale.ai](https://forge.evolutionaryscale.ai) | No (defaults to local ESM3-open) |

Set in `.env` file or in the app sidebar at runtime.

---

## Architecture

```
User prompt (plain English)
        │
        ▼  Claude (Anthropic API)
   NL Parser ──► PromptSpec (residues, keywords, motifs, length)
        │
        ▼  ESM3 (local esm3-open or Forge API)
  ESM3 Backend ──► Raw candidates (ESMProtein objects)
        │
        ▼  ESM2 (HuggingFace facebook/esm2_t6_8M_UR50D)
 Result Processor ──► Ranked CandidateResult list
        │              (composite: 0.5×pTM + 0.3×pLDDT + 0.2×ESM2)
        ▼
  Results Panel ──► Table · 3D viewer · FASTA/PDB download · Refinement controls
```

---

## Scoring

| Score | Meaning | Range |
|---|---|---|
| **pTM** | Predicted TM-score — structural fold quality | 0–1 |
| **pLDDT** | Per-residue confidence in predicted structure | 0–100 |
| **ESM2 LL** | Masked marginal log-likelihood — sequence naturalness / fitness proxy | ~−3 to 0 |
| **Composite** | `0.5×pTM + 0.3×(pLDDT/100) + 0.2×ESM2_norm` | 0–1 |

---

## Iterative Refinement

After each generation round, click **🔬 Refine** on any of the top-5 candidates to configure the next round:

- **pLDDT threshold** — Fix residues ESM3 was confident about; regenerate uncertain regions
- **New keywords** — Add InterPro function terms (e.g. "high thermostability")
- **SS8 hint** — Free-text secondary structure instructions
- **SASA bias** — Push toward more buried core or more exposed surface
- **Condense** — Reduce protein length while preserving active site geometry
- **Free text** — Any additional instruction Claude will interpret

Each round's results are preserved; use the round breadcrumb to navigate your chain-of-thought history.

---

## License & Attribution

ESM3-open is released under the [EvolutionaryScale Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement) (non-commercial academic use).

Cite the original paper if you use this tool in research:
> Hayes et al. (2025). *Simulating 500 million years of evolution with a language model.* Science.
