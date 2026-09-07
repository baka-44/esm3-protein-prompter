"""
ui/concatemer_panel.py — the Concatemer composer.

Parameter entry that scientists will iterate on, so it earns a panel; the search itself is a
batch job in shape, but it runs in ~1s for thousands of candidates, so it runs inline rather than
through the GPU job queue.

Nothing here is spatial — picking an ordering is not posing — so there is deliberately no 3D
canvas. The funnel is the most important thing on the page: without it an empty result is
indistinguishable from a bug, and there is no way to tell an over-aggressive gate from a design
space that genuinely has nothing in it.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from concatemer.digest import digest
from concatemer.features import FEATURES
from concatemer.pipeline import run, to_csv, to_fasta
from concatemer.spec import (
    PRESET_RULES, CleavageRule, ConcatemerSpec, Peptide, Spacer, VectorContext,
    encoding_capacity,
)

DEFAULT_PEPTIDES = pd.DataFrame([
    {"name": "GHK", "sequence": "GHK", "min_copies": 2, "max_copies": 6},
    {"name": "GQPR", "sequence": "GQPR", "min_copies": 1, "max_copies": 5},
])
DEFAULT_SPACERS = pd.DataFrame([
    {"name": "GAR", "sequence": "GAR"},
    {"name": "GGAR", "sequence": "GGAR"},
])


@st.cache_data(show_spinner=False, max_entries=8)
def _run_cached(spec_json: str, max_candidates: int):
    """Keyed on the serialised spec so edits re-run but re-renders do not."""
    return run(ConcatemerSpec.from_dict(json.loads(spec_json)), max_candidates=max_candidates)


def _vector_inputs() -> VectorContext:
    """
    Construct context. Only the EA/EA flag is consumed today — it drives the signal-cleavage gate.

    The 5'UTR / signal CDS / 3'UTR sequences were collected here for start-codon accessibility,
    which has been withdrawn from this page: folding is O(n^3) and costs ~4.4s per candidate at a
    200 aa cargo, so running it over a candidate set is untenable and a large shortlist breaches
    the Cloud Run request timeout. `concatemer/rna.py` is intact and tested; it returns as its own
    entry point where a user picks a shortlist out of the exported CSV, and that page collects the
    sequences it needs. Leaving dead inputs here would only invite someone to fill them in and
    wonder what happened.
    """
    ste13 = st.checkbox(
        "Vector pre-pro retains the EA/EA spacer (Ste13 processing)", value=False, key="cc_ste13",
        help="Ste13 removes N-terminal X-Ala dipeptides processively, so with EAEA present it "
             "trims into a cargo whose second residue is alanine — a live risk with GAR-style "
             "spacers. Many modern vectors delete EAEA because the processing is often incomplete.",
    )
    return VectorContext(signal_ste13=ste13)


def _build_spec(peptides_df, spacers_df, rules, lo, hi, max_units,
                vector: VectorContext | None = None) -> ConcatemerSpec:
    peps = [Peptide(str(r["name"]).strip(), str(r["sequence"]).strip(),
                    int(r["min_copies"] or 0), int(r["max_copies"] or 1))
            for _, r in peptides_df.iterrows() if str(r.get("sequence", "")).strip()]
    spacers = [Spacer(str(r["name"]).strip(), str(r["sequence"]).strip())
               for _, r in spacers_df.iterrows() if str(r.get("sequence", "")).strip()]
    return ConcatemerSpec(peptides=peps, spacers=spacers, rules=rules,
                          length_min=int(lo), length_max=int(hi), max_units=int(max_units),
                          vector=vector or VectorContext())


def render_concatemer(user_email: str | None = None) -> None:
    st.markdown("### 🧷 Concatemer composer")
    st.caption(
        "Assemble bioactive peptides into a secretable carrier chain, then screen which ones are "
        "**likely** to survive expression and give the peptides back on digestion. The product "
        "is the hydrolysate, so the chain is designed not to fold — it is designed to be made "
        "and cut."
    )

    with st.sidebar:
        st.markdown("#### Cleavage chemistry")
        preset_names = st.multiselect(
            "Enzymes", list(PRESET_RULES), default=["trypsin"],
            help="Kex2 is what Pichia already runs in the Golgi — a construct containing KR is "
                 "processed during secretion whether or not that was the intent.",
        )
        rules = [PRESET_RULES[n] for n in preset_names]

        with st.expander("Custom rule"):
            cm = st.text_input("Motif (regex)", value="", key="cc_motif",
                               placeholder="e.g. KR  or  [KR]")
            cs = st.radio("Cut side", ["C", "N"], horizontal=True, key="cc_side",
                          help="C = after the motif (trypsin, Kex2). N = before it (Asp-N).")
            cb = st.text_input("Blocked by at P1'", value="P", key="cc_blocked",
                               help="Proline abolishes both trypsin and Kex2.")
            if cm.strip():
                rules = rules + [CleavageRule("custom", cm.strip(), cs, blocked_by=cb)]

        st.markdown("#### Size envelope")
        lo, hi = st.slider("Chain length (aa)", 20, 600, (60, 160), step=10)
        max_units = st.number_input("Max peptide copies in total", 2, 100, 20, step=1)
        max_candidates = st.select_slider("Candidate cap", [500, 1000, 3000, 5000, 10000],
                                          value=3000)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("**Peptides** — `min_copies` ≥ 1 makes one mandatory; copy number sets the "
                    "delivered blend ratio.")
        peptides_df = st.data_editor(DEFAULT_PEPTIDES, num_rows="dynamic", hide_index=True,
                                     use_container_width=True, key="cc_peptides")
    with c2:
        st.markdown("**Spacers** — prefer sequences free of S and T.")
        st.caption("S/T is the +2 of every N-glycosylation sequon *and* the O-mannosylation "
                   "target, so excluding it removes both. This rules out (GGGGS)ₙ.")
        spacers_df = st.data_editor(DEFAULT_SPACERS, num_rows="dynamic", hide_index=True,
                                    use_container_width=True, key="cc_spacers")

    vector = _vector_inputs()

    caps = [(str(r["name"]), str(r["sequence"]).strip(), int(r["max_copies"] or 1))
            for _, r in peptides_df.iterrows() if str(r.get("sequence", "")).strip()]
    tight = [(n, encoding_capacity(q), m) for n, q, m in caps if encoding_capacity(q) < m * 4]
    if tight:
        st.caption("⚠️ Codon head-room: "
                   + "; ".join(f"**{n}** has {c} distinct encodings for up to {m} copies"
                               for n, c, m in tight)
                   + " — repeats may be hard to de-duplicate at the DNA level.")

    if not st.button("Assemble and screen", type="primary", use_container_width=True):
        return

    spec = _build_spec(peptides_df, spacers_df, rules, lo, hi, max_units, vector)
    errs = spec.errors()
    if errs:
        for e in errs:
            st.error(e)
        return

    with st.spinner("Enumerating, digesting and screening…"):
        res = _run_cached(spec.to_json(), int(max_candidates))
    _render_results(res, spec)


def _render_results(res, spec: ConcatemerSpec) -> None:
    st.divider()
    counts = dict(res.funnel)
    f1, f2, f3 = st.columns(3)
    f1.metric("Assembled", counts.get("assembled", 0))
    f2.metric("Passed the gate", counts.get("passed failure gates", 0))
    f3.metric("Below the gate", counts.get("below the gate", 0))

    if res.stats.get("truncated_candidates") or res.stats.get("truncated_multisets"):
        st.info(f"Search truncated at the candidate cap — {res.stats.get('multisets', 0)} "
                f"multisets were enumerated. Raise the cap or tighten the length envelope to see "
                f"more of the space.")

    census = res.failure_census()
    if census:
        with st.expander(f"Why {counts.get('below the gate', 0)} candidates were gated out", expanded=not res.passed):
            for reason, n in census.items():
                st.markdown(f"- **{n}** — {reason}")

    if not res.passed:
        st.warning("Nothing passed the gate. The census above says which liability is doing the "
                   "work — usually a spacer that cannot be excised by the chosen chemistry, or a "
                   "peptide with an internal cleavage site.")
        return

    rows = []
    for r in res.passed:
        c = res.candidates[r.candidate_id]
        rows.append({
            "Rank": r.rank, "ID": r.candidate_id, "Len": c.length,
            "Exact %": round(r.values["pct_copies_exact"], 1),
            "Payload %": round(r.values["payload_fraction_mass"], 1),
            "Unintended": int(r.values["n_unintended"]),
            "Arch": c.architecture, "Spacer": c.spacer or "—",
            "Composition": ";".join(f"{k}×{v}" for k, v in c.counts.items()),
            "Worst axis": r.worst_group, "Top liability": r.reasons[0] if r.reasons else "",
        })
    st.markdown(f"#### {len(rows)} candidates passed")
    st.caption("Ordered by the **product** metrics first — whether the design delivers the "
               "peptides — then by manufacturability. There is no composite score: its weights "
               "would be invented, and a natural-protein corpus cannot supply them for a "
               "synthetic concatemer.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=340)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ All candidates + features (CSV)", data=to_csv(res),
                           file_name="concatemer_candidates.csv", mime="text/csv",
                           use_container_width=True)
    with d2:
        st.download_button("⬇️ Ranked sequences (FASTA)", data=to_fasta(res),
                           file_name="concatemer_candidates.fasta", mime="text/plain",
                           use_container_width=True)

    st.markdown("#### Inspect a candidate")
    pick = st.selectbox("Candidate", [r.candidate_id for r in res.passed[:200]],
                        format_func=lambda cid: f"#{res.candidates[cid].candidate_id} — "
                                                f"{next(r.rank for r in res.passed if r.candidate_id == cid)}")
    _render_detail(res, spec, pick)


def _render_detail(res, spec: ConcatemerSpec, cid: str) -> None:
    cand = res.candidates[cid]
    row = next(r for r in res.passed if r.candidate_id == cid)
    rep = digest(cand.sequence, spec, layout=cand.layout)

    st.code(cand.sequence, language=None)
    st.caption(f"{cand.length} aa · {cand.architecture} · spacer {cand.spacer or 'none'} · "
               f"{cand.n_units} peptide copies")

    a, b = st.columns([2, 3], gap="large")
    with a:
        st.markdown("**Digest products**")
        st.caption("Molar yield is per copy — the delivered blend, which is not the designed "
                   "copy ratio when sites differ in efficiency.")
        st.dataframe(pd.DataFrame([
            {"Peptide": p.name, "Designed": rep.designed_copies.get(p.name, 0),
             "Released exact": rep.released_exact.get(p.name, 0),
             "Molar yield": rep.molar_yield.get(p.name, 0.0),
             "Impaired sites": rep.impaired_bounding.get(p.name, 0)}
            for p in spec.peptides]), hide_index=True, use_container_width=True)
        st.markdown(f"Sites — **{rep.n_clean}** clean · **{rep.n_impaired}** impaired · "
                    f"**{rep.n_blocked}** blocked")
        if rep.n_unintended:
            st.warning(f"{rep.n_unintended} unintended fragment(s): "
                       + ", ".join(sorted({f.seq for f in rep.fragments if f.kind == 'unintended'})[:8]))
    with b:
        st.markdown("**Features**")
        st.caption(f"Weakest axis: **{row.worst_group}** — {', '.join(row.reasons)}")
        by_group: dict[str, list] = {}
        for f in FEATURES:
            by_group.setdefault(f.group, []).append(
                {"Feature": f.key, "Value": row.values.get(f.key), "Better": f.direction,
                 "Spec": f.spec_id or "—"})
        for g, items in by_group.items():
            with st.expander(f"{g}  ·  group rank {row.group_ranks.get(g, '—')}",
                             expanded=(g == row.worst_group)):
                st.dataframe(pd.DataFrame(items), hide_index=True, use_container_width=True)
