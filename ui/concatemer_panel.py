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
    PRESET_RULES, RULE_GROUPS, CleavageRule, ConcatemerSpec, Peptide, Spacer, VectorContext,
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


def _cell_text(value, default: str = "") -> str:
    """
    A data_editor cell as text.

    An empty cell arrives as float('nan'), and str(nan) is the string "nan" — so a blank name
    silently became a peptide called "nan" rather than an obvious error.
    """
    if value is None:
        return default
    text = str(value).strip()
    return default if text.lower() in ("", "nan", "none", "<na>") else text


def _cell_int(value, default: int) -> int:
    """
    A data_editor cell as an int, tolerating blanks.

    `int(value or default)` looks safe and is not: an empty numeric cell is float('nan'), NaN is
    TRUTHY, so `nan or 0` evaluates to nan and int(nan) raises. That crashed the whole panel for
    anyone who added a peptide row and left a copy-number blank, which is the normal way a
    dynamic data_editor row starts out.
    """
    try:
        if value is None:
            return default
        number = float(value)
        if number != number:                      # NaN
            return default
        return int(number)
    except (TypeError, ValueError):
        return default


def _build_spec(peptides_df, spacers_df, rules, lo, hi, max_units,
                vector: VectorContext | None = None) -> ConcatemerSpec:
    peps = []
    for _, r in peptides_df.iterrows():
        seq = _cell_text(r.get("sequence"))
        if not seq:
            continue                              # blank row from the dynamic editor
        peps.append(Peptide(_cell_text(r.get("name"), seq.upper()), seq,
                            _cell_int(r.get("min_copies"), 0),
                            _cell_int(r.get("max_copies"), 1)))
    spacers = []
    for _, r in spacers_df.iterrows():
        seq = _cell_text(r.get("sequence"))
        if seq:
            spacers.append(Spacer(_cell_text(r.get("name"), seq.upper()), seq))
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
        rules = _cleavage_inputs()

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


def _rule_detail(r: CleavageRule) -> None:
    """Everything needed to choose an enzyme: what it recognises, where it cuts, what it costs."""
    where = "after" if r.side == "C" else "before"
    bits = [f"Cuts **{where}** `{r.motif}` — consensus **{r.consensus or r.motif}**"]
    if r.requires_p1prime:
        bits.append(f"P1′ **must be** {' or '.join(r.requires_p1prime)}")
    if r.blocked_by:
        bits.append(f"blocked by **{'/'.join(r.blocked_by)}** at P1′")
    if r.impaired_by:
        bits.append(f"slowed by **{'/'.join(r.impaired_by)}** at P1′")
    if r.trim_c_basic:
        bits.append("followed by **Kex1** carboxypeptidase trimming")
    bits.append(f"recognition site **{r.site_length} residue(s)** per junction")
    st.markdown("\n".join(f"- {b}" for b in bits))
    if r.note:
        st.caption(r.note)


def _cleavage_inputs() -> list[CleavageRule]:
    """
    Enzyme picker plus an accumulating list of custom rules.

    Presets are grouped by the trade-off that actually decides the choice: broad-specificity
    enzymes add no recognition site and so cost no payload, but cut wherever their residue
    appears; high-selectivity fusion proteases never cut in the wrong place but spend 5-6
    residues at every junction on sequence you do not sell.
    """
    st.markdown("#### Cleavage chemistry")
    ordered = [n for names in RULE_GROUPS.values() for n in names]
    group_of = {n: g for g, names in RULE_GROUPS.items() for n in names}

    picked = st.multiselect(
        "Enzymes", ordered, default=["trypsin"], key="cc_enzymes",
        format_func=lambda n: f"{n} · {PRESET_RULES[n].consensus}",
        help="Mix freely — the digest simulator applies every selected rule and reports the "
             "worst status where two land on the same bond.",
    )
    rules = [PRESET_RULES[n] for n in picked]

    for n in picked:
        r = PRESET_RULES[n]
        with st.expander(f"{r.name} · {r.consensus}"):
            st.caption(group_of.get(n, ""))
            _rule_detail(r)

    # A multi-residue C-side recogniser cuts after its OWN site, so in a tandem layout the site
    # stays attached to the peptide upstream of it. These enzymes were built for a single
    # carrier->product junction, where that site lands on the carrier and is thrown away.
    stranded = [n for n in picked
                if PRESET_RULES[n].side == "C" and PRESET_RULES[n].site_length >= 3
                and not PRESET_RULES[n].trim_c_basic]
    if stranded:
        one = len(stranded) == 1
        st.warning(
            f"**{', '.join(stranded)}** {'cuts' if one else 'cut'} after "
            f"{'its' if one else 'their'} own recognition site, so in a tandem concatemer that "
            f"site stays on the **upstream** peptide — every copy but the last comes back "
            f"extended. Check the digest products before committing. A one-residue recogniser "
            f"(trypsin, Lys-C) avoids this entirely when the peptide's own terminus is the cut "
            f"site, and CPB can trim a basic residue off afterwards."
        )
    if picked:
        tax = sum(PRESET_RULES[n].site_length for n in picked)
        if tax > 3:
            st.caption(f"Recognition sites total **{tax} residues per junction** — across ten "
                       f"junctions, {tax * 10} residues of chain that is not product.")

    # ── custom rules: accumulate, so a mixed digest can be described ──
    st.session_state.setdefault("cc_custom", [])
    with st.expander(f"Custom rules ({len(st.session_state['cc_custom'])})"):
        with st.form("cc_custom_form", clear_on_submit=True):
            name = st.text_input("Name", placeholder="my_protease")
            motif = st.text_input("Motif (regex)", placeholder="KR   [KR]   ENLYFQ")
            side = st.radio("Cut side", ["C", "N"], horizontal=True,
                            help="C = after the motif (trypsin, Kex2, TEV). "
                                 "N = before it (Asp-N).")
            c1, c2 = st.columns(2)
            with c1:
                blocked = st.text_input("Blocked at P1′", value="P",
                                        help="Residues that abolish cleavage. Proline blocks "
                                             "trypsin and Kex2.")
            with c2:
                requires = st.text_input("Required at P1′", value="",
                                         help="Leave empty unless the enzyme demands specific "
                                              "residues, as TEV demands G or S.")
            if st.form_submit_button("Add rule", use_container_width=True):
                _add_custom_rule(name, motif, side, blocked, requires)

        for i, d in enumerate(list(st.session_state["cc_custom"])):
            row, drop = st.columns([5, 1])
            with row:
                st.markdown(f"**{d['name']}** — `{d['motif']}` cut {d['side']}-side"
                            + (f", P1′ must be {d['requires_p1prime']}" if d["requires_p1prime"] else "")
                            + (f", blocked by {d['blocked_by']}" if d["blocked_by"] else ""))
            with drop:
                if st.button("✕", key=f"cc_rm_{i}", help="Remove this rule"):
                    st.session_state["cc_custom"].pop(i)
                    st.rerun()

    return rules + [CleavageRule(**d) for d in st.session_state["cc_custom"]]


def _add_custom_rule(name: str, motif: str, side: str, blocked: str, requires: str) -> None:
    """Validate and append. A bad regex here would otherwise surface as an opaque failure
    much later, inside find_sites, on every candidate at once."""
    name, motif = name.strip() or "custom", motif.strip()
    if not motif:
        st.warning("A motif is required.")
        return
    candidate = CleavageRule(name=name, motif=motif, side=side,
                             blocked_by=blocked.strip().upper(),
                             requires_p1prime=requires.strip().upper())
    errs = candidate.errors()
    if errs:
        for e in errs:
            st.error(e)
        return
    if any(d["name"] == name for d in st.session_state["cc_custom"]):
        st.warning(f"A rule named {name!r} is already added.")
        return
    st.session_state["cc_custom"].append({
        "name": name, "motif": motif, "side": side,
        "blocked_by": candidate.blocked_by, "requires_p1prime": candidate.requires_p1prime,
    })
