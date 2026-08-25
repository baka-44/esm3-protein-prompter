"""
The designed backbone must be persisted and referenced alongside the ESMFold refold.

These are different artefacts and the distinction is not cosmetic. The RF3 backbone holds the
composed pose exactly — on job fe1e7863f08d it reproduced the posed composite to 0.04 Å, with
the torso-to-mount placement preserved to 0.01 Å. candidate_N.pdb is re-predicted from the
sequence alone and never sees that pose; on the same job ESMFold pivoted the torso 22-53 Å
away. Surfacing only the refold meant designs were being judged by a QC artefact.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.worker import Candidate, _write_results  # noqa: E402


class _Storage:
    """Captures write_output calls instead of touching GCS."""

    def __init__(self):
        self.written: dict[str, bytes] = {}
        self.calls: list[str] = []

    def write_output(self, job_id, name, data, content_type=None):
        self.calls.append(name)
        self.written[name] = data
        return f"gs://test/{job_id}/{name}"


class _Manifest:
    class preset:
        value = "borrowed_bodies"
    params = {"contig": "A17-79,6-11,B114-453,5-10,A93-178"}


def _cand(rank, backbone_path, seq="ACDEF"):
    return Candidate(sequence=seq, pdb=f"ESMFOLD-{rank}", rank=rank,
                     parent_backbone_path=backbone_path)


@pytest.fixture
def backbones(tmp_path):
    paths = []
    for i in (1, 2):
        p = tmp_path / f"rf3_model_{i}.pdb"
        p.write_text(f"REMARK designed backbone {i}\nEND\n")
        paths.append(str(p))
    return paths


def test_designed_backbone_is_persisted_and_referenced(backbones, tmp_path):
    st = _Storage()
    top = [_cand(1, backbones[0]), _cand(2, backbones[0]),
           _cand(3, backbones[1]), _cand(4, backbones[1])]
    res = _write_results("job1", _Manifest(), top, st)

    assert "design_1.pdb" in st.written and "design_2.pdb" in st.written
    assert st.written["design_1.pdb"] == b"REMARK designed backbone 1\nEND\n"

    refs = [c["design_pdb"] for c in res["candidates"]]
    assert refs == ["design_1.pdb", "design_1.pdb", "design_2.pdb", "design_2.pdb"]


def test_shared_backbones_are_uploaded_once(backbones):
    """m candidates share one backbone — uploading per candidate would be m x the bytes."""
    st = _Storage()
    top = [_cand(i, backbones[0]) for i in range(1, 5)]
    _write_results("job2", _Manifest(), top, st)
    assert st.calls.count("design_1.pdb") == 1


def test_esmfold_refold_is_still_written_separately(backbones):
    """The refold stays — it is the QC measurement — but it is no longer the only structure."""
    st = _Storage()
    _write_results("job3", _Manifest(), [_cand(1, backbones[0])], st)
    assert st.written["candidate_1.pdb"] == b"ESMFOLD-1"
    assert "design_1.pdb" in st.written


def test_presets_without_a_generated_backbone_report_none():
    """MPNN-only presets hold the input backbone fixed, so there is no separate design."""
    st = _Storage()
    res = _write_results("job4", _Manifest(), [_cand(1, "")], st)
    assert res["candidates"][0]["design_pdb"] is None
    assert not any(n.startswith("design_") for n in st.written)


def test_a_missing_backbone_file_does_not_fail_the_job(tmp_path):
    st = _Storage()
    res = _write_results("job5", _Manifest(),
                         [_cand(1, str(tmp_path / "gone.pdb"))], st)
    assert res["candidates"][0]["design_pdb"] is None
    assert res["candidates"][0]["pdb"] == "candidate_1.pdb"     # results still written


# ── frame alignment ────────────────────────────────────────────────────────────

import json  # noqa: E402

import numpy as np  # noqa: E402

from proteinredesign.worker import _align_design_to_input  # noqa: E402


def _write_pdb(path, coords, chain="A", start=1):
    lines = []
    for i, (x, y, z) in enumerate(coords):
        n = start + i
        for nm, d in (("N", -0.5), ("CA", 0.0), ("C", 0.5)):
            lines.append(
                "%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
                % ("ATOM", len(lines) + 1, nm, "", "ALA", chain, n, "",
                   x + d, y, z, 1.0, 0.0, nm[0]))
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")


def _setup(tmp_path, rot, tran, n=8):
    """Input at known coords; design = the same thing moved by (rot, tran)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=8.0, size=(n, 3))
    inp = str(tmp_path / "input.pdb")
    des = str(tmp_path / "design.pdb")
    _write_pdb(inp, pts, chain="A", start=10)
    _write_pdb(des, pts @ rot.T + tran, chain="A", start=1)
    json.dump({"diffused_index_map": {f"A{10 + i}": f"A{1 + i}" for i in range(n)}},
              open(str(tmp_path / "design.json"), "w"))
    return inp, des, pts


def _ca(path):
    out = []
    for ln in open(path):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            out.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return np.array(out)


def test_design_is_moved_into_the_input_frame(tmp_path):
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0], [0, 0, 1]])
    inp, des, pts = _setup(tmp_path, rot, np.array([25.0, -12.0, 7.0]))

    before = np.sqrt(((_ca(des) - pts) ** 2).sum(1).mean())
    assert before > 20.0                                   # starts in a different frame

    _align_design_to_input(des, inp)
    after = np.sqrt(((_ca(des) - pts) ** 2).sum(1).mean())
    assert after < 0.01, f"still {after:.3f} Å from the input frame"


def test_alignment_preserves_internal_geometry(tmp_path):
    """Re-framing is rigid — it must not distort the design it is moving."""
    rot = np.eye(3)
    inp, des, _ = _setup(tmp_path, rot, np.array([40.0, 0.0, 0.0]))
    d0 = _ca(des)
    pair0 = np.linalg.norm(d0[0] - d0[-1])
    _align_design_to_input(des, inp)
    d1 = _ca(des)
    assert abs(np.linalg.norm(d1[0] - d1[-1]) - pair0) < 1e-3


def test_no_index_map_leaves_the_file_untouched(tmp_path):
    inp, des, _ = _setup(tmp_path, np.eye(3), np.array([30.0, 0.0, 0.0]))
    os.remove(str(tmp_path / "design.json"))
    before = open(des).read()
    _align_design_to_input(des, inp)
    assert open(des).read() == before


def test_too_few_anchors_leaves_the_file_untouched(tmp_path):
    inp, des, _ = _setup(tmp_path, np.eye(3), np.array([30.0, 0.0, 0.0]))
    json.dump({"diffused_index_map": {"A10": "A1", "A11": "A2"}},   # only 2 anchors
              open(str(tmp_path / "design.json"), "w"))
    before = open(des).read()
    _align_design_to_input(des, inp)
    assert open(des).read() == before


def test_a_broken_input_never_raises(tmp_path):
    inp, des, _ = _setup(tmp_path, np.eye(3), np.array([30.0, 0.0, 0.0]))
    assert _align_design_to_input(des, str(tmp_path / "nope.pdb")) == des
