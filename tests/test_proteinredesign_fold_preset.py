"""
Tests for the fold-only preset: ESMFold + catalytic-geometry QC with no design step.

Covers the manifest/routing contract, residue renumbering into the parent frame, the
site-spec parser, and the worker handler end-to-end with ESMFold stubbed out.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign import worker  # noqa: E402
from proteinredesign.manifest import (  # noqa: E402
    NO_INPUT_PDB_PRESETS, JobManifest, Preset,
)
from proteinredesign.submit import _job_name_for_preset  # noqa: E402


def _atom(serial, name, resname, num, x, y, z, b=90.0):
    return ("%-6s%5d %-4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
            % ("ATOM", serial, name, "", resname, "A", num, "", x, y, z, 1.0, b, name[0]))


# A minimal Ser-His-Asp relay numbered 1..N, as ESMFold would emit for a construct slice.
_STUB_PDB = "\n".join([
    _atom(1, "OG", "SER", 272, 0, 0, 0), _atom(2, "CA", "SER", 272, 0, -1.4, 0),
    _atom(3, "NE2", "HIS", 100, 3.0, 0, 0), _atom(4, "ND1", "HIS", 100, 4.4, 1.1, 0),
    _atom(5, "CA", "HIS", 100, 3.0, -1.4, 0),
    _atom(6, "OD1", "ASP", 62, 6.9, 1.6, 0), _atom(7, "CA", "ASP", 62, 6.9, 0.2, 0),
    "END",
]) + "\n"


def test_fold_preset_needs_no_input_pdb_and_routes_to_rf3_worker():
    m = JobManifest(preset=Preset.FOLD_SEQUENCES, user_email="u", pdb_uri="", params={})
    assert Preset.FOLD_SEQUENCES in NO_INPUT_PDB_PRESETS
    assert m.requires_input_pdb() is False
    # ESMFold + GPU live in the rf3 image, so fold jobs must land there.
    assert _job_name_for_preset(Preset.FOLD_SEQUENCES) == "proteinredesign-rf3-worker"


def test_design_presets_still_require_an_input_pdb():
    m = JobManifest(preset=Preset.BORROWED_BODIES, user_email="u", pdb_uri="gs://x", params={})
    assert m.requires_input_pdb() is True


def test_renumber_shifts_residues_into_parent_frame():
    out = worker._renumber_pdb(_STUB_PDB, offset=113)
    nums = sorted({int(l[22:26]) for l in out.splitlines() if l.startswith("ATOM")})
    # local 62/100/272 -> Kex2 precursor 175/213/385
    assert nums == [175, 213, 385]


def test_renumber_is_a_noop_without_offset():
    assert worker._renumber_pdb(_STUB_PDB, offset=0) == _STUB_PDB


def test_site_spec_parser_builds_catalytic_site():
    site = worker._catalytic_site_from_spec("k", {
        "nucleophile": ["A", 385], "base": ["A", 213], "acid": ["A", 175],
        "oxyanion": [["A", 314]], "metal_sites": {"Ca1": [["A", 135], ["A", 184]]},
    })
    assert site.nucleophile == ("A", 385) and site.base == ("A", 213)
    assert site.oxyanion == [("A", 314)]
    assert site.metal_sites["Ca1"] == [("A", 135), ("A", 184)]


class _FakeStore:
    def __init__(self):
        self.written = {}

    def write_output(self, job_id, name, data, content_type=None):
        self.written[name] = data
        return f"gs://out/{job_id}/{name}"


class _FakeJobs:
    def update_job(self, *a, **k):
        pass


def test_fold_handler_folds_renumbers_and_runs_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "run_esmfold", lambda seq: (_STUB_PDB, 88.5))
    m = JobManifest(
        preset=Preset.FOLD_SEQUENCES, user_email="u", pdb_uri="",
        params={
            "sequences": {"Kex2_cat": "MSEQ"},
            "offsets": {"Kex2_cat": 113},
            "catalytic_sites": {"Kex2_cat": {
                "nucleophile": ["A", 385], "base": ["A", 213], "acid": ["A", 175]}},
        })
    store = _FakeStore()
    res = worker._run_fold_sequences(m, str(tmp_path), _FakeJobs(), store)

    assert res["count"] == 1
    rec = res["folds"][0]
    assert rec["mean_plddt"] == 88.5 and rec["pdb"] == "Kex2_cat.pdb"
    # geometry ran against the renumbered (parent-frame) model and found the intact relay
    assert rec["geometry_pass"] is True, rec
    assert rec["min_catalytic_plddt"] == 90.0
    assert "Kex2_cat.pdb" in store.written and b"ATOM" in store.written["Kex2_cat.pdb"]
    saved = json.loads(store.written["results.json"])
    assert saved["preset"] == "fold_sequences"


def test_fold_handler_without_site_spec_still_returns_the_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "run_esmfold", lambda seq: (_STUB_PDB, 71.0))
    m = JobManifest(preset=Preset.FOLD_SEQUENCES, user_email="u", pdb_uri="",
                    params={"sequences": {"plain": "MSEQ"}})
    rec = worker._run_fold_sequences(m, str(tmp_path), _FakeJobs(), _FakeStore())["folds"][0]
    assert rec["mean_plddt"] == 71.0 and rec["geometry"] is None


def test_fold_handler_requires_sequences(tmp_path):
    m = JobManifest(preset=Preset.FOLD_SEQUENCES, user_email="u", pdb_uri="", params={})
    with pytest.raises(ValueError, match="sequences"):
        worker._run_fold_sequences(m, str(tmp_path), _FakeJobs(), _FakeStore())


def test_submit_fold_rejects_empty_input():
    from proteinredesign.submit import submit_fold
    with pytest.raises(ValueError, match="at least one"):
        submit_fold(sequences={}, user_email="u")
    with pytest.raises(ValueError, match="empty sequence"):
        submit_fold(sequences={"a": "  "}, user_email="u")
