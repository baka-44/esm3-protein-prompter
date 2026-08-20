"""
Unit tests for proteinredesign.graft — the model-agnostic graft package (Borrowed Bodies)
and its RF3 adapter (to_engine_params).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from proteinredesign.graft import (  # noqa: E402
    Fragment,
    GraftPackage,
    GraftSpec,
    Linker,
    to_engine_params,
)


def _spec() -> GraftSpec:
    # torso-1 / linker / mount (all-atom) / linker / torso-2  →  A1-15,14,B5-9,10,A30-46
    return GraftSpec(
        chain_order=[
            Fragment("TORSO-1", "torso", "A", 1, 15, fixed_atoms="BKBN"),
            Linker(14, 14),
            Fragment("MOUNT-1", "mount", "B", 5, 9, fixed_atoms="ALL"),
            Linker(8, 12),
            Fragment("TORSO-2", "torso", "A", 30, 46, fixed_atoms="BKBN"),
        ],
        repack_residues=[{"chain": "A", "author_num": 15}, {"chain": "B", "author_num": 5}],
        k=4, m=2,
        provenance={"torso_pdb": "1crn.pdb", "mount_pdb": "x.pdb"},
    )


def _pkg() -> GraftPackage:
    return GraftPackage(spec=_spec(), composite_pdb=b"ATOM  ...\nEND\n")


# ── contig + adapter ─────────────────────────────────────────────────────────────

def test_contig_is_indexed_multisegment():
    p = to_engine_params(_pkg())
    assert p["contig"] == "A1-15,14,B5-9,8-12,A30-46"


def test_select_fixed_atoms_only_for_all_atom_fragments():
    p = to_engine_params(_pkg())
    # mount B5-9 are ALL-atom; torso is BKBN (backbone via contig, not select_fixed_atoms).
    assert p["select_fixed_atoms"] == {f"B{n}": "ALL" for n in range(5, 10)}


def test_motif_residues_cover_all_fixed_and_repack_tokens():
    p = to_engine_params(_pkg())
    n_fixed = (15 - 1 + 1) + (9 - 5 + 1) + (46 - 30 + 1)   # 15 + 5 + 17
    assert len(p["motif_residues"]) == n_fixed
    assert p["repack_residues"] == ["A15", "B5"]
    assert p["k"] == 4 and p["m"] == 2


def test_no_select_fixed_atoms_when_all_backbone():
    spec = GraftSpec(chain_order=[
        Fragment("T1", "torso", "A", 1, 10, "BKBN"),
        Linker(5, 5),
        Fragment("T2", "torso", "A", 20, 30, "BKBN"),
    ])
    p = to_engine_params(GraftPackage(spec=spec, composite_pdb=b"X"))
    assert p["select_fixed_atoms"] is None
    assert p["contig"] == "A1-10,5,A20-30"


# ── package zip roundtrip ──────────────────────────────────────────────────────

def test_package_roundtrip():
    pkg = _pkg()
    data = pkg.to_bytes()
    back = GraftPackage.from_bytes(data)
    assert back.composite_pdb == pkg.composite_pdb
    assert to_engine_params(back)["contig"] == to_engine_params(pkg)["contig"]
    assert back.spec.k == 4 and back.spec.repack_residues == pkg.spec.repack_residues


def test_from_bytes_rejects_non_graft_zip():
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", b"not a graft")
    with pytest.raises(ValueError):
        GraftPackage.from_bytes(buf.getvalue())


# ── validation (CC13 export gate) ──────────────────────────────────────────────

def test_validate_ok():
    assert _spec().validate() == []


def test_validate_needs_two_fragments():
    spec = GraftSpec(chain_order=[Fragment("T1", "torso", "A", 1, 10)])
    assert spec.validate()  # non-empty → invalid


def test_validate_must_end_on_fragment():
    spec = GraftSpec(chain_order=[
        Fragment("T1", "torso", "A", 1, 10), Linker(5, 5),
        Fragment("T2", "torso", "A", 20, 30), Linker(5, 5),
    ])
    assert any("end on" in e for e in spec.validate())


def test_adapter_rejects_invalid_spec():
    spec = GraftSpec(chain_order=[Fragment("T1", "torso", "A", 1, 10)])
    with pytest.raises(ValueError):
        to_engine_params(GraftPackage(spec=spec, composite_pdb=b"X"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
