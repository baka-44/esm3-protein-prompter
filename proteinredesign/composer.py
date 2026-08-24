"""
proteinredesign/composer.py — Borrowed Bodies compose geometry (Phase-1 cockpit-lite).

Pure-CPU structural manipulation that turns a mount PDB + a torso PDB + the user's operations
(retain / cut / pose) into a `graft.GraftPackage`: the torso is cut into two flanks and the mount
is posed between them, all in one coordinate frame, with the N→C chain order, linkers, and repack
shell derived. The interactive 3D editor (Phase 2) drives these same operations; here they are
called with typed inputs.

Conventions follow Biopython: a rigid transform is (rot 3×3, tran 3) with
`new = old @ rot + tran` (so Bio.PDB `atom.transform(rot, tran)` and `Superimposer.rotran` compose
directly).
"""

from __future__ import annotations

import numpy as np

from proteinredesign.graft import Fragment, GraftPackage, GraftSpec, Linker
from utils.pdb_utils import get_residues, parse_pdb

_IDENTITY = (np.eye(3), np.zeros(3))


# ── chain / residue helpers ────────────────────────────────────────────────────

def _chain_residues(pdb_source, chain_id: str | None):
    """Ordered protein residues of the chosen chain (+ the chain id actually used)."""
    residues = get_residues(pdb_source, chain_id=None)
    if not residues:
        raise ValueError("No protein residues (ATOM records) in the structure.")
    chains: dict[str, list] = {}
    for r in residues:
        chains.setdefault(r.get_parent().id, []).append(r)
    if chain_id is None:
        chain_id = next(iter(chains))
    if chain_id not in chains:
        raise ValueError(f"Chain '{chain_id}' not found (have: {', '.join(chains)}).")
    return chains[chain_id], chain_id


def _range_set(ranges: list[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for a, b in ranges:
        out.update(range(a, b + 1))
    return out


def _contiguous_blocks(nums: list[int]) -> list[tuple[int, int]]:
    """Sorted residue numbers → list of contiguous (start, end) blocks."""
    nums = sorted(set(nums))
    blocks: list[tuple[int, int]] = []
    for n in nums:
        if blocks and n == blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], n)
        else:
            blocks.append((n, n))
    return blocks


def cut_torso(residue_nums: list[int], p1: int, p2: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Excise the middle between the two cut points (CC5): the residues strictly between p1 and p2
    disappear, leaving TORSO-1 (…≤p1) and TORSO-2 (≥p2…). p1/p2 are kept as the flank ends.
    Returns ((flank1_start, p1), (p2, flank2_end)).
    """
    lo, hi = min(residue_nums), max(residue_nums)
    a, b = sorted((p1, p2))
    if not (lo <= a < b <= hi):
        raise ValueError(f"Cut points {p1},{p2} must lie within the chain {lo}-{hi} and be distinct.")
    return (lo, a), (b, hi)


# ── pose / snap-to-fit ─────────────────────────────────────────────────────────

def _ca(residues, num: int):
    for r in residues:
        if r.id[1] == num and "CA" in r:
            return np.array(r["CA"].coord, dtype=float)
    return None


def snap_to_fit(
    torso_residues, flank1_end: int, flank2_start: int,
    mount_residues, mount_n: int, mount_c: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rigid transform (CC10) placing the mount so its termini close the two linkers: superpose the
    mount's (N-term, C-term) CA onto the torso's (flank1 C-end, flank2 N-start) CA (Kabsch on the
    two point pairs). Directionality: TORSO-1(C)→MOUNT(N), MOUNT(C)→TORSO-2(N) (CC8).
    """
    from Bio.SVDSuperimposer import SVDSuperimposer

    t1 = _ca(torso_residues, flank1_end)
    t2 = _ca(torso_residues, flank2_start)
    mn = _ca(mount_residues, mount_n)
    mc = _ca(mount_residues, mount_c)
    if any(x is None for x in (t1, t2, mn, mc)):
        raise ValueError("Could not find CA atoms for the terminus pairs used by snap-to-fit.")
    fixed = np.array([t1, t2])       # where the mount ends should land
    moving = np.array([mn, mc])      # the mount's termini
    sup = SVDSuperimposer()
    sup.set(fixed, moving)
    sup.run()
    rot, tran = sup.get_rotran()     # moving @ rot + tran ≈ fixed
    return rot, tran


def _euler_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler angles (degrees) → 3×3 for the row-vector convention (v @ R)."""
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T


def _apply_manual(rot, tran, mount_res, mount_keep_set, nudge, rotate):
    """
    Compose a manual transform (rotate about the mount centre by `rotate` deg, then translate by
    `nudge` Å) on top of the base pose (rot, tran), folded back into a single (rot', tran') that
    maps the mount's INPUT coords to the final placement (new = old @ rot' + tran').
    """
    nudge = np.array(nudge, dtype=float)
    coords = np.array([a.coord for r in mount_res if r.id[1] in mount_keep_set for a in r], dtype=float)
    if coords.size == 0:
        return rot, tran + nudge
    c1 = coords.mean(0) @ rot + tran           # centre of the base-posed mount
    Rm = _euler_matrix(*rotate)
    return rot @ Rm, tran @ Rm - c1 @ Rm + c1 + nudge


# ── compose ────────────────────────────────────────────────────────────────────

def _write_composite(torso_struct, torso_keep: set[int], torso_chain: str,
                     mount_struct, mount_keep: set[int], mount_chain: str,
                     rot: np.ndarray, tran: np.ndarray,
                     out_torso_chain: str, out_mount_chain: str) -> bytes:
    """
    Write a composite PDB: kept torso residues on `out_torso_chain` (original coords/numbering) +
    kept mount residues on `out_mount_chain` (posed by rot/tran, original numbering).
    """
    from io import StringIO

    from Bio.PDB import PDBIO, Select

    # Pose the mount in place.
    for atom in mount_struct.get_atoms():
        atom.set_coord(atom.coord @ rot + tran)

    class _Keep(Select):
        def __init__(self, chain_id, keep, new_chain):
            self.chain_id, self.keep, self.new_chain = chain_id, keep, new_chain

        def accept_chain(self, chain):
            return 1 if chain.id == self.chain_id else 0

        def accept_residue(self, residue):
            return 1 if (residue.id[0] == " " and residue.id[1] in self.keep) else 0

    io = PDBIO()
    atom_lines: list[str] = []
    for struct, ch, keep, new_ch in (
        (torso_struct, torso_chain, torso_keep, out_torso_chain),
        (mount_struct, mount_chain, mount_keep, out_mount_chain),
    ):
        # Relabel the chain so torso→A and mount→B in the composite.
        for model in struct:
            for chain in model:
                if chain.id == ch:
                    chain.id = new_ch
        io.set_structure(struct)
        sink = StringIO()
        io.save(sink, _Keep(new_ch, keep, new_ch))
        atom_lines += [l for l in sink.getvalue().splitlines() if l.startswith(("ATOM", "HETATM"))]

    # Each body was written with its own serials starting at 1 — renumber strictly increasing
    # across the whole composite (RF3's PDB reader rejects non-increasing atom IDs).
    renumbered = []
    for serial, line in enumerate(atom_lines, start=1):
        renumbered.append(f"{line[:6]}{serial:>5}{line[11:]}")
    return ("\n".join(renumbered) + "\nEND\n").encode()


def _auto_repack(composite_pdb: bytes, torso_chain: str, mount_chain: str,
                 junction_residues: list[tuple[str, int]], shell: float) -> list[dict]:
    """
    Repack shell (CC9): fixed-fragment residues (a) contacting the other body (within `shell` Å,
    heavy atoms) or (b) flanking a linker junction. Backbone stays fixed; MPNN may re-identify them.
    """
    residues = get_residues(composite_pdb, chain_id=None)
    torso = [r for r in residues if r.get_parent().id == torso_chain]
    mount = [r for r in residues if r.get_parent().id == mount_chain]

    def atoms(r):
        return np.array([a.coord for a in r], dtype=float)

    repack: set[tuple[str, int]] = set(junction_residues)
    m_atoms = [atoms(r) for r in mount]
    for r in torso:
        ta = atoms(r)
        if any(np.min(np.linalg.norm(ta[:, None, :] - ma[None, :, :], axis=2)) <= shell for ma in m_atoms):
            repack.add((torso_chain, r.id[1]))
    t_atoms = [atoms(r) for r in torso]
    for r in mount:
        ma = atoms(r)
        if any(np.min(np.linalg.norm(ma[:, None, :] - ta[None, :, :], axis=2)) <= shell for ta in t_atoms):
            repack.add((mount_chain, r.id[1]))
    return [{"chain": c, "author_num": n} for c, n in sorted(repack)]


def compose_graft(
    *,
    torso_pdb: bytes,
    mount_pdb: bytes,
    torso_cut: tuple[int, int],
    mount_keep: list[tuple[int, int]],
    torso_chain: str | None = None,
    mount_chain: str | None = None,
    mount_fixed_atoms: str = "ALL",
    mount_termini: tuple[int, int] | None = None,     # (N-term num, C-term num); default fragment ends
    pose: tuple[np.ndarray, np.ndarray] | None = None,  # explicit (rot, tran); overrides repose
    repose: bool = True,                                # True → snap-to-fit; False → keep input coords
    nudge: tuple[float, float, float] = (0.0, 0.0, 0.0),  # manual translation (Å) on top of the base pose (Phase 2)
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),  # manual rotation (deg, about the mount centre) on top
    mount_transform: dict | None = None,  # {"rot":[9 row-major], "tran":[3], "about":[3]} — a rigid
    # transform applied to the base-posed mount about `about` (from the live 3D canvas). new = R·(p-c)+c+t.
    linker_lengths: tuple[tuple[int, int], tuple[int, int]] = ((3, 8), (3, 8)),
    repack_shell: float = 5.0,
    k: int = 5,
    m: int = 3,
) -> GraftPackage:
    """
    Domain-insertion compose: cut the torso into two flanks, keep the chosen mount residues, pose
    the mount between the flanks (snap-to-fit by default), and emit a GraftPackage
    (TORSO-1 / linker / MOUNT / linker / TORSO-2). See module docstring + CC1–CC17.
    """
    OUT_T, OUT_M = "A", "B"

    torso_res, torso_chain = _chain_residues(torso_pdb, torso_chain)
    mount_res, mount_chain = _chain_residues(mount_pdb, mount_chain)

    (f1_start, f1_end), (f2_start, f2_end) = cut_torso([r.id[1] for r in torso_res], *torso_cut)
    torso_keep = set(range(f1_start, f1_end + 1)) | set(range(f2_start, f2_end + 1))

    mount_keep_set = _range_set(mount_keep)
    mount_blocks = _contiguous_blocks([n for n in (r.id[1] for r in mount_res) if n in mount_keep_set])
    if not mount_blocks:
        raise ValueError("No mount residues kept — check the retain ranges.")
    m_lo, m_hi = mount_blocks[0][0], mount_blocks[-1][1]
    mn, mc = mount_termini or (m_lo, m_hi)

    # Pose: explicit transform > snap-to-fit > keep input coords (no re-pose). The last is for
    # same-frame inputs (e.g. reinserting a loop from the SAME PDB) — snap-to-fit would displace
    # an already-native mount and break its geometry.
    if pose is not None:
        rot, tran = pose
    elif repose:
        rot, tran = snap_to_fit(torso_res, f1_end, f2_start, mount_res, mn, mc)
    else:
        rot, tran = _IDENTITY

    # Manual nudge (Phase 2): translation + rotation-about-mount-centre, composed on top of the
    # base pose. Lets the user slide/rotate the mount out of a clash that snap-to-fit couldn't.
    if any(nudge) or any(rotate):
        rot, tran = _apply_manual(rot, tran, mount_res, mount_keep_set, nudge, rotate)

    # Live canvas transform (Phase 2 M2): a rigid rotation R (about point `about`) + translation t,
    # applied to the base-posed mount. Fold into (rot, tran) the same way _apply_manual does, using
    # Rm = Rᵀ (client sends a column-vector matrix R with new = R·v; row-vector form is v @ Rᵀ).
    if mount_transform:
        R = np.array(mount_transform["rot"], dtype=float).reshape(3, 3)
        t = np.array(mount_transform["tran"], dtype=float)
        c1 = np.array(mount_transform["about"], dtype=float)
        Rm = R.T
        rot, tran = rot @ Rm, tran @ Rm - c1 @ Rm + c1 + t

    torso_struct = parse_pdb(torso_pdb)
    mount_struct = parse_pdb(mount_pdb)
    composite = _write_composite(
        torso_struct, torso_keep, torso_chain, mount_struct, mount_keep_set, mount_chain,
        rot, tran, OUT_T, OUT_M,
    )

    # N→C order (CC7): TORSO-1 / L1 / MOUNT(s) / L2 / TORSO-2. Multiple mount blocks become
    # multiple MOUNT-i fragments joined by short designed connectors (schema-ready — CC17).
    chain_order: list = [Fragment("TORSO-1", "torso", OUT_T, f1_start, f1_end, "BKBN")]
    junctions: list[tuple[str, int]] = [(OUT_T, f1_end)]
    chain_order.append(Linker(*linker_lengths[0]))
    for i, (b_lo, b_hi) in enumerate(mount_blocks, start=1):
        if i > 1:
            chain_order.append(Linker(3, 8))  # intra-mount connector between kept blocks
        chain_order.append(Fragment(f"MOUNT-{i}", "mount", OUT_M, b_lo, b_hi, mount_fixed_atoms))
        junctions += [(OUT_M, b_lo), (OUT_M, b_hi)]
    chain_order.append(Linker(*linker_lengths[1]))
    chain_order.append(Fragment("TORSO-2", "torso", OUT_T, f2_start, f2_end, "BKBN"))
    junctions.append((OUT_T, f2_start))

    repack = _auto_repack(composite, OUT_T, OUT_M, junctions, repack_shell)

    spec = GraftSpec(
        chain_order=chain_order,
        repack_residues=repack,
        k=k, m=m,
        provenance={
            "torso_chain": torso_chain, "mount_chain": mount_chain,
            "torso_cut": list(torso_cut), "mount_keep": [list(r) for r in mount_keep],
            "posed_by": ("explicit" if pose is not None else "snap_to_fit" if repose else "keep_input"),
        },
    )
    return GraftPackage(spec=spec, composite_pdb=composite)


# ── Simple fusion compose (no excision) ───────────────────────────────────────

def compose_fusion(
    *,
    mount_pdb: bytes,
    torso_pdb: bytes,
    mount_chain: str | None = None,
    torso_chain: str | None = None,
    mount_keep: list[tuple[int, int]] | None = None,   # default: the whole chain
    torso_keep: list[tuple[int, int]] | None = None,   # default: the whole chain
    chassis_terminus: str = "C",        # "C" → MOUNT–linker–TORSO; "N" → TORSO–linker–MOUNT
    mount_fixed_atoms: str = "ALL",
    nudge: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mount_transform: dict | None = None,
    auto_separate: bool = True,         # sane non-clashing starting pose when none is given
    linker_length: tuple[int, int] = (4, 12),
    repack_shell: float = 5.0,
    k: int = 5,
    m: int = 3,
    catalytic_site: dict | None = None,
) -> GraftPackage:
    """
    Fuse two intact bodies end-to-end and let RF3 design only the connection between them.

    Unlike compose_graft this performs NO excision and NO insertion: both chains are kept whole,
    the user places them relative to one another, and the engine fills the single gap while
    holding both bodies fixed.

    Why this mode exists: an enzyme whose maturation depends on an N-terminal propeptide cannot
    be loop-inserted. The propeptide's autocatalytic cleavage would sever the chain, dropping the
    whole upstream flank — including half the chassis — as a separate polypeptide. Such enzymes
    must carry the chassis on a terminus, and (for an N-terminal propeptide) specifically the
    C-terminus, so that autoprocessing releases the propeptide and leaves mount+chassis intact.
    Hence `chassis_terminus="C"` is the default.

    This is a weaker topological claim than insertion, so the design work has to earn its keep at
    the interface: judge the output on buried surface area and compactness, not merely on whether
    the linker closed.
    """
    OUT_T, OUT_M = "A", "B"
    if chassis_terminus not in ("C", "N"):
        raise ValueError("chassis_terminus must be 'C' (mount first) or 'N' (torso first)")

    torso_res, torso_chain = _chain_residues(torso_pdb, torso_chain)
    mount_res, mount_chain = _chain_residues(mount_pdb, mount_chain)

    def _whole(residues, keep):
        nums = [r.id[1] for r in residues]
        return _range_set(keep) if keep else set(nums)

    mount_keep_set = _whole(mount_res, mount_keep)
    torso_keep_set = _whole(torso_res, torso_keep)
    m_blocks = _contiguous_blocks([n for n in (r.id[1] for r in mount_res) if n in mount_keep_set])
    t_blocks = _contiguous_blocks([n for n in (r.id[1] for r in torso_res) if n in torso_keep_set])
    if not m_blocks or not t_blocks:
        raise ValueError("Both bodies must retain at least one residue.")
    if len(m_blocks) > 1 or len(t_blocks) > 1:
        raise ValueError("Fusion mode expects one contiguous block per body; use compose_graft "
                         "for multi-segment grafts.")
    (m_lo, m_hi), (t_lo, t_hi) = m_blocks[0], t_blocks[0]

    # Starting pose: no snap-to-fit (there is no cavity to snap into). What matters for a fusion
    # is the distance between the two JUNCTION TERMINI — the ends the linker has to bridge — not
    # the separation of the centroids. Placing by centroid puts elongated bodies absurdly far
    # apart at the junction, so translate along the terminus-to-terminus axis to a span the
    # linker can actually close, then back off only as far as needed to clear a clash.
    rot, tran = _IDENTITY
    if auto_separate and not (mount_transform or any(nudge) or any(rotate)):
        m_junc_num = m_hi if chassis_terminus == "C" else m_lo
        t_junc_num = t_lo if chassis_terminus == "C" else t_hi
        m_j = _ca(mount_res, m_junc_num)
        t_j = _ca(torso_res, t_junc_num)
        if m_j is not None and t_j is not None:
            m_atoms = np.array([a.coord for r in mount_res if r.id[1] in mount_keep_set for a in r])
            t_atoms = np.array([a.coord for r in torso_res if r.id[1] in torso_keep_set for a in r])
            axis = m_j - t_j
            n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
            target = max(6.0, linker_length[1] * 2.5)     # a span the longest linker can close
            for _ in range(40):                            # nudge outward until no steric overlap
                cand = (t_j + axis * target) - m_j
                moved = m_atoms + cand
                d = np.linalg.norm(moved[:, None, :] - t_atoms[None, :, :], axis=-1).min() \
                    if (len(moved) * len(t_atoms) <= 4_000_000) else 99.0
                if d >= 3.2:
                    break
                target += 2.0
            tran = (t_j + axis * target) - m_j

    if any(nudge) or any(rotate):
        rot, tran = _apply_manual(rot, tran, mount_res, mount_keep_set, nudge, rotate)
    if mount_transform:
        R = np.array(mount_transform["rot"], dtype=float).reshape(3, 3)
        tt = np.array(mount_transform["tran"], dtype=float)
        c1 = np.array(mount_transform["about"], dtype=float)
        Rm = R.T
        rot, tran = rot @ Rm, tran @ Rm - c1 @ Rm + c1 + tt

    composite = _write_composite(
        parse_pdb(torso_pdb), torso_keep_set, torso_chain,
        parse_pdb(mount_pdb), mount_keep_set, mount_chain,
        rot, tran, OUT_T, OUT_M,
    )

    mount_frag = Fragment("MOUNT-1", "mount", OUT_M, m_lo, m_hi, mount_fixed_atoms)
    torso_frag = Fragment("TORSO-1", "torso", OUT_T, t_lo, t_hi, "BKBN")
    if chassis_terminus == "C":       # MOUNT — linker — TORSO (chassis C-terminal)
        chain_order = [mount_frag, Linker(*linker_length), torso_frag]
        junctions = [(OUT_M, m_hi), (OUT_T, t_lo)]
    else:                             # TORSO — linker — MOUNT
        chain_order = [torso_frag, Linker(*linker_length), mount_frag]
        junctions = [(OUT_T, t_hi), (OUT_M, m_lo)]

    spec = GraftSpec(
        chain_order=chain_order,
        repack_residues=_auto_repack(composite, OUT_T, OUT_M, junctions, repack_shell),
        k=k, m=m,
        catalytic_site=catalytic_site or {},
        provenance={
            "mode": "fusion", "chassis_terminus": chassis_terminus,
            "torso_chain": torso_chain, "mount_chain": mount_chain,
            "posed_by": "manual" if (mount_transform or any(nudge) or any(rotate))
                        else ("auto_separate" if auto_separate else "keep_input"),
        },
    )
    return GraftPackage(spec=spec, composite_pdb=composite)
