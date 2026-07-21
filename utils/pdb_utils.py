"""
pdb_utils.py — PDB file parsing and backbone coordinate extraction.

Provides helpers to:
  - Parse a PDB file into a Biopython structure
  - Extract backbone atom coordinates (N, CA, C, O) for specified residue indices
  - Return a coordinate array compatible with ESMProtein.coordinates format
    (shape: [protein_length, 37, 3], padded with NaN for non-motif positions)
"""

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ESM3 uses 37 heavy atoms per residue (standard residue atom ordering).
# Backbone atoms N, CA, C, O are at indices 0, 1, 2, 3 in this ordering.
ESM_ATOM_N = 0
ESM_ATOM_CA = 1
ESM_ATOM_C = 2
ESM_ATOM_O = 3
ESM_NUM_ATOMS = 37

# Biopython atom names for backbone
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def parse_pdb(pdb_source: str | bytes | Path) -> "Structure":
    """
    Parse a PDB file and return a Biopython Structure object.

    Args:
        pdb_source: Path to a .pdb file, or raw PDB bytes/string content.

    Returns:
        Biopython Structure object.
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError as e:
        raise RuntimeError("biopython is required. Run: pip install biopython") from e

    parser = PDBParser(QUIET=True)

    def _is_existing_file(src) -> bool:
        # A multi-line PDB-content string makes Path(src).exists() raise ENAMETOOLONG;
        # treat any such error as "not a path" so we parse it as raw content instead.
        try:
            return Path(src).exists()
        except (OSError, ValueError):
            return False

    if isinstance(pdb_source, (str, Path)) and _is_existing_file(pdb_source):
        structure = parser.get_structure("protein", str(pdb_source))
    else:
        # Treat as raw content
        if isinstance(pdb_source, str):
            pdb_source = pdb_source.encode()
        handle = io.StringIO(pdb_source.decode(errors="replace"))
        structure = parser.get_structure("protein", handle)

    return structure


def get_residues(pdb_source: str | bytes | Path, chain_id: str | None = None) -> list:
    """
    Return an ordered list of Biopython Residue objects from the first model.

    Args:
        pdb_source: PDB file path or raw content.
        chain_id:   If given, only residues from this chain are returned.
                    If None, all chains are returned in order.

    Returns:
        List of Biopython Residue objects (HETATM residues excluded).
    """
    structure = parse_pdb(pdb_source)
    model = structure[0]  # first model

    residues = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain:
            # Skip HETATM (waters, ligands)
            if residue.id[0] != " ":
                continue
            residues.append(residue)

    return residues


def extract_backbone_coordinates(
    pdb_source: str | bytes | Path,
    protein_length: int,
    motif_residue_indices: list[int],
    chain_id: str | None = None,
) -> np.ndarray:
    """
    Extract backbone atom coordinates for specified residue positions and return
    an array compatible with ESMProtein.coordinates.

    ESMProtein.coordinates shape: (protein_length, 37, 3)
    Non-motif positions are filled with NaN (ESM3 treats NaN as unspecified).

    Args:
        pdb_source:            PDB file path or raw content.
        protein_length:        Total length of the protein to be generated.
        motif_residue_indices: 0-based indices (in the generated protein) where
                               backbone coordinates should be applied.
                               These map 1-to-1 with the PDB residues in order.
        chain_id:              Chain to extract from (None = all chains in order).

    Returns:
        numpy array of shape (protein_length, 37, 3), dtype float32.
        NaN where coordinates are not specified.
    """
    pdb_residues = get_residues(pdb_source, chain_id=chain_id)

    if len(motif_residue_indices) > len(pdb_residues):
        raise ValueError(
            f"Requested {len(motif_residue_indices)} motif residues but PDB only has "
            f"{len(pdb_residues)} residues."
        )

    coords = np.full((protein_length, ESM_NUM_ATOMS, 3), fill_value=np.nan, dtype=np.float32)

    for out_idx, pdb_res_pos in enumerate(motif_residue_indices):
        if pdb_res_pos >= protein_length:
            warnings.warn(
                f"Motif residue index {pdb_res_pos} exceeds protein_length {protein_length}. "
                f"Skipping."
            )
            continue

        pdb_residue = pdb_residues[out_idx]

        for atom_name, atom_slot in zip(
            BACKBONE_ATOM_NAMES,
            (ESM_ATOM_N, ESM_ATOM_CA, ESM_ATOM_C, ESM_ATOM_O),
        ):
            if atom_name in pdb_residue:
                atom = pdb_residue[atom_name]
                coords[pdb_res_pos, atom_slot, :] = atom.get_vector().get_array()
            else:
                warnings.warn(
                    f"Atom '{atom_name}' missing in residue {pdb_residue.get_resname()} "
                    f"at PDB position {pdb_residue.id[1]}. Leaving NaN."
                )

    return coords


def get_sequence_from_pdb(
    pdb_source: str | bytes | Path,
    chain_id: str | None = None,
) -> str:
    """
    Extract the amino acid sequence from a PDB file.

    Returns:
        Single-letter amino acid string.
    """
    try:
        from Bio.PDB.Polypeptide import protein_letters_3to1
    except ImportError as e:
        raise RuntimeError("biopython is required. Run: pip install biopython") from e

    residues = get_residues(pdb_source, chain_id=chain_id)
    seq = []
    for res in residues:
        resname = res.get_resname().strip()
        aa = protein_letters_3to1.get(resname, "X")
        seq.append(aa)
    return "".join(seq)


def extract_full_backbone(
    pdb_source: str | bytes | Path,
    chain_id: str | None = None,
) -> tuple[str, np.ndarray]:
    """
    Extract backbone coordinates for ALL residues in the PDB.

    Used for full structure conditioning (inverse-folding mode): passes backbone
    N/CA/C/O for every residue so ESM3 has structural context at masked positions.

    Returns:
        sequence: str          — single-letter AA sequence from ATOM records
        coords:   np.ndarray   — shape (L, 37, 3), float32.
                                 N/CA/C/O filled for every residue;
                                 side-chain slots (4-36) remain NaN.
    """
    residues = get_residues(pdb_source, chain_id=chain_id)
    sequence = get_sequence_from_pdb(pdb_source, chain_id=chain_id)
    L = len(residues)

    coords = np.full((L, ESM_NUM_ATOMS, 3), fill_value=np.nan, dtype=np.float32)

    for i, residue in enumerate(residues):
        for atom_name, atom_slot in zip(
            BACKBONE_ATOM_NAMES,
            (ESM_ATOM_N, ESM_ATOM_CA, ESM_ATOM_C, ESM_ATOM_O),
        ):
            if atom_name in residue:
                coords[i, atom_slot, :] = residue[atom_name].get_vector().get_array()

    return sequence, coords


def extract_motif_by_source_indices(
    pdb_source: str | bytes | Path,
    target_length: int,
    source_indices: list[int],
    target_indices: list[int],
    chain_id: str | None = None,
) -> np.ndarray:
    """
    Extract backbone N/CA/C/O from specific PDB residues and place them at
    specified positions in a new (shorter) protein coordinate array.

    Used for scaffold condensation: active site residues from the original PDB
    (at source_indices) are anchored at remapped positions (target_indices) in
    a shorter protein of target_length.

    Args:
        pdb_source:     PDB file path or raw content.
        target_length:  Length of the condensed protein to generate.
        source_indices: 0-based positions in the PDB to read coordinates from.
        target_indices: 0-based positions in the new protein to write to.
                        Must be same length as source_indices (1-to-1 mapping).
        chain_id:       Chain to extract from (None = all chains in order).

    Returns:
        numpy array of shape (target_length, 37, 3), dtype float32.
        NaN at all positions except the motif residues.
    """
    residues = get_residues(pdb_source, chain_id=chain_id)
    coords = np.full((target_length, ESM_NUM_ATOMS, 3), fill_value=np.nan, dtype=np.float32)

    for src_idx, tgt_idx in zip(source_indices, target_indices):
        if src_idx >= len(residues):
            warnings.warn(
                f"Source index {src_idx} exceeds PDB length {len(residues)} — skipping."
            )
            continue
        if tgt_idx >= target_length:
            warnings.warn(
                f"Target index {tgt_idx} exceeds target_length {target_length} — skipping."
            )
            continue
        residue = residues[src_idx]
        for atom_name, atom_slot in zip(
            BACKBONE_ATOM_NAMES,
            (ESM_ATOM_N, ESM_ATOM_CA, ESM_ATOM_C, ESM_ATOM_O),
        ):
            if atom_name in residue:
                coords[tgt_idx, atom_slot, :] = residue[atom_name].get_vector().get_array()

    return coords


def pdb_bytes_to_string(pdb_bytes: bytes) -> str:
    """Decode PDB bytes to string, stripping null bytes."""
    return pdb_bytes.decode(errors="replace").replace("\x00", "")


# ── Ligand detection (preset #2 — D8/B6) ───────────────────────────────────────
# Waters, common crystallographic ions, and cryo/buffer additives are excluded from
# the ligand picker by default so scientists only confirm among plausible ligands.
# Not exhaustive — a curated, adjustable list, not a comprehensive PDB ontology.
_SOLVENT_ION_ADDITIVE_BLOCKLIST = frozenset({
    # waters
    "HOH", "WAT", "DOD", "H2O", "OH2",
    # common monoatomic/small ions
    "NA", "K", "CL", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU", "CU1", "NI", "CO",
    "CD", "HG", "LI", "RB", "CS", "BA", "SR", "AL", "PB", "AG", "AU", "PT", "BR",
    "IOD", "F", "GD", "LA", "CE", "EU", "TB", "YB", "3CO", "3NI", "NH4",
    # common cryoprotectant / crystallization buffer additives
    "GOL", "EDO", "PEG", "PG4", "1PE", "PGE", "MPD", "DMS", "ACT", "TRS", "BME",
    "CIT", "FMT", "SO4", "PO4", "NO3", "IMD", "BOG", "EPE", "MES", "BU2", "BU3",
    "12P", "P6G", "PEP", "UNX", "UNL", "ACE",
})


@dataclass
class HetGroup:
    """A candidate ligand: one HETATM residue group in the uploaded PDB."""

    resname: str
    chain_id: str
    res_seq: int   # PDB author residue number
    atom_count: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.resname, self.chain_id, self.res_seq)

    def label(self) -> str:
        return f"{self.resname} · chain {self.chain_id} · residue {self.res_seq} · {self.atom_count} atoms"


def get_hetatm_groups(pdb_source: str | bytes | Path, chain_id: str | None = None) -> list["HetGroup"]:
    """
    Return candidate ligand HETATM groups in the PDB, excluding common solvents,
    ions, and crystallization additives (see _SOLVENT_ION_ADDITIVE_BLOCKLIST).

    Args:
        pdb_source: PDB file path or raw content.
        chain_id:   If given, only groups on this chain are returned.

    Returns:
        List of HetGroup, one per distinct (resname, chain, residue number).
    """
    structure = parse_pdb(pdb_source)
    model = structure[0]

    groups: list[HetGroup] = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain:
            if residue.id[0] == " ":
                continue  # standard amino acid — not a HETATM group
            resname = residue.get_resname().strip()
            if resname in _SOLVENT_ION_ADDITIVE_BLOCKLIST:
                continue
            groups.append(HetGroup(
                resname=resname,
                chain_id=chain.id,
                res_seq=residue.id[1],
                atom_count=sum(1 for _ in residue.get_atoms()),
            ))
    return groups


def filter_pdb_keep_ligand(pdb_source: str | bytes | Path, ligand: tuple[str, str, int]) -> bytes:
    """
    Return PDB bytes with all standard protein ATOM records kept, plus HETATM
    records ONLY for the selected ligand group. All other HETATMs (waters, other
    buffer/ions, unselected ligand candidates) are dropped, so LigandMPNN's
    automatic ligand-context parsing sees only the intended ligand.

    Args:
        pdb_source: PDB file path or raw content.
        ligand:     (resname, chain_id, res_seq) identifying the group to keep —
                    as returned by HetGroup.key from get_hetatm_groups().

    Returns:
        PDB file content as bytes.
    """
    from Bio.PDB import PDBIO, Select

    resname, chain_id, res_seq = ligand
    structure = parse_pdb(pdb_source)

    class _LigandSelect(Select):
        def accept_residue(self, residue):
            if residue.id[0] == " ":
                return True  # keep all standard protein residues
            return (
                residue.get_resname().strip() == resname
                and residue.get_parent().id == chain_id
                and residue.id[1] == res_seq
            )

    io_writer = PDBIO()
    io_writer.set_structure(structure)
    buf = io.StringIO()
    io_writer.save(buf, select=_LigandSelect())
    return buf.getvalue().encode()
