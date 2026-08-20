"""
proteinredesign/submit.py — job submission orchestration (frontend side).

Ties together: config builder (validate) → upload PDB + manifest to GCS →
create Firestore job record → trigger the Cloud Run Job. Keeps the Streamlit
UI thin. All GCP-touching imports are lazy so the module imports without the
backend configured (local dev / ESM3-only use).
"""

from __future__ import annotations

import os

from proteinredesign.config_builders.preset1 import Preset1Config, build_preset1_config
from proteinredesign.config_builders.preset2 import Preset2Config, build_preset2_config
from proteinredesign.config_builders.preset5 import Preset5Config, build_preset5_config
from proteinredesign.config_builders.preset6 import Preset6Config, build_preset6_config
from proteinredesign.manifest import JobManifest, MPNN_ONLY_PRESETS, Preset


def backend_configured() -> bool:
    """True only when the proteinredesign GCP backend env is present (buckets + project)."""
    from config import PROTEINREDESIGN_INPUTS_BUCKET, PROTEINREDESIGN_OUTPUTS_BUCKET, GCP_PROJECT
    return bool(PROTEINREDESIGN_INPUTS_BUCKET and PROTEINREDESIGN_OUTPUTS_BUCKET and GCP_PROJECT)


def submit_preset1(
    *,
    pdb_bytes: bytes,
    pdb_filename: str,
    fixed_residues_str: str,
    chain_id: str | None,
    user_email: str,
    num_outputs: int = 10,
):
    """
    Validate inputs, persist them, and launch a preset #1 job.

    Returns (JobRecord, Preset1Config). Raises ConfigError on invalid input
    (before anything is uploaded).
    """
    cfg: Preset1Config = build_preset1_config(
        pdb_bytes, fixed_residues_str, chain_id=chain_id
    )

    manifest = JobManifest(
        preset=Preset.FIXED_BACKBONE_REDESIGN,
        user_email=user_email,
        pdb_uri="",  # set after upload
        params=cfg.to_params(),
        num_outputs=num_outputs,
    )

    from proteinredesign import jobstore, storage

    manifest.pdb_uri = storage.put_input_pdb(
        manifest.job_id, pdb_bytes, filename=pdb_filename or "input.pdb"
    )
    manifest_uri = storage.write_manifest(manifest.job_id, manifest.to_json())

    rec = jobstore.create_job(
        manifest,
        manifest_uri=manifest_uri,
        title=f"Fixed-backbone redesign · {cfg.mapping_summary or fixed_residues_str}",
    )

    _trigger_job(manifest_uri, preset=manifest.preset)
    return rec, cfg


def submit_preset2(
    *,
    pdb_bytes: bytes,
    pdb_filename: str,
    ligand_key: tuple[str, str, int],
    fixed_residues_str: str,
    chain_id: str | None,
    user_email: str,
    num_outputs: int = 10,
):
    """
    Validate inputs, persist them, and launch a preset #2 (ligand-aware redesign) job.

    Returns (JobRecord, Preset2Config). Raises ConfigError on invalid input
    (before anything is uploaded). Uploads the FILTERED PDB (only the chosen
    ligand's HETATM kept) — not the raw upload — so the worker never has to
    repeat the ligand-selection/filtering step.
    """
    cfg: Preset2Config = build_preset2_config(
        pdb_bytes, ligand_key, fixed_residues_str, chain_id=chain_id
    )

    manifest = JobManifest(
        preset=Preset.LIGAND_AWARE_REDESIGN,
        user_email=user_email,
        pdb_uri="",  # set after upload
        params=cfg.to_params(),
        num_outputs=num_outputs,
    )

    from proteinredesign import jobstore, storage

    manifest.pdb_uri = storage.put_input_pdb(
        manifest.job_id, cfg.filtered_pdb_bytes, filename=pdb_filename or "input.pdb"
    )
    manifest_uri = storage.write_manifest(manifest.job_id, manifest.to_json())

    rec = jobstore.create_job(
        manifest,
        manifest_uri=manifest_uri,
        title=f"Ligand-aware redesign · {cfg.ligand.resname} ({cfg.ligand.chain_id}#{cfg.ligand.res_seq})",
    )

    _trigger_job(manifest_uri, preset=manifest.preset)
    return rec, cfg


def submit_preset5(
    *,
    pdb_bytes: bytes,
    pdb_filename: str,
    partial_t: float,
    k: int,
    m: int,
    chain_id: str | None,
    user_email: str,
):
    """
    Validate inputs, persist them, and launch a scaffold-diversification job (RF3
    partial diffusion → MPNN → ESMFold QC). Routes to the **rf3-worker** Cloud Run
    Job (Python 3.12 / foundry image), not the MPNN-only worker (D11).

    Returns (JobRecord, Preset5Config). Raises ConfigError on invalid input (before
    anything is uploaded). The fan-out is K×M raw designs; `num_outputs` is set to
    K×M so every QC-passed candidate is returned, ranked (D10.3).
    """
    cfg: Preset5Config = build_preset5_config(
        pdb_bytes, partial_t, k, m, chain_id=chain_id
    )

    manifest = JobManifest(
        preset=Preset.SCAFFOLD_DIVERSIFICATION,
        user_email=user_email,
        pdb_uri="",  # set after upload
        params=cfg.to_params(),
        num_outputs=cfg.total_designs,
    )

    from proteinredesign import jobstore, storage

    manifest.pdb_uri = storage.put_input_pdb(
        manifest.job_id, pdb_bytes, filename=pdb_filename or "input.pdb"
    )
    manifest_uri = storage.write_manifest(manifest.job_id, manifest.to_json())

    rec = jobstore.create_job(
        manifest,
        manifest_uri=manifest_uri,
        title=(f"Scaffold diversification · chain {cfg.chain_id} · "
               f"{cfg.k}×{cfg.m} · {cfg.partial_t:g}Å"),
    )

    _trigger_job(manifest_uri, preset=manifest.preset)
    return rec, cfg


def submit_preset6(
    *,
    pdb_bytes: bytes,
    pdb_filename: str,
    catalytic_residues_str: str,
    fixed_atoms_mode: str,
    ligand_key: tuple[str, str, int] | None,
    length_min: int,
    length_max: int,
    k: int,
    m: int,
    chain_id: str | None,
    user_email: str,
):
    """
    Validate inputs, persist them, and launch an enzyme active-site scaffolding job (RF3
    all-atom scaffold → MPNN → ESMFold + motif-RMSD QC). Routes to the rf3-worker (D11).

    Returns (JobRecord, Preset6Config). Uploads the FILTERED PDB (protein + only the chosen
    cofactor) when a ligand is selected, so RF3/LigandMPNN see only the intended ligand.
    """
    cfg: Preset6Config = build_preset6_config(
        pdb_bytes, catalytic_residues_str,
        fixed_atoms_mode=fixed_atoms_mode, ligand_key=ligand_key,
        length_min=length_min, length_max=length_max, k=k, m=m, chain_id=chain_id,
    )

    manifest = JobManifest(
        preset=Preset.ENZYME_ACTIVE_SITE,
        user_email=user_email,
        pdb_uri="",  # set after upload
        params=cfg.to_params(),
        num_outputs=cfg.total_designs,
    )

    from proteinredesign import jobstore, storage

    upload_bytes = cfg.filtered_pdb_bytes if cfg.filtered_pdb_bytes is not None else pdb_bytes
    manifest.pdb_uri = storage.put_input_pdb(
        manifest.job_id, upload_bytes, filename=pdb_filename or "input.pdb"
    )
    manifest_uri = storage.write_manifest(manifest.job_id, manifest.to_json())

    lig = f" · {cfg.ligand.resname}" if cfg.ligand is not None else ""
    rec = jobstore.create_job(
        manifest,
        manifest_uri=manifest_uri,
        title=(f"Enzyme active-site scaffolding · {cfg.mapping_summary or catalytic_residues_str}"
               f"{lig} · {cfg.length_min}-{cfg.length_max} aa · {cfg.k}×{cfg.m}"),
    )

    _trigger_job(manifest_uri, preset=manifest.preset)
    return rec, cfg


def _job_name_for_preset(preset: "Preset") -> str:
    """
    Route a preset to its Cloud Run Job (D11: two images / two jobs).
    - MPNN-only presets (#1/#2) → the Python-3.10 `mpnn-worker` job (default).
    - RF3 presets (#8, later #3/#6) → the Python-3.12 `rf3-worker` job.
    A Cloud Run Job's image is fixed in its template, so routing lives here.
    """
    if preset in MPNN_ONLY_PRESETS:
        return os.getenv("PROTEINREDESIGN_JOB_NAME", "proteinredesign-worker")
    return os.getenv("PROTEINREDESIGN_RF3_JOB_NAME", "proteinredesign-rf3-worker")


def _trigger_job(manifest_uri: str, preset: "Preset | None" = None) -> None:
    """Launch the Cloud Run Job with a per-execution PROTEINREDESIGN_MANIFEST_URI override."""
    from config import GCP_PROJECT

    region = os.getenv("GCP_REGION", "us-central1")
    job_name = (
        _job_name_for_preset(preset) if preset is not None
        else os.getenv("PROTEINREDESIGN_JOB_NAME", "proteinredesign-worker")
    )

    from google.cloud import run_v2

    client = run_v2.JobsClient()
    name = f"projects/{GCP_PROJECT}/locations/{region}/jobs/{job_name}"
    overrides = run_v2.RunJobRequest.Overrides(
        container_overrides=[
            run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=[run_v2.EnvVar(name="PROTEINREDESIGN_MANIFEST_URI", value=manifest_uri)]
            )
        ]
    )
    client.run_job(request=run_v2.RunJobRequest(name=name, overrides=overrides))
