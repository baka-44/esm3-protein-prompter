"""
cofold/submit.py — job submission orchestration (frontend side).

Ties together: config builder (validate) → upload PDB + manifest to GCS →
create Firestore job record → trigger the Cloud Run Job. Keeps the Streamlit
UI thin. All GCP-touching imports are lazy so the module imports without the
backend configured (local dev / ESM3-only use).
"""

from __future__ import annotations

import os

from cofold.config_builders.preset1 import Preset1Config, build_preset1_config
from cofold.manifest import JobManifest, Preset


def backend_configured() -> bool:
    """True only when the cofold GCP backend env is present (buckets + project)."""
    from config import COFOLD_INPUTS_BUCKET, COFOLD_OUTPUTS_BUCKET, GCP_PROJECT
    return bool(COFOLD_INPUTS_BUCKET and COFOLD_OUTPUTS_BUCKET and GCP_PROJECT)


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

    from cofold import jobstore, storage

    manifest.pdb_uri = storage.put_input_pdb(
        manifest.job_id, pdb_bytes, filename=pdb_filename or "input.pdb"
    )
    manifest_uri = storage.write_manifest(manifest.job_id, manifest.to_json())

    rec = jobstore.create_job(
        manifest,
        manifest_uri=manifest_uri,
        title=f"Fixed-backbone redesign · {cfg.mapping_summary or fixed_residues_str}",
    )

    _trigger_job(manifest_uri)
    return rec, cfg


def _trigger_job(manifest_uri: str) -> None:
    """Launch the Cloud Run Job with a per-execution COFOLD_MANIFEST_URI override."""
    from config import GCP_PROJECT

    region = os.getenv("GCP_REGION", "us-central1")
    job_name = os.getenv("COFOLD_JOB_NAME", "cofold-worker")

    from google.cloud import run_v2

    client = run_v2.JobsClient()
    name = f"projects/{GCP_PROJECT}/locations/{region}/jobs/{job_name}"
    overrides = run_v2.RunJobRequest.Overrides(
        container_overrides=[
            run_v2.RunJobRequest.Overrides.ContainerOverride(
                env=[run_v2.EnvVar(name="COFOLD_MANIFEST_URI", value=manifest_uri)]
            )
        ]
    )
    client.run_job(request=run_v2.RunJobRequest(name=name, overrides=overrides))
