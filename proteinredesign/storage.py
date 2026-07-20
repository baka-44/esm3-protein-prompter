"""
proteinredesign/storage.py — Google Cloud Storage I/O for the generation backend.

Buckets (from env, provisioned by Terraform; no lifecycle deletion — D4):
  PROTEINREDESIGN_INPUTS_BUCKET   — uploaded PDBs + job manifests
  PROTEINREDESIGN_OUTPUTS_BUCKET  — generated PDBs / sequences / scores JSON
  PROTEINREDESIGN_WEIGHTS_BUCKET  — model weights (mounted or downloaded — A6)

The frontend uses put_input_pdb / write_manifest; the worker uses read_manifest /
download helpers / write_output; both use the small gs:// URI helpers.
"""

from __future__ import annotations

import json
import os
from typing import Any

from config import (
    PROTEINREDESIGN_INPUTS_BUCKET,
    PROTEINREDESIGN_OUTPUTS_BUCKET,
    PROTEINREDESIGN_WEIGHTS_BUCKET,
    PROTEINREDESIGN_WEIGHTS_MOUNT,
    get_gcs_client,
)


# ── gs:// URI helpers ─────────────────────────────────────────────────────────

def parse_gs_uri(uri: str) -> tuple[str, str]:
    """gs://bucket/path/to/obj → ('bucket', 'path/to/obj')."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri}")
    bucket, _, blob = uri[len("gs://"):].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Malformed gs:// URI: {uri}")
    return bucket, blob


def _gs_uri(bucket: str, blob: str) -> str:
    return f"gs://{bucket}/{blob}"


def _require(bucket: str | None, which: str) -> str:
    if not bucket:
        raise RuntimeError(
            f"{which} bucket is not configured. Set the corresponding PROTEINREDESIGN_*_BUCKET env var."
        )
    return bucket


# ── Generic blob I/O ──────────────────────────────────────────────────────────

def upload_bytes(bucket: str, blob_name: str, data: bytes, content_type: str | None = None) -> str:
    client = get_gcs_client()
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return _gs_uri(bucket, blob_name)


def download_bytes(uri: str) -> bytes:
    bucket, blob_name = parse_gs_uri(uri)
    client = get_gcs_client()
    return client.bucket(bucket).blob(blob_name).download_as_bytes()


def write_json(bucket: str, blob_name: str, obj: Any) -> str:
    return upload_bytes(
        bucket, blob_name, json.dumps(obj, indent=2).encode(), content_type="application/json"
    )


def read_json(uri: str) -> Any:
    return json.loads(download_bytes(uri).decode())


def download_to_path(uri: str, local_path: str) -> str:
    bucket, blob_name = parse_gs_uri(uri)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    client = get_gcs_client()
    client.bucket(bucket).blob(blob_name).download_to_filename(local_path)
    return local_path


# ── High-level helpers used by frontend / worker ──────────────────────────────

def put_input_pdb(job_id: str, pdb_bytes: bytes, filename: str = "input.pdb") -> str:
    """Upload the user's PDB into the inputs bucket; returns its gs:// URI."""
    bucket = _require(PROTEINREDESIGN_INPUTS_BUCKET, "inputs")
    return upload_bytes(bucket, f"jobs/{job_id}/{filename}", pdb_bytes, content_type="chemical/x-pdb")


def write_manifest(job_id: str, manifest_json: str) -> str:
    bucket = _require(PROTEINREDESIGN_INPUTS_BUCKET, "inputs")
    return upload_bytes(
        bucket, f"jobs/{job_id}/manifest.json", manifest_json.encode(),
        content_type="application/json",
    )


def read_manifest(uri: str) -> str:
    return download_bytes(uri).decode()


def write_output(job_id: str, name: str, data: bytes, content_type: str | None = None) -> str:
    """Write a result artifact under the job's output prefix; returns its gs:// URI."""
    bucket = _require(PROTEINREDESIGN_OUTPUTS_BUCKET, "outputs")
    return upload_bytes(bucket, f"jobs/{job_id}/{name}", data, content_type=content_type)


def output_uri(job_id: str, name: str) -> str:
    return _gs_uri(_require(PROTEINREDESIGN_OUTPUTS_BUCKET, "outputs"), f"jobs/{job_id}/{name}")


def list_outputs(job_id: str) -> list[str]:
    """List gs:// URIs of all artifacts under a job's output prefix."""
    bucket = _require(PROTEINREDESIGN_OUTPUTS_BUCKET, "outputs")
    client = get_gcs_client()
    prefix = f"jobs/{job_id}/"
    return [_gs_uri(bucket, b.name) for b in client.bucket(bucket).list_blobs(prefix=prefix)]


# ── Weights (A6: mounted from GCS, else downloaded to a local cache) ───────────

def ensure_weights(subdir: str, local_cache: str = "/tmp/proteinredesign-weights") -> str:
    """
    Return a local path to a weights directory/file.

    - If PROTEINREDESIGN_WEIGHTS_MOUNT is set (e.g. a gcsfuse mount), return the path under it.
    - Otherwise download the blob(s) under `subdir` from the weights bucket into a
      local cache and return that path.
    """
    if PROTEINREDESIGN_WEIGHTS_MOUNT:
        return os.path.join(PROTEINREDESIGN_WEIGHTS_MOUNT, subdir)

    bucket = _require(PROTEINREDESIGN_WEIGHTS_BUCKET, "weights")
    dest_root = os.path.join(local_cache, subdir)
    if os.path.exists(dest_root) and os.listdir(dest_root):
        return dest_root  # already cached

    client = get_gcs_client()
    for blob in client.bucket(bucket).list_blobs(prefix=f"{subdir}/"):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(f"{subdir}/"):]
        dest = os.path.join(dest_root, rel)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        blob.download_to_filename(dest)
    return dest_root
