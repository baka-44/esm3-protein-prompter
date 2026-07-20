"""
proteinredesign/jobstore.py — Firestore-backed job store.

Job records are durable (survive frontend refresh / re-login — B7) and
**user-linked** (tagged by owner email — A7; no per-user quotas). The frontend
job dashboard lists a user's own jobs and polls their status; the worker updates
status through the lifecycle queued → running → done/failed.

Firestore collection name from config.PROTEINREDESIGN_JOBS_COLLECTION (default "proteinredesign_jobs").
Document id = job_id.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from config import PROTEINREDESIGN_JOBS_COLLECTION, get_firestore_client
from proteinredesign.manifest import JobManifest, JobStatus, Preset


@dataclass
class JobRecord:
    """A row in the job dashboard. Mirrors + tracks a JobManifest through its lifecycle."""

    job_id: str
    user_email: str
    preset: str
    status: str = JobStatus.QUEUED.value
    stage: str = ""                 # human-readable current stage, e.g. "ProteinMPNN"
    progress: float = 0.0           # 0.0–1.0
    title: str = ""                 # short label for the dashboard row
    manifest_uri: str = ""          # gs:// URI of the manifest
    result_uri: str = ""            # gs:// URI of the results JSON when done
    error: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.DONE.value, JobStatus.FAILED.value)


def _collection():
    return get_firestore_client().collection(PROTEINREDESIGN_JOBS_COLLECTION)


def create_job(manifest: JobManifest, manifest_uri: str = "", title: str = "") -> JobRecord:
    """Create a QUEUED job record from a manifest and persist it."""
    rec = JobRecord(
        job_id=manifest.job_id,
        user_email=manifest.user_email,
        preset=manifest.preset.value,
        status=JobStatus.QUEUED.value,
        title=title or _default_title(manifest.preset),
        manifest_uri=manifest_uri,
        created_at=manifest.created_at,
        updated_at=time.time(),
    )
    _collection().document(rec.job_id).set(rec.to_dict())
    return rec


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    result_uri: str | None = None,
    error: str | None = None,
) -> None:
    """Patch mutable fields on a job record."""
    patch: dict[str, Any] = {"updated_at": time.time()}
    if status is not None:
        patch["status"] = status
    if stage is not None:
        patch["stage"] = stage
    if progress is not None:
        patch["progress"] = float(progress)
    if result_uri is not None:
        patch["result_uri"] = result_uri
    if error is not None:
        patch["error"] = error
    _collection().document(job_id).update(patch)


def get_job(job_id: str) -> JobRecord | None:
    snap = _collection().document(job_id).get()
    return JobRecord.from_dict(snap.to_dict()) if snap.exists else None


def list_jobs_for_user(user_email: str, limit: int = 50) -> list[JobRecord]:
    """Most-recent-first jobs owned by this user (dashboard view)."""
    try:
        from google.cloud.firestore import Query
        direction = Query.DESCENDING
    except ImportError:
        direction = "DESCENDING"
    q = (
        _collection()
        .where("user_email", "==", user_email)
        .order_by("created_at", direction=direction)
        .limit(limit)
    )
    return [JobRecord.from_dict(doc.to_dict()) for doc in q.stream()]


def _default_title(preset: Preset) -> str:
    return {
        Preset.FIXED_BACKBONE_REDESIGN: "Fixed-backbone redesign",
        Preset.LIGAND_AWARE_REDESIGN: "Ligand-aware redesign",
        Preset.MOTIF_SCAFFOLDING: "Motif scaffolding",
        Preset.ENZYME_ACTIVE_SITE: "Enzyme active-site scaffolding",
        Preset.SCAFFOLD_DIVERSIFICATION: "Scaffold diversification",
    }.get(preset, preset.value)
