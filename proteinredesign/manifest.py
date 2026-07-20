"""
proteinredesign/manifest.py — Job manifest schema for the RFdiffusion/MPNN backend.

A JobManifest is the self-contained input spec for one generation job. It is
written to GCS and referenced by a Firestore job record (see jobstore.py). The
worker reads the manifest, runs the pipeline, and writes results back to GCS.

Kept dependency-free (stdlib only) so it is trivially importable by both the
frontend (job submission) and the worker (job execution), and unit-testable.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Preset(str, Enum):
    """MVP task presets (D6). Values are stable string ids used in storage."""

    FIXED_BACKBONE_REDESIGN = "fixed_backbone_redesign"     # #1  MPNN-only
    LIGAND_AWARE_REDESIGN = "ligand_aware_redesign"         # #2  LigandMPNN
    MOTIF_SCAFFOLDING = "motif_scaffolding"                 # #3  RF3
    ENZYME_ACTIVE_SITE = "enzyme_active_site"               # #6  RF3-all-atom
    SCAFFOLD_DIVERSIFICATION = "scaffold_diversification"   # #8  RF3 partial diffusion


class JobStatus(str, Enum):
    """Job lifecycle states (mirrored in the Firestore job record)."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# Presets that do NOT require RFdiffusion (MPNN-only) — see taxonomy in the design doc.
MPNN_ONLY_PRESETS: frozenset[Preset] = frozenset(
    {Preset.FIXED_BACKBONE_REDESIGN, Preset.LIGAND_AWARE_REDESIGN}
)


@dataclass
class JobManifest:
    """
    Self-contained spec for one generation job.

    Attributes:
        preset:      Which task preset this job runs.
        user_email:  Owner (from OAuth). Jobs are user-linked (A7); no quotas.
        pdb_uri:     gs:// URI of the uploaded input PDB.
        params:      Preset-specific parameters produced by the config builder
                     (e.g. {"chain_id": "A", "fixed_positions": {"A": [12, 27]}}).
        num_outputs: Number of final QC-passed ranked outputs to return (B4 = 10).
        job_id:      Short unique id (also used as the GCS output prefix).
        created_at:  Unix timestamp (UTC seconds).
    """

    preset: Preset
    user_email: str
    pdb_uri: str
    params: dict[str, Any] = field(default_factory=dict)
    num_outputs: int = 10
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["preset"] = self.preset.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobManifest":
        return cls(
            preset=Preset(d["preset"]),
            user_email=d["user_email"],
            pdb_uri=d["pdb_uri"],
            params=d.get("params", {}),
            num_outputs=int(d.get("num_outputs", 10)),
            job_id=d.get("job_id") or uuid.uuid4().hex[:12],
            created_at=float(d.get("created_at", time.time())),
        )

    @classmethod
    def from_json(cls, s: str) -> "JobManifest":
        return cls.from_dict(json.loads(s))

    def output_prefix(self) -> str:
        """GCS object prefix for this job's outputs (under the outputs bucket)."""
        return f"jobs/{self.job_id}"

    def requires_rfdiffusion(self) -> bool:
        return self.preset not in MPNN_ONLY_PRESETS
