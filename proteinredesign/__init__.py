"""
proteinredesign — RFdiffusion + MPNN generation backend.

A self-hosted, commercially-clean (BSD/MIT/CC-BY-4.0) generation pipeline that
runs as async GPU jobs (Cloud Run Jobs), decoupled from the Streamlit frontend.

Design + all decisions: docs/plans/rfdiffusion_mpnn_backend.md (D1–D9).

Package layout:
  manifest.py          — job manifest schema (Preset, JobStatus, JobManifest)
  config_builders/     — per-preset deterministic config builders (code, never LLM)
  storage.py           — GCS I/O + weights-mount helper
  jobstore.py          — Firestore job records (user-tagged, status lifecycle)
  worker.py            — pipeline entrypoint (MPNN → scoring → ESMFold QC → rank)

MVP scope (D6): presets #1, #2, #3, #6, #8 on RFdiffusion3 + ProteinMPNN family
+ ESMFold. Increment 1 (this code) implements preset #1 (fixed-backbone redesign),
which is MPNN-only and proves the end-to-end async spine.
"""

__all__ = ["manifest", "config_builders"]
