variable "project_id" {
  type        = string
  description = "Dedicated GCP project for the cofold backend (created in its own folder — D4)."
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "bucket_prefix" {
  type        = string
  description = "Prefix for the (globally-unique) GCS bucket names, e.g. \"phyx44-cofold\"."
}

variable "artifact_repo" {
  type    = string
  default = "cofold"
}

variable "job_name" {
  type    = string
  default = "cofold-worker"
}

variable "worker_image" {
  type        = string
  description = "Full Artifact Registry image ref for the worker (built from cofold/Dockerfile.worker)."
  # e.g. us-central1-docker.pkg.dev/<project>/cofold/worker:latest
}

variable "frontend_service_account" {
  type        = string
  description = "Email of the Streamlit frontend's runtime SA (existing prot-prompt service). Granted job-trigger + bucket + Firestore access."
}
