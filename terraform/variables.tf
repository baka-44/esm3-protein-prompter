variable "project_id" {
  type        = string
  description = "Dedicated GCP project for the proteinredesign backend (created in its own folder — D4)."
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "bucket_prefix" {
  type        = string
  description = "Prefix for the (globally-unique) GCS bucket names, e.g. \"phyx44-proteinredesign\"."
}

variable "artifact_repo" {
  type    = string
  default = "proteinredesign"
}

variable "job_name" {
  type    = string
  default = "proteinredesign-worker"
}

variable "worker_image" {
  type        = string
  description = "Full Artifact Registry image ref for the worker (built from proteinredesign/Dockerfile.worker)."
  # e.g. us-central1-docker.pkg.dev/<project>/proteinredesign/worker:latest
}

variable "frontend_service_account" {
  type        = string
  description = "Email of the Streamlit frontend's runtime SA (existing prot-prompt service). Granted job-trigger + bucket + Firestore access."
}
