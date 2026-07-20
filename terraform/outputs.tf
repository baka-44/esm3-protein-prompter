output "inputs_bucket" {
  value = google_storage_bucket.inputs.name
}

output "outputs_bucket" {
  value = google_storage_bucket.outputs.name
}

output "weights_bucket" {
  value = google_storage_bucket.weights.name
}

output "worker_service_account" {
  value = google_service_account.worker.email
}

output "job_name" {
  value = google_cloud_run_v2_job.worker.name
}

output "artifact_registry_repo" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.cofold.repository_id}"
}

# Env vars to set on the existing Streamlit (prot-prompt) Cloud Run service so it
# can submit jobs and read the dashboard:
#   GCP_PROJECT, GCP_REGION, COFOLD_JOB_NAME,
#   COFOLD_INPUTS_BUCKET, COFOLD_OUTPUTS_BUCKET, COFOLD_WEIGHTS_BUCKET
output "frontend_env_hint" {
  value = {
    GCP_PROJECT           = var.project_id
    GCP_REGION            = var.region
    COFOLD_JOB_NAME       = var.job_name
    COFOLD_INPUTS_BUCKET  = google_storage_bucket.inputs.name
    COFOLD_OUTPUTS_BUCKET = google_storage_bucket.outputs.name
    COFOLD_WEIGHTS_BUCKET = google_storage_bucket.weights.name
  }
}
