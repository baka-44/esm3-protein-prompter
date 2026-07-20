# proteinredesign backend infra (M0). Provisions everything WITHIN an already-created
# dedicated project (var.project_id) that lives in its own folder with billing
# attached (D4 — done out-of-band so this applies without org-admin rights).
#
# Scale-to-zero by construction: the worker is a Cloud Run *Job* (runs on demand,
# nothing standing). Buckets have NO lifecycle deletion (D4 — keep fold data + weights).

locals {
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "firestore.googleapis.com",
    "cloudscheduler.googleapis.com",   # M5 review reminder
    "cloudfunctions.googleapis.com",   # M5 review reminder
    "iam.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)
  service  = each.value

  disable_on_destroy = false
}

# ── Storage — inputs / outputs / weights (no lifecycle deletion — D4) ──────────
resource "google_storage_bucket" "inputs" {
  name                        = "${var.bucket_prefix}-inputs"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
  depends_on                  = [google_project_service.apis]
}

resource "google_storage_bucket" "outputs" {
  name                        = "${var.bucket_prefix}-outputs"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
  depends_on                  = [google_project_service.apis]
}

resource "google_storage_bucket" "weights" {
  name                        = "${var.bucket_prefix}-weights"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
  depends_on                  = [google_project_service.apis]
}

# ── Firestore (Native) — job store ────────────────────────────────────────────
resource "google_firestore_database" "jobs" {
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  depends_on  = [google_project_service.apis]
}

# ── Artifact Registry — worker image ──────────────────────────────────────────
resource "google_artifact_registry_repository" "proteinredesign" {
  repository_id = var.artifact_repo
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ── Service account for the worker job ────────────────────────────────────────
resource "google_service_account" "worker" {
  account_id   = "proteinredesign-worker"
  display_name = "proteinredesign generation worker"
}

# Worker: read/write buckets + Firestore.
resource "google_storage_bucket_iam_member" "worker_inputs" {
  bucket = google_storage_bucket.inputs.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_storage_bucket_iam_member" "worker_weights" {
  bucket = google_storage_bucket.weights.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_storage_bucket_iam_member" "worker_outputs" {
  bucket = google_storage_bucket.outputs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_project_iam_member" "worker_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.worker.email}"
}

# ── Frontend SA: write inputs, read outputs, Firestore, trigger the job ────────
resource "google_storage_bucket_iam_member" "frontend_inputs" {
  bucket = google_storage_bucket.inputs.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.frontend_service_account}"
}

resource "google_storage_bucket_iam_member" "frontend_outputs" {
  bucket = google_storage_bucket.outputs.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.frontend_service_account}"
}

resource "google_project_iam_member" "frontend_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${var.frontend_service_account}"
}

resource "google_project_iam_member" "frontend_run_invoker" {
  project = var.project_id
  role    = "roles/run.developer" # execute Cloud Run Jobs with overrides
  member  = "serviceAccount:${var.frontend_service_account}"
}

resource "google_service_account_iam_member" "frontend_act_as_worker" {
  service_account_id = google_service_account.worker.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${var.frontend_service_account}"
}

# ── Cloud Run Job — GPU worker (scale-to-zero; L4) ────────────────────────────
resource "google_cloud_run_v2_job" "worker" {
  name                = var.job_name
  location            = var.region
  deletion_protection = false

  template {
    template {
      timeout         = "3600s"
      max_retries     = 1
      service_account = google_service_account.worker.email

      gpu_zonal_redundancy_disabled = true
      node_selector {
        accelerator = "nvidia-l4"
      }

      containers {
        image = var.worker_image
        resources {
          limits = {
            cpu              = "8"
            memory           = "32Gi"
            "nvidia.com/gpu" = "1"
          }
        }
        env {
          name  = "GCP_PROJECT"
          value = var.project_id
        }
        env {
          name  = "GCP_REGION"
          value = var.region
        }
        env {
          name  = "PROTEINREDESIGN_INPUTS_BUCKET"
          value = google_storage_bucket.inputs.name
        }
        env {
          name  = "PROTEINREDESIGN_OUTPUTS_BUCKET"
          value = google_storage_bucket.outputs.name
        }
        env {
          name  = "PROTEINREDESIGN_WEIGHTS_BUCKET"
          value = google_storage_bucket.weights.name
        }
      }
    }
  }

  depends_on = [google_project_service.apis]
}
