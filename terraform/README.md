# proteinredesign backend — Terraform (M0)

Provisions the RFdiffusion/MPNN backend inside a **dedicated project** (D4).

## Done out-of-band first (needs org/billing rights)
1. Create a dedicated **project** under the org (no folders in this org); attach billing
   (D4 — single audit line). E.g.
   `gcloud projects create phyx44-proteinredesign-v1 --organization=626836951011`
   then `gcloud billing projects link phyx44-proteinredesign-v1 --billing-account=01EF81-EA4098-55A36D`.
2. Request **NVIDIA L4 GPU quota** in the region (lead time).
3. Build & push the worker image:
   ```bash
   gcloud builds submit --tag <region>-docker.pkg.dev/<project>/proteinredesign/worker \
     -f proteinredesign/Dockerfile.worker .
   ```
   (Or build locally and `docker push`. The Artifact Registry repo is created by Terraform,
   so push after the first apply — or create the repo first.)
4. Upload model weights to `gs://<bucket_prefix>-weights/` under `mpnn/` and `esmfold/` (A6).

## Apply
```bash
cd terraform
cat > terraform.tfvars <<'EOF'
project_id               = "phyx44-proteinredesign-v1"
region                   = "us-central1"
bucket_prefix            = "phyx44-proteinredesign"
worker_image             = "us-central1-docker.pkg.dev/phyx44-proteinredesign-v1/proteinredesign/worker:latest"
frontend_service_account = "<prot-prompt runtime SA email>"
EOF

terraform init
terraform plan
terraform apply
```

## After apply
- Set the env vars from `terraform output frontend_env_hint` on the existing **prot-prompt**
  Cloud Run service so the Streamlit frontend can submit jobs + read the dashboard.
- Confirm scale-to-zero (nothing runs until a job is triggered) and the single billing line.

## Not included yet (later increments)
- M5: Cloud Scheduler (6-month) → Cloud Function email reminder (D4); budget alerts declined (D9).
- RFdiffusion3 + presets #2/#3/#6/#8 (M2–M4).
