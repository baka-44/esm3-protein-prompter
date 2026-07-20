# cofold backend — Terraform (M0)

Provisions the RFdiffusion/MPNN backend inside a **dedicated project** (D4).

## Done out-of-band first (needs org/billing rights)
1. Create a **folder** and a **project** inside it; attach billing (D4 — single audit line).
2. Request **NVIDIA L4 GPU quota** in the region (lead time).
3. Build & push the worker image:
   ```bash
   gcloud builds submit --tag <region>-docker.pkg.dev/<project>/cofold/worker \
     -f cofold/Dockerfile.worker .
   ```
   (Or build locally and `docker push`. The Artifact Registry repo is created by Terraform,
   so push after the first apply — or create the repo first.)
4. Upload model weights to `gs://<bucket_prefix>-weights/` under `mpnn/` and `esmfold/` (A6).

## Apply
```bash
cd terraform
cat > terraform.tfvars <<'EOF'
project_id               = "phyx44-cofold"
region                   = "us-central1"
bucket_prefix            = "phyx44-cofold"
worker_image             = "us-central1-docker.pkg.dev/phyx44-cofold/cofold/worker:latest"
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
