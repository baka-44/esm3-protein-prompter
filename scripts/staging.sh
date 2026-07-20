#!/usr/bin/env bash
#
# staging.sh — Cloud Run revision-tag staging flow (Option A) for the prot-prompt service.
#
# Validate a new revision at its own URL before it takes production traffic, then
# promote the SAME validated revision (build once, promote — no rebuild for prod).
#
#   scripts/staging.sh deploy     # build + deploy a "staging"-tagged revision, no prod traffic
#   scripts/staging.sh url        # print the staging URL to validate
#   scripts/staging.sh promote    # shift 100% prod traffic to the validated staging revision
#   scripts/staging.sh rollback   # list revisions and shift traffic to a chosen previous one
#
# Env overrides: SERVICE, PROJECT, REGION.
set -euo pipefail

SERVICE="${SERVICE:-prot-prompt}"
PROJECT="${PROJECT:-phyx44-pp-codonlm-v1}"
REGION="${REGION:-us-central1}"

SECRETS="ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,FORGE_API_TOKEN=FORGE_API_TOKEN:latest,GOOGLE_CLIENT_ID=GOOGLE_CLIENT_ID:latest,GOOGLE_CLIENT_SECRET=GOOGLE_CLIENT_SECRET:latest,OAUTH_REDIRECT_URI=OAUTH_REDIRECT_URI:latest,ALLOWED_EMAILS=ALLOWED_EMAILS:latest"

_base=(--project "$PROJECT" --region "$REGION")

cmd="${1:-}"
case "$cmd" in
  deploy)
    echo ">> Deploying a 'staging'-tagged revision (no production traffic)…"
    gcloud run deploy "$SERVICE" "${_base[@]}" \
      --source . \
      --no-traffic --tag staging \
      --memory 4Gi --cpu 2 --timeout 600 \
      --set-secrets="$SECRETS"
    # Re-apply IAM (project practice: re-apply after every deploy).
    gcloud run services add-iam-policy-binding "$SERVICE" "${_base[@]}" \
      --member="allUsers" --role="roles/run.invoker" >/dev/null
    echo
    "$0" url
    ;;

  url)
    gcloud run services describe "$SERVICE" "${_base[@]}" \
      --format='value(status.traffic[].url)' | tr ';' '\n' | grep -i staging \
      || { echo "No staging-tagged revision found. Run: scripts/staging.sh deploy"; exit 1; }
    ;;

  promote)
    echo ">> Promoting the staging-tagged revision to 100% production traffic…"
    gcloud run services update-traffic "$SERVICE" "${_base[@]}" --to-tags staging=100
    ;;

  rollback)
    echo "Recent revisions:"
    gcloud run revisions list --service "$SERVICE" "${_base[@]}" --limit 10
    echo
    read -r -p "Revision to route 100% traffic to: " rev
    [ -n "$rev" ] && gcloud run services update-traffic "$SERVICE" "${_base[@]}" --to-revisions "$rev=100"
    ;;

  *)
    echo "usage: scripts/staging.sh {deploy|url|promote|rollback}" >&2
    exit 2
    ;;
esac
