#!/bin/bash
# Creates the ACA job that supplies [self-hosted, aca] runners for this repo.
#
# Usage:
#   GITHUB_PAT=ghp_xxx bash .github/aca-runner/setup.sh
#
# Optional overrides: JOB_NAME, CPU, MEMORY, MAX_EXECUTIONS, IMAGE_TAG,
#                     RESOURCE_GROUP, ACA_ENV, ACR, REPO
#
# Prereqs:
#   - az login, with Contributor on the resource group below
#   - GITHUB_PAT with repo scope and admin on REPO. It is used both to
#     register runners (entrypoint.sh) and for KEDA queue polling.
#   - The shared ACA environment already exists. It is provisioned by
#     aintropy-engine-product/aca-runner/setup.sh, which also created the
#     resource group, Log Analytics workspace, and ACR. This script only adds
#     a job to it and never mutates those shared resources.
#
# Why this repo builds its own runner image rather than reusing
# aintropy-engine-product's: kuration's CI runs `import cv2` and
# `import faiss`, which need libgl1/libglib2.0-0/libgomp1. That image ships
# none of them, and carries Azure CLI + kubectl that this repo never calls.
# See the Dockerfile.
#
# Re-run this after changing the Dockerfile or entrypoint.sh: it rebuilds and
# pushes the image. If the job already exists, only the image is refreshed —
# in-flight runners finish their job on the old image, and the next replica
# picks up the new one.
#
# Emergency rollback: set runs-on: ubuntu-latest in the affected workflows.
# CI resumes on GitHub-hosted runners within minutes.

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-kurious-rg-dev}"
ACA_ENV="${ACA_ENV:-kurious-cicd-env}"
ACR="${ACR:-kuriouscr001}"
IMAGE_TAG="${IMAGE_TAG:-kurious-ns/kuration-runner:latest}"
RUNNER_IMAGE="${ACR}.azurecr.io/${IMAGE_TAG}"
REPO="${REPO:-aintropy-ai/kuration-sdk}"
JOB_NAME="${JOB_NAME:-github-runner-kuration-sdk}"

# 4 CPU / 8Gi is the Consumption-profile ceiling. ci.yml queues ~30
# self-hosted jobs per run and several install torch, so this is sized for
# the heaviest job rather than the median one.
CPU="${CPU:-4}"
MEMORY="${MEMORY:-8Gi}"
MAX_EXECUTIONS="${MAX_EXECUTIONS:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${GITHUB_PAT:-}" ]; then
  echo "ERROR: GITHUB_PAT must be set"
  exit 1
fi

# Fail before touching Azure if the PAT cannot mint a registration token,
# otherwise the job is created and every replica dies in entrypoint.sh.
echo "=== 1. Verify PAT can register runners on ${REPO} ==="
HTTP_CODE=$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
  -H "Authorization: token ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${REPO}/actions/runners/registration-token")

if [ "$HTTP_CODE" != "201" ]; then
  echo "ERROR: registration-token returned HTTP ${HTTP_CODE} (expected 201)."
  echo "The PAT needs repo scope and admin on ${REPO}."
  exit 1
fi
echo "OK"

echo "=== 2. Verify shared ACA environment '${ACA_ENV}' exists ==="
echo "Subscription: $(az account show --query 'name' -o tsv 2>/dev/null || echo '(az account show failed — not logged in?)')"
echo "Tenant:       $(az account show --query 'tenantId' -o tsv 2>/dev/null || echo '?')"

# Surface the real az error rather than reporting every failure as "not found":
# a wrong subscription, a missing containerapp extension, an expired login and
# an absent environment all fail here for different reasons.
if ! ENV_ERR=$(az containerapp env show \
      --name "$ACA_ENV" --resource-group "$RESOURCE_GROUP" -o none 2>&1); then
  echo "ERROR: could not read ACA environment '${ACA_ENV}' in '${RESOURCE_GROUP}'."
  echo "az said:"
  echo "${ENV_ERR}" | sed 's/^/  /'
  echo ""
  echo "Common causes, in order:"
  echo "  1. Wrong subscription selected. The environment lives wherever the"
  echo "     engine-product 'github-runner' job runs:"
  echo "       az account list -o table"
  echo "       az account set --subscription <name-or-id>"
  echo "  2. Environment lives elsewhere. Find it, then re-run with overrides:"
  echo "       az resource list --resource-type Microsoft.App/managedEnvironments \\"
  echo "         --query '[].{name:name,group:resourceGroup,location:location}' -o table"
  echo "       RESOURCE_GROUP=<rg> ACA_ENV=<env> GITHUB_PAT=... bash ${BASH_SOURCE[0]}"
  echo "  3. containerapp extension missing:  az extension add -n containerapp"
  echo ""
  echo "The environment is created by aintropy-engine-product/aca-runner/setup.sh;"
  echo "this script only adds a job to it."
  exit 1
fi
echo "OK"

echo "=== 3. Build and push runner image ==="
az acr build \
  --registry "$ACR" \
  --image "$IMAGE_TAG" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  "$SCRIPT_DIR"

if az containerapp job show --name "$JOB_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "=== 4. Job '${JOB_NAME}' exists — image refreshed, leaving config alone ==="
  echo "Delete it first if you need to change CPU/memory/scale settings:"
  echo "  az containerapp job delete -n ${JOB_NAME} -g ${RESOURCE_GROUP} --yes"
  exit 0
fi

echo "=== 4. Create ACA Job '${JOB_NAME}' ==="
az containerapp job create \
  --name "$JOB_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ACA_ENV" \
  --trigger-type "Event" \
  --replica-timeout 3600 \
  --replica-retry-limit 1 \
  --replica-completion-count 1 \
  --parallelism 1 \
  --min-executions 0 \
  --max-executions "$MAX_EXECUTIONS" \
  --image "$RUNNER_IMAGE" \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --registry-server "${ACR}.azurecr.io" \
  --registry-identity system \
  --secrets "github-pat=${GITHUB_PAT}" \
  --env-vars "GITHUB_PAT=secretref:github-pat" "RUNNER_REPO=${REPO}" \
  --scale-rule-name "github-runner-scaler" \
  --scale-rule-type "github-runner" \
  `# enableEtags is deliberately left at the extension default (true): the` \
  `# scaler sends If-None-Match and a 304 costs it almost nothing, which` \
  `# keeps steady-state polling well under its 3s HTTP timeout. Do not set` \
  `# it to false — that forces a full response on every poll. The timeout` \
  `# this fleet actually hit came from job-list size, fixed by splitting` \
  `# ci.yml; see .github/aca-runner/ci-workflow-split-plan.md.` \
  --scale-rule-metadata \
      "githubAPIURL=https://api.github.com" \
      "owner=${REPO%/*}" \
      "runnerScope=repo" \
      "repos=${REPO##*/}" \
      "labels=self-hosted,aca" \
      "targetWorkflowQueueLength=1" \
  --scale-rule-auth "personalAccessToken=github-pat"

echo "=== 5. Assign AcrPull to job managed identity ==="
PRINCIPAL_ID=$(az containerapp job show \
  --name "$JOB_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query identity.principalId -o tsv)

ACR_ID=$(az acr show --name "$ACR" --query id -o tsv)

az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role AcrPull \
  --scope "$ACR_ID"

echo ""
echo "=== Done ==="
echo "ACA job '${JOB_NAME}' created in '${ACA_ENV}'."
echo "KEDA polls ${REPO} and scales to ${MAX_EXECUTIONS} runners"
echo "(${CPU} CPU / ${MEMORY} each) for jobs labelled: self-hosted,aca"
echo ""
echo "Verify by re-running a queued workflow, then:"
echo "  az containerapp job execution list -n ${JOB_NAME} -g ${RESOURCE_GROUP} -o table"
echo ""
echo "NOTE: GITHUB_PAT is used for both runner registration and KEDA polling,"
echo "and its expiry takes this repo's CI down. Long-term: a GitHub App, or two"
echo "fine-grained PATs (Actions:read for KEDA, administration:write for"
echo "registration), owned by a machine account rather than a person."
