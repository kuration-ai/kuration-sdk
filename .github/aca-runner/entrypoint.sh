#!/bin/bash
# Registers one ephemeral self-hosted runner, runs a single job, exits.
# KEDA starts a fresh replica per queued job, so each job gets a clean
# container and no state carries between them.
set -e

REPO="${RUNNER_REPO:-aintropy-ai/kuration-sdk}"
API="https://api.github.com/repos/${REPO}/actions/runners/registration-token"

echo "Runner target repo: ${REPO}"

if [ -z "${GITHUB_PAT:-}" ]; then
  echo "ERROR: GITHUB_PAT is not set."
  exit 1
fi

echo "Requesting runner registration token..."
TOKEN=$(curl -sS -X POST \
  -H "Authorization: token ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github+json" \
  "${API}" | jq -r .token)

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
  echo "ERROR: Failed to get registration token. Check GITHUB_PAT scope and"
  echo "that it has admin on ${REPO}."
  exit 1
fi

echo "Configuring runner..."
./config.sh \
  --url "https://github.com/${REPO}" \
  --token "$TOKEN" \
  --name "kuration-runner-$(hostname)" \
  --labels "self-hosted,aca,Linux,X64" \
  --ephemeral \
  --unattended \
  --disableupdate

echo "Starting runner..."
./run.sh
