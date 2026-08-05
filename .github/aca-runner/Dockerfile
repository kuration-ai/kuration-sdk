FROM ubuntu:22.04

ARG RUNNER_VERSION=2.334.0
ARG RUNNER_ARCH=x64

ENV DEBIAN_FRONTEND=noninteractive

# Runner prerequisites plus the system libraries kuration's optional extras
# link against.
#
#   libgl1, libglib2.0-0  Required. opencv-python (non-headless) is pulled in
#                         by [pixel-shift], [action-state] and [slam], and
#                         ci.yml runs `import cv2` for each. Without these it
#                         fails at import (opencv 5.x: libxcb.so.1, which
#                         libgl1 pulls in transitively). Verified by building
#                         this image and importing cv2 against a control image
#                         using the engine-product package set, where it fails.
#   libgomp1              Defensive. faiss-cpu ([slam]) links OpenMP, and
#                         libgomp1 is absent from ubuntu:22.04 — it arrives
#                         only as a transitive dep. faiss-cpu 1.15 was observed
#                         importing without it explicitly listed, so this is
#                         insurance against a wheel that does need it, not a
#                         confirmed requirement.
#
# torch, av, pyarrow, h5py, ray and the rest ship complete manylinux wheels.
#
# Deliberately no Azure CLI, kubectl, or docker: this image only runs pytest.
# Deliberately no sudo — the runner user is unprivileged, so jobs needing
# `apt-get install` (e.g. `playwright install --with-deps`) must stay on
# GitHub-hosted runners. ci.yml documents that for the notebook-ui job.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git jq unzip tar \
    python3 \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# GitHub Actions runner — installed as root, ownership dropped below.
RUN mkdir -p /runner && cd /runner \
    && curl -fsSL \
       "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz" \
       | tar -xz \
    && ./bin/installdependencies.sh

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# actions/setup-python installs interpreters (3.11-3.14) into the tool cache,
# so it must exist and be writable by the unprivileged runner user.
ENV RUNNER_TOOL_CACHE=/opt/hostedtoolcache
RUN mkdir -p "$RUNNER_TOOL_CACHE"

RUN useradd -m -s /bin/bash runner \
    && chown -R runner:runner /runner "$RUNNER_TOOL_CACHE"

WORKDIR /runner
USER runner

ENTRYPOINT ["/entrypoint.sh"]
