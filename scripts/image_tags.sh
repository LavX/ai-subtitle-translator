#!/usr/bin/env bash
# Compute the image tags for build-docker.yml. latest moves only for a final release tag
# (vX.Y.Z) or an explicit push_latest=true, never for a release candidate or a build.
set -euo pipefail

if [ -n "${VERSION_INPUT:-}" ]; then
  VERSION="${VERSION_INPUT}"
elif [[ "${GITHUB_REF}" == refs/tags/v* ]]; then
  VERSION="${GITHUB_REF#refs/tags/}"
else
  VERSION="dev-${SHORT_SHA}"
fi
VERSION="${VERSION#v}"

BASE="${REGISTRY}/${IMAGE_NAME}"
TAGS="${BASE}:${VERSION},${BASE}:v${VERSION},${BASE}:sha-${SHORT_SHA},${BASE}:py${PYTHON_VERSION}${JIT_SUFFIX}"

if [ "${PUSH_LATEST:-false}" = "true" ] || [[ "${GITHUB_REF}" =~ ^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  TAGS="${TAGS},${BASE}:latest"
fi

echo "tags=${TAGS}"
echo "version=${VERSION}"
