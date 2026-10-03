#!/usr/bin/env bash
# Register AMD's signed Core SDK repository, without changing the host driver.
# All package profiles use this one repository and the companion release pin.
# Temporary files are private to this invocation; nothing shares /tmp names.
set -euo pipefail

rocm_repo_work="$(mktemp -d /tmp/llaminar-rocm-repository.XXXXXXXX)"
trap 'rm -rf -- "${rocm_repo_work}"' EXIT
curl -fsSL --retry 5 --connect-timeout 30 \
    https://stable.repo.amd.com/rocm/gpg/packages.gpg \
    -o "${rocm_repo_work}/packages.gpg"
gpg --batch --yes --dearmor -o "${rocm_repo_work}/amdrocm.gpg" \
    "${rocm_repo_work}/packages.gpg"
install -d -m 755 /etc/apt/keyrings /etc/apt/sources.list.d
install -m 644 "${rocm_repo_work}/amdrocm.gpg" /etc/apt/keyrings/amdrocm.gpg
printf '%s\n' \
    'Types: deb' \
    'URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/' \
    'Suites: stable' \
    'Components: main' \
    'Architectures: amd64' \
    'Signed-By: /etc/apt/keyrings/amdrocm.gpg' \
    > "${rocm_repo_work}/amdrocm-stable.sources"
install -m 644 "${rocm_repo_work}/amdrocm-stable.sources" \
    /etc/apt/sources.list.d/amdrocm-stable.sources
