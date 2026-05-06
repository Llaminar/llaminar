#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    repo_root="$1"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/.." && pwd)"
fi

pattern="KernelFactory::getOrCreatePreparedEmbeddingWeights|getOrCreatePreparedEmbeddingWeights\(|KernelFactory::getPreparedEmbeddingWeights|getPreparedEmbeddingWeights\(|KernelFactory::preparedEmbeddingRegistrySize|preparedEmbeddingRegistrySize\(|prepared_embedding_registry_|PreparedEmbeddingKey"

scan_paths=(
    "${repo_root}/src/v2"
    "${repo_root}/tests/v2"
)

if command -v rg >/dev/null 2>&1; then
    if rg -n "${pattern}" "${scan_paths[@]}" >&2; then
        echo "" >&2
        echo "KernelFactory prepared embedding registry APIs are deleted." >&2
        echo "Route embedding preparation through PreparedWeightStore or KernelFactory::prepareEmbeddingHandleLocal()." >&2
        exit 1
    fi
else
    if grep -R -n -E "${pattern}" "${scan_paths[@]}" >&2; then
        echo "" >&2
        echo "KernelFactory prepared embedding registry APIs are deleted." >&2
        echo "Route embedding preparation through PreparedWeightStore or KernelFactory::prepareEmbeddingHandleLocal()." >&2
        exit 1
    fi
fi

echo "Prepared-embedding registry guard passed: no source or test references to deleted KernelFactory APIs."