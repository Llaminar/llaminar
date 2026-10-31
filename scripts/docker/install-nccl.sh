#!/usr/bin/env bash
# Build the canonical NCCL dependency with resumed CUDA graph capture support.
#
# The capture repair fixes strong-stream membership; the live-row extension
# clips native payloads without changing transport or reduction order. A distinct SONAME prevents a system
# NCCL package from silently replacing the tested implementation at runtime.
# Native GIN auto selection must resolve its sentinel even when the probe finds
# no compatible devices. Its actual-source proof ships to model-free preflight.
# Devcontainers and release builders use this same installer; runtime images
# copy its shared library from the builder, without a runtime compiler.
#
# NCCL_INSTALL_PREFIX selects the installation prefix (default /usr/local).
# Docker's CUDA installer derives NVCC_GENCODE from Llaminar's CUDAARCHS
# contract, so the collective fatbin and application kernels name the same
# supported devices. A standalone local invocation may omit it deliberately to
# request NCCL's complete CUDA-toolkit architecture set.
set -euo pipefail

nccl_revision=dbc86fd06e8b0c4517b95d8958a09ccacf9520c9
nccl_tag=v2.28.9-1
nccl_version=2.28.9
nccl_prefix="${NCCL_INSTALL_PREFIX:-/usr/local}"
nccl_patch="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches/nccl-capture-reentry.patch"
nccl_rows_patch="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches/nccl-device-live-rows.patch"
nccl_gin_patch="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches/nccl-gin-auto-selection.patch"
nccl_thread_patch="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches/nccl-created-thread-ownership.patch"
nccl_thread_probe="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/nccl-thread-lifetime/proof.py"
nccl_gin_probe="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/nccl-gin-selection/proof.py"
nccl_shm_patch="$(dirname -- "${nccl_patch}")/nccl-partial-file-cleanup.patch"
nccl_shm_probe="$(dirname -- "${nccl_gin_probe}")/../shared-file-lifecycle/proof.py"
nccl_shm_report="${nccl_prefix}/share/llaminar-nccl/shared-file-lifecycle.json"
nccl_library="${nccl_prefix}/lib/libllaminar_nccl.so.${nccl_version}"
nccl_marker="${nccl_prefix}/lib/llaminar-nccl-build.txt"
nccl_gin_report="${nccl_prefix}/share/llaminar-nccl/gin-selection.json"
nccl_thread_report="${nccl_prefix}/share/llaminar-nccl/thread-lifetime.json"
nccl_identity="${nccl_revision}:$(sha256sum "${nccl_patch}" | cut -d ' ' -f 1):$(sha256sum "${nccl_rows_patch}" | cut -d ' ' -f 1):$(sha256sum "${nccl_gin_patch}" | cut -d ' ' -f 1):$(python3 "${nccl_gin_probe}" identity):${NVCC_GENCODE:-toolkit-default}"

# Only an exact source/patch/architecture receipt can reuse an installation.
# This tiny patch identity has no relationship to model-weight cache hashing.
if [[ -f "${nccl_library}" && -f "${nccl_marker}" && -f "${nccl_gin_report}" && -f "${nccl_thread_report}" && -f "${nccl_shm_report}" &&
      "$(<"${nccl_marker}")" == "${nccl_identity}" ]]; then
    python3 "${nccl_gin_probe}" verify --report "${nccl_gin_report}" \
        --library "${nccl_library}" --receipt "${nccl_marker}"
    python3 "${nccl_thread_probe}" verify --report "${nccl_thread_report}" \
        --library "${nccl_library}" --receipt "${nccl_marker}"
    python3 "${nccl_shm_probe}" verify --backend nccl --report "${nccl_shm_report}" \
        --library "${nccl_library}" --receipt "${nccl_marker}"
    echo "NCCL capture-reentry dependency already installed: ${nccl_library}"
    exit 0
fi

nccl_build_dir="$(mktemp -d /tmp/llaminar-nccl-build.XXXXXXXX)"
# This is exclusively the unique directory created above, never a caller's
# source checkout, model cache, or shared build tree.
trap 'rm -rf -- "${nccl_build_dir}"' EXIT
git clone --depth 1 --branch "${nccl_tag}" \
    https://github.com/NVIDIA/nccl.git "${nccl_build_dir}/source"
[[ "$(git -C "${nccl_build_dir}/source" rev-parse HEAD)" == "${nccl_revision}" ]]
git -C "${nccl_build_dir}/source" apply --check "${nccl_patch}"
git -C "${nccl_build_dir}/source" apply "${nccl_patch}"
git -C "${nccl_build_dir}/source" apply --check "${nccl_rows_patch}"
git -C "${nccl_build_dir}/source" apply "${nccl_rows_patch}"
git -C "${nccl_build_dir}/source" apply --check "${nccl_gin_patch}"
git -C "${nccl_build_dir}/source" apply "${nccl_gin_patch}"
git -C "${nccl_build_dir}/source" apply --check "${nccl_thread_patch}"
git -C "${nccl_build_dir}/source" apply "${nccl_thread_patch}"
git -C "${nccl_build_dir}/source" apply --check "${nccl_shm_patch}"
git -C "${nccl_build_dir}/source" apply "${nccl_shm_patch}"
make -C "${nccl_build_dir}/source" -j"$(nproc)" src.build \
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" LIBNAME=libllaminar_nccl.so
printf '%s\n' "${nccl_identity}" > "${nccl_build_dir}/build-receipt.txt"
python3 "${nccl_gin_probe}" compile --source "${nccl_build_dir}/source" \
    --library "${nccl_build_dir}/source/build/lib/libllaminar_nccl.so.${nccl_version}" \
    --receipt "${nccl_build_dir}/build-receipt.txt" \
    --sdk "${CUDA_HOME:-/usr/local/cuda}" --compiler "${CXX:-g++}" \
    --report "${nccl_build_dir}/gin-selection.json"
python3 "${nccl_thread_probe}" compile --source "${nccl_build_dir}/source" \
    --library "${nccl_build_dir}/source/build/lib/libllaminar_nccl.so.${nccl_version}" \
    --receipt "${nccl_build_dir}/build-receipt.txt" \
    --sdk "${CUDA_HOME:-/usr/local/cuda}" --compiler "${CXX:-g++}" \
    --report "${nccl_build_dir}/thread-lifetime.json"

python3 "${nccl_shm_probe}" compile --backend nccl --source "${nccl_build_dir}/source" \
    --library "${nccl_build_dir}/source/build/lib/libllaminar_nccl.so.${nccl_version}" \
    --receipt "${nccl_build_dir}/build-receipt.txt" \
    --sdk "${CUDA_HOME:-/usr/local/cuda}" --compiler "${CXX:-g++}" \
    --architecture "${NVCC_GENCODE:-toolkit-default}" \
    --report "${nccl_build_dir}/shared-file-lifecycle.json"

install -d "${nccl_prefix}/lib" "${nccl_prefix}/include/llaminar-nccl" \
    "${nccl_prefix}/share/licenses/llaminar-nccl" "${nccl_prefix}/share/llaminar-nccl"
install -m 644 "${nccl_build_dir}/source/LICENSE.txt" "${nccl_prefix}/share/licenses/llaminar-nccl/LICENSE.txt"
install -m 644 "${nccl_build_dir}/source/build/lib/libllaminar_nccl.so.${nccl_version}" "${nccl_library}"
install -m 644 "${nccl_build_dir}/source/build/include/nccl.h" "${nccl_prefix}/include/llaminar-nccl/nccl.h"
install -m 644 "${nccl_build_dir}/gin-selection.json" "${nccl_gin_report}"
install -m 644 "${nccl_build_dir}/thread-lifetime.json" "${nccl_thread_report}"
install -m 644 "${nccl_build_dir}/shared-file-lifecycle.json" "${nccl_shm_report}"
ln -sfn "libllaminar_nccl.so.${nccl_version}" "${nccl_prefix}/lib/libllaminar_nccl.so.2"
ln -sfn libllaminar_nccl.so.2 "${nccl_prefix}/lib/libllaminar_nccl.so"
printf '%s\n' "${nccl_identity}" > "${nccl_marker}"
if [[ "${nccl_prefix}" == /usr/local ]]; then ldconfig; fi
