#!/usr/bin/env bash
# Build the canonical NCCL dependency with resumed CUDA graph capture support.
#
# This patch changes capture-time strong-stream membership, not collective
# kernels, transport selection, or replay. A distinct SONAME prevents a system
# NCCL package from silently replacing the tested implementation at runtime.
# Devcontainers and release builders use this same installer; runtime images
# copy its shared library from the builder, without a runtime compiler.
#
# NCCL_INSTALL_PREFIX selects the installation prefix (default /usr/local).
# NVCC_GENCODE may restrict a local build to its actual device architectures;
# absent that setting, NCCL's complete CUDA-toolkit architecture set is built.
set -euo pipefail

nccl_revision=dbc86fd06e8b0c4517b95d8958a09ccacf9520c9
nccl_tag=v2.28.9-1
nccl_version=2.28.9
nccl_prefix="${NCCL_INSTALL_PREFIX:-/usr/local}"
nccl_patch="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/patches/nccl-capture-reentry.patch"
nccl_library="${nccl_prefix}/lib/libllaminar_nccl.so.${nccl_version}"
nccl_marker="${nccl_prefix}/lib/llaminar-nccl-build.txt"
nccl_identity="${nccl_revision}:$(sha256sum "${nccl_patch}" | cut -d ' ' -f 1):${NVCC_GENCODE:-toolkit-default}"

# Only an exact source/patch/architecture receipt can reuse an installation.
# This tiny patch identity has no relationship to model-weight cache hashing.
if [[ -f "${nccl_library}" && -f "${nccl_marker}" &&
      "$(<"${nccl_marker}")" == "${nccl_identity}" ]]; then
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
make -C "${nccl_build_dir}/source" -j"$(nproc)" src.build \
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" LIBNAME=libllaminar_nccl.so

install -d "${nccl_prefix}/lib" "${nccl_prefix}/include/llaminar-nccl" \
    "${nccl_prefix}/share/licenses/llaminar-nccl"
install -m 644 "${nccl_build_dir}/source/LICENSE.txt" "${nccl_prefix}/share/licenses/llaminar-nccl/LICENSE.txt"
install -m 644 "${nccl_build_dir}/source/build/lib/libllaminar_nccl.so.${nccl_version}" "${nccl_library}"
install -m 644 "${nccl_build_dir}/source/build/include/nccl.h" "${nccl_prefix}/include/llaminar-nccl/nccl.h"
ln -sfn "libllaminar_nccl.so.${nccl_version}" "${nccl_prefix}/lib/libllaminar_nccl.so.2"
ln -sfn libllaminar_nccl.so.2 "${nccl_prefix}/lib/libllaminar_nccl.so"
printf '%s\n' "${nccl_identity}" > "${nccl_marker}"
if [[ "${nccl_prefix}" == /usr/local ]]; then ldconfig; fi
