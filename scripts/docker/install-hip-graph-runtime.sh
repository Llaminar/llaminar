#!/usr/bin/env bash
# Build the canonical HIP runtime with race-free captured-graph identity.
#
# Independent TP graph builders must stay concurrent. ROCm's scheduler must
# identify actual nodes, not racy diagnostic labels. This source repair keeps
# native packet capture and replay enabled and changes no device kernel.
#
# Both development and release builders call this after installing ROCm's
# compiler/SDK. The release image copies the resulting standard-SONAME DSO
# from its builder; it never downloads sources or runs a compiler at startup.
# HIP_RUNTIME_INSTALL_PREFIX may stage an isolated installation for validation.
set -euo pipefail

hip_clr_revision=fe5035afc8713dfc6adedd3c00c4306c93a160f8
hip_common_revision=bc9af25177f96c0fea93198b89cf4c3cf08f3ea3
hip_sdk_root="${ROCM_PATH:-/opt/rocm}"
hip_prefix="${HIP_RUNTIME_INSTALL_PREFIX:-${hip_sdk_root}}"
hip_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
hip_patch="${hip_script_dir}/patches/rocm-hip-graph-node-identity.patch"
hip_library="${hip_prefix}/lib/libamdhip64.so.7.2.70204"
hip_marker="${hip_prefix}/share/llaminar/hip-graph-runtime.txt"

# This source pin is an ABI dependency, not an independently upgradable plugin.
# Reject a future SDK override until its source revision and repair are reviewed.
if [[ ! -r "${hip_sdk_root}/.info/version" ||
      "$(<"${hip_sdk_root}/.info/version")" != 7.2.4 ]]; then
    echo "HIP graph-identity runtime requires the pinned ROCm 7.2.4 SDK" >&2
    exit 1
fi
hip_identity="${hip_clr_revision}:${hip_common_revision}:$(sha256sum "${hip_patch}" | cut -d ' ' -f 1)"

# Only tiny dependency receipts are hashed, never model files. Including the
# installed DSO digest detects an apt reinstall that replaces the repaired DSO
# but leaves our receipt intact. Reuse does not hide a changed runtime.
if [[ -f "${hip_library}" && -f "${hip_marker}" &&
      "$(<"${hip_marker}")" == "${hip_identity}:$(sha256sum "${hip_library}" | cut -d ' ' -f 1)" ]]; then
    echo "HIP graph-identity runtime already installed: ${hip_library}"
    exit 0
fi

for dependency in cmake ninja ccache curl git python3; do
    command -v "${dependency}" >/dev/null || { echo "Missing build dependency: ${dependency}" >&2; exit 1; }
done
if [[ ! -f "${hip_sdk_root}/lib/llvm/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
    echo "HIP runtime build requires the matching rocm-llvm-dev package" >&2
    exit 1
fi

hip_build_dir="$(mktemp -d /tmp/llaminar-hip-graph-build.XXXXXXXX)"
# The only recursive deletion is our own freshly-created temporary workspace.
trap 'rm -rf -- "${hip_build_dir}"' EXIT
mkdir -p "${hip_build_dir}/clr" "${hip_build_dir}/hip"
curl -fsSL --retry 3 --connect-timeout 30 \
    "https://codeload.github.com/ROCm/clr/tar.gz/${hip_clr_revision}" \
    -o "${hip_build_dir}/clr.tar.gz"
curl -fsSL --retry 3 --connect-timeout 30 \
    "https://codeload.github.com/ROCm/HIP/tar.gz/${hip_common_revision}" \
    -o "${hip_build_dir}/hip.tar.gz"
tar -xzf "${hip_build_dir}/clr.tar.gz" --strip-components=1 -C "${hip_build_dir}/clr"
tar -xzf "${hip_build_dir}/hip.tar.gz" --strip-components=1 -C "${hip_build_dir}/hip"
git -C "${hip_build_dir}/clr" apply --check "${hip_patch}"
git -C "${hip_build_dir}/clr" apply "${hip_patch}"
python3 -m venv "${hip_build_dir}/venv"
"${hip_build_dir}/venv/bin/pip" install --disable-pip-version-check CppHeaderParser==2.7.4

cmake -S "${hip_build_dir}/clr" -B "${hip_build_dir}/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${hip_sdk_root}/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="${hip_sdk_root}/llvm/bin/clang++" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
    -DHIP_COMMON_DIR="${hip_build_dir}/hip" \
    -DPython3_EXECUTABLE="${hip_build_dir}/venv/bin/python" \
    -DROCM_PATH="${hip_sdk_root}" -DCMAKE_PREFIX_PATH="${hip_sdk_root}" \
    -DROCM_PATCH_VERSION=70204 -D__HIP_ENABLE_PCH=ON
cmake --build "${hip_build_dir}/build" --target amdhip64 --parallel

hip_licenses="${hip_prefix}/share/licenses/llaminar-hip"
install -d "${hip_prefix}/lib" "${hip_prefix}/share/llaminar" "${hip_licenses}"
# Rename a fresh inode into place: never truncate a DSO mapped by a live process.
install -m 755 "${hip_build_dir}/build/hipamd/lib/libamdhip64.so.7.2.70204" "${hip_library}.new"
mv -f -- "${hip_library}.new" "${hip_library}"
ln -sfn libamdhip64.so.7.2.70204 "${hip_prefix}/lib/libamdhip64.so.7"
ln -sfn libamdhip64.so.7 "${hip_prefix}/lib/libamdhip64.so"
install -m 644 "${hip_build_dir}/clr/LICENSE.md" "${hip_licenses}/LICENSE.md"
install -m 644 "${hip_patch}" "${hip_licenses}/rocm-hip-graph-node-identity.patch"
printf '%s:%s\n' "${hip_identity}" "$(sha256sum "${hip_library}" | cut -d ' ' -f 1)" > "${hip_marker}"
echo "HIP graph-identity runtime installed: ${hip_library}"
