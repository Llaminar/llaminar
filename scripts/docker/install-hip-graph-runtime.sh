#!/usr/bin/env bash
# Build the canonical HIP runtime with complete, concurrency-preserving graphs.
#
# The pinned monorepo release already owns race-free graph identities, queue
# placement, ordered packet publication and launch entry/exit dependencies.
# Do not stack obsolete backports on those upstream implementations. Remaining
# repairs publish consumed empty-segment completions (ROCm/rocm-systems#12638),
# backport #9643's TLS-safe native host-thread stack admission and #9652's SDMA
# sharing when streams outnumber physical engines. Ordering-only SDMA events
# retain their exact DMA completion rather than waiting behind unrelated
# captured compute frontiers on a shared hardware queue. The graph ROI
# repair preserves independent progress for waits and opaque child/callback
# work. Packet capture, concurrent builders and the existing queue bound remain
# enabled; finite compute graphs retain the upstream single-queue optimization.
#
# Both development and release builders call this after installing ROCm's
# compiler/SDK. The release image copies the resulting standard-SONAME DSO
# from its builder; it never downloads sources or runs a compiler at startup.
# The package-stamped filename and both public SONAME links must resolve to
# that same artifact. Otherwise ldconfig can silently resurrect a stock DSO
# even though a separately named repaired file still has a valid receipt.
# HIP_RUNTIME_INSTALL_PREFIX may stage an isolated installation for validation.
set -euo pipefail

hip_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${hip_script_dir}/rocm-release.env"
hip_sdk_root="${ROCM_PATH:-${LLAMINAR_ROCM_SDK_PREFIX}}"
hip_prefix="${HIP_RUNTIME_INSTALL_PREFIX:-${hip_sdk_root}}"
hip_patches=(
    "${hip_script_dir}/patches/rocm-hip-graph-empty-segment-publication.patch"
    "${hip_script_dir}/patches/rocm-hip-host-queue-native-stack.patch"
    "${hip_script_dir}/patches/rocm-hip-sdma-stream-sharing.patch"
    "${hip_script_dir}/patches/rocm-hip-sdma-event-publication.patch"
    "${hip_script_dir}/patches/rocm-hip-graph-progress-safe-collapse.patch"
)
hip_library="${hip_prefix}/lib/libamdhip64.so.${LLAMINAR_HIP_LIBRARY_VERSION}"
# ROCm's package uses a zero Git stamp. Preserve its filename as an alias, not
# a competing stock artifact: ldconfig prefers the longer versioned filename.
hip_package_library="${hip_library}-0000000"
hip_marker="${hip_prefix}/share/llaminar/hip-graph-runtime.txt"

# This source pin is an ABI dependency, not an independently upgradable plugin.
# Reject a future SDK override until its source revision and repair are reviewed.
if [[ ! -r "${hip_sdk_root}/.info/version" ||
      "$(<"${hip_sdk_root}/.info/version")" != "${LLAMINAR_ROCM_VERSION}" ]]; then
    echo "HIP graph runtime requires the pinned ROCm ${LLAMINAR_ROCM_VERSION} SDK" >&2
    exit 1
fi
hip_identity="${LLAMINAR_ROCM_VERSION}:${LLAMINAR_ROCM_DEB_VERSION}:${LLAMINAR_ROCM_SYSTEMS_REVISION}:${LLAMINAR_HIP_LIBRARY_VERSION}"
for hip_patch in "${hip_patches[@]}"; do
    hip_identity+=":$(sha256sum "${hip_patch}" | cut -d ' ' -f 1)"
done

# Only tiny dependency receipts are hashed, never model files. Including the
# installed DSO digest detects an apt reinstall that replaces the repaired DSO
# but leaves our receipt intact. Reuse does not hide a changed runtime.
if [[ -f "${hip_library}" && -f "${hip_marker}" &&
      "${hip_library}" -ef "${hip_package_library}" &&
      "${hip_library}" -ef "${hip_prefix}/lib/libamdhip64.so.7" &&
      "${hip_library}" -ef "${hip_prefix}/lib/libamdhip64.so" &&
      "$(<"${hip_marker}")" == "${hip_identity}:$(sha256sum "${hip_library}" | cut -d ' ' -f 1)" ]]; then
    echo "HIP graph runtime already installed: ${hip_library}"
    exit 0
fi

for dependency in cmake ninja ccache curl git python3; do
    command -v "${dependency}" >/dev/null || { echo "Missing build dependency: ${dependency}" >&2; exit 1; }
done
if [[ ! -f "${hip_sdk_root}/lib/llvm/lib/cmake/llvm/LLVMConfig.cmake" ]]; then
    echo "HIP runtime build requires the matching amdrocm-llvm-dev${LLAMINAR_ROCM_SERIES} package" >&2
    exit 1
fi

hip_build_dir="$(mktemp -d /tmp/llaminar-hip-graph-build.XXXXXXXX)"
# The only recursive deletion is our own freshly-created temporary workspace.
trap 'rm -rf -- "${hip_build_dir}"' EXIT
mkdir -p "${hip_build_dir}/systems"
curl -fsSL --retry 3 --connect-timeout 30 \
    "https://codeload.github.com/ROCm/rocm-systems/tar.gz/${LLAMINAR_ROCM_SYSTEMS_REVISION}" \
    -o "${hip_build_dir}/systems.tar.gz"
# Extract only the release's paired CLR/HIP trees. Building outside a Git
# checkout also prevents upstream version discovery from fetching other refs.
tar -xzf "${hip_build_dir}/systems.tar.gz" --strip-components=1 \
    -C "${hip_build_dir}/systems" --wildcards '*/projects/clr' '*/projects/hip'
for hip_patch in "${hip_patches[@]}"; do
    git -C "${hip_build_dir}/systems" apply --check --directory=projects/clr "${hip_patch}"
    git -C "${hip_build_dir}/systems" apply --directory=projects/clr "${hip_patch}"
done
python3 -m venv "${hip_build_dir}/venv"
"${hip_build_dir}/venv/bin/pip" install --disable-pip-version-check CppHeaderParser==2.7.4

cmake -S "${hip_build_dir}/systems/projects/clr" -B "${hip_build_dir}/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${hip_sdk_root}/lib/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="${hip_sdk_root}/lib/llvm/bin/clang++" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd \
    -DHIP_COMMON_DIR="${hip_build_dir}/systems/projects/hip" \
    -DPython3_EXECUTABLE="${hip_build_dir}/venv/bin/python" \
    -DROCM_PATH="${hip_sdk_root}" -DCMAKE_PREFIX_PATH="${hip_sdk_root}" \
    -DROCM_PATCH_VERSION="${LLAMINAR_HIP_BUILD_ID}" \
    -DTHEROCK_FLAG_STAMP_LIBRARY_GIT_VERSIONS=OFF -D__HIP_ENABLE_PCH=ON
cmake --build "${hip_build_dir}/build" --target amdhip64 --parallel

hip_licenses="${hip_prefix}/share/licenses/llaminar-hip"
install -d "${hip_prefix}/lib" "${hip_prefix}/share/llaminar" "${hip_licenses}"
# Rename a fresh inode into place: never truncate a DSO mapped by a live process.
install -m 755 "${hip_build_dir}/build/hipamd/lib/libamdhip64.so.${LLAMINAR_HIP_LIBRARY_VERSION}" "${hip_library}.new"
mv -f -- "${hip_library}.new" "${hip_library}"
# Retire the distribution candidate atomically. Both names now identify one
# physical DSO even if a later package/loader operation retargets the SONAME.
# Renaming a symlink also preserves any old inode mapped by an existing client.
ln -sfn "libamdhip64.so.${LLAMINAR_HIP_LIBRARY_VERSION}" "${hip_package_library}.new"
mv -f -- "${hip_package_library}.new" "${hip_package_library}"
ln -sfn "libamdhip64.so.${LLAMINAR_HIP_LIBRARY_VERSION}" "${hip_prefix}/lib/libamdhip64.so.7"
ln -sfn libamdhip64.so.7 "${hip_prefix}/lib/libamdhip64.so"
install -m 644 "${hip_build_dir}/systems/projects/clr/LICENSE.md" "${hip_licenses}/LICENSE.md"
for hip_patch in "${hip_patches[@]}"; do
    install -m 644 "${hip_patch}" "${hip_licenses}/$(basename -- "${hip_patch}")"
done
printf '%s:%s\n' "${hip_identity}" "$(sha256sum "${hip_library}" | cut -d ' ' -f 1)" > "${hip_marker}"
echo "HIP graph runtime installed: ${hip_library}"
