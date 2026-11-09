#!/usr/bin/env bash
# Build the exact SDK source's legacy SMI ABI required by pinned RCCL7.2.4.
# AMD-SMI remains separate and unchanged. No driver or GPU query is installed/run.
# Only component libraries, headers and CMake exports are published; legacy CLI
# programs from the source project's dev install remain private to this build.
set -euo pipefail
smi_scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${smi_scripts}/rocm-release.env"
smi_sdk="${ROCM_PATH:-${LLAMINAR_ROCM_SDK_PREFIX}}"
smi_prefix="${ROCM_SMI_INSTALL_PREFIX:-${smi_sdk}}"
smi_library="${smi_prefix}/lib/librocm_smi64.so.1.0"
smi_marker="${smi_prefix}/share/llaminar/rocm-smi-compat.txt"
smi_identity="${LLAMINAR_ROCM_VERSION}:${LLAMINAR_ROCM_DEB_VERSION}:${LLAMINAR_ROCM_SYSTEMS_REVISION}:legacy-smi-1.0"
[[ -r "${smi_sdk}/.info/version" && "$(<"${smi_sdk}/.info/version")" == "${LLAMINAR_ROCM_VERSION}" ]] || {
    echo "Legacy SMI requires the pinned SDK" >&2; exit 1;
}
if [[ -f "${smi_library}" && ! -L "${smi_library}" && -f "${smi_marker}" &&
      "$(<"${smi_marker}")" == "${smi_identity}:$(sha256sum "${smi_library}" | cut -d ' ' -f 1)" &&
      "${smi_library}" -ef "${smi_prefix}/lib/librocm_smi64.so.1" &&
      "${smi_library}" -ef "${smi_prefix}/lib/librocm_smi64.so" &&
      -f "${smi_prefix}/include/rocm_smi/rocm_smi.h" &&
      -f "${smi_prefix}/lib/cmake/rocm_smi/rocm_smi-config.cmake" ]]; then
    if [[ -n "${ROCM_SMI_STAGE_PREFIX:-}" ]]; then
        smi_requested="${ROCM_SMI_STAGE_PREFIX}"
        [[ "${smi_requested}" == /* && ! -L "${smi_requested}" &&
           ( ! -e "${smi_requested}" || -d "${smi_requested}" ) ]] || {
            echo "SMI export requires an absolute owned directory" >&2; exit 1;
        }
        smi_stage="$(realpath -m -- "${smi_requested}")"
        smi_installed="$(realpath -e -- "${smi_prefix}")"
        [[ "${smi_stage}" != "${smi_installed}" && "${smi_stage}" != "${smi_installed}/"* ]] || {
            echo "SMI export must be outside the installed SDK" >&2; exit 1;
        }
        for smi_part in lib share; do
            [[ ! -e "${smi_stage}/${smi_part}" && ! -L "${smi_stage}/${smi_part}" ]] || {
                echo "SMI export destination is occupied" >&2; exit 1;
            }
        done
        [[ "$(readlink "${smi_prefix}/lib/librocm_smi64.so.1")" == librocm_smi64.so.1.0 &&
           "$(readlink "${smi_prefix}/lib/librocm_smi64.so")" == librocm_smi64.so.1 ]] || {
            echo "SMI export aliases are not self-contained" >&2; exit 1;
        }
        install -d "${smi_stage}/lib" "${smi_stage}/share/llaminar" "${smi_stage}/share/licenses"
        cp -P "${smi_prefix}/lib/librocm_smi64.so"* "${smi_stage}/lib/"
        cp "${smi_marker}" "${smi_stage}/share/llaminar/"
        cp -a "${smi_prefix}/share/licenses/llaminar-rocm-smi" "${smi_stage}/share/licenses/"
    fi
    echo "Legacy SMI component authenticated: ${smi_library}"
    exit 0
fi
[[ -z "${ROCM_SMI_STAGE_PREFIX:-}" ]] || {
    echo "SMI export requires an authenticated builder component" >&2; exit 1;
}
smi_work="$(mktemp -d /tmp/llaminar-rocm-smi.XXXXXXXX)"
trap 'rm -rf -- "${smi_work}"' EXIT
mkdir -p "${smi_work}/systems"
curl -fsSL --retry 3 --connect-timeout 30 \
    "https://codeload.github.com/ROCm/rocm-systems/tar.gz/${LLAMINAR_ROCM_SYSTEMS_REVISION}" -o "${smi_work}/systems.tar.gz"
tar -xzf "${smi_work}/systems.tar.gz" --strip-components=1 -C "${smi_work}/systems" --wildcards '*/projects/rocm-smi-lib'
cmake -S "${smi_work}/systems/projects/rocm-smi-lib" -B "${smi_work}/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${smi_sdk}/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER="${smi_sdk}/lib/llvm/bin/clang++" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_PREFIX_PATH="${smi_sdk}" -DCMAKE_INSTALL_PREFIX="${smi_work}/component" \
    -DCMAKE_INSTALL_LIBDIR=lib -DROCM_SMI_DISABLE=OFF -DBUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON -DROCM_PATCH_VERSION=0 -DENABLE_LDCONFIG=OFF -DCMAKE_EXPORT_PACKAGE_REGISTRY=OFF
cmake --build "${smi_work}/build" --parallel
cmake --install "${smi_work}/build" --component dev
# Publish only this source component's library/export/header closure, not its
# command-line applications. CMake's export includes OAM, so keep that build DSO
# with the dev metadata; RCCL's runtime component needs only rocm_smi64.
install -d "${smi_prefix}/lib/cmake" "${smi_prefix}/include" "${smi_prefix}/share/llaminar" "${smi_prefix}/share/licenses/llaminar-rocm-smi"
for smi_built in "${smi_work}/component/lib/"*.so*; do
    if [[ -L "${smi_built}" ]]; then continue; fi
    install -m 755 "${smi_built}" "${smi_prefix}/lib/$(basename "${smi_built}").new"
    mv -f -- "${smi_prefix}/lib/$(basename "${smi_built}").new" "${smi_prefix}/lib/$(basename "${smi_built}")"
done
for smi_built in "${smi_work}/component/lib/"*.so*; do
    [[ -L "${smi_built}" ]] || continue
    ln -sfn "$(readlink "${smi_built}")" "${smi_prefix}/lib/$(basename "${smi_built}")"
done
cp -a "${smi_work}/component/lib/cmake/rocm_smi" "${smi_prefix}/lib/cmake/"
cp -a "${smi_work}/component/include/rocm_smi" "${smi_prefix}/include/"
cp -a "${smi_work}/component/include/oam" "${smi_prefix}/include/"
install -m 644 "${smi_work}/systems/projects/rocm-smi-lib/LICENSE.md" "${smi_prefix}/share/licenses/llaminar-rocm-smi/LICENSE.md"
[[ -f "${smi_library}" ]] || { echo "Legacy SMI ABI filename mismatch" >&2; exit 1; }
printf '%s:%s\n' "${smi_identity}" "$(sha256sum "${smi_library}" | cut -d ' ' -f 1)" > "${smi_marker}"
echo "Legacy SMI installed: ${smi_library}"
