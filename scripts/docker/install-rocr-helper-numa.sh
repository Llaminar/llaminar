#!/usr/bin/env bash
# Build pinned ROCr with helper-NUMA and deferred SDMA profiling ownership fixes.
# Preserve helper affinity, original command profiling and exact native completion.
# Builder replay authenticates the original/candidate native source and DSO.
# Installed/export paths only verify evidence and never create a hidden build.
set -euo pipefail
rocr_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${rocr_script_dir}/rocm-release.env"
rocr_sdk="${ROCM_PATH:-${LLAMINAR_ROCM_SDK_PREFIX}}"
rocr_prefix="${ROCR_RUNTIME_INSTALL_PREFIX:-${rocr_sdk}}"
rocr_patch="${rocr_script_dir}/patches/rocm-rocr-helper-numa.patch"
rocr_proof="${rocr_script_dir}/rocr-helper-numa/proof.py"
rocr_fixture="${rocr_script_dir}/rocr-helper-numa/contract.cpp.in"
rocr_sdma_patch="${rocr_script_dir}/patches/rocm-rocr-deferred-sdma-profiling.patch"
rocr_sdma_proof="${rocr_script_dir}/rocr-deferred-sdma/proof.py"
rocr_sdma_fixture="${rocr_script_dir}/rocr-deferred-sdma/contract.cpp.in"
rocr_library="${rocr_prefix}/lib/libhsa-runtime64.so.${LLAMINAR_ROCR_LIBRARY_VERSION}"
rocr_marker="${rocr_prefix}/share/llaminar/rocr-helper-numa.txt"
rocr_receipt="${rocr_prefix}/share/llaminar/rocr-helper-numa-proof.json"
rocr_sdma_receipt="${rocr_prefix}/share/llaminar/rocr-deferred-sdma-proof.json"
[[ -r "${rocr_sdk}/.info/version" && "$(<"${rocr_sdk}/.info/version")" == "${LLAMINAR_ROCM_VERSION}" ]] || {
    echo "ROCr helper repair requires the pinned SDK" >&2; exit 1;
}
rocr_identity="${LLAMINAR_ROCM_VERSION}:${LLAMINAR_ROCM_DEB_VERSION}:${LLAMINAR_ROCM_SYSTEMS_REVISION}:${LLAMINAR_ROCR_LIBRARY_VERSION}"
for rocr_input in "${rocr_patch}" "${rocr_proof}" "${rocr_fixture}" \
    "${rocr_sdma_patch}" "${rocr_sdma_proof}" "${rocr_sdma_fixture}"; do
    rocr_identity+=":$(sha256sum "${rocr_input}" | cut -d ' ' -f 1)"
done
if [[ -f "${rocr_library}" && ! -L "${rocr_library}" && -f "${rocr_marker}" &&
      "${rocr_library}" -ef "${rocr_prefix}/lib/libhsa-runtime64.so.1" &&
      "${rocr_library}" -ef "${rocr_prefix}/lib/libhsa-runtime64.so" &&
      "$(<"${rocr_marker}")" == "${rocr_identity}:$(sha256sum "${rocr_library}" | cut -d ' ' -f 1)" ]]; then
    python3 "${rocr_proof}" --verify "${rocr_receipt}" --library "${rocr_library}" --source-identity "${rocr_identity}"
    python3 "${rocr_sdma_proof}" --verify "${rocr_sdma_receipt}" --library "${rocr_library}" --source-identity "${rocr_identity}"
    if [[ -n "${ROCR_RUNTIME_STAGE_PREFIX:-}" ]]; then
        rocr_requested="${ROCR_RUNTIME_STAGE_PREFIX}"
        [[ "${rocr_requested}" == /* && ! -L "${rocr_requested}" &&
           ( ! -e "${rocr_requested}" || -d "${rocr_requested}" ) ]] || {
            echo "ROCr export requires an absolute owned directory" >&2; exit 1;
        }
        rocr_stage="$(realpath -m -- "${rocr_requested}")"
        rocr_installed="$(realpath -e -- "${rocr_prefix}")"
        [[ "${rocr_stage}" != "${rocr_installed}" && "${rocr_stage}" != "${rocr_installed}/"* ]] || {
            echo "ROCr export must be outside the installed SDK" >&2; exit 1;
        }
        for rocr_component in lib share; do
            [[ ! -e "${rocr_stage}/${rocr_component}" && ! -L "${rocr_stage}/${rocr_component}" ]] || {
                echo "ROCr export destination already owns ${rocr_component}" >&2; exit 1;
            }
        done
        [[ "$(readlink "${rocr_prefix}/lib/libhsa-runtime64.so.1")" == "libhsa-runtime64.so.${LLAMINAR_ROCR_LIBRARY_VERSION}" &&
           "$(readlink "${rocr_prefix}/lib/libhsa-runtime64.so")" == libhsa-runtime64.so.1 ]] || {
            echo "ROCr export requires canonical local aliases" >&2; exit 1;
        }
        cmp -s "${rocr_patch}" "${rocr_prefix}/share/licenses/llaminar-rocr/rocm-rocr-helper-numa.patch" || {
            echo "ROCr installed repair differs from export authority" >&2; exit 1;
        }
        cmp -s "${rocr_sdma_patch}" "${rocr_prefix}/share/licenses/llaminar-rocr/rocm-rocr-deferred-sdma-profiling.patch" || {
            echo "ROCr installed deferred SDMA repair differs from export authority" >&2; exit 1;
        }
        install -d "${rocr_stage}/lib" "${rocr_stage}/share/llaminar" "${rocr_stage}/share/licenses"
        cp -P "${rocr_library}" "${rocr_prefix}/lib/libhsa-runtime64.so.1" "${rocr_prefix}/lib/libhsa-runtime64.so" "${rocr_stage}/lib/"
        cp "${rocr_marker}" "${rocr_receipt}" "${rocr_sdma_receipt}" "${rocr_stage}/share/llaminar/"
        cp -a "${rocr_prefix}/share/licenses/llaminar-rocr" "${rocr_stage}/share/licenses/"
    fi
    echo "ROCr helper-NUMA component authenticated: ${rocr_library}"
    exit 0
fi
[[ -z "${ROCR_RUNTIME_STAGE_PREFIX:-}" ]] || {
    echo "ROCr export requires an authenticated builder installation" >&2; exit 1;
}
for rocr_dependency in cmake ninja ccache curl git python3; do
    command -v "${rocr_dependency}" >/dev/null || { echo "Missing dependency: ${rocr_dependency}" >&2; exit 1; }
done
rocr_work="$(mktemp -d /tmp/llaminar-rocr-helper.XXXXXXXX)"
trap 'rm -rf -- "${rocr_work}"' EXIT
mkdir -p "${rocr_work}/systems"
curl -fsSL --retry 3 --connect-timeout 30 \
    "https://codeload.github.com/ROCm/rocm-systems/tar.gz/${LLAMINAR_ROCM_SYSTEMS_REVISION}" -o "${rocr_work}/systems.tar.gz"
tar -xzf "${rocr_work}/systems.tar.gz" --strip-components=1 -C "${rocr_work}/systems" --wildcards '*/projects/rocr-runtime'
git -C "${rocr_work}/systems" apply --check "${rocr_patch}"
git -C "${rocr_work}/systems" apply "${rocr_patch}"
git -C "${rocr_work}/systems" apply --check "${rocr_sdma_patch}"
git -C "${rocr_work}/systems" apply "${rocr_sdma_patch}"
cmake -S "${rocr_work}/systems/projects/rocr-runtime" -B "${rocr_work}/build" -G Ninja \
    -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${rocr_sdk}/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER="${rocr_sdk}/lib/llvm/bin/clang++" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_PREFIX_PATH="${rocr_sdk}" -DCMAKE_INSTALL_PREFIX="${rocr_prefix}" -DCMAKE_INSTALL_LIBDIR=lib \
    -DROCM_PATH="${rocr_sdk}" -DROCM_PATCH_VERSION=0 -DBUILD_SHARED_LIBS=ON -DBUILD_ROCR=ON -DENABLE_LDCONFIG=OFF
cmake --build "${rocr_work}/build" --target hsa-runtime64 --parallel
# Prove the actual built candidate before publishing it into the SDK.
rocr_built="${rocr_work}/build/rocr/lib/libhsa-runtime64.so.${LLAMINAR_ROCR_LIBRARY_VERSION}"
python3 "${rocr_proof}" --source-root "${rocr_work}/systems" --patch "${rocr_patch}" \
    --compiler "${rocr_sdk}/lib/llvm/bin/clang++" --library "${rocr_built}" --source-identity "${rocr_identity}" \
    --output "${rocr_work}/proof.json"
python3 "${rocr_sdma_proof}" --source-root "${rocr_work}/systems" --patch "${rocr_sdma_patch}" \
    --compiler "${rocr_sdk}/lib/llvm/bin/clang++" --library "${rocr_built}" --source-identity "${rocr_identity}" \
    --output "${rocr_work}/sdma-proof.json"
install -d "${rocr_prefix}/lib" "${rocr_prefix}/share/llaminar" "${rocr_prefix}/share/licenses/llaminar-rocr"
install -m 755 "${rocr_built}" "${rocr_library}.new"
mv -f -- "${rocr_library}.new" "${rocr_library}"
for rocr_alias in libhsa-runtime64.so.1 libhsa-runtime64.so; do
    rocr_target="libhsa-runtime64.so.${LLAMINAR_ROCR_LIBRARY_VERSION}"
    if [[ "${rocr_alias}" == libhsa-runtime64.so ]]; then rocr_target=libhsa-runtime64.so.1; fi
    ln -sfn "${rocr_target}" "${rocr_prefix}/lib/${rocr_alias}.new"
    mv -f -- "${rocr_prefix}/lib/${rocr_alias}.new" "${rocr_prefix}/lib/${rocr_alias}"
done
install -m 644 "${rocr_work}/proof.json" "${rocr_receipt}"
install -m 644 "${rocr_work}/sdma-proof.json" "${rocr_sdma_receipt}"
install -m 644 "${rocr_sdma_patch}" "${rocr_prefix}/share/licenses/llaminar-rocr/rocm-rocr-deferred-sdma-profiling.patch"
install -m 644 "${rocr_patch}" "${rocr_prefix}/share/licenses/llaminar-rocr/rocm-rocr-helper-numa.patch"
install -m 644 "${rocr_work}/systems/projects/rocr-runtime/runtime/hsa-runtime/LICENSE.md" "${rocr_prefix}/share/licenses/llaminar-rocr/LICENSE.md"
printf '%s:%s\n' "${rocr_identity}" "$(sha256sum "${rocr_library}" | cut -d ' ' -f 1)" > "${rocr_marker}"
echo "ROCr helper-NUMA runtime installed: ${rocr_library}"
