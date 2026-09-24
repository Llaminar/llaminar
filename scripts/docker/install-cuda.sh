#!/usr/bin/env bash
# install-cuda.sh
#
# Install the NVIDIA CUDA 13 build/runtime libraries from NVIDIA's official
# apt repository. Matches NCCL to the CUDA 13 series.
#
# MODE=full     (default) — minimal builder set: nvcc + CUDA runtime,
#                            cuBLAS/NCCL headers and shared libraries.
# MODE=runtime             — shared libraries only. Slim runtime stage.
# INSTALL_CUDA_PROFILERS=1 — additionally install the CUDA-series-matched
#                            Nsight Compute, Nsight Systems, and CUPTI packages.
#                            Development images enable this; release builders
#                            remain deliberately lean.
#
# The host driver is injected at container run time via
# `--gpus=all` + nvidia-container-toolkit; we never install drivers inside
# the image.
set -euo pipefail

MODE="${MODE:-full}"
INSTALL_CUDA_PROFILERS="${INSTALL_CUDA_PROFILERS:-0}"
export DEBIAN_FRONTEND=noninteractive

APT_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)

## Restrict the patched NCCL fatbin to Llaminar's declared CUDA contract.
#
# CUDAARCHS is exported by the Dockerfile from LLAMINAR_CUDA_ARCHS, the same
# value supplied to CMake's CMAKE_CUDA_ARCHITECTURES.  Leaving NCCL's
# NVCC_GENCODE unset makes its upstream Makefile compile every architecture
# known to the installed toolkit, including devices for which this Llaminar
# image has no kernels.  Apart from wasting a cold build, that makes the
# collective library advertise a broader device set than the application.
#
# CMake's broad selectors intentionally remain broad: NCCL's upstream default
# is the only equivalent representation for ``all``, ``all-major`` and
# ``native``.  Concrete CUDA architectures may include a suffix such as 90a;
# nvcc accepts that spelling in both compute and sm targets.
configure_nccl_gencode() {
    local requested_architectures="${CUDAARCHS:-}"
    local architecture
    local -a gencodes=()
    local -a architectures=()

    case "${requested_architectures}" in
        ""|all|all-major|native)
            return 0
            ;;
    esac

    IFS=';' read -r -a architectures <<< "${requested_architectures}"
    for architecture in "${architectures[@]}"; do
        # CMake accepts 86-real/86-virtual spellings.  NCCL's Makefile needs
        # the underlying compute capability because it constructs the exact
        # SASS target itself.
        architecture="${architecture%%-*}"
        if [[ ! "${architecture}" =~ ^[0-9]+[A-Za-z]?$ ]]; then
            echo "Unsupported CUDAARCHS entry '${architecture}' for NCCL. " \
                 "Use concrete capabilities (for example 86 or 90a), or all/all-major/native." >&2
            return 1
        fi
        gencodes+=("-gencode arch=compute_${architecture},code=sm_${architecture}")
    done

    if ((${#gencodes[@]} == 0)); then
        echo "CUDAARCHS must not resolve to an empty NCCL target set" >&2
        return 1
    fi
    export NVCC_GENCODE="${gencodes[*]}"
    echo "==> [nccl] CUDA targets: ${CUDAARCHS}"
}

# --- Register NVIDIA apt repo --------------------------------------------
curl -fsSL --retry 5 --retry-delay 5 -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/cuda-keyring.deb
rm /tmp/cuda-keyring.deb
apt-get "${APT_OPTS[@]}" update

if [[ "${MODE}" == "full" ]]; then
    # Build-minimal CUDA: avoid cuda-toolkit-13-0, which also pulls Nsight,
    # OpenJDK, GTK, profilers, docs, and other tools not needed to compile
    # Llaminar.
    apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
        --allow-change-held-packages \
        cuda-nvcc-13-0 \
        cuda-cudart-dev-13-0 \
        libcublas-dev-13-0 \
        "libnccl2=*+cuda13.0" \
        "libnccl-dev=*+cuda13.0"

    # The package supplies tooling; production loads the separately named,
    # pinned source build that supports repeated capture into a native parent.
    configure_nccl_gencode
    bash "$(dirname -- "${BASH_SOURCE[0]}")/install-nccl.sh"

    if [[ "${INSTALL_CUDA_PROFILERS}" == "1" ]]; then
        apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
            cuda-cupti-13-0 \
            cuda-nsight-compute-13-0 \
            cuda-nsight-systems-13-0
    fi
else
    # Runtime libraries only. These are the minimum a dynamically-linked
    # CUDA-enabled llaminar2 binary needs at process start. The host driver is
    # injected by nvidia-container-toolkit at docker run time.
    apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
        --allow-change-held-packages \
        cuda-cudart-13-0 \
        libcublas-13-0
fi

rm -rf /var/lib/apt/lists/*
