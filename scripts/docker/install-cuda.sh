#!/usr/bin/env bash
# install-cuda.sh
#
# Install the NVIDIA CUDA 13 build/runtime libraries from NVIDIA's official
# apt repository. Matches NCCL to the CUDA 13 series.
#
# MODE=full     (default) — minimal builder set: nvcc + CUDA runtime,
#                            cuBLAS/NCCL headers and shared libraries.
# MODE=runtime             — shared libraries only. Slim runtime stage.
#
# The host driver is injected at container run time via
# `--gpus=all` + nvidia-container-toolkit; we never install drivers inside
# the image.
set -euo pipefail

MODE="${MODE:-full}"
export DEBIAN_FRONTEND=noninteractive

APT_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)

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
else
    # Runtime libraries only. These are the minimum a dynamically-linked
    # CUDA-enabled llaminar2 binary needs at process start. The host driver is
    # injected by nvidia-container-toolkit at docker run time.
    apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
        --allow-change-held-packages \
        cuda-cudart-13-0 \
        libcublas-13-0 \
        "libnccl2=*+cuda13.0"
fi

rm -rf /var/lib/apt/lists/*
