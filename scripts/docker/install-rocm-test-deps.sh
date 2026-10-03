#!/usr/bin/env bash
# Install the ROCm annotation ABI used by the kernel test/tuning harnesses.
#
# Both the compiler builder and installed test runner need the same package
# closure. Their existing signed ROCm repository selects its compatible ABI.
# Core SDK packages ROCTX and rocprofiler-register in its lightweight base
# profiler closure (also required by BLAS), not the full profiling toolchain.
# The annotation
# library is usable without a GPU or an attached profiler. Do not install the
# full profiling SDK/toolchain, or call this installer from the serving image.
# Keeping test-only dependencies separate also preserves the expensive CUDA,
# HIP, RCCL and oneDNN cache layers when the test harness gains a dependency.
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/rocm-release.env"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    "amdrocm-profiler-base${LLAMINAR_ROCM_SERIES}=${LLAMINAR_ROCM_DEB_VERSION}"
apt-get clean
rm -rf /var/lib/apt/lists/*
