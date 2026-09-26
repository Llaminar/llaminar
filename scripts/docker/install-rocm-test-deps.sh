#!/usr/bin/env bash
# Install the ROCm annotation ABI used by the kernel test/tuning harnesses.
#
# Both the compiler builder and installed test runner need the same package
# closure. Their existing pinned ROCm repository selects its compatible ABI;
# apt also installs its small rocprofiler-register dependency. The annotation
# library is usable without a GPU or an attached profiler. Do not install the
# full profiling SDK/toolchain, or call this installer from the serving image.
# Keeping test-only dependencies separate also preserves the expensive CUDA,
# HIP, RCCL and oneDNN cache layers when the test harness gains a dependency.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends rocprofiler-sdk-roctx
apt-get clean
rm -rf /var/lib/apt/lists/*
