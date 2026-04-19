# syntax=docker/dockerfile:1.6
#
# Llaminar — CUDA + ROCm runtime image
#
# Two-stage build:
#   1. builder — Ubuntu 24.04 + full CUDA 13 toolkit + ROCm 7.1.1 + C++23
#      toolchain, compiles `llaminar2` (Release).
#   2. runtime — Ubuntu 24.04 + CUDA 13 shared libs + ROCm 7.1.1 user-space +
#      compiled binary only (no compilers, no -dev packages).
#
# All dependency-install logic lives in scripts/docker/install-*.sh, which are
# shared with .devcontainer/Dockerfile. Edit those scripts — not this file —
# to change what gets installed.
#
# Runtime usage (both CUDA + ROCm available; pick per invocation with -d):
#   docker run --gpus all --rm -it \
#       -v /path/to/models:/models:ro \
#       -p 8080:8080 \
#       ghcr.io/llaminar/llaminar:latest \
#       --serve --port 8080 -d cuda:0 -m /models/Qwen2.5-1.5B-Instruct-Q8_0.gguf
#
# ROCm-only host:
#   docker run --device=/dev/kfd --device=/dev/dri \
#       --group-add video --group-add render \
#       --ipc=host --shm-size=16G \
#       -v /path/to/models:/models:ro \
#       ghcr.io/llaminar/llaminar:latest -d rocm:0 -m /models/<gguf>

ARG CUTLASS_VERSION=v4.2.1
ARG LLAMINAR_BUILD_TYPE=Release
ARG LLAMINAR_CUDA_ARCHS="75;80;86;89;90"

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM ubuntu:24.04 AS builder

ARG CUTLASS_VERSION
ARG LLAMINAR_BUILD_TYPE
ARG LLAMINAR_CUDA_ARCHS

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    CUDAARCHS=${LLAMINAR_CUDA_ARCHS} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib \
    ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    CUTLASS_DIR=/opt/cutlass \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=20G \
    CCACHE_COMPRESS=1 \
    CCACHE_COMPRESSLEVEL=6 \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime,pch_defines,locale,system_headers \
    CCACHE_COMPILERCHECK=content \
    CCACHE_NOHASHDIR=1 \
    CCACHE_BASEDIR=/src

COPY scripts/docker /tmp/install-scripts
RUN MODE=build  /tmp/install-scripts/install-system-deps.sh
RUN MODE=full   /tmp/install-scripts/install-cuda.sh
RUN MODE=full   /tmp/install-scripts/install-rocm.sh
RUN CUTLASS_VERSION=${CUTLASS_VERSION} /tmp/install-scripts/install-cutlass.sh
RUN rm -rf /tmp/install-scripts

# `docker build` cannot use `--gpus`, so nvidia-container-runtime never
# injects the real /usr/lib/x86_64-linux-gnu/libcuda.so.1 driver lib.
# Without it, any CUDA-linked binary (e.g. our parity test executables)
# fails to start with exit 127 ("error while loading shared libraries:
# libcuda.so.1") — which breaks the cmake POST_BUILD --gtest_list_tests
# discovery step in tests/v2/cmake/V2ParityTestDiscovery.cmake.
#
# Resolve the SONAME against the CUDA toolkit's link-time stub for build
# time only. At `docker run --gpus all` time, the real driver gets
# injected at /usr/lib/x86_64-linux-gnu/libcuda.so.1 and wins via
# ld.so.cache, so this stub is invisible at actual test execution time.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/zz-cuda-stubs.conf && \
    ldconfig

WORKDIR /src

# Python dependencies for the reference tests + parity gates. Pulls the
# CPU-only PyTorch wheel (~250 MB) plus our transformers fork. Cached as a
# separate layer keyed only on requirements.txt so source edits don't
# invalidate it.
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY CMakeLists.txt ./CMakeLists.txt
COPY .githooks ./.githooks
COPY jinja ./jinja
COPY cmake ./cmake
COPY external/vendor ./external/vendor
COPY python ./python

# Integration build — what CI drives for unit, parity, and E2E tests. Has
# debug symbols, assertions active, tensor verification enabled.
RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build_v2_integration -S src/v2 -G Ninja \
        -DCMAKE_BUILD_TYPE=Integration \
        -DHAVE_CUDA=ON \
        -DHAVE_ROCM=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
 && cmake --build build_v2_integration --parallel

# Release build — what the runtime image ships. Optimized, no assertions,
# only the llaminar2 target (skip test binaries).
RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build_v2_release -S src/v2 -G Ninja \
        -DCMAKE_BUILD_TYPE=${LLAMINAR_BUILD_TYPE} \
        -DHAVE_CUDA=ON \
        -DHAVE_ROCM=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
 && cmake --build build_v2_release --parallel --target llaminar2

# Strip debug info from every executable + .a/.so in both build trees.
# Integration test binaries average ~370 MB unstripped (mostly CUDA fatbin
# + DWARF), ~85 MB stripped — a 4x shrink across ~600 binaries dominates
# the builder image size. We keep assertions / sanitizers (those are
# compile-flag controlled, not stripped), so test fidelity is unchanged.
# `--strip-debug` (not `--strip-all`) preserves the symbol table, so
# stack traces from gtest / gdb attach still resolve function names.
RUN find build_v2_integration build_v2_release \
        \( -type f -executable -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
        -not -path '*/CMakeFiles/*' \
        -print0 \
    | xargs -0 -r -P "$(nproc)" -n 32 strip --strip-debug 2>/dev/null || true

# CI runs `docker run --group-add render --group-add video` against this
# builder image; docker resolves --group-add names from the image's /etc/group,
# not the host's. Ensure those groups exist so the gates can attach to
# /dev/dri/renderD* and /dev/kfd. Placed last to keep the cmake layer cache
# valid on Dockerfile edits.
RUN groupadd -f render && groupadd -f video

# OpenMPI refuses to run as root unless explicitly opted in. The CI gates run
# the test binaries as root inside the container and many of them call
# mpirun() under the hood, so set the bypass env vars at image scope.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# =============================================================================
# Stage 2: Runtime — slim image with only shared libs + the binary
# =============================================================================
FROM ubuntu:24.04 AS runtime

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=dev

LABEL org.opencontainers.image.title="Llaminar" \
      org.opencontainers.image.description="High-performance LLM inference engine (CUDA + ROCm)" \
      org.opencontainers.image.source="https://github.com/llaminar/llaminar" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    CUDA_HOME=/usr/local/cuda \
    ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/usr/local/cuda/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib

COPY scripts/docker /tmp/install-scripts
RUN MODE=runtime /tmp/install-scripts/install-system-deps.sh
RUN MODE=runtime /tmp/install-scripts/install-cuda.sh
RUN MODE=runtime /tmp/install-scripts/install-rocm.sh
RUN rm -rf /tmp/install-scripts

# Copy just the compiled binary. Dynamic linker resolves libs from the apt
# packages installed above (CUDA shared libs, ROCm runtime, MPI, OpenBLAS).
COPY --from=builder /src/build_v2_release/llaminar2 /usr/local/bin/llaminar2

# Non-root user for the runtime. GPU devices on the host expose render/video
# group ownership; join those so /dev/kfd + /dev/dri work for ROCm.
RUN groupadd -f render \
 && groupadd -f video \
 && useradd -m -s /bin/bash -G render,video llaminar

USER llaminar
WORKDIR /home/llaminar

VOLUME ["/models"]
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llaminar2"]
CMD ["--help"]
