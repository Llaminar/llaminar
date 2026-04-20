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

# ---------------------------------------------------------------------------
# OneDNN — clone + build in a dedicated layer BEFORE the source COPYs.
#
# This layer's cache key depends only on the install scripts, the base image,
# and the pinned tag below. Source edits (src/, tests/, CMakeLists.txt) do
# NOT invalidate it, so the ~2-minute clone + configure + compile only re-runs
# when ONEDNN_GIT_REF is bumped here OR when an earlier layer changes.
#
# The src/v2/CMakeLists.txt OneDNN integration script detects the prebuilt
# tree at external/onednn/build/include/oneapi/dnnl/dnnl.hpp and skips its
# own clone+build entirely. Keep ONEDNN_GIT_REF in sync with the pin in
# src/v2/CMakeLists.txt (search for ONEDNN_GIT_REF).
# ---------------------------------------------------------------------------
ARG ONEDNN_GIT_REF=v3.11.3
RUN set -e; \
    mkdir -p /src/external; \
    git clone --depth 1 --branch ${ONEDNN_GIT_REF} \
        https://github.com/uxlfoundation/oneDNN.git /src/external/onednn; \
    cmake -B /src/external/onednn/build -S /src/external/onednn \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/src/external/onednn/build \
        -DDNNL_CPU_RUNTIME=OMP \
        -DDNNL_BUILD_TESTS=OFF \
        -DDNNL_BUILD_EXAMPLES=OFF \
        -DDNNL_EXPERIMENTAL_UKERNEL=ON \
        -DCMAKE_CXX_FLAGS=-march=native \
        -DCMAKE_C_FLAGS=-march=native; \
    cmake --build /src/external/onednn/build --parallel --target install; \
    git -C /src/external/onednn rev-parse HEAD \
        > /src/external/onednn/build/.llaminar-onednn-commit; \
    # Drop OneDNN's intermediate .o / .d files but keep the installed lib +
    # headers + .git (cmake checks `git remote get-url origin` to validate
    # the checkout matches the expected upstream URL).
    find /src/external/onednn/build \
        \( -name '*.o' -o -name '*.d' -o -name CMakeFiles \) \
        -prune -exec rm -rf {} +

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
# CI helper scripts (parity/perf trend chart generation, etc.). Copied as a
# separate cheap layer so post-build CI steps that invoke
# `python3 scripts/ci/summarize_*_trends.py` inside this image find them.
COPY scripts/ci ./scripts/ci

# Integration build — what CI drives for unit, parity, and E2E tests. Has
# debug symbols, assertions active, tensor verification enabled.
#
# Strip + intermediate cleanup happens INSIDE this RUN so the committed
# layer is already minimal. Splitting build / strip / clean across multiple
# RUNs would commit a 150 GB+ snapshot first (BuildKit's `exporting layers`
# step has to write the full overlay diff), then a smaller delta — total
# export time scales with the largest intermediate, not the final size.
#
# Stripping (--strip-debug, not --strip-all) keeps the symbol table so
# stack traces from gtest / gdb attach still resolve function names.
# Removing .o / .d / .gch files is safe: ctest never re-invokes the
# compiler at test time.
RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build_v2_integration -S src/v2 -G Ninja \
        -DCMAKE_BUILD_TYPE=Integration \
        -DHAVE_CUDA=ON \
        -DHAVE_ROCM=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
 && cmake --build build_v2_integration --parallel \
 && find build_v2_integration \
        \( -type f -executable -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
        -not -path '*/CMakeFiles/*' \
        -print0 \
    | xargs -0 -r -P "$(nproc)" -n 32 strip --strip-debug 2>/dev/null || true \
 && find build_v2_integration \
        \( -name '*.o' -o -name '*.d' -o -name '*.gch' -o -name '*.cmake_pch.hxx' \) \
        -delete \
 && find build_v2_integration -depth -type d -name CMakeFiles -exec rm -rf {} + \
 && rm -rf build_v2_integration/Testing build_v2_integration/_deps/*-build/CMakeFiles

# Release build — what the runtime image ships. Optimized, no assertions,
# only the llaminar2 target (skip test binaries). Same in-RUN cleanup.
RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build_v2_release -S src/v2 -G Ninja \
        -DCMAKE_BUILD_TYPE=${LLAMINAR_BUILD_TYPE} \
        -DHAVE_CUDA=ON \
        -DHAVE_ROCM=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
 && cmake --build build_v2_release --parallel --target llaminar2 \
 && find build_v2_release \
        \( -type f -executable -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
        -not -path '*/CMakeFiles/*' \
        -print0 \
    | xargs -0 -r -P "$(nproc)" -n 32 strip --strip-debug 2>/dev/null || true \
 && find build_v2_release \
        \( -name '*.o' -o -name '*.d' -o -name '*.gch' -o -name '*.cmake_pch.hxx' \) \
        -delete \
 && find build_v2_release -depth -type d -name CMakeFiles -exec rm -rf {} +

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
