# syntax=docker/dockerfile:1.6
#
# Llaminar — CUDA + ROCm runtime image
#
# Two-stage build:
#   1. builder — Ubuntu 24.04 + full CUDA 13 toolkit + ROCm 7.2.4 + C++23
#      toolchain, compiles `llaminar2` (Release).
#   2. runtime — Ubuntu 24.04 + CUDA 13 shared libs + ROCm 7.2.4 user-space +
#      compiled binary only (no compilers, no -dev packages).
#
# All dependency-install logic lives in scripts/docker/install-*.sh, which are
# shared with .devcontainer/Dockerfile. Edit those scripts — not this file —
# to change what gets installed.
# Both output images declare their source, ISA, backend set and role. The
# builder additionally records whether its Integration installation was
# skipped; CI rejects skipped tests and still authenticates/executes the real
# installed test inventory. Metadata never substitutes for a successful gate.
#
# Runtime usage (both CUDA + ROCm available; pick per invocation with -d):
#   docker run --gpus all --rm -it \
#       --security-opt seccomp=unconfined \
#       --cap-add SYS_NICE --cap-add SYS_PTRACE \
#       --shm-size=16G \
#       -v /path/to/models:/models:ro \
#       -p 8080:8080 \
#       ghcr.io/llaminar/llaminar:latest \
#       --serve --port 8080 -d cuda:0 -m /models/Qwen2.5-1.5B-Instruct-Q8_0.gguf
#
# ROCm-only host:
#   docker run --device=/dev/kfd --device=/dev/dri \
#       --group-add "$(stat -c '%g' /dev/kfd)" \
#       --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
#       --security-opt seccomp=unconfined \
#       --cap-add SYS_NICE --cap-add SYS_PTRACE \
#       --shm-size=16G \
#       -v /path/to/models:/models:ro \
#       ghcr.io/llaminar/llaminar:latest -d rocm:0 -m /models/<gguf>

ARG CUTLASS_VERSION=v4.2.1
ARG NINJA_VERSION=1.13.0
ARG LLAMINAR_BUILD_TYPE=Release
ARG LLAMINAR_CPU_ISA=AVX512
ARG LLAMINAR_CUDA_ARCHS="80;86;89;90"
ARG LLAMINAR_ENABLE_CUDA=ON
ARG LLAMINAR_ENABLE_ROCM=ON
ARG LLAMINAR_SKIP_INTEGRATION=0
ARG LLAMINAR_BUILD_MODEL_PARITY_MATRICES=ON
ARG LLAMINAR_TEST_RUNNER_INVENTORY=full-matrix
ARG LLAMINAR_BUILD_RCCL_FROM_SOURCE=ON
ARG RCCL_GIT_REF=rocm-7.2.4
ARG RCCL_GPU_TARGETS=gfx906
ARG ROCM_RUNTIME_GPU_TARGETS=
ARG LLAMINAR_SOURCE_TREE=

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM ubuntu:24.04 AS toolchain

ARG CUTLASS_VERSION
ARG NINJA_VERSION
ARG LLAMINAR_BUILD_TYPE
ARG LLAMINAR_CUDA_ARCHS
ARG LLAMINAR_ENABLE_CUDA
ARG LLAMINAR_ENABLE_ROCM
ARG LLAMINAR_SKIP_INTEGRATION
ARG LLAMINAR_BUILD_RCCL_FROM_SOURCE
ARG RCCL_GIT_REF
ARG RCCL_GPU_TARGETS

ENV DEBIAN_FRONTEND=noninteractive \
    CUDAARCHS=${LLAMINAR_CUDA_ARCHS} \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib \
    ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    CUTLASS_DIR=/opt/cutlass \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=50G \
    CCACHE_COMPRESS=1 \
    CCACHE_COMPRESSLEVEL=6 \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime,pch_defines,locale,system_headers \
    CCACHE_COMPILERCHECK=content \
    CCACHE_NOHASHDIR=1 \
    CCACHE_BASEDIR=/src

COPY scripts/docker/install-system-deps.sh \
     scripts/docker/install-cuda.sh \
     scripts/docker/install-nccl.sh \
     scripts/docker/install-rocm.sh \
     scripts/docker/install-hip-graph-runtime.sh \
     scripts/docker/install-cutlass.sh \
     /tmp/install-scripts/
# CUDA's NCCL installer consumes this patch during early toolchain setup.
COPY scripts/docker/patches/nccl-capture-reentry.patch \
     /tmp/install-scripts/patches/nccl-capture-reentry.patch
COPY scripts/docker/patches/rocm-hip-graph-node-identity.patch \
     /tmp/install-scripts/patches/rocm-hip-graph-node-identity.patch
RUN NINJA_VERSION=${NINJA_VERSION} MODE=build \
    /tmp/install-scripts/install-system-deps.sh
RUN if [ "${LLAMINAR_ENABLE_CUDA}" = "ON" ]; then \
        MODE=full /tmp/install-scripts/install-cuda.sh; \
    else \
        echo "==> [cuda] disabled for this build"; \
    fi
RUN if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
        MODE=build /tmp/install-scripts/install-rocm.sh; \
    else \
        echo "==> [rocm] disabled for this build"; \
    fi
RUN if [ "${LLAMINAR_ENABLE_CUDA}" = "ON" ]; then \
        CUTLASS_VERSION=${CUTLASS_VERSION} /tmp/install-scripts/install-cutlass.sh; \
    else \
        echo "==> [cutlass] skipped because CUDA is disabled"; \
    fi
RUN rm -rf /tmp/install-scripts

WORKDIR /src

# RCCL — build in a dedicated layer so the ROCm collective library is visible,
# cacheable, and not hidden inside Llaminar's CMake configure step. The source
# build is enabled by default for release images because packaged ROCm 7.2.4
# RCCL has no usable gfx906 binaries on MI50/MI60 systems.
# MSCCL generated kernels are optional for standard collectives and make source
# builds dramatically slower, so release images default them off.
ARG RCCL_ENABLE_MSCCL_KERNEL=OFF
# The typed collective backend exposes these five dtypes and SUM/MIN/MAX.
# Keep direct Docker builds and the release wrapper on the same support set;
# an explicit empty override remains the full upstream RCCL developer build.
ARG RCCL_ONLY_FUNCS=default
COPY scripts/docker/rccl-functions.txt /src/rccl-functions.txt
# The capture patch is consumed only by the RCCL source build. Stage it before
# the ISA boundary below, so its immutable collective layer is shared by both
# shipping CPU variants.
COPY scripts/docker/apply-rccl-capture-patch.sh /tmp/install-scripts/
COPY scripts/docker/patches/rccl-hip-capture-event-wait.patch \
     /tmp/install-scripts/patches/rccl-hip-capture-event-wait.patch
# ccache hashes the compiler and full command line, so one 50 GB shared cache
# is correct across both ISA lanes. It lives in the persistent named BuildKit
# worker hosted by the one host Docker daemon; locked sharing keeps independent
# builds coherent without exposing Docker's graph store to another daemon.
RUN --mount=type=cache,id=llaminar-ccache,target=/root/.ccache,sharing=locked \
    set -e; \
    rccl_started_epoch="$(date +%s)"; \
    rccl_build_funcs="${RCCL_ONLY_FUNCS}"; \
    if [ "${rccl_build_funcs}" = "default" ]; then rccl_build_funcs="$(cat /src/rccl-functions.txt)"; fi; \
    if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ] && [ "${LLAMINAR_BUILD_RCCL_FROM_SOURCE}" = "ON" ]; then \
        RCCL_TARBALL_URL="https://codeload.github.com/ROCm/rccl/tar.gz/refs/tags/${RCCL_GIT_REF}"; \
        echo "==> [rccl] fetch ${RCCL_GIT_REF} from ${RCCL_TARBALL_URL}"; \
        for attempt in 1 2 3 4; do \
            rm -rf /src/external/rccl; \
            mkdir -p /src/external/rccl; \
            echo "==> [rccl] download attempt ${attempt}/4"; \
            if timeout 600s wget --progress=dot:giga \
                --dns-timeout=30 --connect-timeout=30 --read-timeout=60 \
                --tries=2 --waitretry=10 --retry-connrefused \
                -O /tmp/rccl.tar.gz "${RCCL_TARBALL_URL}" \
                && tar -xzf /tmp/rccl.tar.gz -C /src/external/rccl --strip-components=1; then \
                break; \
            fi; \
            if [ "${attempt}" = "4" ]; then exit 1; fi; \
            rm -f /tmp/rccl.tar.gz; \
            sleep $((attempt * 15)); \
        done; \
        rm -f /tmp/rccl.tar.gz; \
        printf '%s\n' "${RCCL_GIT_REF}" > /src/external/rccl/.llaminar-rccl-source-ref; \
        bash /tmp/install-scripts/apply-rccl-capture-patch.sh /src/external/rccl; \
        echo "==> [rccl] configure for GPU_TARGETS=${RCCL_GPU_TARGETS}"; \
        if [ -n "${rccl_build_funcs}" ]; then \
            echo "==> [rccl] ONLY_FUNCS=${rccl_build_funcs}"; \
            cmake -B /src/external/rccl/build -S /src/external/rccl -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
                -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
                -DCMAKE_PREFIX_PATH=/opt/rocm \
                -DROCM_PATH=/opt/rocm \
                -DGPU_TARGETS="${RCCL_GPU_TARGETS}" \
                -DENABLE_MSCCL_KERNEL="${RCCL_ENABLE_MSCCL_KERNEL}" \
                -DONLY_FUNCS="${rccl_build_funcs}" \
                -DBUILD_TESTS=OFF; \
        else \
            cmake -B /src/external/rccl/build -S /src/external/rccl -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
                -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
                -DCMAKE_PREFIX_PATH=/opt/rocm \
                -DROCM_PATH=/opt/rocm \
                -DGPU_TARGETS="${RCCL_GPU_TARGETS}" \
                -DENABLE_MSCCL_KERNEL="${RCCL_ENABLE_MSCCL_KERNEL}" \
                -DBUILD_TESTS=OFF; \
        fi; \
        echo "==> [rccl] build (parallel)"; \
        cmake --build /src/external/rccl/build --parallel --verbose; \
        if [ ! -e /src/external/rccl/build/librccl.so.1.0 ]; then \
            rccl_lib="$(find /src/external/rccl/build -maxdepth 3 -name 'librccl.so*' -type f | sort -V | tail -1)"; \
            test -n "${rccl_lib}"; \
            cp -P "${rccl_lib}" /src/external/rccl/build/librccl.so.1.0; \
        fi; \
        ln -sf librccl.so.1.0 /src/external/rccl/build/librccl.so.1; \
        ln -sf librccl.so.1 /src/external/rccl/build/librccl.so; \
        rccl_capture_patch_sha="$(sha256sum /tmp/install-scripts/patches/rccl-hip-capture-event-wait.patch | cut -d ' ' -f 1)"; \
        printf '%s\n' "${RCCL_GIT_REF}-capture-${rccl_capture_patch_sha}" \
            > /src/external/rccl/build/.llaminar-rccl-commit; \
        echo "==> [rccl] done; elapsed_seconds=$(( $(date +%s) - rccl_started_epoch )); library: $(readlink -f /src/external/rccl/build/librccl.so.1.0)"; \
    elif [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
        echo "==> [rccl] source build disabled; runtime image will stage packaged RCCL"; \
    else \
        echo "==> [rccl] skipped because ROCm is disabled"; \
    fi

# Install the selected collective dependency before compiling either binary.
# The runtime stage copies these same bytes; tests must not accidentally load
# the packaged gfx906 library merely because their checkout lives under /src.
RUN set -e; \
    if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ] && [ "${LLAMINAR_BUILD_RCCL_FROM_SOURCE}" = "ON" ]; then \
        install -m 0755 /src/external/rccl/build/librccl.so.1.0 /usr/local/lib/librccl.so.1.0; \
        ln -sf librccl.so.1.0 /usr/local/lib/librccl.so.1; \
        ln -sf librccl.so.1 /usr/local/lib/librccl.so; \
        ldconfig; \
    fi

# RCCL capture preparation consumes this helper directory above.  Removing it
# only after that dependency has been materialized keeps the toolchain image
# reproducible without leaking build helpers into later layers.
RUN rm -rf /tmp/install-scripts

# ---------------------------------------------------------------------------
# OneDNN — clone + build in a dedicated ISA-specific layer before source COPYs.
#
# The CPU ISA is deliberately first consumed only after CUDA, ROCm, CUTLASS
# and RCCL have all been materialized. Those dependencies are byte-identical
# across AVX2 and AVX512 images, so their BuildKit records can be reused between
# shipping lanes. Source edits (src/, tests/, CMakeLists.txt) do not invalidate
# this layer; it rebuilds only when its pin, ISA, or shared toolchain input
# changes.
#
# The src/v2/CMakeLists.txt OneDNN integration script detects the prebuilt
# ISA-specific tree at external/onednn/build-<isa>/include/oneapi/dnnl/dnnl.hpp
# and skips its own clone+build entirely. Keep ONEDNN_GIT_REF in sync with the
# pin in src/v2/CMakeLists.txt (search for ONEDNN_GIT_REF).
# ---------------------------------------------------------------------------
ARG LLAMINAR_CPU_ISA
ARG ONEDNN_GIT_REF=v3.11.3
RUN set -e; \
    onednn_started_epoch="$(date +%s)"; \
    case "${LLAMINAR_CPU_ISA}" in \
        AVX512) ONEDNN_ISA_SUFFIX=avx512; ONEDNN_CPU_FLAGS="-msse4.1 -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mpopcnt -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni" ;; \
        AVX2) ONEDNN_ISA_SUFFIX=avx2; ONEDNN_CPU_FLAGS="-msse4.1 -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mpopcnt" ;; \
        *) echo "Unsupported LLAMINAR_CPU_ISA='${LLAMINAR_CPU_ISA}'. Use AVX512 or AVX2." >&2; exit 1 ;; \
    esac; \
    ONEDNN_BUILD_DIR="/src/external/onednn/build-${ONEDNN_ISA_SUFFIX}"; \
    mkdir -p /src/external; \
    ONEDNN_TARBALL_URL="https://codeload.github.com/uxlfoundation/oneDNN/tar.gz/refs/tags/${ONEDNN_GIT_REF}"; \
    echo "==> [onednn] fetch ${ONEDNN_GIT_REF} from ${ONEDNN_TARBALL_URL}"; \
    for attempt in 1 2 3 4; do \
        rm -rf /src/external/onednn; \
        mkdir -p /src/external/onednn; \
        echo "==> [onednn] download attempt ${attempt}/4"; \
        if timeout 600s wget --progress=dot:giga \
            --dns-timeout=30 --connect-timeout=30 --read-timeout=60 \
            --tries=2 --waitretry=10 --retry-connrefused \
            -O /tmp/onednn.tar.gz "${ONEDNN_TARBALL_URL}" \
            && tar -xzf /tmp/onednn.tar.gz -C /src/external/onednn --strip-components=1; then \
            break; \
        fi; \
        if [ "${attempt}" = "4" ]; then exit 1; fi; \
        rm -f /tmp/onednn.tar.gz; \
        sleep $((attempt * 15)); \
    done; \
    rm -f /tmp/onednn.tar.gz; \
    printf '%s\n' "${ONEDNN_GIT_REF}" > /src/external/onednn/.llaminar-onednn-source-ref; \
    cmake -B "${ONEDNN_BUILD_DIR}" -S /src/external/onednn \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${ONEDNN_BUILD_DIR}" \
        -DDNNL_CPU_RUNTIME=OMP \
        -DDNNL_BUILD_TESTS=OFF \
        -DDNNL_BUILD_EXAMPLES=OFF \
        -DDNNL_EXPERIMENTAL_UKERNEL=ON \
        -DCMAKE_CXX_FLAGS="${ONEDNN_CPU_FLAGS}" \
        -DCMAKE_C_FLAGS="${ONEDNN_CPU_FLAGS}"; \
    cmake --build "${ONEDNN_BUILD_DIR}" --parallel --target install; \
    printf '%s\n' "${ONEDNN_GIT_REF};isa=${LLAMINAR_CPU_ISA}" \
        > "${ONEDNN_BUILD_DIR}/.llaminar-onednn-commit"; \
    # Drop OneDNN's intermediate .o / .d files but keep the installed lib +
    # headers + source ref marker used by CMake cache validation.
    find "${ONEDNN_BUILD_DIR}" \
        \( -name '*.o' -o -name '*.d' -o -name CMakeFiles \) \
        -prune -exec rm -rf {} +; \
    echo "==> [onednn] done; elapsed_seconds=$(( $(date +%s) - onednn_started_epoch ))"

# Python dependencies for the reference tests + parity gates. Pulls the
# CPU-only PyTorch wheel (~250 MB) plus our transformers fork. Cached as a
# separate layer keyed only on requirements.txt so source edits don't
# invalidate it.
COPY requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -r requirements.txt

FROM toolchain AS builder

# The full production pipeline needs matrix executables for typed E2E discovery.
# The develop image gate is intentionally model-free, so it selects the smaller
# inventory below. Declare this only at the builder boundary: toolchain layers
# must remain reusable across the two policies as well as both CPU ISAs.
ARG LLAMINAR_BUILD_MODEL_PARITY_MATRICES
ARG LLAMINAR_TEST_RUNNER_INVENTORY
RUN case "${LLAMINAR_BUILD_MODEL_PARITY_MATRICES}:${LLAMINAR_TEST_RUNNER_INVENTORY}" in \
        ON:full-matrix|OFF:model-free) ;; \
        *) echo "Invalid test-runner inventory: matrices=${LLAMINAR_BUILD_MODEL_PARITY_MATRICES}, inventory=${LLAMINAR_TEST_RUNNER_INVENTORY}" >&2; exit 1 ;; \
    esac

# Some preflight regressions reuse the native kernel tuning harness. Its
# ROCTX annotation dependency is test tooling, not an inference dependency.
# Install it after the expensive toolchain cache boundary and use this same
# small package closure in the installed test runner below.
COPY scripts/docker/install-rocm-test-deps.sh /tmp/install-rocm-test-deps.sh
RUN if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
        bash /tmp/install-rocm-test-deps.sh; \
    fi \
 && rm /tmp/install-rocm-test-deps.sh

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
# Unit/static tests exercise top-level helper scripts directly, so the builder
# test image needs the full script tree rather than only CI summary helpers.
COPY scripts ./scripts
COPY .agents ./.agents
COPY .github ./.github
COPY .devcontainer ./.devcontainer
COPY AGENTS.md README.md ./
COPY benchmarks/production ./benchmarks/production

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
RUN --mount=type=cache,id=llaminar-ccache,target=/root/.ccache,sharing=locked \
    if [ "${LLAMINAR_SKIP_INTEGRATION}" = "1" ]; then \
        echo "==> [integration] skipped (LLAMINAR_SKIP_INTEGRATION=1)"; \
    else \
        integration_started_epoch="$(date +%s)"; \
        RCCL_CMAKE_ARGS="-DLLAMINAR_BUILD_RCCL_FROM_SOURCE=OFF"; \
        if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ] && [ "${LLAMINAR_BUILD_RCCL_FROM_SOURCE}" = "ON" ]; then \
            RCCL_CMAKE_ARGS="${RCCL_CMAKE_ARGS} -DRCCL_INCLUDE_DIR=/src/external/rccl/src/include -DRCCL_LIBRARY=/usr/local/lib/librccl.so.1"; \
        fi; \
        echo "==> [integration] cmake configure" \
     && cmake -B build_v2_integration -S src/v2 -G Ninja \
            -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
            -DCMAKE_BUILD_TYPE=Integration \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DHAVE_CUDA="${LLAMINAR_ENABLE_CUDA}" \
            -DHAVE_ROCM="${LLAMINAR_ENABLE_ROCM}" \
            -DLLAMINAR_CPU_ISA="${LLAMINAR_CPU_ISA}" \
            -DLLAMINAR_SHARED_CORE=ON \
            -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
            ${RCCL_CMAKE_ARGS} \
     && case "${LLAMINAR_BUILD_MODEL_PARITY_MATRICES}" in \
            ON) integration_targets="v2_unit_gate v2_production_test_preflight_gate v2_model_parity_matrices" ;; \
            OFF) integration_targets="v2_unit_gate v2_production_test_preflight_gate" ;; \
            *) echo "Unsupported LLAMINAR_BUILD_MODEL_PARITY_MATRICES='${LLAMINAR_BUILD_MODEL_PARITY_MATRICES}'. Use ON or OFF." >&2; exit 1 ;; \
        esac \
     && echo "==> [integration] cmake build (parallel; matrices=${LLAMINAR_BUILD_MODEL_PARITY_MATRICES})" \
     && cmake --build build_v2_integration --parallel \
            --target ${integration_targets} \
     && echo "==> [integration] strip --strip-debug on executables/.a/.so (parallel, $(nproc) jobs)" \
     && { find build_v2_integration \
              \( -type f -executable -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
              -not -path '*/CMakeFiles/*' \
              -print0 \
          | xargs -0 -r -P "$(nproc)" -n 32 strip --strip-debug 2>/dev/null || true; } \
     && echo "==> [integration] removing intermediates (.o/.d/.gch/CMakeFiles)" \
     && mv build_v2_integration/CMakeFiles/rules.ninja build_v2_integration/installed-rules.ninja \
     && find build_v2_integration \
            \( -name '*.o' -o -name '*.d' -o -name '*.gch' -o -name '*.cmake_pch.hxx' \) \
            -delete \
     && find build_v2_integration -depth -type d -name CMakeFiles -exec rm -rf {} + \
     # Keep the small compiler-launcher receipt used by the build-contract
     # Unit test. Objects remain discarded; this is evidence, not a rebuild tree.
     && mkdir build_v2_integration/CMakeFiles \
     && mv build_v2_integration/installed-rules.ninja build_v2_integration/CMakeFiles/rules.ninja \
     && rm -rf build_v2_integration/Testing build_v2_integration/_deps/*-build/CMakeFiles \
     && echo "==> [integration] done; elapsed_seconds=$(( $(date +%s) - integration_started_epoch )); final size: $(du -sh build_v2_integration | cut -f1)"; \
    fi

# This receipt permits the campaign driver to use installed binaries without
# reconstructing stripped objects. Both complete model-free gates still run.
RUN if [ "${LLAMINAR_SKIP_INTEGRATION}" != "1" ]; then \
      python3 scripts/ci/prebuilt_test_image.py --build-dir build_v2_integration \
        --receipt /src/installed-tests.json --seal; \
    fi

# Release build — what the runtime image ships. Optimized, no assertions,
# only the llaminar2 target (skip test binaries). Same in-RUN cleanup.
RUN --mount=type=cache,id=llaminar-ccache,target=/root/.ccache,sharing=locked \
    release_started_epoch="$(date +%s)"; \
    RCCL_CMAKE_ARGS="-DLLAMINAR_BUILD_RCCL_FROM_SOURCE=OFF"; \
    if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ] && [ "${LLAMINAR_BUILD_RCCL_FROM_SOURCE}" = "ON" ]; then \
        RCCL_CMAKE_ARGS="${RCCL_CMAKE_ARGS} -DRCCL_INCLUDE_DIR=/src/external/rccl/src/include -DRCCL_LIBRARY=/usr/local/lib/librccl.so.1"; \
    fi; \
    echo "==> [release] cmake configure" \
 && cmake -B build_v2_release -S src/v2 -G Ninja \
        -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)" \
        -DCMAKE_BUILD_TYPE=${LLAMINAR_BUILD_TYPE} \
        -DHAVE_CUDA="${LLAMINAR_ENABLE_CUDA}" \
        -DHAVE_ROCM="${LLAMINAR_ENABLE_ROCM}" \
        -DLLAMINAR_CPU_ISA="${LLAMINAR_CPU_ISA}" \
        -DLLAMINAR_SHARED_CORE=ON \
        -DLLAMINAR_BUILD_TESTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
        ${RCCL_CMAKE_ARGS} \
 && echo "==> [release] cmake build --target llaminar2 (parallel)" \
 && cmake --build build_v2_release --parallel --target llaminar2 \
 && if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ] && [ ! -e external/rccl/build/librccl.so.1.0 ]; then \
        echo "==> [release] source-built RCCL not present; staging system RCCL"; \
        mkdir -p external/rccl/build; \
        cp -P /opt/rocm/lib/librccl.so* external/rccl/build/; \
    fi \
 && echo "==> [release] strip --strip-debug on executables/.a/.so (parallel, $(nproc) jobs)" \
 && { find build_v2_release \
          \( -type f -executable -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
          -not -path '*/CMakeFiles/*' \
          -print0 \
      | xargs -0 -r -P "$(nproc)" -n 32 strip --strip-debug 2>/dev/null || true; } \
 && echo "==> [release] removing intermediates (.o/.d/.gch/CMakeFiles)" \
 && find build_v2_release \
        \( -name '*.o' -o -name '*.d' -o -name '*.gch' -o -name '*.cmake_pch.hxx' \) \
        -delete \
 && find build_v2_release -depth -type d -name CMakeFiles -exec rm -rf {} + \
 && mkdir -p /src/runtime-bin /src/runtime-libs /src/runtime-licenses \
 && cp build_v2_release/llaminar2 /src/runtime-bin/llaminar2 \
 && cp build_v2_release/libllaminar2_core.so /src/runtime-libs/ \
 && if [ "${LLAMINAR_ENABLE_CUDA}" = "ON" ]; then cp -P /usr/local/lib/libllaminar_nccl.so* /src/runtime-libs/; fi \
 && if [ "${LLAMINAR_ENABLE_CUDA}" = "ON" ]; then cp -r /usr/local/share/licenses/llaminar-nccl /src/runtime-licenses/; fi \
 && if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
        cp -L /opt/rocm/lib/libamdhip64.so.7 /src/runtime-libs/libamdhip64.so.7; \
        cp -r /opt/rocm/share/licenses/llaminar-hip /src/runtime-licenses/; \
    fi \
 && cp "external/onednn/build-$(printf '%s' "${LLAMINAR_CPU_ISA}" | tr '[:upper:]' '[:lower:]')/lib/libdnnl.so.3.11" /src/runtime-libs/ \
 && if [ -e external/rccl/build/librccl.so.1.0 ]; then cp external/rccl/build/librccl.so.1.0 /src/runtime-libs/; fi \
 && echo "==> [release] done; elapsed_seconds=$(( $(date +%s) - release_started_epoch )); final size: $(du -sh build_v2_release | cut -f1)"

# The source-identity Unit constructs a tiny Git fixture using the real ignore
# policy. Materialize this source-policy input after compilation so a revision
# label cannot invalidate compilation. The test-runner stage copies this sealed
# workspace verbatim and owns its user setup and public image identity.
COPY Dockerfile .dockerignore .gitignore ./

# =============================================================================
# Stage 1a: Test workspace — retain only the executable CTest closure
# =============================================================================
#
# ``builder`` is deliberately a compiler stage. It retains transient Release
# artifacts, static backend archives and fetched dependency source that are
# useful while producing images, but are not part of an installed test's
# runtime closure. The receipt created above seals CTest registrations, the
# Integration executables and ``libllaminar2_core.so``; it does not admit the
# files removed here. Runtime has already copied its own launcher and shared
# libraries from ``builder``, so this cleanup cannot affect the published image.
#
# Keep the complete source tree and Integration CMake metadata: source-policy
# tests and CTest's generated registration include them by absolute path. The
# test binaries' OneDNN RUNPATH falls through to the identical runtime library
# in ``/usr/local/lib`` once the build-time source tree is absent.
FROM builder AS test-workspace

RUN set -e; \
    rm -rf /src/build_v2_release \
           /src/runtime-bin \
           /src/runtime-libs \
           /src/runtime-licenses \
           /src/external/onednn \
           /src/external/rccl \
           /src/build_v2_integration/_deps; \
    rm -f /src/build_v2_integration/libcuda_backend.a \
          /src/build_v2_integration/librocm_backend.a \
          /src/build_v2_integration/libdevice_benchmarks.a \
          /src/build_v2_integration/libllaminar2_core.a

# =============================================================================
# Stage 2: Runtime — slim image with only shared libs + the binary
# =============================================================================
FROM ubuntu:24.04 AS runtime

ARG LLAMINAR_BUILD_TYPE
ARG LLAMINAR_ENABLE_CUDA=ON
ARG LLAMINAR_ENABLE_ROCM=ON
ARG ROCM_RUNTIME_GPU_TARGETS

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    LLAMINAR_ENABLE_CUDA=${LLAMINAR_ENABLE_CUDA} \
    LLAMINAR_ENABLE_ROCM=${LLAMINAR_ENABLE_ROCM} \
    CUDA_HOME=/usr/local/cuda \
    ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    ROCM_RUNTIME_GPU_TARGETS=${ROCM_RUNTIME_GPU_TARGETS} \
    PATH=/usr/local/cuda/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/opt/rocm/lib

COPY scripts/docker/install-system-deps.sh \
     scripts/docker/install-cuda.sh \
     scripts/docker/install-rocm-runtime.sh \
     scripts/docker/prune-runtime-image.sh \
     /tmp/install-scripts/
RUN MODE=runtime /tmp/install-scripts/install-system-deps.sh
RUN if [ "${LLAMINAR_ENABLE_CUDA}" = "ON" ]; then \
        MODE=runtime /tmp/install-scripts/install-cuda.sh; \
    else \
        echo "==> [cuda] runtime disabled for this image"; \
    fi
RUN if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
        bash /tmp/install-scripts/install-rocm-runtime.sh \
     && bash /tmp/install-scripts/prune-runtime-image.sh; \
    else \
        echo "==> [rocm] runtime disabled for this image"; \
    fi
RUN rm -rf /tmp/install-scripts

# The base runtime packages do not vary by CPU ISA.  Keep the ISA argument and
# its visible runtime contract below their cache boundary so AVX2 and AVX512
# share CUDA/ROCm/MPI installation layers on the retained BuildKit worker.
ARG LLAMINAR_CPU_ISA
ENV LLAMINAR_CPU_ISA=${LLAMINAR_CPU_ISA}

# Copy the compiled launcher, core shared library, and vendored shared
# dependencies. The remaining dynamic libraries resolve from the apt packages
# installed above (CUDA shared libs, ROCm runtime, MPI, OpenBLAS).
COPY --from=builder /src/runtime-bin/ /usr/local/bin/
COPY --from=builder /src/runtime-libs/ /usr/local/lib/
COPY --from=builder /src/runtime-licenses/ /usr/local/share/licenses/
RUN ln -sf libdnnl.so.3.11 /usr/local/lib/libdnnl.so.3 \
 && ln -sf libdnnl.so.3 /usr/local/lib/libdnnl.so \
 && if [ -e /usr/local/lib/librccl.so.1.0 ]; then \
        ln -sf librccl.so.1.0 /usr/local/lib/librccl.so.1; \
        ln -sf librccl.so.1 /usr/local/lib/librccl.so; \
    fi \
 && ldconfig

# Non-root user for the runtime. GPU devices on the host expose render/video
# group ownership; join those so /dev/kfd + /dev/dri work for ROCm.
RUN groupadd -f render \
 && groupadd -f video \
 && useradd -m -s /bin/bash -G render,video llaminar

# Revision labels must not force reinstalling CUDA/ROCm on every source commit.
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=dev
ARG LLAMINAR_SOURCE_TREE
LABEL org.opencontainers.image.title="Llaminar" \
      org.opencontainers.image.description="High-performance LLM inference engine (CUDA + ROCm)" \
      org.opencontainers.image.source="https://github.com/llaminar/llaminar" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.llaminar.image_role="runtime" \
      org.llaminar.cpu_isa="${LLAMINAR_CPU_ISA}" \
      org.llaminar.source_tree="${LLAMINAR_SOURCE_TREE}" \
      org.llaminar.build_type="${LLAMINAR_BUILD_TYPE}" \
      org.llaminar.cuda="${LLAMINAR_ENABLE_CUDA}" \
      org.llaminar.rocm="${LLAMINAR_ENABLE_ROCM}"

USER llaminar
WORKDIR /home/llaminar

VOLUME ["/models"]
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llaminar2"]
CMD ["--help"]

# =============================================================================
# Stage 3: Test runner — runtime-equivalent libraries plus sealed test evidence
# =============================================================================
#
# The compiler/toolchain stage is expensive to import into Docker's image
# store (~20 GB), yet CI only needs it to produce artifacts.  The gate must run
# exactly those sealed Integration binaries, but it does not need compilers,
# headers or object files.  Derive the test runner from the actual runtime so
# it exercises the same CUDA/ROCm/MPI shared-library closure, then add only the
# CTest/Python harness and immutable workspace used by the registered gates.
# Building this target first also imports the runtime's layers; the subsequent
# runtime target is therefore a metadata/final-layer transaction rather than a
# second independent multi-gigabyte import.
FROM runtime AS test-runner

USER root

ARG LLAMINAR_BUILD_TYPE
ARG LLAMINAR_CPU_ISA
ARG LLAMINAR_ENABLE_CUDA
ARG LLAMINAR_ENABLE_ROCM
ARG LLAMINAR_SKIP_INTEGRATION
ARG LLAMINAR_TEST_RUNNER_INVENTORY
ARG LLAMINAR_TEST_UID=1000
ARG LLAMINAR_TEST_GID=1000
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=dev
ARG LLAMINAR_SOURCE_TREE

# CTest executes the sealed CMake registration and the source-identity test
# creates a private Git fixture.  A small number of Unit scripts also compile
# generated selectors in private temporary directories, summarize JSON through
# ``jq``, and invoke the same pinned Ninja path recorded by CMake. Retain only
# that fixture closure (``g++``, ``jq``, and the pinned Ninja binary), not the
# production toolchain, SDK headers, or build objects. The runtime already
# provides Python and OpenMPI; these are the only host tools absent from its
# production closure.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      jq \
 && rm -rf /var/lib/apt/lists/*

# The compiler-stage annotations must also resolve when sealed test binaries
# execute here. This layer does not enter the production runtime image.
COPY scripts/docker/install-rocm-test-deps.sh /tmp/install-rocm-test-deps.sh
RUN if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
        bash /tmp/install-rocm-test-deps.sh; \
    fi \
 && rm /tmp/install-rocm-test-deps.sh

# CTest's sealed registration names the workspace-pinned Ninja by its absolute
# /usr/local path. Copy that tiny exact executable rather than substituting the
# distribution version and silently testing a different generator contract.
COPY --from=toolchain /usr/local/bin/ninja /usr/local/bin/ninja

# Preserve the exact source/build paths encoded in CTest and binary RUNPATHs.
# Copying the sealed workspace wholesale avoids a second hand-maintained list
# of test scripts, CMake metadata and source-policy fixtures. ``test-workspace``
# has already removed compiler-only/redundant artifacts, so this is materially
# smaller than importing the compiler image while retaining every receipt-bound
# test file.
COPY --from=test-workspace /src/ /src/
COPY --from=builder /usr/local/lib/python3.12/dist-packages/ \
     /usr/local/lib/python3.12/dist-packages/

# Test executables carry the Integration core in their CTest-relative build
# tree.  ``runtime`` also contains the Release core at /usr/local/lib for the
# serving executable.  Resolve the test-local library first: mixing a sealed
# Integration executable with that Release library can silently change tensor
# behavior or fail at a newer symbol.  This ordering is test-runner-only and
# cannot affect a published runtime image.
ENV LD_LIBRARY_PATH=/src/build_v2_integration:/usr/local/lib:/usr/local/cuda/lib64:/opt/rocm/lib

# Docker resolves group names inside the test-runner image.  Establish the host
# render/video groups and a matching ephemeral CI identity before CTest creates
# its own logs or reference scratch data.  Runtime ownership remains isolated:
# this layer is never published as a serving artifact.
RUN set -e; \
    groupadd -f render; \
    groupadd -f video; \
    if ! getent group "${LLAMINAR_TEST_GID}" >/dev/null; then \
        groupadd --gid "${LLAMINAR_TEST_GID}" llaminar-ci; \
    fi; \
    if ! getent passwd "${LLAMINAR_TEST_UID}" >/dev/null; then \
        useradd --uid "${LLAMINAR_TEST_UID}" --gid "${LLAMINAR_TEST_GID}" \
            --create-home --shell /bin/bash llaminar-ci; \
    fi; \
    find /src -type d -exec chown "${LLAMINAR_TEST_UID}:${LLAMINAR_TEST_GID}" {} +

ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

WORKDIR /src

# The test driver supplies its complete argv explicitly.  Clear the runtime
# launcher entrypoint so a test command cannot accidentally invoke inference.
ENTRYPOINT []
CMD ["bash"]

LABEL org.opencontainers.image.title="Llaminar test runner" \
      org.opencontainers.image.description="Sealed Unit and ProductionTestPreflight runner" \
      org.opencontainers.image.source="https://github.com/llaminar/llaminar" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.llaminar.image_role="test-runner" \
      org.llaminar.cpu_isa="${LLAMINAR_CPU_ISA}" \
      org.llaminar.source_tree="${LLAMINAR_SOURCE_TREE}" \
      org.llaminar.build_type="${LLAMINAR_BUILD_TYPE}" \
      org.llaminar.cuda="${LLAMINAR_ENABLE_CUDA}" \
      org.llaminar.rocm="${LLAMINAR_ENABLE_ROCM}" \
      org.llaminar.integration_skipped="${LLAMINAR_SKIP_INTEGRATION}" \
      org.llaminar.test_runner_inventory="${LLAMINAR_TEST_RUNNER_INVENTORY}"
