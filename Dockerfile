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
#       --security-opt seccomp=unconfined \
#       --cap-add SYS_NICE --cap-add SYS_PTRACE \
#       --ipc=host --shm-size=16G \
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
#       --ipc=host --shm-size=16G \
#       -v /path/to/models:/models:ro \
#       ghcr.io/llaminar/llaminar:latest -d rocm:0 -m /models/<gguf>

ARG CUTLASS_VERSION=v4.2.1
ARG LLAMINAR_BUILD_TYPE=Release
ARG LLAMINAR_CUDA_ARCHS="80;86;89;90"
ARG LLAMINAR_ENABLE_CUDA=ON
ARG LLAMINAR_ENABLE_ROCM=ON
ARG LLAMINAR_SKIP_INTEGRATION=0
ARG LLAMINAR_BUILD_RCCL_FROM_SOURCE=ON
ARG RCCL_GIT_REF=rocm-7.1.1
ARG RCCL_GPU_TARGETS=gfx906
ARG ROCM_RUNTIME_GPU_TARGETS=

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM ubuntu:24.04 AS builder

ARG CUTLASS_VERSION
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
    CCACHE_MAXSIZE=20G \
    CCACHE_COMPRESS=1 \
    CCACHE_COMPRESSLEVEL=6 \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime,pch_defines,locale,system_headers \
    CCACHE_COMPILERCHECK=content \
    CCACHE_NOHASHDIR=1 \
    CCACHE_BASEDIR=/src

COPY scripts/docker/install-system-deps.sh \
     scripts/docker/install-cuda.sh \
     scripts/docker/install-rocm.sh \
     scripts/docker/install-cutlass.sh \
     /tmp/install-scripts/
RUN MODE=build  /tmp/install-scripts/install-system-deps.sh
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
    printf '%s\n' "${ONEDNN_GIT_REF}" \
        > /src/external/onednn/build/.llaminar-onednn-commit; \
    # Drop OneDNN's intermediate .o / .d files but keep the installed lib +
    # headers + source ref marker used by CMake cache validation.
    find /src/external/onednn/build \
        \( -name '*.o' -o -name '*.d' -o -name CMakeFiles \) \
        -prune -exec rm -rf {} +

# RCCL — build in a dedicated layer so the ROCm collective library is visible,
# cacheable, and not hidden inside Llaminar's CMake configure step. The source
# build is enabled by default for release images because Ubuntu's packaged
# ROCm 7.1.1 RCCL has broken gfx906 binaries on MI50/MI60 systems.
# MSCCL generated kernels are optional for standard collectives and make source
# builds dramatically slower, so release images default them off.
ARG RCCL_ENABLE_MSCCL_KERNEL=OFF
ARG RCCL_ONLY_FUNCS=
RUN --mount=type=cache,target=/root/.ccache \
    set -e; \
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
        echo "==> [rccl] configure for GPU_TARGETS=${RCCL_GPU_TARGETS}"; \
        if [ -n "${RCCL_ONLY_FUNCS}" ]; then \
            echo "==> [rccl] ONLY_FUNCS=${RCCL_ONLY_FUNCS}"; \
            cmake -B /src/external/rccl/build -S /src/external/rccl -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
                -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
                -DCMAKE_PREFIX_PATH=/opt/rocm \
                -DROCM_PATH=/opt/rocm \
                -DGPU_TARGETS="${RCCL_GPU_TARGETS}" \
                -DENABLE_MSCCL_KERNEL="${RCCL_ENABLE_MSCCL_KERNEL}" \
                -DONLY_FUNCS="${RCCL_ONLY_FUNCS}" \
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
        printf '%s\n' "${RCCL_GIT_REF}" \
            > /src/external/rccl/build/.llaminar-rccl-commit; \
        echo "==> [rccl] done; library: $(readlink -f /src/external/rccl/build/librccl.so.1.0)"; \
    elif [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ]; then \
        echo "==> [rccl] source build disabled; runtime image will stage packaged RCCL"; \
    else \
        echo "==> [rccl] skipped because ROCm is disabled"; \
    fi

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
# Unit/static tests exercise top-level helper scripts directly, so the builder
# test image needs the full script tree rather than only CI summary helpers.
COPY scripts ./scripts

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
    if [ "${LLAMINAR_SKIP_INTEGRATION}" = "1" ]; then \
        echo "==> [integration] skipped (LLAMINAR_SKIP_INTEGRATION=1)"; \
    else \
        RCCL_CMAKE_ARGS="-DLLAMINAR_BUILD_RCCL_FROM_SOURCE=OFF"; \
        if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ] && [ "${LLAMINAR_BUILD_RCCL_FROM_SOURCE}" = "ON" ]; then \
            RCCL_CMAKE_ARGS="${RCCL_CMAKE_ARGS} -DRCCL_INCLUDE_DIR=/src/external/rccl/src/include -DRCCL_LIBRARY=/src/external/rccl/build/librccl.so"; \
        fi; \
        echo "==> [integration] cmake configure" \
     && cmake -B build_v2_integration -S src/v2 -G Ninja \
            -DCMAKE_BUILD_TYPE=Integration \
            -DHAVE_CUDA="${LLAMINAR_ENABLE_CUDA}" \
            -DHAVE_ROCM="${LLAMINAR_ENABLE_ROCM}" \
            -DLLAMINAR_SHARED_CORE=ON \
            -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
            ${RCCL_CMAKE_ARGS} \
     && echo "==> [integration] cmake build (parallel)" \
     && cmake --build build_v2_integration --parallel \
     && echo "==> [integration] strip --strip-debug on executables/.a/.so (parallel, $(nproc) jobs)" \
     && { find build_v2_integration \
              \( -type f -executable -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \) \
              -not -path '*/CMakeFiles/*' \
              -print0 \
          | xargs -0 -r -P "$(nproc)" -n 32 strip --strip-debug 2>/dev/null || true; } \
     && echo "==> [integration] removing intermediates (.o/.d/.gch/CMakeFiles)" \
     && find build_v2_integration \
            \( -name '*.o' -o -name '*.d' -o -name '*.gch' -o -name '*.cmake_pch.hxx' \) \
            -delete \
     && find build_v2_integration -depth -type d -name CMakeFiles -exec rm -rf {} + \
     && rm -rf build_v2_integration/Testing build_v2_integration/_deps/*-build/CMakeFiles \
     && echo "==> [integration] done; final size: $(du -sh build_v2_integration | cut -f1)"; \
    fi

# Release build — what the runtime image ships. Optimized, no assertions,
# only the llaminar2 target (skip test binaries). Same in-RUN cleanup.
RUN --mount=type=cache,target=/root/.ccache \
    RCCL_CMAKE_ARGS="-DLLAMINAR_BUILD_RCCL_FROM_SOURCE=OFF"; \
    if [ "${LLAMINAR_ENABLE_ROCM}" = "ON" ] && [ "${LLAMINAR_BUILD_RCCL_FROM_SOURCE}" = "ON" ]; then \
        RCCL_CMAKE_ARGS="${RCCL_CMAKE_ARGS} -DRCCL_INCLUDE_DIR=/src/external/rccl/src/include -DRCCL_LIBRARY=/src/external/rccl/build/librccl.so"; \
    fi; \
    echo "==> [release] cmake configure" \
 && cmake -B build_v2_release -S src/v2 -G Ninja \
        -DCMAKE_BUILD_TYPE=${LLAMINAR_BUILD_TYPE} \
        -DHAVE_CUDA="${LLAMINAR_ENABLE_CUDA}" \
        -DHAVE_ROCM="${LLAMINAR_ENABLE_ROCM}" \
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
 && mkdir -p /src/runtime-bin /src/runtime-libs \
 && cp build_v2_release/llaminar2 /src/runtime-bin/llaminar2 \
 && cp build_v2_release/libllaminar2_core.so /src/runtime-libs/ \
 && cp external/onednn/build/lib/libdnnl.so.3.11 /src/runtime-libs/ \
 && if [ -e external/rccl/build/librccl.so.1.0 ]; then cp external/rccl/build/librccl.so.1.0 /src/runtime-libs/; fi \
 && echo "==> [release] done; final size: $(du -sh build_v2_release | cut -f1)"

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
ARG LLAMINAR_ENABLE_CUDA=ON
ARG LLAMINAR_ENABLE_ROCM=ON
ARG ROCM_RUNTIME_GPU_TARGETS

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

# Copy the compiled launcher, core shared library, and vendored shared
# dependencies. The remaining dynamic libraries resolve from the apt packages
# installed above (CUDA shared libs, ROCm runtime, MPI, OpenBLAS).
COPY --from=builder /src/runtime-bin/ /usr/local/bin/
COPY --from=builder /src/runtime-libs/ /usr/local/lib/
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

USER llaminar
WORKDIR /home/llaminar

VOLUME ["/models"]
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/llaminar2"]
CMD ["--help"]
