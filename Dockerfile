# syntax=docker/dockerfile:1.6
#
# Llaminar — CUDA runtime image
#
# Multi-stage build:
#   1. builder: CUDA 13 devel + OpenMPI + OpenBLAS + GCC 14 toolchain,
#      compiles `llaminar2` in Release configuration.
#   2. runtime: CUDA 13 runtime + minimal libs + compiled binary.
#
# The image does NOT bundle models. Mount your GGUF weights into /models at
# container run time and point `-m` at them.
#
# Example:
#   docker run --gpus all --rm -it \
#       -v /path/to/models:/models:ro \
#       -p 8080:8080 \
#       ghcr.io/llaminar/llaminar:latest \
#       --serve --port 8080 -d cuda:0 -m /models/Qwen2.5-1.5B-Instruct-Q8_0.gguf
#

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS builder

ARG CUTLASS_VERSION=v4.2.1
ARG LLAMINAR_BUILD_TYPE=Release
ARG LLAMINAR_CUDA_ARCHS="75;80;86;89;90"

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    CUDAARCHS=${LLAMINAR_CUDA_ARCHS}

# Core build deps plus NCCL (CUDA 13). The nvidia/cuda:*-devel-* base image
# has the CUDA apt repo configured but ships without Ubuntu's `universe`
# repository enabled, so cmake / ninja-build / libopenblas-dev / libgtest-dev
# etc. are invisible. Enable universe explicitly before the main install.
#
# GCC 14 is required for the project's C++23 usage (std::print).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        software-properties-common \
    && add-apt-repository -y universe \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        cmake \
        curl \
        g++-14 \
        gcc-14 \
        gfortran-14 \
        git \
        jq \
        libgmock-dev \
        libgtest-dev \
        libhwloc-dev \
        libnccl2 \
        libnccl-dev \
        libnuma-dev \
        libopenblas-dev \
        libopenmpi-dev \
        libscalapack-openmpi-dev \
        libssl-dev \
        ninja-build \
        openmpi-bin \
        openmpi-common \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Default GCC = 14 for C++23 support.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-14 \
        --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-14 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-14

# Build & install GoogleTest headers+libs.
RUN cmake -S /usr/src/googletest -B /tmp/gtest-build \
    && cmake --build /tmp/gtest-build --target install --parallel \
    && rm -rf /tmp/gtest-build

# CUTLASS (header-only tensor-core templates).
RUN git clone --branch "${CUTLASS_VERSION}" --depth 1 \
        https://github.com/NVIDIA/cutlass.git /opt/cutlass
ENV CUTLASS_DIR=/opt/cutlass

WORKDIR /src

# Copy only what we need to build (exclusions in .dockerignore).
COPY src ./src
COPY tests ./tests
COPY CMakeLists.txt ./CMakeLists.txt
COPY .githooks ./.githooks

# Compile Release. Tests stay in-tree but are not exercised here; the
# build-and-test step in CI runs ctest against an Integration build on a
# separate job with GPU access.
RUN --mount=type=cache,target=/root/.ccache \
    cmake -B build -S src/v2 -G Ninja \
        -DCMAKE_BUILD_TYPE=${LLAMINAR_BUILD_TYPE} \
        -DHAVE_CUDA=ON \
        -DHAVE_ROCM=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${LLAMINAR_CUDA_ARCHS}" \
    && cmake --build build --parallel --target llaminar2

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04 AS runtime

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=dev

LABEL org.opencontainers.image.title="Llaminar" \
      org.opencontainers.image.description="High-performance LLM inference engine (CUDA runtime)" \
      org.opencontainers.image.source="https://github.com/llaminar/llaminar" \
      org.opencontainers.image.licenses="See LICENSE in repository" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    LLAMINAR_MODELS_DIR=/models \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/llaminar/lib:/usr/lib/x86_64-linux-gnu

# Minimal runtime deps. stdc++ from gcc-14 runtime provides <print>/<format>.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libhwloc15 \
        libnuma1 \
        libopenblas0 \
        libopenmpi3t64 \
        libstdc++-14-dev \
        openmpi-bin \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# NCCL runtime (must match builder's major).
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        'libnccl2=*+cuda13.0' \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

# Copy compiled binary from builder stage.
COPY --from=builder /src/build/llaminar2 /opt/llaminar/bin/llaminar2

# Non-root runtime user.
RUN groupadd --system llaminar \
    && useradd --system --gid llaminar --home /home/llaminar --create-home --shell /bin/bash llaminar \
    && mkdir -p /models \
    && chown -R llaminar:llaminar /models /opt/llaminar

USER llaminar
WORKDIR /home/llaminar

ENV PATH="/opt/llaminar/bin:${PATH}"

EXPOSE 8080

ENTRYPOINT ["llaminar2"]
CMD ["--help"]
