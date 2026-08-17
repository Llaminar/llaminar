#!/usr/bin/env bash
# install-system-deps.sh
#
# Install the base system + C++23 build toolchain used by all Llaminar images.
#
# MODE=build    (default) — full compiler toolchain and -dev packages.
# MODE=runtime             — minimal libraries needed to run the compiled
#                            llaminar2 binary (slim release image).
#
# Shared by:
#   * .devcontainer/Dockerfile  (development image)
#   * Dockerfile (builder stage + runtime stage)
#
# The nvidia/cuda and rocm base images ship without Ubuntu's `universe` and
# `multiverse` repositories enabled, so some build packages are invisible
# without the explicit `add-apt-repository` calls below.
set -euo pipefail

MODE="${MODE:-build}"
NINJA_VERSION="${NINJA_VERSION:-1.13.0}"
export DEBIAN_FRONTEND=noninteractive

# Make apt resilient to transient network failures. Self-hosted runners and
# some corporate egress paths see occasional timeouts to archive.ubuntu.com,
# so retry each individual package fetch up to 5 times before giving up.
APT_OPTS=(
    -o Acquire::Retries=5
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
)

# --- Bootstrap: packages required to enable `universe` --------------------
apt-get "${APT_OPTS[@]}" update
apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
    --allow-change-held-packages \
    ca-certificates \
    gnupg \
    software-properties-common \
    curl \
    wget
add-apt-repository -y universe
add-apt-repository -y multiverse
apt-get "${APT_OPTS[@]}" update

if [[ "${MODE}" == "build" ]]; then
    # --- Full build toolchain --------------------------------------------
    apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
        --allow-change-held-packages \
        bc \
        build-essential \
        ccache \
        cmake \
        g++-14 \
        gcc-14 \
        gfortran-14 \
        gdb \
        git \
        gpg \
        gpg-agent \
        jq \
        libeigen3-dev \
        libgmock-dev \
        libgtest-dev \
        libhwloc-dev \
        libnuma-dev \
        libopenblas-dev \
        libopenmpi-dev \
        libscalapack-openmpi-dev \
        libssl-dev \
        make \
        ninja-build \
        openmpi-bin \
        openmpi-common \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv

    # Ubuntu 24.04 ships Ninja 1.11 while the Python/reference environment
    # installs the newer PyPI Ninja binary. Mixing those executables against
    # one build tree changes Ninja's recorded command hashes and forces a full
    # rebuild on every switch. Install one pinned binary as both the PATH and
    # /usr/bin authority so CMake caches and activated virtualenvs agree.
    python3 -m pip install \
        --break-system-packages \
        --no-cache-dir \
        "ninja==${NINJA_VERSION}"
    NINJA_PYPI_BIN="$(python3 -c \
        'import os, ninja; print(os.path.join(ninja.BIN_DIR, "ninja"))')"
    test -x "${NINJA_PYPI_BIN}"
    case "$("${NINJA_PYPI_BIN}" --version)" in
        "${NINJA_VERSION}"*) ;;
        *) echo "Pinned Ninja binary does not report ${NINJA_VERSION}" >&2; exit 1 ;;
    esac
    install -m 0755 "${NINJA_PYPI_BIN}" /usr/bin/ninja

    # Build & install GoogleTest headers+libs so CMake find_package(GTest) works.
    cmake -S /usr/src/googletest -B /tmp/googletest-build
    cmake --build /tmp/googletest-build --target install --parallel
    rm -rf /tmp/googletest-build

    # Default GCC = 14 for full C++23 support (std::print, std::println).
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-14 \
        --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-14 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-14
else
    # --- Slim runtime libraries ------------------------------------------
    # Only what the compiled binary needs at process start. No compilers, no
    # -dev packages, no GTest. Keeps the runtime image as small as possible.
    apt-get "${APT_OPTS[@]}" install -y --no-install-recommends \
        --allow-change-held-packages \
        libgomp1 \
        libhwloc15 \
        libnuma1 \
        libopenblas0 \
        libopenmpi3 \
        libscalapack-openmpi2.2 \
        libssl3 \
        libstdc++6 \
        openmpi-bin \
        python3
fi

rm -rf /var/lib/apt/lists/*
