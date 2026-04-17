# Llaminar development container.
#
# Shares its system-deps install logic with the release Dockerfile via
# scripts/docker/install-*.sh — edit those to change installed packages, not
# this file.
FROM ubuntu:24.04

ARG INSTALL_CUDA=false
ARG INSTALL_ROCM=false
ENV INSTALL_CUDA=${INSTALL_CUDA}
ENV INSTALL_ROCM=${INSTALL_ROCM}

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_PARALLEL_LEVEL=56

# CRITICAL: install sed/grep/coreutils FIRST — VS Code's UID-remap process
# and some later install steps need them available before anything else.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        coreutils \
        findutils \
        gpg \
        gpg-agent \
        grep \
        sed \
        sudo \
        wget \
    && rm -rf /var/lib/apt/lists/*

# --- Shared deps scripts ---------------------------------------------------
# Copy the shared installers and run the same ones the release image uses,
# so the dev environment tracks production 1:1.
COPY scripts/docker /tmp/install-scripts
RUN chmod +x /tmp/install-scripts/*.sh
RUN MODE=build /tmp/install-scripts/install-system-deps.sh

# Dev-only extras not in the release image: debuggers, editors, perf tools,
# graphics libs for GPU debug, linux-tools for `perf`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bc \
        bsdmainutils \
        htop \
        libcurl4-openssl-dev \
        libdrm-dev \
        libgbm-dev \
        libjson-c-dev \
        libxml2-dev \
        linux-tools-common \
        linux-tools-generic \
        lsb-release \
        nano \
        passwd \
        procps \
        python3.12-venv \
        strace \
        tree \
        unzip \
        util-linux \
        vim \
        xterm \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Fix perf symlink for the Ubuntu 24.04 linux-tools-$(uname -r) bug.
RUN KERNEL_VERSION=$(uname -r) && \
    INSTALLED_VERSION=$(ls -1 /usr/lib/linux-tools/ 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+' | head -n1) && \
    if [ -n "$INSTALLED_VERSION" ] && [ -n "$KERNEL_VERSION" ]; then \
        mkdir -p /usr/lib/linux-tools/$KERNEL_VERSION && \
        ln -sf /usr/lib/linux-tools/$INSTALLED_VERSION/perf /usr/lib/linux-tools/$KERNEL_VERSION/perf; \
    fi

# CUTLASS (header-only Tensor Core templates).
RUN /tmp/install-scripts/install-cutlass.sh
ENV CUTLASS_DIR=/opt/cutlass

# Conditional CUDA 13.
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
        MODE=full /tmp/install-scripts/install-cuda.sh; \
    else \
        echo "Skipping CUDA installation"; \
    fi

# Conditional ROCm 7.1.1 (gfx906 Tensile overlay handled inside script).
# Also clones our P2P-fix amdgpu DKMS fork — used to rebuild the host driver
# when PCIe-switch P2P is silently disabled on mainline kernels.
RUN if [ "$INSTALL_ROCM" = "true" ]; then \
        MODE=full /tmp/install-scripts/install-rocm.sh && \
        git clone --depth 1 --branch fix/pex-pcie-switch-p2p-enable \
            https://github.com/dbsanfte/amdgpu.git /opt/amdgpu-dkms; \
    else \
        echo "Skipping ROCm installation"; \
    fi

# Latest stable CMake from Kitware repo (newer than Ubuntu's default).
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
      | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    echo 'deb https://apt.kitware.com/ubuntu/ noble main' \
      | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*

# Done with install scripts.
RUN rm -rf /tmp/install-scripts

# --- Dev user --------------------------------------------------------------
RUN groupadd -f render && \
    groupadd -f video && \
    useradd -m -s /bin/bash vscode && \
    usermod -aG sudo vscode && \
    usermod -aG render,video vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /workspace && chown vscode:vscode /workspace

USER vscode
WORKDIR /workspace

# MPI, CUDA, ROCm env for vscode shells.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    CUDA_HOME=/usr/local/cuda \
    ROCM_HOME=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/home/vscode/.local/bin:/usr/local/cuda/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/rocm/lib

# vcpkg + uv for convenience in shells.
RUN git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg && \
    ~/vcpkg/bootstrap-vcpkg.sh && \
    echo 'export PATH="$HOME/vcpkg:$PATH"' >> ~/.bashrc
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias la="ls -la"' >> ~/.bashrc && \
    echo 'alias uvx="uv run --with"' >> ~/.bashrc

EXPOSE 8080 3000 5000
CMD ["sleep", "infinity"]
