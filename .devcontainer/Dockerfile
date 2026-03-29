# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

ARG INSTALL_CUDA=false
ARG INSTALL_ROCM=false
ENV INSTALL_CUDA=${INSTALL_CUDA}
ENV INSTALL_ROCM=${INSTALL_ROCM}

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# CMake default parallelism
ENV CMAKE_BUILD_PARALLEL_LEVEL=56

# CRITICAL: Install sed, grep, and core utilities FIRST before anything else
# These are required by VS Code's UID update process and Intel oneAPI repo setup
RUN apt-get update && apt-get install -y \
    wget \
    sudo \
    sed \
    grep \
    coreutils \
    findutils \
    gpg \
    gpg-agent \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities (sed, grep, coreutils, findutils already installed above)
    util-linux \
    procps \
    passwd \
    xterm \
    # Build essentials (GCC 14 for full C++23 support including <print>)
    build-essential \
    gcc-14 \
    g++-14 \
    gfortran-14 \
    gdb \
    cmake \
    make \
    ninja-build \
    pkg-config \
    bsdmainutils \
    strace \
    ccache \
    # MPI dependencies for hyper-scalability
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    # Linear algebra libraries
    libopenblas-dev \
    libscalapack-openmpi-dev \
    # NUMA support
    libnuma-dev \
    libhwloc-dev \
    # Google Test / gMock for unit tests
    libgtest-dev \
    libgmock-dev \
    # Graphics libraries (for GPU compute / transfer)
    libgbm-dev \
    libdrm-dev \
    # Additional tools
    git \
    curl \
    wget \
    sudo \
    vim \
    nano \
    tree \
    htop \
    unzip \
    zip \
    ca-certificates \
    gnupg \
    lsb-release \
    linux-tools-generic \
    linux-tools-common \ 
    linux-tools-$(uname -r) \
    # Python for scripting and tooling
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3.12-venv \
    # Additional libraries that might be useful
    libssl-dev \
    libcurl4-openssl-dev \
    libjson-c-dev \
    libxml2-dev \
    libeigen3-dev \
    # Node.js for MCP servers (will be installed via features)
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set GCC 14 as default compiler (for full C++23 support including std::print)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-14 \
        --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-14 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-14

# Fix perf symlink for Ubuntu 24.04 bug
# Detect installed linux-tools version and create proper symlink
RUN KERNEL_VERSION=$(uname -r) && \
    INSTALLED_VERSION=$(ls -1 /usr/lib/linux-tools/ | grep -E '^[0-9]+\.[0-9]+\.[0-9]+-[0-9]+' | head -n1) && \
    if [ -n "$INSTALLED_VERSION" ] && [ -n "$KERNEL_VERSION" ]; then \
        mkdir -p /usr/lib/linux-tools/$KERNEL_VERSION && \
        ln -sf /usr/lib/linux-tools/$INSTALLED_VERSION/perf /usr/lib/linux-tools/$KERNEL_VERSION/perf; \
        echo "Created perf symlink: $INSTALLED_VERSION -> $KERNEL_VERSION"; \
    fi

# Build and install GoogleTest libraries (CMake find_package friendly)
RUN cmake -S /usr/src/googletest -B /tmp/googletest-build \
    && cmake --build /tmp/googletest-build --target install \
    && rm -rf /tmp/googletest-build

# Install NVIDIA CUTLASS (Tensor Core library) - header-only, no build needed
# CUTLASS 4.2.1 supports CUDA 12.x and provides modern Tensor Core APIs
RUN git clone --branch v4.2.1 --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass 

# Set CUTLASS environment variable for CMake
ENV CUTLASS_DIR=/opt/cutlass

# Install CUDA 13.0 toolkit (conditional)
# NOTE: Do NOT install nvidia-driver-* inside container - the host driver is
# injected at runtime via --gpus=all and nvidia-container-toolkit
# Pin libnccl2/libnccl-dev to +cuda13.0 variant to avoid NCCL/CUDA version
# mismatch (older NCCL built for cuda12.0 reports "stub library" errors)
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-13-0 'libnccl2=*+cuda13.0' 'libnccl-dev=*+cuda13.0' && \
    rm cuda-keyring.deb; \
    else \
    echo "Skipping CUDA installation"; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Install ROCm 7.1.1 (conditional)
# ROCm 7.1.1 dropped gfx906 (MI50) support, so we copy Tensile files from Arch Linux package
RUN if [ "$INSTALL_ROCM" = "true" ]; then \
    # Download and install amdgpu-install tool for ROCm 7.1.1
    wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb -O /tmp/amdgpu-install.deb && \
    apt-get update && \
    apt-get install -y /tmp/amdgpu-install.deb && \
    # Install ROCm 7.1.1 (no DKMS - kernel driver comes from host)
    amdgpu-install --usecase=rocm --no-dkms -y && \
    # Download rocBLAS from Arch Linux (has gfx906 Tensile files)
    apt-get install -y zstd && \
    wget -O /tmp/rocblas-arch.pkg.tar.zst "https://archlinux.org/packages/extra/x86_64/rocblas/download" && \
    cd /tmp && tar -I zstd -xf rocblas-arch.pkg.tar.zst && \
    # Copy gfx906 Tensile files to ROCm installation
    cp /tmp/opt/rocm/lib/rocblas/library/*gfx906* /opt/rocm/lib/rocblas/library/ && \
    # Clone our amdgpu DKMS fork with P2P fix for PCIe switches on mainline kernels.
    # The upstream DKMS gates CONFIG_HSA_AMD_P2P on CONFIG_DMABUF_MOVENOTIFY which
    # doesn't exist in mainline kernels >= 5.12, silently breaking P2P.
    # To rebuild host DKMS with the fix:
    #   sudo dkms remove amdgpu/6.12.12-2187269.24.04 --all
    #   sudo cp /opt/amdgpu-dkms/drivers/gpu/drm/amd/dkms/dkms-config.sh \
    #        /usr/src/amdgpu-*/amd/dkms/dkms-config.sh
    #   sudo dkms install amdgpu/6.12.12-2187269.24.04
    git clone --depth 1 --branch fix/pex-pcie-switch-p2p-enable \
        https://github.com/dbsanfte/amdgpu.git /opt/amdgpu-dkms && \
    # Cleanup
    rm -rf /tmp/amdgpu-install.deb /tmp/rocblas-arch.pkg.tar.zst /tmp/opt /tmp/usr && \
    apt-get autoremove -y && \
    apt-get clean; \
    else \
    echo "Skipping ROCm installation"; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Install specific CMake version (latest stable)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    echo 'deb https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -f render && \
    groupadd -f video && \
    useradd -m -s /bin/bash vscode && \
    usermod -aG sudo vscode && \
    usermod -aG render,video vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Verify user was created successfully
RUN id vscode && grep vscode /etc/passwd

# Set up workspace directory
RUN mkdir -p /workspace && \
    chown vscode:vscode /workspace

# Switch to non-root user
USER vscode

# Set working directory
WORKDIR /workspace

# Set environment variables for MPI
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Set ROCm environment variables
ENV ROCM_HOME=/opt/rocm
ENV PATH="${ROCM_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_HOME}/lib:${LD_LIBRARY_PATH}"
ENV HIP_PATH="${ROCM_HOME}"

# Set PATH to include user local bin directory
ENV PATH="/home/vscode/.local/bin:$PATH"

# Note: Python packages will be installed in a virtual environment
# See .devcontainer/setup-python-env.sh which is run via postCreateCommand

# Install vcpkg (C++ package manager) properly
RUN git clone https://github.com/Microsoft/vcpkg.git ~/vcpkg && \
    ~/vcpkg/bootstrap-vcpkg.sh && \
    echo 'export PATH="$HOME/vcpkg:$PATH"' >> ~/.bashrc

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up shell environment
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias la="ls -la"' >> ~/.bashrc && \
    echo 'alias uvx="uv run --with"' >> ~/.bashrc

# Expose commonly used ports
EXPOSE 8080 3000 5000

# Keep container running (Dev Containers will override this)
CMD ["sleep", "infinity"]