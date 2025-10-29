# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

ARG INSTALL_CUDA=false
ARG INSTALL_ROCM=false
ENV INSTALL_CUDA=${INSTALL_CUDA}
ENV INSTALL_ROCM=${INSTALL_ROCM}

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# CRITICAL: Install sed, grep, and core utilities FIRST before anything else
# These are required by VS Code's UID update process
RUN apt-get update && apt-get install -y \
    sed \
    grep \
    coreutils \
    findutils \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities (sed, grep, coreutils, findutils already installed above)
    util-linux \
    procps \
    passwd \
    xterm \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
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
    # Linear algebra libraries for COSMA
    libopenblas-dev \
    libscalapack-openmpi-dev \
    # NUMA support
    libnuma-dev \
    # Google Test / gMock for unit tests
    libgtest-dev \
    libgmock-dev \
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
    && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    sudo ln -s /usr/lib/linux-tools-6.8.0-86/perf /usr/lib/linux-tools/6.14.0-33-generic/perf

# Build and install GoogleTest libraries (CMake find_package friendly)
RUN cmake -S /usr/src/googletest -B /tmp/googletest-build \
    && cmake --build /tmp/googletest-build --target install \
    && rm -rf /tmp/googletest-build

# Install CUDA 12.9 (conditional)
RUN if [ "$INSTALL_CUDA" = "true" ]; then \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-9 && \
    rm cuda-keyring.deb; \
    else \
    echo "Skipping CUDA installation"; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Install ROCm 6.4 (conditional)
RUN if [ "$INSTALL_ROCM" = "true" ]; then \
    mkdir -p --mode=0755 /etc/apt/keyrings && \
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4.2 noble main" \
    | tee /etc/apt/sources.list.d/rocm.list && \
    echo 'Package: *' \
    | tee /etc/apt/preferences.d/rocm-pin-600 && \
    echo 'Pin: release o=repo.radeon.com' \
    | tee -a /etc/apt/preferences.d/rocm-pin-600 && \
    echo 'Pin-Priority: 600' \
    | tee -a /etc/apt/preferences.d/rocm-pin-600 && \
    apt-get update && \
    apt-get install -y rocm && \
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