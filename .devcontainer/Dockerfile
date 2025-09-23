# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gdb \
    cmake \
    make \
    ninja-build \
    pkg-config \
    bsdmainutils \
    strace \
    # MPI dependencies for hyper-scalability
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    # Linear algebra libraries for COSMA
    libopenblas-dev \
    libscalapack-openmpi-dev \
    # NUMA support
    libnuma-dev \
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
    # Python for scripting and tooling
    python3 \
    python3-pip \
    python3-dev \
    # Additional libraries that might be useful
    libssl-dev \
    libcurl4-openssl-dev \
    libjson-c-dev \
    libxml2-dev \
    libeigen3-dev \
    # Node.js for MCP servers (will be installed via features)
    && rm -rf /var/lib/apt/lists/*

# Install CUDA Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-0 cuda-drivers && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Install ROCm and AMD drivers
RUN wget https://repo.radeon.com/amdgpu-install/7.0.1/ubuntu/jammy/amdgpu-install_7.0.1.70001-1_all.deb && \
    apt install -y ./amdgpu-install_7.0.1.70001-1_all.deb && \
    apt-get update && \
    apt-get install -y python3-setuptools python3-wheel && \
    apt-get install -y rocm && \
    rm -f amdgpu-install_7.0.1.70001-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Install specific CMake version (latest stable)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    echo 'deb https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash vscode && \
    usermod -aG sudo vscode && \
    usermod -aG render,video vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

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

# Install Python packages that might be useful
RUN pip3 install --user \
    conan \
    numpy \
    matplotlib \
    pytest

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

# Set default command
CMD ["/bin/bash"]