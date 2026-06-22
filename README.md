# 🚿 Llaminar
An LLM inferencing engine in C++.

Llaminar tries to solve a variety of problems encountered in other projects:

* **Tensor and Pipeline Parallelism:** natively supported, mix and match heterogenous domains.
* **Multiple vendors:** Mix and match CPU, ROCm and CUDA, simultaneously and natively.
* **Easy scaling:** Built from the ground-up on OpenMPI so you can scale out across a cluster. NUMA-aware. 
* **IaC-like experience:** Plan, then deploy.

## Installation

### Ubuntu 24.04 Mixed-GPU Host

The release container is built for machines that may use NVIDIA CUDA and AMD
ROCm in the same process. It ships the Llaminar binary plus CUDA 13.0
user-space libraries, NCCL for CUDA 13.0, and ROCm 7.1.1 user-space libraries.
It does not ship kernel drivers.

On the host you need:

- Ubuntu 24.04 on x86_64.
- Docker Engine with the Buildx plugin.
- NVIDIA Linux driver `580.95.05` or newer for CUDA 13.0 Update 2.
- NVIDIA Container Toolkit configured for Docker.
- AMDGPU DKMS kernel driver from the ROCm 7.1.1 stack.

OpenMPI and libnuma are hard Llaminar dependencies. The Docker images include
them; source builds should install `openmpi-bin`, `libopenmpi-dev`, and
`libnuma-dev`.

You do not need to install the full CUDA Toolkit or the full ROCm user-space
stack on the host. Those user-space libraries are in the image. Pick the image
variant that matches the backends you want to expose: CPU-only images do not
need GPU devices, CUDA images need NVIDIA Container Toolkit, ROCm images need
the AMDGPU kernel driver and `/dev/kfd` plus `/dev/dri`, and the combined image
needs both ecosystems.

1. Install Docker Engine:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
```

Log out and back in after adding your user to the `docker` group, or keep using
`sudo docker` until the group membership is active.

2. Install an NVIDIA driver new enough for CUDA 13.0:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
curl -fsSL -o /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-drivers
sudo reboot
```

After reboot, confirm the installed driver is `580.95.05` or newer:

```bash
nvidia-smi
```

3. Install and configure NVIDIA Container Toolkit for Docker:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify Docker can inject the NVIDIA driver libraries:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

4. Install the AMDGPU DKMS driver for ROCm containers:

```bash
sudo apt-get update
sudo apt-get install -y "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
curl -fsSL -o /tmp/amdgpu-install.deb \
  https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb
sudo apt-get install -y /tmp/amdgpu-install.deb
sudo amdgpu-install --usecase=dkms -y
sudo usermod -aG render,video "$USER"
sudo reboot
```

After reboot, confirm the AMD device nodes exist:

```bash
ls -l /dev/kfd /dev/dri/render*
```

5. Build the local release image:

```bash
scripts/docker/build-runtime-image.sh --tag llaminar:local --cuda-archs "80;86;89;90"
```

Use the semicolon-separated CUDA architecture list for the NVIDIA GPUs you plan
to run. Common values are `80` for A100, `86` for RTX 30/A10, `89` for RTX
40/L4/L40, and `90` for H100/H200.

Build smaller backend-specific images when the target machine only needs one
GPU ecosystem:

```bash
scripts/docker/build-runtime-image.sh --variant cpu  --tag llaminar:cpu
scripts/docker/build-runtime-image.sh --variant cuda --tag llaminar:cuda --cuda-archs "80;86;89;90"
scripts/docker/build-runtime-image.sh --variant rocm --tag llaminar:rocm
```

6. Verify the Llaminar image can use both GPU ecosystems:

```bash
export AMD_KFD_GID="$(stat -c '%g' /dev/kfd)"
export AMD_RENDER_GID="$(stat -c '%g' "$(find /dev/dri -maxdepth 1 -name 'renderD*' | head -n1)")"

docker run --rm --gpus all \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --cap-add SYS_PTRACE \
  llaminar:local --help

docker run --rm \
  --gpus all \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add "$AMD_KFD_GID" \
  --group-add "$AMD_RENDER_GID" \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --cap-add SYS_PTRACE \
  --entrypoint rocminfo \
  llaminar:local
```

Llaminar does not require `--privileged` for normal container runs. It does
require a few targeted Docker permissions:

- `--shm-size=16g` gives OpenMPI, NCCL, and RCCL enough `/dev/shm` for
  tensor-parallel collectives. Avoid `--ipc=host` unless the host `/dev/shm`
  is known to be large enough; Docker's `--shm-size` does not resize host IPC.
- `--security-opt seccomp=unconfined` allows Linux NUMA policy syscalls
  (`mbind`, `set_mempolicy`, `get_mempolicy`, and `move_pages`) so CPU
  execution can bind and verify model pages on the intended NUMA node.
- `--cap-add SYS_NICE` allows the MPI/NUMA runtime to apply placement and
  scheduling policy without Docker capability denials.
- `--cap-add SYS_PTRACE` is required on common ROCm Docker hosts for AMD GPU
  runtime/debug interfaces used through `/dev/kfd`.

When CPU model-page NUMA binding is requested, Llaminar fails model loading by
default if binding cannot be applied. Set `LLAMINAR_ALLOW_NUMA_BIND_FALLBACK=1`
only when you explicitly accept degraded CPU NUMA placement.

### Running Llaminar

Set `MODEL_DIR` to a directory containing GGUF models:

```bash
export MODEL_DIR=/path/to/models
```

CUDA:

```bash
docker run --rm -it \
  --gpus all \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --cap-add SYS_PTRACE \
  --shm-size=16g \
  -v "$MODEL_DIR":/models:ro \
  -p 8080:8080 \
  llaminar:local \
  serve --port 8080 -d cuda:0 \
  -m /models/qwen2.5-1.5b-instruct-q8_0.gguf
```

ROCm on a mixed NVIDIA + AMD host:

```bash
export AMD_KFD_GID="$(stat -c '%g' /dev/kfd)"
export AMD_RENDER_GID="$(stat -c '%g' "$(find /dev/dri -maxdepth 1 -name 'renderD*' | head -n1)")"

docker run --rm -it \
  --gpus all \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add "$AMD_KFD_GID" \
  --group-add "$AMD_RENDER_GID" \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --cap-add SYS_PTRACE \
  --shm-size=16g \
  -v "$MODEL_DIR":/models:ro \
  -p 8080:8080 \
  llaminar:local \
  serve --port 8080 -d rocm:0 \
  -m /models/qwen2.5-1.5b-instruct-q8_0.gguf
```

CPU from the same release image:

```bash
docker run --rm -it \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --shm-size=16g \
  -v "$MODEL_DIR":/models:ro \
  -p 8080:8080 \
  llaminar:local \
  serve --port 8080 -d cpu \
  -m /models/qwen2.5-1.5b-instruct-q8_0.gguf
```

Reference docs:
- NVIDIA CUDA release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- Docker Engine install guide: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- AMD ROCm Docker container guide: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

## The Llaminar Philosophy

* Tensors want to be open and free: so is Llaminar.
* Tensors want to be sliced, sharded, and pipelined: Llaminar lets them be.
* Tensors want to run on a variety of hardware types without artificial handicaps: Llaminar helps them to do so.
