# 🚿 Llaminar
An LLM inferencing engine in C++, with custom quantised kernels for CPU AVX512-VNNI, CUDA `sm86`, and ROCm `gfx906`.

Llaminar tries to solve a variety of problems encountered in other projects:

* **Tensor and Pipeline Parallelism:** natively supported, mix and match heterogenous domains.
* **Multiple vendors:** Mix and match CPU, ROCm and CUDA, simultaneously and natively.
* **Easy scaling:** Built from the ground-up on OpenMPI with the goal of enabling scaling across clusters of machines. NUMA-aware.
* **IaC-like experience:** Plan, then deploy.

Llaminar is **experimental** and very much in an **alpha** stage of development. Use it with that in mind and expect the odd segfault.

## Supported Hardware

Llaminar supports:

* CPU inferencing (AVX512-VNNI is **required**, AVX2 coming soon)
* CUDA inferencing (RTX-3090 / `sm86` initial support for now)
* ROCm inferencing (`gfx906` only for now)
* All of the above simultaneously
* Tensor Parallel / Pipeline Parallel / MoE Expert Parallel (WiP)

## Supported Models

Llaminar supports the following model architectures initially:

* Qwen 2.5 (dense)
* Qwen 3 (dense)
* Qwen 3.5/3.6 (dense and MoE)

## Benchmarks

Latest benchmarks can be found here: 

https://github.com/Llaminar/llaminar/blob/develop/benchmark_results/e126900d/benchmark_results.csv

## Quickstart

### Building Llaminar

Llaminar uses a predefined devcontainer and the recommended development environment is vscode on a Linux machine with AVX512-VNNI, and access to gfx906 / sm86 hardware.

Open vscode in the devcontainer, and run the Build Integration / Build Release vscode tasks with `CTRL + Shift + P`.

### Running Llaminar

Set these once before running the one-liners below:

```bash
export MODEL_DIR=/opt/llaminar-models
export MODEL_DENSE="$MODEL_DIR/Qwen3.6-27B-Q4_K_S.gguf"
export MODEL_MOE="$MODEL_DIR/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
export MODEL_PP_DENSE="$MODEL_DIR/Qwen3.5-27B-Q4_K_M.gguf"
export LLAMINAR_CPU_IMAGE=ghcr.io/llaminar/llaminar:develop-cpu-latest
export LLAMINAR_CUDA_IMAGE=ghcr.io/llaminar/llaminar:develop-cuda13.0-latest
export LLAMINAR_ROCM_IMAGE=ghcr.io/llaminar/llaminar:develop-rocm7.1.1-latest
export LLAMINAR_FULL_IMAGE=ghcr.io/llaminar/llaminar:develop-latest
# docker run pulls these public GHCR images automatically when needed.
export AMD_KFD_GID="$(stat -c '%g' /dev/kfd 2>/dev/null || true)"
export AMD_RENDER_GID="$(stat -c '%g' "$(find /dev/dri -maxdepth 1 -name 'renderD*' 2>/dev/null | head -n1)" 2>/dev/null || true)"

COMMON_RUN=(--rm -it --network bridge --ulimit core=-1 --user 0:0 --security-opt seccomp=unconfined --cap-add SYS_NICE --cap-add SYS_PTRACE --shm-size=16g -v "$MODEL_DIR:$MODEL_DIR:ro")
CUDA_RUN=(--gpus all)
ROCM_RUN=(--device /dev/kfd --device /dev/dri --group-add "$AMD_KFD_GID" --group-add "$AMD_RENDER_GID")
PREFIX_FLAGS=(--prefix-cache --prefix-cache-storage ram --prefix-cache-ram-budget-mb 1024 --prefix-cache-terminal-state auto)
MOE_PREFIX_FLAGS=("${PREFIX_FLAGS[@]}" --prefix-cache-moe-policy placement-fingerprint)
MTP_FLAGS=(--mtp --mtp-draft-tokens 2 --mtp-depth-policy fixed --mtp-verify-mode greedy)
```

#### CPU Cross-socket TP/EP

```bash
docker run "${COMMON_RUN[@]}" -p 8080:8080 "$LLAMINAR_CPU_IMAGE" serve --host 0.0.0.0 --port 8080 -d cpu "${MOE_PREFIX_FLAGS[@]}" -m "$MODEL_MOE"
```

#### CUDA SingleDevice

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" -p 8080:8080 "$LLAMINAR_CUDA_IMAGE" serve --host 0.0.0.0 --port 8080 -d cuda:0 "${PREFIX_FLAGS[@]}" "${MTP_FLAGS[@]}" -m "$MODEL_DENSE"
```

#### CUDA TensorParallel tp=2

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" -p 8080:8080 "$LLAMINAR_CUDA_IMAGE" serve --host 0.0.0.0 --port 8080 --tp-devices cuda:0,cuda:1 "${MOE_PREFIX_FLAGS[@]}" -m "$MODEL_MOE"
```

#### ROCm SingleDevice

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO -p 8080:8080 "$LLAMINAR_ROCM_IMAGE" serve --host 0.0.0.0 --port 8080 -d rocm:0 "${PREFIX_FLAGS[@]}" "${MTP_FLAGS[@]}" -m "$MODEL_DENSE"
```

#### ROCm TensorParallel tp=2

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO -p 8080:8080 "$LLAMINAR_ROCM_IMAGE" serve --host 0.0.0.0 --port 8080 --tp-devices rocm:0,rocm:1 "${MOE_PREFIX_FLAGS[@]}" -m "$MODEL_MOE"
```

#### ROCm TensorParallel tp=4

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO -p 8080:8080 "$LLAMINAR_ROCM_IMAGE" serve --host 0.0.0.0 --port 8080 --tp-devices rocm:0,rocm:1,rocm:2,rocm:3 "${MOE_PREFIX_FLAGS[@]}" -m "$MODEL_MOE"
```

#### CUDA+ROCm Pipeline Parallel

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" "${ROCM_RUN[@]}" -p 8080:8080 "$LLAMINAR_FULL_IMAGE" serve --host 0.0.0.0 --port 8080 --define-domain cuda_pp=cuda:0 --define-domain rocm_pp=rocm:0 --pp-stage 0=cuda_pp:0-31 --pp-stage 1=rocm_pp:32-63 -m "$MODEL_PP_DENSE"
```

#### CUDA+ROCm Host-staged Tensor Parallel tp=2

This uses the E2E model files and the validated host-staged cross-vendor TP
shape. The release-container server E2E matrix does not currently include this
exact Quickstart header as a full server case.

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" "${ROCM_RUN[@]}" -p 8080:8080 "$LLAMINAR_FULL_IMAGE" serve --host 0.0.0.0 --port 8080 --tp-devices cuda:0,rocm:0 --backend host "${MOE_PREFIX_FLAGS[@]}" -m "$MODEL_MOE"
```

## Llaminar Architecture

Llaminar V2 is a kernel-centric inference runtime for CPU, CUDA, ROCm, and
mixed-vendor deployments. Model-specific graph builders declare the exact compute
stages needed for a forward pass, and the runtime binds those stages to devices,
buffers, NUMA nodes, MPI ranks, and collective backends.

The high-level pipeline is:

```text
CLI/YAML
  -> OrchestrationConfig
  -> RankExecutionPlan
  -> GraphConfig
  -> DeviceGraphOrchestrator / RankOrchestrator
  -> ComputeGraph
  -> DeviceGraphExecutor
```

`OrchestrationConfig` is the user-facing plan: devices, tensor parallelism,
pipeline stages, backend preference, batch shape, sequence limits, and model
path. `ExecutionPlanBuilder` turns that into a per-rank `RankExecutionPlan`
with parsed runtime values, local device assignments, pipeline ranges, shard
ownership, and the NUMA node for that rank. From there, model-specific config
builders create a `GraphConfig`, and the runtime chooses either a
`DeviceGraphOrchestrator` for one device or a `RankOrchestrator` for local
multi-device tensor or pipeline parallelism.

### MPI, NUMA, and CPU Scaling

OpenMPI and libnuma are hard runtime dependencies. Llaminar treats CPU sockets
as first-class execution domains, not as a flat pile of cores. Each MPI rank is
planned with an explicit host, rank id, socket/NUMA assignment, and shard
contract. The launcher bootstraps MPI, configures OpenMP placement, pins work
to sockets, and uses NUMA-aware allocation so CPU weights and activation pages
live where the kernels that consume them run.

Cross-socket work uses the same distributed execution model as cross-rank work:
local compute stages produce partial results, then collective stages reconcile
them. CPU tensor parallelism uses MPI collectives such as `MPI_Allreduce`,
`MPI_Allgather`, and variable-count gather stages to combine row-parallel
projections, logits shards, or pipeline handoffs. NUMA binding is considered a
correctness and performance contract. If model-page binding is requested and
cannot be applied or verified, Llaminar fails instead of silently accepting
remote-memory execution.

### Graphs and Compute Stages

Model execution is represented as a declarative `ComputeGraph`: a DAG of
`IComputeStage` nodes with named inputs, outputs, dependencies, and device
placement. Stages are the unit of real work. Examples include embedding, RMS
norm, fused QKV projection, RoPE, KV-cache append, attention, SwiGLU, residual
add, LM head projection, all-reduce, all-gather, and pipeline send/receive.

The graph system is model-agnostic. `GraphBuilderRegistry` maps an architecture
name such as `qwen2` or `qwen3` to an `IGraphBuilder`, while
`SchemaFactoryRegistry` provides the weight sharding and stage schema for that
architecture. Adding a model means registering a schema and graph builder; the
orchestration, MPI, memory, and collective layers remain shared.

`DeviceGraphExecutor` runs the graph through a common stage loop controlled by
`StageRunPolicy`. That loop handles buffer coherence, device uploads, output
ownership, validation, profiling, snapshots, and collective interception. This
keeps prefill, decode, parity testing, and fast cached decode on the same
execution semantics, with different policy knobs rather than separate
hand-written pipelines.

### Collectives Across CPU, CUDA, and ROCm

Graphs declare collectives abstractly. A model graph says "all-reduce this
buffer" or "all-gather these logits"; it does not hardcode MPI, NCCL, RCCL, or
host staging. At execution time, `CollectiveContext` and `BackendRouter` inspect
the participating devices and select the concrete backend.

- CPU and cross-node collectives use OpenMPI.
- Same-vendor CUDA groups use NCCL.
- Same-vendor ROCm groups use RCCL.
- Heterogeneous CPU/GPU or CUDA/ROCm paths use the available host or direct
  transfer path selected by the router.

This is what lets a single Llaminar process run CPU-only, CUDA-only,
ROCm-only, or mixed CUDA+ROCm inference. Tensor-parallel domains can be local
to a rank, spread across ranks, or composed with pipeline-parallel stages. The
graph sees the same logical collective stages in each case; only the runtime
routing changes.

### GPU Graph Capture

For inference, Llaminar optimizes the hot decode path with CUDA and HIP graph
capture where the backend and stage sequence support it. Prefill and decode
build normal `ComputeGraph` objects first. Stable GPU segments can then be
warmed, captured, cached, and replayed so later tokens avoid repeated kernel
launch overhead. Dynamic state such as token positions, KV-cache counters,
router decisions, logits buffers, and MTP verifier state is explicitly staged
for capture-safe replay rather than being read from stale host pointers.

Collectives are handled carefully around graph capture. NCCL and RCCL
collectives may run through segmented graph execution when the policy allows
it; otherwise the executor leaves non-capturable stages outside the captured
segments and runs them manually in order. Unsupported capture configurations
fall back to the normal fast-decode path with diagnostics instead of producing
silently incorrect replay.

The result is one execution model that scales down to a single CPU socket and
up to heterogeneous multi-GPU, multi-socket, and multi-rank deployments while
keeping placement, collectives, and graph replay explicit.

## Running Llaminar

### Ubuntu 24.04 Mixed-GPU Host

The release container is built for machines that may use NVIDIA CUDA and AMD
ROCm in the same process. It ships the Llaminar binary plus CUDA 13.0
user-space libraries, NCCL for CUDA 13.0, and ROCm 7.1.1 user-space libraries.
It does not ship kernel drivers.

On the host you need:

- AVX512-VNNI support (AVX2 support is coming soon)
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

5. Pull the public GHCR runtime images:

```bash
export LLAMINAR_CPU_IMAGE=ghcr.io/llaminar/llaminar:develop-cpu-latest
export LLAMINAR_CUDA_IMAGE=ghcr.io/llaminar/llaminar:develop-cuda13.0-latest
export LLAMINAR_ROCM_IMAGE=ghcr.io/llaminar/llaminar:develop-rocm7.1.1-latest
export LLAMINAR_FULL_IMAGE=ghcr.io/llaminar/llaminar:develop-latest

docker pull "$LLAMINAR_CPU_IMAGE"
docker pull "$LLAMINAR_CUDA_IMAGE"
docker pull "$LLAMINAR_ROCM_IMAGE"
docker pull "$LLAMINAR_FULL_IMAGE"
```

Docker also pulls these public images automatically on first `docker run`. Use
the CPU, CUDA, or ROCm image when the target machine only needs one backend; use
the full image for mixed CUDA+ROCm runs.

To build images locally instead of pulling GHCR, use the release image build
script:

```bash
scripts/docker/build-runtime-image.sh --tag llaminar:local --cuda-archs "80;86;89;90"
```

Use the semicolon-separated CUDA architecture list for the NVIDIA GPUs in your
local build. Common values are `80` for A100, `86` for RTX 30/A10, `89` for RTX
40/L4/L40, and `90` for H100/H200.

Backend-specific local builds are also available:

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
  "$LLAMINAR_FULL_IMAGE" --help

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
  "$LLAMINAR_FULL_IMAGE"
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

The image entrypoint is `llaminar2`, so the command after the image name is
`benchmark`, `serve`, or another Llaminar subcommand. The examples below use the
same model paths and Docker runtime settings as the release-container E2E
harness:

- Docker bridge networking with explicit port publishing for `serve`.
- Private `/dev/shm` sized to `16g` for OpenMPI, NCCL, and RCCL.
- `seccomp=unconfined` for strict NUMA binding and verification.
- `SYS_NICE` for MPI/NUMA placement and `SYS_PTRACE` for ROCm hosts.
- Root inside the container, matching the release E2E runs.

Set `MODEL_DIR` to the host directory containing the GGUF files. The E2E runner
uses `/opt/llaminar-models`, and the examples keep that same path inside the
container:

```bash
export MODEL_DIR=/opt/llaminar-models

export MODEL_SMALL="$MODEL_DIR/qwen2.5-1.5b-instruct-q8_0.gguf"
export MODEL_CPU_DENSE="$MODEL_DIR/Qwen3.6-27B-Q4_K_S.gguf"
export MODEL_PP_DENSE="$MODEL_DIR/Qwen3.5-27B-Q4_K_M.gguf"
export MODEL_TP_MOE="$MODEL_DIR/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
```

Use the public GHCR develop release aliases:

```bash
export LLAMINAR_CPU_IMAGE=ghcr.io/llaminar/llaminar:develop-cpu-latest
export LLAMINAR_CUDA_IMAGE=ghcr.io/llaminar/llaminar:develop-cuda13.0-latest
export LLAMINAR_ROCM_IMAGE=ghcr.io/llaminar/llaminar:develop-rocm7.1.1-latest
export LLAMINAR_FULL_IMAGE=ghcr.io/llaminar/llaminar:develop-latest
```

For local builds, override these variables with tags such as `llaminar:cpu`,
`llaminar:cuda`, `llaminar:rocm`, or `llaminar:local`.

For compact copy/paste examples, define the common Docker arguments once:

```bash
COMMON_RUN=(
  --rm -it
  --network bridge
  --ulimit core=-1
  --user 0:0
  --security-opt seccomp=unconfined
  --cap-add SYS_NICE
  --cap-add SYS_PTRACE
  --shm-size=16g
  -v "$MODEL_DIR:$MODEL_DIR:ro"
)

CUDA_RUN=(--gpus all)
```

#### CPU-only image

Single CPU socket, using `cpu:0`:

```bash
docker run "${COMMON_RUN[@]}" \
  "$LLAMINAR_CPU_IMAGE" \
  benchmark -d cpu:0 -m "$MODEL_CPU_DENSE"

docker run "${COMMON_RUN[@]}" -p 8080:8080 \
  "$LLAMINAR_CPU_IMAGE" \
  serve --host 0.0.0.0 --port 8080 -d cpu:0 -m "$MODEL_SMALL"
```

All CPU sockets, using `-d cpu` for node-local tensor parallel CPU execution:

```bash
docker run "${COMMON_RUN[@]}" \
  "$LLAMINAR_CPU_IMAGE" \
  benchmark -d cpu -m "$MODEL_CPU_DENSE"

docker run "${COMMON_RUN[@]}" -p 8080:8080 \
  "$LLAMINAR_CPU_IMAGE" \
  serve --host 0.0.0.0 --port 8080 -d cpu -m "$MODEL_SMALL"
```

#### CUDA image

Single CUDA device:

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" \
  "$LLAMINAR_CUDA_IMAGE" \
  benchmark -d cuda:0 -m "$MODEL_SMALL"

docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" -p 8080:8080 \
  "$LLAMINAR_CUDA_IMAGE" \
  serve --host 0.0.0.0 --port 8080 -d cuda:0 -m "$MODEL_SMALL"
```

Pipeline parallel across two CUDA devices. This uses the same 64-layer
`Qwen3.5-27B-Q4_K_M.gguf` split that is tested in the CUDA+ROCm pipeline E2E
case, with both stages placed on CUDA devices:

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" \
  "$LLAMINAR_CUDA_IMAGE" \
  benchmark \
  --define-domain cuda_pp0=cuda:0 \
  --define-domain cuda_pp1=cuda:1 \
  --pp-stage 0=cuda_pp0:0-31 \
  --pp-stage 1=cuda_pp1:32-63 \
  -m "$MODEL_PP_DENSE"

docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" -p 8080:8080 \
  "$LLAMINAR_CUDA_IMAGE" \
  serve --host 0.0.0.0 --port 8080 \
  --define-domain cuda_pp0=cuda:0 \
  --define-domain cuda_pp1=cuda:1 \
  --pp-stage 0=cuda_pp0:0-31 \
  --pp-stage 1=cuda_pp1:32-63 \
  -m "$MODEL_PP_DENSE"
```

Tensor parallel across two CUDA devices. The two entries in `--tp-devices`
select TP=2. This model and TP2 CUDA shape are in the release E2E matrix:

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" \
  "$LLAMINAR_CUDA_IMAGE" \
  benchmark --tp-devices cuda:0,cuda:1 -m "$MODEL_TP_MOE"

docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" -p 8080:8080 \
  "$LLAMINAR_CUDA_IMAGE" \
  serve --host 0.0.0.0 --port 8080 \
  --tp-devices cuda:0,cuda:1 \
  -m "$MODEL_TP_MOE"
```

Tensor parallel across four CUDA devices. The four entries in `--tp-devices`
select TP=4, using the same tested MoE model and TP command shape extended to
four CUDA devices:

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" \
  "$LLAMINAR_CUDA_IMAGE" \
  benchmark --tp-devices cuda:0,cuda:1,cuda:2,cuda:3 -m "$MODEL_TP_MOE"

docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" -p 8080:8080 \
  "$LLAMINAR_CUDA_IMAGE" \
  serve --host 0.0.0.0 --port 8080 \
  --tp-devices cuda:0,cuda:1,cuda:2,cuda:3 \
  -m "$MODEL_TP_MOE"
```

#### ROCm image

Define the ROCm device arguments on hosts with AMD GPUs:

```bash
export AMD_KFD_GID="$(stat -c '%g' /dev/kfd)"
export AMD_RENDER_GID="$(stat -c '%g' "$(find /dev/dri -maxdepth 1 -name 'renderD*' | head -n1)")"
ROCM_RUN=(
  --device /dev/kfd
  --device /dev/dri
  --group-add "$AMD_KFD_GID"
  --group-add "$AMD_RENDER_GID"
)
```

Single ROCm device. The ROCm E2E run also sets `NCCL_DEBUG=INFO` and
`RCCL_LOG_LEVEL=INFO`; they are included here for the same diagnostics:

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  benchmark -d rocm:0 -m "$MODEL_SMALL"

docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" -p 8080:8080 \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  serve --host 0.0.0.0 --port 8080 -d rocm:0 -m "$MODEL_SMALL"
```

Pipeline parallel across two ROCm devices. This uses the same 64-layer
`Qwen3.5-27B-Q4_K_M.gguf` split that is tested in the CUDA+ROCm pipeline E2E
case, with both stages placed on ROCm devices:

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  benchmark \
  --define-domain rocm_pp0=rocm:0 \
  --define-domain rocm_pp1=rocm:1 \
  --pp-stage 0=rocm_pp0:0-31 \
  --pp-stage 1=rocm_pp1:32-63 \
  -m "$MODEL_PP_DENSE"

docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" -p 8080:8080 \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  serve --host 0.0.0.0 --port 8080 \
  --define-domain rocm_pp0=rocm:0 \
  --define-domain rocm_pp1=rocm:1 \
  --pp-stage 0=rocm_pp0:0-31 \
  --pp-stage 1=rocm_pp1:32-63 \
  -m "$MODEL_PP_DENSE"
```

Tensor parallel across two ROCm devices. The two entries in `--tp-devices`
select TP=2. This model and TP2 ROCm shape are in the release E2E matrix:

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  benchmark --tp-devices rocm:0,rocm:1 -m "$MODEL_TP_MOE"

docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" -p 8080:8080 \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  serve --host 0.0.0.0 --port 8080 \
  --tp-devices rocm:0,rocm:1 \
  -m "$MODEL_TP_MOE"
```

Tensor parallel across four ROCm devices. The four entries in `--tp-devices`
select TP=4. This model and TP4 ROCm shape are in the release E2E matrix:

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  benchmark --tp-devices rocm:0,rocm:1,rocm:2,rocm:3 -m "$MODEL_TP_MOE"

docker run "${COMMON_RUN[@]}" "${ROCM_RUN[@]}" -p 8080:8080 \
  -e NCCL_DEBUG=INFO -e RCCL_LOG_LEVEL=INFO \
  "$LLAMINAR_ROCM_IMAGE" \
  serve --host 0.0.0.0 --port 8080 \
  --tp-devices rocm:0,rocm:1,rocm:2,rocm:3 \
  -m "$MODEL_TP_MOE"
```

#### CUDA+ROCm image

Pipeline parallel across one CUDA GPU and one ROCm GPU. Define `ROCM_RUN` as in
the ROCm section above first. This is the exact hybrid release E2E topology:
`Qwen3.5-27B-Q4_K_M.gguf`, layers `0-31` on `cuda:0`, and layers `32-63` on
`rocm:0`.

```bash
docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" "${ROCM_RUN[@]}" \
  "$LLAMINAR_FULL_IMAGE" \
  benchmark \
  --define-domain cuda_pp=cuda:0 \
  --define-domain rocm_pp=rocm:0 \
  --pp-stage 0=cuda_pp:0-31 \
  --pp-stage 1=rocm_pp:32-63 \
  -m "$MODEL_PP_DENSE"

docker run "${COMMON_RUN[@]}" "${CUDA_RUN[@]}" "${ROCM_RUN[@]}" -p 8080:8080 \
  "$LLAMINAR_FULL_IMAGE" \
  serve --host 0.0.0.0 --port 8080 \
  --define-domain cuda_pp=cuda:0 \
  --define-domain rocm_pp=rocm:0 \
  --pp-stage 0=cuda_pp:0-31 \
  --pp-stage 1=rocm_pp:32-63 \
  -m "$MODEL_PP_DENSE"
```

The release E2E matrix currently exercises CUDA TP2 and ROCm TP2/TP4 directly.
The homogeneous CUDA/ROCm PP2 examples above reuse the same tested PP model,
layer split, and domain syntax as the hybrid CUDA+ROCm PP case.

Reference docs:
- NVIDIA CUDA release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- Docker Engine install guide: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- AMD ROCm Docker container guide: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

## The Llaminar Philosophy

* Tensors want to be open and free: so is Llaminar.
* Tensors want to be sliced, sharded, and pipelined: Llaminar lets them be.
* Tensors want to run on a variety of hardware types without artificial handicaps: Llaminar helps them to do so.
