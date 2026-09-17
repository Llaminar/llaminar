# 🚿 Llaminar
An LLM inferencing engine in C++, with custom quantised kernels for CPU AVX512-VNNI / AVX2, CUDA `sm86`, and ROCm `gfx906`.

Llaminar tries to solve a variety of problems encountered in other projects:

* **Tensor and Pipeline Parallelism:** natively supported, mix and match heterogenous domains.
* **Multiple vendors:** Mix and match CPU, ROCm and CUDA, simultaneously and natively.
* **Easy scaling:** Built from the ground-up on OpenMPI with the goal of enabling scaling across clusters of machines. NUMA-aware.
* **IaC-like experience:** Plan, then deploy.

Llaminar is **experimental** and very much in an **alpha** stage of development. Use it with that in mind and expect the odd segfault.

## Supported Hardware

Llaminar supports:

* CPU inferencing (AVX512-VNNI and AVX2 runtime images)
* CUDA inferencing (RTX-3090 / `sm86` initial support for now)
* ROCm inferencing (`gfx906` only for now)
* All of the above simultaneously
* Tensor Parallel / Pipeline Parallel / MoE routed-expert placement (replicated,
  apportioned, or tensor-sharded compute with explicit row assignment) (WiP)

## Supported Models

Llaminar supports the following model architectures initially:

* Qwen 2.5 (dense)
* Qwen 3 (dense)
* Qwen 3.5/3.6 (dense and MoE)

## Benchmarks

For homogeneous multi-GPU MoE, `--moe-hot-expert-cache` is a replica-cache
upper bound. Physical admission fits the largest positive cache alongside
the complete model and graph allocations; it never disables Dynamic movement
to fit. The resolved capacity is reported during setup.

Production benchmarks use the canonical model-parity cells tagged for E2E
certification. The [production CI guide](docs/production-ci.md) describes the
local/hosted pipeline, independent AVX512/AVX2 image certificates, and checked-in
[high-water marks](benchmarks/production/high_water.json). Official successful
runs commit their compact result JSON under `benchmarks/production/results/`.
Both full E2E server suites pass before either image's benchmarks run; ISA-specific
high-water marks and certificates are never reused across the two images.
For targeted cross-host iteration, the remote runner can prioritize failing or
unseen canonical cases without reducing the required certification matrix;
see the CI guide's Azure workflow.

The [Llaminar testing workflow](.agents/llaminar-testing/SKILL.md) covers Unit
and production-preflight gates, reviewed HTTP token regression (MTP off and
dynamic depth), explicit diagnostic numerical model parity,
HTTP/remote-MPI E2E, and image/benchmark certification.

## Quickstart

### Optional tuning and certification corpora

All published corpora live in [Llaminar/corpora](https://github.com/Llaminar/corpora),
pinned here as the optional `corpora/` submodule. Normal source checkouts,
builds, and Unit/preflight tests do not fetch or require these large datasets.
Only initialize them when working with corpus evidence:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init -- corpora
git -C corpora lfs pull --include 'native_vnni_dispatch/cpu/**' --exclude ''
```

Narrow the include pattern to the generation you need. Publish new corpus
families and their LFS objects in that repository, then update this repository's
submodule pointer. See `corpora/README.md` after initialization for publication
details. Corpus data is excluded from Docker build contexts.

### Building Llaminar

Llaminar uses a predefined devcontainer and the recommended development environment is vscode on a Linux machine with AVX512-VNNI or AVX2, and access to gfx906 / sm86 hardware.

Open vscode in the devcontainer, and run the Build Integration / Build Release vscode tasks with `CTRL + Shift + P`.

For terminal-only development over SSH, use `llaminar` to enter the same
devcontainer with Codex CLI, `llaminar shell` for a persistent shell, or
`llaminar rebuild` to recreate the environment without VS Code. See [the SSH
and Codex workflow](.devcontainer/SSH_CODEX.md) for setup and recovery details.

The image pins one Ninja release for both the system and workspace tools.
Resolve the devcontainer's active executable while configuring and always
build through CMake, so an existing tree keeps using that same tool. Mixing
Ninja executables can make their command-log hashes differ and cause a
needless full rebuild.

The image also builds the patched NCCL capture dependency through
`scripts/docker/install-nccl.sh`. Native CUDA builds outside the devcontainer
must install that dependency first; an unpatched system NCCL does not support
the retained-parent capture lifecycle.

```bash
LLAMINAR_NINJA_BIN="$(command -v ninja)"
cmake -B build_v2_integration -S src/v2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Integration \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="${LLAMINAR_NINJA_BIN}"
cmake --build build_v2_integration --parallel
```

### Running Llaminar

The runtime image contains CPU, CUDA, and ROCm support. Select it for the
**host CPU ISA**, not for the accelerator: use the unsuffixed tag on an
AVX-512 host and `-avx2` on an AVX2 host. Pin a release tag or digest in a
deployment; the `develop` tags below are for trying the current build.

```bash
export MODEL_DIR=/opt/llaminar-models
export QWEN38=/models/Qwen3.8-27B-IQ4_XS.gguf
export QWEN36_MOE=/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf
export LLAMINAR_AVX512=ghcr.io/llaminar/llaminar:develop
export LLAMINAR_AVX2=ghcr.io/llaminar/llaminar:develop-avx2

# Choose exactly one for this host, then pull it.
export LLAMINAR_IMAGE="$LLAMINAR_AVX512" # AVX-512 host
# export LLAMINAR_IMAGE="$LLAMINAR_AVX2" # AVX2 host
docker pull "$LLAMINAR_IMAGE"

COMMON_RUN=(
  --rm --network host --ipc=host
  --security-opt seccomp=unconfined
  --cap-add SYS_NICE --cap-add SYS_PTRACE
  -v "$MODEL_DIR:/models:ro"
)
```

The image runs as a non-root user. ROCm needs KFD/DRI device access and the
actual host GIDs; this discovers every local render node without assuming a
particular card or render-device number:

```bash
ROCM_DEVICE_ARGS=(--device=/dev/kfd --device=/dev/dri)
for node in /dev/kfd /dev/dri/card* /dev/dri/renderD*; do
  [[ -e "$node" ]] && ROCM_DEVICE_ARGS+=(--group-add "$(stat -c '%g' "$node")")
done
```

#### Serve Qwen 3.8 locally

Production model activations are currently FP32. The examples use FP16 KV,
tiered prefix caching, and dynamic-depth MTP. With host networking, the server
listens directly on the selected port; do not add `-p`.

**One ROCm card (32K context):**

```bash
docker run "${COMMON_RUN[@]}" "${ROCM_DEVICE_ARGS[@]}" \
  --name qwen38-rocm "$LLAMINAR_IMAGE" serve \
  -m "$QWEN38" -d rocm:0 \
  --context-length 32768 \
  --activation-precision fp32 --kv-cache-precision fp16 \
  --prefix-cache --prefix-cache-storage tiered \
  --mtp --mtp-depth-policy dynamic \
  --host 0.0.0.0 --port 8080
```

**One CUDA card (8K context on a 24 GiB RTX 3090):**

```bash
docker run "${COMMON_RUN[@]}" --gpus all \
  --name qwen38-cuda "$LLAMINAR_IMAGE" serve \
  -m "$QWEN38" -d cuda:0 \
  --context-length 8192 \
  --activation-precision fp32 --kv-cache-precision fp16 \
  --prefix-cache --prefix-cache-storage tiered \
  --mtp --mtp-depth-policy dynamic \
  --host 0.0.0.0 --port 8080
```

**Two cards with homogeneous TP:** use one vendor per tensor-parallel
communicator. Do not combine CUDA and ROCm in a dense TP group.

```bash
# 2 x ROCm
docker run "${COMMON_RUN[@]}" "${ROCM_DEVICE_ARGS[@]}" \
  --name qwen38-rocm-tp2 "$LLAMINAR_IMAGE" serve \
  -m "$QWEN38" --tp 2 --tp-scope rank_local \
  --tp-devices rocm:0,rocm:1 --backend rccl \
  --context-length 32768 \
  --activation-precision fp32 --kv-cache-precision fp16 \
  --prefix-cache --prefix-cache-storage tiered \
  --mtp --mtp-depth-policy dynamic --host 0.0.0.0 --port 8080

# 2 x CUDA
docker run "${COMMON_RUN[@]}" --gpus all \
  --name qwen38-cuda-tp2 "$LLAMINAR_IMAGE" serve \
  -m "$QWEN38" --tp 2 --tp-scope rank_local \
  --tp-devices cuda:0,cuda:1 --backend nccl \
  --context-length 32768 \
  --activation-precision fp32 --kv-cache-precision fp16 \
  --prefix-cache --prefix-cache-storage tiered \
  --mtp --mtp-depth-policy dynamic --host 0.0.0.0 --port 8080
```

#### Plan once, then apply exactly

`plan` and `serve` accept the same inference configuration. Automatic planning
is the default when no explicit placement is supplied, so `--auto` is optional.
Use `--only-backends` and `--only-strategies` for hard constraints;
`--plan-workload` is a ranking horizon, not a request-output limit. A saved
plan is a lossless, apply-only document: `serve --config` does not rerun the
search or reinterpret its constraints.

```bash
mkdir -p plans

docker run "${COMMON_RUN[@]}" "${ROCM_DEVICE_ARGS[@]}" \
  -v "$PWD/plans:/plans" "$LLAMINAR_IMAGE" plan \
  -m "$QWEN38" --only-backends rocm --only-strategies single,tp \
  --context-length 32768 --plan-workload 512,384 \
  --mtp --mtp-depth-policy dynamic --kv-cache-precision fp16 \
  --output /plans/qwen38-rocm.json

docker run "${COMMON_RUN[@]}" "${ROCM_DEVICE_ARGS[@]}" \
  -v "$PWD/plans:/plans:ro" "$LLAMINAR_IMAGE" serve \
  --config /plans/qwen38-rocm.json --host 0.0.0.0 --port 8080
```

#### Remote CPU ExpertOverlay with an MPI hostfile

The cross-host topology uses one local GPU continuation rank and one CPU rank
per remote machine. All ranks run the **same immutable image digest**, have the
same GGUF path, and communicate over private, routable addresses. The hostfile
names physical MPI endpoints; it does not copy an image or GGUF, and it does
not turn a node-local transport into a cross-host transport. A Docker deployment
must therefore launch every MPI daemon *inside* its matching runtime image;
launching host-MPI processes beside containers is not a supported topology.

```text
# cluster/hosts: GPU controller first, then one CPU-only host per line.
10.10.0.10 slots=1
10.10.0.21 slots=1
10.10.0.22 slots=1
```

After the cluster launcher has started the identical runtime image on each of
those machines, run the public frontend on the GPU controller. This is the
exact default-auto policy exercised by the Azure E2E: it requires every host,
permits only ROCm/CPU compute, and selects ExpertOverlay rather than manually
hard-coding expert owners. It gives the planner the topology intent, while the
physical-memory authority decides how many experts each admitted tier can hold.

```bash
llaminar2 plan -m /models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --mpi-hostfile /cluster/hosts \
  --only-backends rocm,cpu --only-strategies expert-overlay \
  --auto-hosts all --context-length 32768 --plan-workload 512,384 \
  --activation-precision fp32 --kv-cache-precision fp16 \
  --prefix-cache --prefix-cache-storage tiered \
  --mtp --mtp-depth-policy dynamic \
  --output /cluster/qwen36-rocm-remote-cpu.json

llaminar2 serve --config /cluster/qwen36-rocm-remote-cpu.json
```

For the managed Azure route, do not hand-roll remote Docker/MPI processes. The
certification runner provisions owned CPU peers, stages the immutable image and
the declared GGUF shards, builds the exact hostfile, runs both `plan`/apply and
direct auto-serve, proves remote CPU expert work in prefill and decode, then
retires the lease:

```bash
python3 scripts/ci/run_production_cross_host_e2e.py \
  --manifest build_v2_integration/production-ci/avx2/cross-host-manifest.json \
  --source-revision "$(git rev-parse HEAD)" \
  --container-image ghcr.io/llaminar/llaminar:develop-avx2 \
  --models /opt/llaminar-models \
  --model-ramdisk-root /mnt/llaminar-production-parity \
  --report parity-results/cross-host-e2e.json \
  --azure-subscription "$AZURE_SUBSCRIPTION" \
  --azure-ssh-source "$AZURE_SSH_SOURCE" \
  --azure-ssh-public-key "$AZURE_SSH_PUBLIC_KEY" \
  --ssh-private-key "$HOME/.ssh/id_ed25519"
```

See [production CI](docs/production-ci.md#azure-resources-for-cross-host-e2e)
for credential, networking, lease-retirement, and retained-debugging policy.

#### Compact expert-tier topology

Use compact `--expert-tier` declarations when you want an authored topology.
Tier names are labels; integer priority is the policy, and smaller is preferred.
Scope, collective, rank ownership, capacity, and safety margin resolve from the
inventory and `PhysicalMemoryAuthority`. Do not hard-code an expert count just
to fill a device. The lowest numeric priority is the continuation tier by
default, so there is no separate `fallback=true` switch.

```bash
llaminar2 serve -m model.gguf \
  --expert-tier 'accelerator=cuda:0,cuda:1;priority=0' \
  --expert-tier 'capacity=cpu;priority=10' \
  --moe-residency-maintenance dynamic \
  --mtp --mtp-depth-policy dynamic
```

#### Send a request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":"Explain ROCm graph capture in two sentences."}],
    "max_tokens": 128,
    "temperature": 0.0,
    "enable_thinking": false
  }'
```

#### Exact completion token IDs

Non-streaming `/v1/chat/completions` requests may set `"return_token_ids": true`.
The response then includes `token_ids.prompt` (the actual templated input) and
`token_ids.completion` (ordered committed output, including stop tokens and any
forced thinking continuation). Counts agree with `usage`; displayed text may
omit tokens used for framing or termination. This option does not change
sampling, MTP, graph execution, or prefix-cache behavior, and performs no extra
model-state download. Streaming requests with this option return HTTP 400.
Without it, the response format and token-storage cost are unchanged.

The independent `"return_runtime_summary": true` option includes a versioned
`runtime_summary` with the completed request's prefix-cache outcome and MTP
statistics. This is the runner's existing terminal observation, not a live-state
probe or a reconstruction from PerfStats. It remains available when profiling
and INFO logging are disabled. Streaming requests with this option return HTTP
400; ordinary responses omit it.
Within `prefix_cache`, `hit` denotes a full hit and `partial_hit` denotes a
partial restore; these flags are mutually exclusive, not an aggregate hit flag.
The optional `expert_movement` member contains the placement owner's completed
edge, economy and host-admission records. Its explicit `model_lifetime` scope
is cumulative: successive responses overlap and must not be summed as separate
request totals. Logical `movement_axis` and physical `direction` are independent;
unknown MPI ranks are `null`, and estimated weight bytes remain labeled as
estimates. This export reads the existing immutable ledger once before request
cleanup, without advancing or waiting for maintenance. Ordinary responses and
INFO logging do not copy this journal. Truncated or malformed owner evidence
fails the requested export rather than being presented as zero movement.

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

Prefill uses captured chunks of at most 512 tokens by default, independently
of the full KV context limit. Use `--prefill-max-bucket-size 1024` (or another
positive row count) to select a different maximum. Smaller chunks reduce
activation/workspace VRAM; larger chunks may improve prefill throughput.
The option replaces the startup `LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES` list with
the canonical buckets up to that size, including the exact requested endpoint.
Without the option, an explicit environment bucket list remains authoritative.
ExpertOverlay's separate `--moe-overlay-prefill-segment-rows` limit may further
bound its chunk size; raise both limits to request larger overlay segments.

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

- An x86_64 CPU with AVX512-VNNI or AVX2. Use runtime image tags that match
  the CPU ISA on the host.
- Ubuntu 24.04 on x86_64.
- Docker Engine with the Buildx plugin.
- NVIDIA Linux driver `580.95.05` or newer for CUDA 13.0 Update 2.
- NVIDIA Container Toolkit configured for Docker.
- AMDGPU DKMS kernel driver from the ROCm 7.1.1 stack.

OpenMPI and libnuma are hard Llaminar dependencies. The Docker images include
them; source builds should install `openmpi-bin`, `libopenmpi-dev`, and
`libnuma-dev`.

You do not need to install the full CUDA Toolkit or the full ROCm user-space
stack on the host. Those user-space libraries are in the image. The full image
also runs on CPU-only cluster members without GPU drivers. Exposing CUDA needs
NVIDIA Container Toolkit; exposing ROCm needs the AMDGPU kernel driver and
`/dev/kfd` plus `/dev/dri`. Supply both ecosystems only when that host will use
both GPU backends. CUDA driver binding is deferred until CUDA preparation, not
required merely to load a combined CPU/CUDA/ROCm executable.

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

5. Pull the independently certified runtime image for this host's ISA:

```bash
export LLAMINAR_CPU_ISA=AVX512  # or AVX2
case "$LLAMINAR_CPU_ISA" in
  AVX512) LLAMINAR_IMAGE_TAG_SUFFIX="" ;;
  AVX2)   LLAMINAR_IMAGE_TAG_SUFFIX="-avx2" ;;
  *) echo "LLAMINAR_CPU_ISA must be AVX512 or AVX2" >&2; exit 1 ;;
esac

# Use the immutable tag emitted by a successful production CI/release run.
export LLAMINAR_IMAGE_TAG="replace-with-certified-tag"
export LLAMINAR_FULL_IMAGE="ghcr.io/llaminar/llaminar:${LLAMINAR_IMAGE_TAG}${LLAMINAR_IMAGE_TAG_SUFFIX}"
export LLAMINAR_CPU_IMAGE="$LLAMINAR_FULL_IMAGE"
export LLAMINAR_CUDA_IMAGE="$LLAMINAR_FULL_IMAGE"
export LLAMINAR_ROCM_IMAGE="$LLAMINAR_FULL_IMAGE"

docker pull "$LLAMINAR_FULL_IMAGE"
```

Docker also pulls the image automatically on first `docker run`. Official
certification ships full CPU/CUDA/ROCm images for both ISAs; the example backend
variables name that same artifact. Backend selection remains a runtime CLI
choice. Unsuffixed tags are AVX512; append `-avx2` for independently certified
AVX2 images. Local backend-subset builds below do not carry these certificates.

To build images locally instead of pulling GHCR, use the release image build
script:

```bash
scripts/docker/build-runtime-image.sh --cpu-isa "$LLAMINAR_CPU_ISA" --tag llaminar:local --cuda-archs "80;86;89;90"
```

Use the semicolon-separated CUDA architecture list for the NVIDIA GPUs in your
local build. Common values are `80` for A100, `86` for RTX 30/A10, `89` for RTX
40/L4/L40, and `90` for H100/H200. `--cpu-isa AVX512` is the default; pass
`--cpu-isa AVX2` for an AVX2-compatible local image.

Backend-specific local builds are also available:

```bash
scripts/docker/build-runtime-image.sh --variant cpu  --cpu-isa "$LLAMINAR_CPU_ISA" --tag llaminar:cpu
scripts/docker/build-runtime-image.sh --variant cuda --cpu-isa "$LLAMINAR_CPU_ISA" --tag llaminar:cuda --cuda-archs "80;86;89;90"
scripts/docker/build-runtime-image.sh --variant rocm --cpu-isa "$LLAMINAR_CPU_ISA" --tag llaminar:rocm
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
export MODEL_CPU_DENSE="$MODEL_DIR/Qwen3.8-27B-IQ4_XS.gguf"
export MODEL_PP_DENSE="$MODEL_DIR/Qwen3.5-27B-Q4_K_M.gguf"
export MODEL_TP_MOE="$MODEL_DIR/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
```

Use the immutable tag from a successful production CI/release run:

```bash
export LLAMINAR_CPU_ISA=AVX512  # or AVX2
case "$LLAMINAR_CPU_ISA" in
  AVX512) LLAMINAR_IMAGE_TAG_SUFFIX="" ;;
  AVX2)   LLAMINAR_IMAGE_TAG_SUFFIX="-avx2" ;;
  *) echo "LLAMINAR_CPU_ISA must be AVX512 or AVX2" >&2; exit 1 ;;
esac

# Use the immutable tag emitted by a successful production CI/release run.
export LLAMINAR_IMAGE_TAG="replace-with-certified-tag"
export LLAMINAR_FULL_IMAGE="ghcr.io/llaminar/llaminar:${LLAMINAR_IMAGE_TAG}${LLAMINAR_IMAGE_TAG_SUFFIX}"
export LLAMINAR_CPU_IMAGE="$LLAMINAR_FULL_IMAGE"
export LLAMINAR_CUDA_IMAGE="$LLAMINAR_FULL_IMAGE"
export LLAMINAR_ROCM_IMAGE="$LLAMINAR_FULL_IMAGE"
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

Examples above demonstrate CLI topology syntax; they are not an inventory of
current E2E certificates. Eligibility lives in the canonical typed model-parity
definitions. List it with:

```bash
cmake --build build_v2_integration --parallel --target v2_model_parity_matrices
python3 scripts/ci/run_model_parity_e2e.py --build-dir build_v2_integration --list
```

Run without `--list` to certify tagged cells through the Release HTTP server,
including the full needle, long-generation, prefix/reset and context-boundary
checks. Both local and container runners consume the same definitions. See the
[parity workflow](tests/v2/integration/parity/README.md#tagged-http--long-context-certification)
for selection, persistent tmpfs staging and evidence requirements.

Reference docs:
- NVIDIA CUDA release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- Docker Engine install guide: https://docs.docker.com/engine/install/ubuntu/
- NVIDIA Container Toolkit install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- AMD ROCm Docker container guide: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html

## The Llaminar Philosophy

* Tensors want to be open and free: so is Llaminar.
* Tensors want to be sliced, sharded, and pipelined: Llaminar lets them be.
* Tensors want to run on a variety of hardware types without artificial handicaps: Llaminar helps them to do so.

## Activation precision

Production inference currently supports FP32 model activations only.
`--activation-precision fp32` is the default; other activation modes fail as
unimplemented. KV-cache precision and model/expert weight formats are separate
settings and retain their own supported formats.
