# Two-host GPU/MPI round-trip experiment

This opt-in, core-free **diagnostic** compares explicit physical buffer policies
under the same retained GPU program. It is not a production collective, a PMA
admission receipt, a general topology implementation, or a model benchmark.
The original production MPI session-owner sources are built and reused; there
is no private process-retirement implementation.

## What is timed

Two original MPI ranks, on two physical hosts, each own their actual local GPU.
One retained cooperative CUDA graph runs all transactions on each GPU:

1. The initiating GPU produces a changing, exactly representable FP32 payload.
2. It release-publishes readiness only after every writer's system fence.
3. The host MPI follower sends exactly the requested bytes, without computing
   values, reading GPU counts, launching per-message kernels or copying tensors.
4. The peer GPU validates the epoch/payload, adds its local FP32 contribution,
   consumes the result, and publishes its return payload.
5. The initiating GPU validates and consumes the returned reduction result.
6. Every source buffer waits for its exact MPI send retirement before reuse.

Per-transaction device-global-timer durations cover the **complete round trip**,
including protocol waits, GPU production/reduction/consumption, and identical
validation overhead. They are not one-way network latency. Samples from the
non-initiating GPU are not used to estimate cross-device clock intervals.
Allocation, poisoning, capture, setup joins, terminal D2H result collection,
guard checks and teardown are outside those durations. No allocation, explicit
CUDA copy, host numeric reduction, host callback or synchronization occurs per
transaction. The final stream synchronization is a diagnostic retirement bound.
MPI/provider internals must be audited separately for implicit staging/sync.

The cooperative program is synthetic and has bounded resident workers with
live-sized grids. It does not prove progress alongside production model graphs
or authorize a persistent waiter that starves other GPU work. The consumer is
simple arithmetic, not a full inference layer. Keep these limits in any report.

## Explicit candidates

- `mapped`: cached `cudaHostAllocMapped` pages, directly accessed by the GPU and MPI.
- `write-combined`: CUDA mapped write-combined pages; separate cache-policy candidate.
- `registered`: aligned ordinary system-memory pages registered with CUDA.
- `device`: `cudaMalloc` buffers; requires **functional CUDA-aware MPI**, otherwise
  the program fails as unqualified before allocation. There is no fallback.

The first three can still use NIC RDMA. Conversely, CUDA awareness alone does
not prove GPU-direct RDMA: retain actual UCX protocol and native transport
observations. Memory aliasing on a coherent integrated GPU does not prove every
allocation is NIC-registerable or has the same cache behavior.

Native allocations here are isolated test-fixture infrastructure. The selected
production endpoint must subsequently use TransferEngine, physical memory
admission, authenticated rank plans, retained lifecycle and live-work contracts.
Do not copy this fixture's allocation or polling code into serving unchanged.

## Build and device-free tests

```sh
cmake -S tests/v2/performance/transfer/mpi_roundtrip -B /tmp/roundtrip-host \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)"
cmake --build /tmp/roundtrip-host --parallel
ctest --test-dir /tmp/roundtrip-host --output-on-failure --parallel
```

For the native build, explicitly select the real CUDA architecture, selected
MPI compiler, matching CUDA host compiler, and the source checkout containing
`src/v2/utils/MPISessionOwner.cpp`. Use `ROUNDTRIP_GPU=ON` and
`LLAMINAR_SOURCE_ROOT=/exact/source/root`. No default-stream, split-compile or
fast-math workaround is enabled; ptxas spills are fatal. Compile the canonical
shipping architecture set as well as the native target before publishing kernel
resource claims. `build-identity.json` identifies the kernel and contract sources.

The exact command-line contract is:

```text
mpi_gpu_roundtrip CANDIDATE BYTES WARMUP SAMPLES ROOT WORKERS OUTPUT_PREFIX
```

`WORKERS` is an explicit positive resident-worker cap or `all` (one worker
per SM). Actual grids remain bounded by live payload work and proven cooperative
residency. Compare one-block small-message latency against wider prefill workers
in separate matched campaigns, never different grids for competing allocations.

Both hosts must have the same executable/dependency closure and corresponding
writable output mount. Original MPI rank count must be two with one rank per
physical host; using two aliases of one Spark GPU is rejected. This fixture
restriction does not constrain Llaminar's supported topologies.

## RDMA experiment launch

Use `run_campaign.py --help` for the retained argument interface. Supply the
built `llaminar_mpi_session`, binary, MPI executable, matching `--daemon` (`orted`),
original hosts and SSH agent. The driver also wraps the remote launch agent with
`llaminar_mpi_session --mpi-daemon --session-base /tmp -- .../orted`, so each
physical daemon receives its own authenticated roots. Wrapping only the local
`mpirun` leaves the remote daemon without that ownership and is rejected.
The SSH agent must enter the corresponding container on the actual remote host. Preserve host-key verification and use narrow experiment-only credentials;
remove any temporary authorization after retiring the experiment.

The containers require the actual `/dev/infiniband` devices and adequate locked
memory limits, alongside the GPU. Keep private IPC; shared IPC requires matching
PID identity. Never change host drivers or network configuration merely to
force a benchmark through. Use the existing observed RDMA interface identities.

`UCX_TLS=rc,self` and explicit `UCX_NET_DEVICES` constrain a host-memory test to
its selected RDMA ports. A CUDA-aware test additionally needs the CUDA memory
transports (`rc,self,cuda`). A `device` campaign requires
`--gpu-direct-rdma yes`, and the binary rejects a device that reports no GPUDirect
RDMA support. Still authenticate the actual message protocol: even UCX's `yes`
setting does not prevent its higher-level protocol from choosing CUDA staging.
A `cuda_copy → network → cuda_copy` trace is an **unqualified direct-device
attempt**, regardless of correct output, and cannot enter the winner comparison.
Neither campaign may silently select TCP for its data plane.
The script binds one CPU core per rank and records every command and UCX option.
`--rndv-threshold 0` is an explicit rendezvous experiment, not a universal claim
that eager protocols are slower. Run the default/eager policy as a separately
named comparison where beneficial. Likewise compare one rail versus both rails
under explicit settings; do not equate port count with doubled bandwidth.

First use an untimed protocol-observation run with `UCX_LOG_LEVEL=info` and
`UCX_PROTO_INFO=y`. Keep the loaded-library versions, selected physical devices,
message memory types and applicable size ranges. For canonical timings, verbose
protocol logging is disabled. Raw NIC counter deltas are useful corroboration,
not a substitute for distinguishing payload from metadata or other jobs' traffic.

The campaign rotates/reverses candidate order and measures both initiating
ranks. It retains raw per-transaction CSVs on the initiating host, each exact
command, logs, validated rows and per-size/direction summaries. Collect evidence
from **both hosts**. A selected candidate failure stops the campaign rather than
quietly reporting timings from the remaining candidates. Overlapping trial ranges
are reported as unresolved, not declared a clear winner. A winner at 16 KiB need
not win at prefill sizes; retain those distinct results and compare the eventual
production workload before installing a dispatch policy.

## Isolated CUDA-aware dependency attempt

`build_experimental_mpi.sh /absolute/experiment/root` builds UCX 1.16.0 and
OpenMPI 4.1.6 in that root only, with archive SHA256 checks. Download the named
archives from their official releases into `deps-src/` first. It does not replace
system/image libraries or modify drivers. Use the resulting stack for **both**
mapped and device candidates; comparing different MPI versions alone does not
isolate allocation-policy cost.

CUDA 13 compatibility adjustments are explicit: the UCX driver function-pointer
spelling uses the versioned five-argument ABI, and its existing CUDA-hook test
uses the SDK's four-argument context-creation API. These changes and all failed
build evidence belong in the experiment receipt. This is not a qualified new
production dependency installation.

## Explicit DMA-BUF qualification

Enable `ROUNDTRIP_DMABUF=ON` in the native build (CUDA13 headers and libibverbs
are required). `cuda_dmabuf_probe HCA [HCA ...]` independently observes device
GDR, device DMA-BUF, host DMA-BUF and VMM capabilities. It actually attempts
allocation/export/import for cached, write-combined and registered host pages;
ordinary and asynchronous device allocations; CPU-/GPU-touched managed memory;
and device/host/observed-NUMA VMM, including the explicit device GDR flag.
Both default and PCIe export flags are tested. Capability zero does not skip a
different allocation's native API call. Invalid-FD negative controls exercise the
actual NIC import API. Run `test_dmabuf_probe.py --audit RECEIPT.jsonl` to reject
missing combinations, incomplete retirement or a false transfer qualification.

A successful `ibv_reg_dmabuf_mr` is **not** proof that GPU-visible payloads move
correctly. The round-trip binary additionally accepts these explicit candidates:

- `mapped-verbs` / `write-combined-verbs`: ordinary `ibv_reg_mr` controls.
- `mapped-dmabuf` / `write-combined-dmabuf`: actual exported CUDA DMA-BUF FDs
  imported separately into each selected HCA. No regular-MR substitution.

Append original `HCA:PORT:GID_INDEX` specifications to a verbs invocation, or
supply `--verbs-rails` to the campaign driver. Observe GID type/address on both
hosts first; the experiment requires explicit active RoCE v2 endpoints. Both
registration modes use identical page-rounded backing, live payload slices,
RC WRITE_WITH_IMM operations, completion handling and retained GPU kernels.
Two preposted immediate-only receive credits prevent a repost race; they are
protocol metadata, not payload bounce buffers. The next receive completion can
precede the previous send completion and must survive the epoch boundary.

MPI owns cold original-rank endpoint exchange and process lifecycle; the isolated
verbs data plane is a diagnostic comparison, **not permission to replace the
production cross-rank MPI/shmemspin authority**. Compare ordinary-MR against
DMA-BUF-MR to isolate registration effects, and separately against the MPI path
to expose protocol overhead. A faster private verbs fixture is not an installed
MPI transport. GPU payloads, guards and complete round-trip timing are checked
by the same kernel in every mode. Source FDs and MRs outlive all native work.

## Acceptance and limits

Every epoch's complete payload is checked on GPU, both receive extents are
verified, guard regions are checked, and terminal GPU consumer values are
compared exactly. Wrong epochs, incomplete messages, unsupported candidates or
transport failures invalidate the result. The transport has a 30-second progress
deadline; fatal errors abort before releasing in-flight buffer ownership.

Run partial and large-to-small payload cases as distinct admitted geometries.
This fixture does not yet certify dynamic device-counted collective geometry,
request reset or general collective membership. Do not call repeated synthetic
transactions a replay of recorded production inputs.

Profile separately. A kernel-replay profiler must **not** replay this persistent
program without also replaying its network peer and host transport; that would
strand the protocol. Use a non-replaying timeline observation or a properly owned
whole-application replay. Keep driver diagnostics around attachment and stop on
new driver failures. Report static registers/spills and unavailable hardware
counters honestly. Do not mix profiler timings with the campaign's raw samples.
