# Two-Spark GPU/MPI round-trip experiment — 2026-11-22

## Decision and qualification boundary

**Use the existing OpenMPI/UCX stack with GPU-mapped host payload storage as the S0 implementation direction.** Adding CUDA-aware MPI provided no measured benefit for those allocations. The explicit `cudaMalloc` candidate is **unqualified** on the observed GB10/driver combination: both devices report `gpuDirectRDMASupported=0`, UCX reports device-allocation CUDA DMA-BUF unavailable, and the earlier CUDA-aware attempt actually used CUDA staging. No driver or installed dependency was replaced.

Within the mapped family, write-combined allocation has the clearest repeatable advantage at 4 MiB. Cached mapped versus write-combined allocation is nearly tied at 16 KiB; do not claim one universal allocation winner. Two RDMA links substantially improve larger transfers. More resident workers improve this particular synthetic producer/consumer, not necessarily a real model kernel.

The fastest measured 16 KiB protocol is native eager **zero-copy copy-out**, not an entirely copy-free receive path. Strict rendezvous is slightly slower and proves direct NIC movement between the mapped payloads. Neither should be misrepresented as CUDA device-memory GPUDirect. Native opaque transport copying is distinct from a CPU numerical reduction or a CUDA D2H/H2D staging pipeline; production admission must still own any transport storage and prove its lifecycle.

**The subsequent host DMA-BUF experiment succeeded end to end**; see the follow-up below. DMA-BUF registration itself showed no consistent performance advantage over ordinary MR registration. A leaner private verbs protocol reduced latency, but it is not an installed cross-rank MPI implementation.

This selects an endpoint family for implementation. It does **not** qualify a production cross-rank collective, DeepSeek inference, general topology, dynamic live counts, repeated graph reset/replay, or achieved GPU resource economy. Nsight profiling is blocked by the canonical startup guard; see below.

## Fixture and provenance

Implementation: [standalone experiment](../../../../tests/v2/performance/transfer/mpi_roundtrip/README.md), branch `feat/rdma-roundtrip`, based on develop `96c3ab43185f0645236790e9ff956e240e428451`. Measurements are bound to the archived source and exact binaries below. No DeepSeek implementer worktree/build was changed during the experiment.

The measured operation is:

1. Initiating GPU generates the live FP32 vector and publishes readiness.
2. Its CPU transport follower submits the exact live MPI extent.
3. Peer GPU verifies the input, adds its contribution and consumes the result.
4. MPI returns the reduced vector.
5. Initiating GPU verifies and consumes it, then records its own elapsed GPU timer.

A captured cooperative kernel performs all epochs in one retained graph launch. Host progress uses `MPI_Irecv`/`MPI_Isend`/`MPI_Test` and transport publication, not numerical reduction or per-epoch GPU submission/copies. Every epoch checks values; both endpoints check final consumers and allocation guards. Initial/final diagnostic copies and synchronization are outside the measured loop. These repetitions are not a production recorded-input replay test.

Both endpoints used:

- `.80` / `zgx-5457` = original rank 0; `.79` / `zgx-5cac` = original rank 1. Both initiating directions were measured.
- NVIDIA GB10, 48 SMs, driver 580.159.03, CUDA 13.0.48, GCC 14, ARM64.
- Image `sha256:9e17dbf8a7a9c6bc4f01f6ba5ba3737b5decbd6be39ca7a341ea96bb16bed9f6`.
- Private IPC, host networking, explicit RDMA device access; no shared PID namespace.
- OpenMPI 4.1.6 / UCX 1.16.0. Stock MPI is not CUDA-aware. The isolated `/work/deps` build is CUDA-aware, with its explicit CUDA 13 compatibility patch retained in the fixture.
- `rocep1s0f1:1`, and additionally `roceP2p1s0f1:1` for dual-link runs. Observed active RoCE Ethernet links, MTU 4096, 200 Gb/s link rate per rail.
- One MPI process per original host, core-bound, canonical local and remote daemon session owners. GB10 CPU/GPU memory is one physical pool, not additive RAM and VRAM.

Final binaries, independently matched on both hosts:

| Artifact | SHA-256 |
|---|---|
| Stock-linked benchmark | `9f1938cb890ceb53e7fc5c3b678cf75836f6d2251f1d56443283641346115700` |
| CUDA-aware-linked benchmark | `1caa5dba89e78a246b973b5a2da8849a59672dcbac35d368ab053caa4f1bd45b` |
| Session helper, both builds | `545f295cf199f70623d854a5d6c93e723f904fbf047c507c7d33767a304fa3c4` |

`final-environment.txt` on each host binds source files, native modules, dependency DSOs and binaries. Each campaign's `request.json` binds its exact binary SHA and UCX environment; each run retains its command, log and raw per-epoch CSV. Older prototype/direct-device probes are separate evidence, not timings of these final binaries.

## Matched results

Values below are **median of trial means**, microseconds for the complete round trip. Main cells have three rotated/reversed allocation-order trials, 100 warmup and 500 measured epochs per trial, in each initiating direction. Parentheses give rank-0 / rank-1 results, not uncertainty bounds. Raw trial ranges remain in each summary; allocation comparisons whose ranges overlap are unresolved.

| Stack / transport / workers | Payload | Cached mapped | Write-combined | Registered pages |
|---|---:|---:|---:|---:|
| Stock, one rail, up to 48 workers | 16 KiB | 29.730 / 30.026 | 29.894 / 29.974 | 30.281 / 30.352 |
| Stock, one rail, up to 48 workers | 256 KiB | 90.703 / 91.285 | 89.480 / 89.926 | 90.656 / 91.249 |
| Stock, one rail, up to 48 workers | 4 MiB | 902.825 / 902.634 | 896.275 / 895.914 | 903.723 / 904.752 |
| CUDA-aware stack, one rail, up to 48 workers | 16 KiB | 29.578 / 29.904 | 29.893 / 29.960 | 30.304 / 30.104 |
| CUDA-aware stack, one rail, up to 48 workers | 256 KiB | 90.846 / 90.776 | 89.884 / 89.850 | 90.911 / 90.344 |
| CUDA-aware stack, one rail, up to 48 workers | 4 MiB | 902.305 / 902.498 | 896.723 / 895.659 | 903.011 / 903.773 |
| Stock, two rails, up to 48 workers | 16 KiB | 28.774 / 28.740 | 28.579 / 28.562 | 29.063 / 29.187 |
| Stock, two rails, up to 48 workers | 256 KiB | 65.481 / 64.858 | 63.780 / 64.011 | 65.387 / 65.188 |
| Stock, two rails, up to 48 workers | 4 MiB | 560.026 / 559.148 | 550.773 / 549.484 | 561.093 / 557.831 |
| Stock, two rails, up to 192 workers | 16 KiB | 28.654 / 28.666 | 28.654 / 28.719 | 29.250 / 29.056 |
| Stock, two rails, up to 192 workers | 256 KiB | 60.001 / 60.289 | 59.675 / 59.143 | 60.313 / 59.572 |
| Stock, two rails, up to 192 workers | 4 MiB | 473.340 / 473.157 | **464.129 / 464.522** | 470.491 / 470.502 |
| Stock, two rails available, native auto protocol | 16 KiB | **26.331 / 26.551** | 26.495 / 26.528 | 27.544 / 27.536 |

All rows except the last set `UCX_RNDV_THRESH=0`. The final row sets `auto`; it is an explicitly different protocol, not a silently substituted endpoint. Auto at larger payloads was **not** a separate timing campaign. No single experiment proves a globally optimal allocation/protocol/worker combination.

At 4 MiB, the best measured complete-path payload rate is approximately **18.1 GB/s**, calculated as `2 × live payload / round-trip time`. It includes GPU production, arithmetic, verification, consumption and network progress. It is **not** NIC line bandwidth. The 192-worker setting is four blocks per observed SM and within measured cooperative residency, not an engine topology/count restriction.

The one-worker 16 KiB control and 68-/1028-byte partial-block cases are retained separately. All **270 matched runs / 135,000 measured round trips**, plus 27,000 warmup epochs, passed. The independent post-run audit checked every CSV's sample count, sequence, positive duration and recomputed mean, every returned correctness flag, owner launch command and submitted live-block count. Both hosts' final standalone contract/evidence CTests passed (2/2 each); the same tests passed on the Mac.

## Protocol and admission observations

`protocol-16384.log` authenticates the fast native policy for mapped host memory:

- `8247..16775`: `multi-frag eager zero-copy copy-out` on `rc_mlx5/rocep1s0f1:1` for the repeated-send protocol.
- Both links are available, but this small-message eager path uses **one** rail.
- The receive copy-out means this is not a fully copy-free protocol.

`protocol-4194304.log` authenticates the larger write-combined path:

- `zero-copy fenced write to remote`.
- `50% on rc_mlx5/rocep1s0f1:1 and 50% on rc_mlx5/roceP2p1s0f1:1`.
- No CUDA-aware stack is required to RDMA these GPU-visible host pages.

The earlier device attempt succeeded numerically but its UCX protocol explicitly said `cuda_copy, fenced write to remote, cuda_copy`. Setting `UCX_IB_GPU_DIRECT_RDMA=yes` was **not sufficient** to prevent that staging. A separate forced device-allocation DMA-BUF probe reported `dmabuf support requested but not found`; no `nvidia_peermem` module was loaded. The final binary additionally rejects `device` when native `gpuDirectRDMASupported=0`; `device-admission.log` records both ranks' rejection before the retained kernel. This is evidence about the observed device-allocation path, not a permanent claim about every GB10 software configuration. It does not establish that CUDA13 host-allocation DMA-BUF is unavailable; that capability and API path are separate.

The first current-binary launch also correctly rejected an unowned remote daemon: `daemon has no original owned location binding`. The campaign now supplies:

```text
--launch-agent '/work/build/llaminar_mpi_session --mpi-daemon --session-base /tmp -- /usr/bin/orted'
```

The CUDA-aware control uses its matching `/work/deps/bin/orted`. Each remote daemon gets fresh host-local roots through the canonical owner; no environment was forged and no ownership check was disabled.

## GPU resource evidence and remaining profiling blocker

The final resource dump covers `sm_80;86;89;90;121`, with fatal ptxas spill diagnostics enabled. Registers are 32 / 32 / 32 / 24 / 28 respectively; all report **zero stack, local and shared memory**. The native occupancy API reports six resident 256-thread blocks per SM for this kernel. Native submitted dimensions are:

- 68 bytes: 1 block; 1028 bytes: 2 blocks.
- 16 KiB: 16 blocks normally, 1 for the explicit control.
- 256 KiB and 4 MiB: 48 or 192 blocks according to the explicit worker cap.
- Always 256 threads/block; no wholly inactive capacity blocks.

`final-sm121.sass` retains exact ISA. These are resource/geometry observations, **not achieved occupancy or memory-efficiency measurements**. The persistent fixture also occupies GPU resources while waiting for network progress; it does not prove overlap economics of a production graph.

The isolated Nsight Systems attempt exited 125 before kernel launch:

```text
Native CPU placement cannot enter a process with existing workers
```

Profiler injection creates threads before the guarded entrypoint. The generated `.nsys-rep` is **not a GPU profile**. No guard was weakened, no bare/unowned MPI path substituted, and no kernel-replay profiler was used against an independently progressing peer. A legitimate profiler-safe entry/attachment path and another exclusive host window are required to close achieved occupancy, memory-efficiency and trace gates. Production qualification remains open.

## Evidence locations, cleanup and handoff

Raw evidence remains outside the source repository:

- Each host: `~/llaminar-rdma-roundtrip/results/` (durable host mount).
- Collected copy: `/tmp/llaminar-rdma-evidence/{zgx-5457,zgx-5cac}/`.
- Independent aggregate audit: `/tmp/llaminar-rdma-evidence/audit.json`.
- Isolated binary/dependency copies: `/tmp/llaminar-rdma-artifacts/`.
- Earlier staging/DMA-BUF probes: `current-device.log`, `dmabuf-probe.log` under the `.80` result root; final rejection is `device-admission.log`.

Campaign directories: `stock-rail1-all`, `aware-rail1-all`, `stock-rail2-all`, `stock-rail2-workers192`, `stock-rail2-auto-16k`, `stock-rail2-worker1-16k`, `stock-partial-live`. Root-0 CSVs reside on `.80`; root-1 CSVs reside on `.79`. Keep both halves together.

Both hosts were exclusively reserved during native runs. All session owners retired, no GPU clients remained, and both owned experiment containers were stopped before release. Only the experiment's authorized-key line on `.79` and private key on `.80` were removed. The coordinating implementer was explicitly notified that **both hosts were free** and could resume its unrelated gate. That campaign released both hosts before the separately coordinated host DMA-BUF follow-up below. Raw result directories remain outside Git.

Next production work remains the S0 plan's typed TransferEngine/PMA ownership, retained transport lifecycle, precise live-extent protocol, graph/order/reset/failure regressions and actual-model certification. Preserve MPI across ranks; these results do not authorize cross-rank NCCL or a second live allocation ledger.

## Host DMA-BUF follow-up — 2026-11-22/23

### Allocation-specific native admission

Both original hosts were tested independently on the same unchanged driver/image.
CUDA reports device GDR **0**, device DMA-BUF **0**, host-allocation DMA-BUF **1**,
VMM GDR **0**, and VMM **1**. Host page size is 4096 bytes on both. The host query
is `CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED`; the device query cannot
substitute for it.

`cuda_dmabuf_probe` actually allocates, exports and calls `ibv_reg_dmabuf_mr`
against each HCA. It never substitutes regular registration. Each host ran the
complete **11 allocations × 2 export flags × 2 HCAs** matrix three times:
**264 case receipts, including 24 successful imports**, plus invalid-FD controls
returning `EBADF` on both HCAs in every run. These probe receipts deliberately
retain `transfer_qualified:false`; the separate payload experiment supplies that
additional evidence.

| Allocation / export flags | Native outcome on both hosts |
|---|---|
| Cached mapped `cudaHostAlloc`, flags 0 | Export and actual DMA-BUF MR import succeed on both HCAs |
| Write-combined mapped `cudaHostAlloc`, flags 0 | Export and actual DMA-BUF MR import succeed on both HCAs |
| Either preceding allocation, PCIe export flag 1 | Export rejects with 801 / `CUDA_ERROR_NOT_SUPPORTED` |
| Registered host pages, flags 0 and 1 | Export rejects with 1 / `CUDA_ERROR_INVALID_VALUE` |
| Ordinary device allocation, flags 0 and 1 | Export rejects with 1 |
| Managed allocation, CPU-touched and GPU-API-initialized controls, flags 0 and 1 | Export rejects with 1; these are not claims of measured physical residency |
| Async device allocation, flags 0 and 1 | Export rejects with 801 |
| Device VMM, flags 0 and 1 | Export rejects with 1 |
| Device VMM with explicit GDR allocation flag | `cuMemCreate` rejects with 101 / `CUDA_ERROR_INVALID_DEVICE`, before export |
| Host VMM and observed-NUMA host VMM, flags 0 and 1 | Export rejects with 1 |

These are the exact tested allocation/API outcomes, not a blanket prediction
about other allocation flags, newer software, or every possible DMA-BUF user.
Rejected combinations did not proceed to a substituted transfer path.

### Payload experiment and audit

Four explicit native candidates use identical page-rounded backing, striping,
RC QPs, `WRITE_WITH_IMM` operations and the same retained GPU kernel:

- `mapped-verbs` / `write-combined-verbs`: actual `ibv_reg_mr` controls.
- `mapped-dmabuf` / `write-combined-dmabuf`: CUDA-exported FDs passed to actual
  `ibv_reg_dmabuf_mr`, independently for transmit/receive buffers on every active HCA.

MPI handles cold original-rank endpoint exchange and canonical process ownership;
it does not carry payloads in these native variants. The ordinary MPI candidates
remain separate matched controls. No CUDA-aware MPI dependency was required.
An explicit RoCE v2 GID index 3 was observed on each selected port; both separate
rails and their combination were exercised. These private verbs variants are
**diagnostic implementations, not replacements for production MPI/shmemspin**.

GPU production, peer verification/reduction/consumption, return transfer and
initiator verification/consumption all passed. The campaign comprised:

| Campaign | Explicit cases | Runs |
|---|---|---:|
| `dual-all` | Six candidates, two rails, worker cap 48, 16 KiB / 256 KiB / 4 MiB | 108 |
| `dual-192` | Same six candidates/shapes, worker cap 192 | 108 |
| `first-rail` | Six candidates, first rail only, cap 48, same three shapes | 108 |
| `second-rail` | Four native candidates, second rail only, cap 48, same three shapes | 72 |
| `partial-live` | Four native candidates, two rails available, 4 / 68 / 1028 / 16380 bytes | 96 |

Every cell has three rotated/reversed trials, both initiating ranks, 100 warmup
and 500 measured epochs. **All 492 runs passed: 246,000 measured round trips plus
49,200 warmup epochs.** They comprise 192 DMA-BUF, 192 ordinary-verbs and 108 MPI
runs. The earlier successful 8-run, 20-sample smoke matrix is separate and excluded
from these totals.

The independent audit joins both hosts' raw CSVs, checks complete unique cells,
recomputes means/percentiles, verifies canonical owner/daemon commands, and checks
the recorded submitted live slices and grids. Both endpoints' native receipt
multisets must match the exact nonempty partition, including one active rail for
a one-word payload. All rounded allocation padding is guarded. This verifies the
fixture's submitted extents; it is not an independent wire packet capture.
At 4/68/1028/16380 bytes the actual recorded grids are 1/1/2/16 blocks. At 16 KiB
the grid remains 16 blocks even with a 192-worker cap.

### Matched performance

Below are median trial means in microseconds, rank 0 / rank 1, for **write-combined
host buffers and two rails**. All MPI controls use strict rendezvous, not the prior
auto/eager policy. Complete cached-buffer and single-rail comparisons, including
trial ranges, remain in the summaries and aggregate audit.

| Worker cap | Payload | MPI | Ordinary-MR verbs | DMA-BUF-MR verbs |
|---|---:|---:|---:|---:|
| 48 | 16 KiB | 28.678 / 28.697 | 21.863 / 22.024 | 22.094 / 22.194 |
| 48 | 256 KiB | 63.979 / 64.732 | 55.722 / 55.733 | 55.805 / 55.683 |
| 48 | 4 MiB | 551.025 / 548.986 | 543.014 / 542.284 | 543.319 / 541.954 |
| 192 | 16 KiB | 28.625 / 28.783 | 22.045 / 22.085 | 22.126 / 22.201 |
| 192 | 256 KiB | 58.791 / 58.493 | 50.580 / 50.478 | 50.389 / 50.632 |
| 192 | 4 MiB | 465.703 / 464.645 | 456.827 / 458.141 | 458.213 / 462.869 |

**DMA-BUF itself is not a demonstrated speedup.** Ordinary-MR and DMA-BUF-MR
results generally overlap, with small direction/cell-specific differences rather
than a consistent winner. For example, at 4 MiB/cap192 the rank-0 ordinary-MR
trial range is 456.686–457.226 µs versus DMA-BUF 456.874–459.502 µs; rank 1 has
456.194–459.753 versus 457.084–463.397 µs. A claim of universal DMA-BUF superiority
would contradict these observations.

The approximately 22 versus 28.7 µs small-message difference is a **protocol**
comparison: retained RC writes with preposted immediate receipts versus MPI's
rendezvous/progress machinery. Both native registration mechanisms obtain it.
It is not proof that replacing host pages with ordinary CUDA device memory works,
or that the private verbs implementation is production-ready.

### Corrections, provenance and retirement

The first DMA-BUF payload smoke exposed a fixture validation error before GPU
launch: rdma-core v50 sets a DMA-BUF MR's `addr` to its **FD offset**, not the
submitted IOVA. The ordinary-MR address check was invalid for that API.
`validateNativeMr` now checks the allocation-specific contract, with focused
positive/negative device-free tests for address, offset and extent. The explicit
IOVA still owns every submitted SGE/remote address. The original failed logs are
retained, not reclassified as hardware failures or successful transfers. Fatal
reporting was also moved ahead of borrowed-buffer retirement so diagnostics are
not suppressed by an aborting destructor. The final campaign ran only after
these corrections. Reference: [rdma-core v50 native registration body](https://github.com/linux-rdma/rdma-core/blob/v50.0/libibverbs/verbs.c).

Both hosts ran the identical final binary, built on `.80` and copied to `.79`:

| Artifact | SHA-256 |
|---|---|
| Payload benchmark | `ac6b5bd5b039b358a6c0546ea04bcc9967659bc12bad7c41ab2e7151bf5d05f4` |
| Native allocation/export/import probe | `b2d5ca160bcd3af595211e6ee80fae7906e395e23da48b87d1ea685cd0b28fcd` |

The session helper remains `545f295c…` from the original campaign. Native libibverbs
is `50.0-2ubuntu0.2`; both actual verbs/provider DSOs are hashed in each final
manifest. Source hashes, build identities, zero-spill compiler output and the
final `sm_80;86;89;90;121` resource/SASS evidence are retained. The kernel remains
32/32/32/24/28 registers, zero stack/local/shared memory. Final standalone tests
passed 3/3 on both hosts and the Mac. The profiler/startup blocker above remains
open; these results do not certify achieved occupancy or production graph replay.

Follow-up artifacts are separate from the original benchmark evidence:

- Both hosts: `~/llaminar-rdma-roundtrip/results/dmabuf/`.
- Local collection: `/tmp/llaminar-dmabuf-evidence/{zgx-5457,zgx-5cac}/`.
- Aggregate: `/tmp/llaminar-dmabuf-evidence/audit.json`, with the audit script and
  source/binary archives alongside it; a durable copy is retained on `.80`.
- Per-host `probe-v2-{1,2,3}.jsonl`, `final-environment.txt`, build identities,
  `final-kernel-resources.txt`, `final-sm121.sass` and `driver-window.log`.
- Rejected initial smoke evidence: `smoke/`, `smoke-v2/`; final smoke: `smoke-v3/`.

The follow-up began only after explicit retirement of the implementer's gate.
After all experiments, no GPU clients remained and no Xid/assert/reset was
observed in the driver window. Every native owner retired; both experiment
containers were stopped and only the owned temporary SSH authorization/private
key were removed. The coordinator was explicitly told both hosts were free.
Subsequent evidence collection uses host files only. Raw measurement artifacts
remain outside Git; no driver/dependency was replaced, and the deleted research
markdown was not recreated.
