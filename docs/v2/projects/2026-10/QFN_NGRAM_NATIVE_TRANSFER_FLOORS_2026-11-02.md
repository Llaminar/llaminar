# QFN native transfer floor investigation

This is a component investigation, not a full lookup, inference, or matrix
certificate. The registered Release fixture uses the same devices, two retained
patterns, payload sizes, three warmups and 25 timed replays for native NCCL/RCCL
and the mapped-channel comparator. Each message is returned to its source.
Exact payload and poisoned inactive-byte checks run outside timing. Both
directions are measured separately. This host intentionally has no direct GPU
P2P; the native libraries choose their transport.

## Retained Release measurement

The first isolated run executed all 28 registrations in 94.3 seconds. Twenty-two
passed. All six native CUDA cases failed the same graph-family memory
observation assertion; their byte/replay checks passed. No test was skipped.
The complete driver window passed with 25 new records and no findings.

Source-device event medians below measure the full round trip after the peer
has been submitted. Wall time additionally includes host submission and both
terminal observations. Neither is a one-way lookup latency.

| Pair / initial direction | Payload per direction | Native event / wall µs | Mapped event / wall µs |
|---|---:|---:|---:|
| MI50 0 → 1 | 64 B | 47.84 / 113.23 | 66.08 / 112.72 |
| MI50 0 → 1 | 1,440 B | 48.16 / 92.04 | 66.08 / 118.14 |
| MI50 0 → 1 | 737,280 B | 290.88 / 488.30 | 336.32 / 528.57 |
| MI50 1 → 0 | 64 B | 49.60 / 96.85 | 66.24 / 112.96 |
| MI50 1 → 0 | 1,440 B | 48.48 / 115.11 | 64.96 / 109.11 |
| MI50 1 → 0 | 737,280 B | 293.76 / 460.94 | 352.48 / 524.99 |
| RTX 3090 0 → MI50 0 | 1,440 B | — | 53.25 / 94.99 |
| RTX 3090 0 → MI50 0 | 737,280 B | — | 313.34 / 493.83 |
| MI50 0 → RTX 3090 0 | 1,440 B | — | 54.72 / 95.39 |
| MI50 0 → RTX 3090 0 | 737,280 B | — | 312.64 / 372.89 |

RCCL reduces the small-payload event interval by roughly 25–28% here. Host wall
times vary more and do not show a universal improvement. Keep both observations;
do not infer full-model speedup from the device interval alone.

CUDA native event medians were 20.48–22.53 µs for 64/1,440-byte messages and
287.74–290.82 µs for 737,280 bytes, versus mapped 39.94–43.01 and
301.06–304.13 µs respectively. These remain **unqualified measurements** because
their memory check failed. They are not installed performance-policy evidence.

## CUDA memory observation

The actual `cudaMemGetInfo` delta around the first native graph instantiation is
54,525,952 bytes (52 MiB) on each device; the second records zero. The fixture
reserves two existing general-graph units, totaling 46,137,344 bytes (44 MiB).
The deltas are not double-counted between those two handles. The observation
does not establish which opaque runtime pool owns the growth.

A diagnostic using `CUDA_MODULE_LOADING=EAGER` reproduces the same 52 MiB
observation. Separate allocator-phase observations report zero CUDA graph
async-allocation and default/current pool bytes; these public pools are not
NCCL's private pool. The growth survives graph/buffer retirement and returns
only when the collective retires. The stack limit also changes from 1,024 to
1,456 bytes, but its numeric difference does not account for device bytes.

A second diagnostic submits the exact original forward/return operation once
during cold setup, using the same communicator, payload buffers, streams and
terminal events. The full 52 MiB moves to that operation; both later graph
instantiations observe zero growth. The bytes still return at collective
retirement. This establishes a first-use collective-lifetime effect, not a
per-graph allocation. It does not identify an unobserved private allocator.
The diagnostic passed its original byte/replay and graph assertions with a
clean driver window in 4.48 seconds. It is not a benchmark or a production
warmup policy.

The original floor fixture used ordinary workers before creating a separate
PMA, and omitted the existing native runtime-preparation boundary. Its repair
uses the original prepared source, runtime and retained graph owners, with
unchanged graph reservation units. No eager warmup, arbitrary limit increase,
or guessed subtraction is being installed. The cold first-use interval must
be reported with its actual mixed lifetime scope. Native qualification of
that repaired fixture is described below; the original six failures stay retained.

## Original-owner functional rerun

The original prepared-source fixture now uses its admitted worker stream and
one contiguous submission sequence **per retained definition**. Alternating
two immutable input patterns does not make the first invocation of the second
definition ordinal two. Both checks retain the production lifecycle guards.

The fresh Integration rerun (`native-fixed-floor-r166`) passes all six ROCm
fixed-payload cases. Its six CUDA cases fail before submission: ordinary NCCL
fixed P2P needs device stack growth that the original prepared source has not
declared. No graph allowance is raised and no late measured context ledger is
introduced. The complete driver window has 25 records and zero findings.
These runs establish functional outcomes, not new Release timing evidence.

The separate completed-lookup transport uses a Broadcast recipe. Dedicated
Broadcast kernel qualification must cover the fixed header/ACK and completed
payload together; success there will not relabel the generic fixed-P2P control
as qualified. The latter stack-growth obligation remains explicit.

## Evidence and next qualification

Local evidence root: `/tmp/qfn-read-ahead-root-integration-r1/`.

- `native-mapped-floor-r155/`: all registrations, samples, XML, summaries and
  complete driver window.
- `native-floor-cuda-memory-diagnostic-r156-r2/`: failing observed/admitted
  values from the original fixture.
- `native-floor-cuda-eager-diagnostic-r156/`: separate eager-module control.
- `native-floor-cuda-pool-phases-r157/`: original cold phase/lifetime observations.
- `native-floor-cuda-original-warm-r159-r2/`: original-operation setup comparison.
  The first attempt ended before native work because its MPI session directory
  had not been created; that failed setup receipt remains alongside the rerun.
- `colocated-and-floor-native-r163/`: invalid stale-ABI floor executable results
  retained as invalid, alongside the independent captured-copy fixture failure.
- `colocated-and-floor-native-r164/`: all ten native finite-parent replays and
  16 mapped/cross-vendor floor cases pass; 12 original-stream fixture rejections.
- `native-fixed-floor-r165/`: original-stream correction exposes the fixture's
  incorrectly shared submission ordinal; clean driver window.
- `native-fixed-floor-r166/`: ROCm six pass; CUDA six retain the original-source
  stack-growth rejection. Both actual registered binaries were rebuilt first.

These fixed-count operations do not qualify the device-authored count/slice
transport needed by pooled ngrams. That path must separately prove zero work,
changing live extents, source offsets, repeated replay, exact receiver payloads,
original graph ownership and retirement. The six-topology campaign then measures
complete lookups and model execution with useful admitted RAM/VRAM caches,
promotion/eviction, MTP and dynamic expert balancing.
