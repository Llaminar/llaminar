# GPU kernel targets and dispatch

Size limit: 15,000 UTF-8 bytes. Experiments belong in linked notes.

## Installed ownership

| Authority | Owns | Lifetime |
|---|---|---|
| [ROCmHardwareMetadata](../../../../src/v2/backends/rocm/ROCmHardwareMetadata.h) | Visible identity, physical limits and native preferred wave width | Process visibility; unchanged on context reset |
| [CUDAHardwareMetadata](../../../../src/v2/backends/cuda/CUDAHardwareMetadata.h) | Visible UUID/PCI identity, SM version/count, warp width, thread/register/shared-memory limits | Process visibility; unchanged on context reset |
| [ROCmCompiledWaveContract](../../../../src/v2/kernels/common/ROCmCompiledWaveContract.h) and compiled kernels | Native instructions, arithmetic, wave width, geometry and layouts | Exact built program |
| [ROCmBackend runtime generation](../../../../src/v2/backends/rocm/ROCmBackend.h) | Runtime resource identity | Changes when native resources are replaced |
| [CUDABackend runtime generation](../../../../src/v2/backends/cuda/CUDABackend.h) | Runtime resource identity | Changes when native resources are replaced |

ROCm startup admits the hardware inventory; CPU-only startup
does not enumerate HIP. Missing/duplicate UUID or PCI, invalid ordinals and
failed queries reject admission without inventing ordinal zero or an empty
inventory. HIP queries may initialize opaque runtime resources; caching their
results grants no memory-capacity credit.
`PhysicalMemoryAuthority` remains the allocation/admission ledger.

Metadata lookup performs no HIP call or mutex acquisition. FA2 binds cached
CU/LDS/thread limits through `bindROCmFA2Hardware` before choosing its captured
query/context transaction. Host recording current-device queries must succeed before lookup or launch. Captured replay performs no
per-token property query, host selection, allocation or metadata refresh.
Legacy precision booleans preserve caller semantics without qualifying native
instructions, wave width or matrix arithmetic.

CUDA follows the same explicit startup contract. Backend public getters,
enumeration, attention, router preparation and NativeVNNI scheduling borrow one
inventory. No ordinal masking, fixed device-count cache or invented SM count is
permitted. Distinct CUDA UUIDs may share a parent PCI endpoint; this identity
rule alone does not qualify execution on partitioned hardware. Router and
attention retain typed, immutable launch bindings derived from the inventory.
Actual function occupancy stays with the loaded program, not physical metadata.

ROCm's admitted wall-clock frequency supplies service telemetry and captured
protocol deadlines. It is distinct from a variable shader clock. TP column
assembly on both backends combines the admitted SM/CU count with the exact
loaded function's occupancy; its resident-grid formula remains unchanged.

Mutable stream, event and workspace bindings have execution owners. They never
belong in physical metadata. Embedding instances use an explicit device owner
and authenticate stream/workspace replacements before committing them. The
projection stream/event pool migration retains CUDA's eight streams/nine events
and ROCm's four streams/five events in their exact device context. Exclusion spans
the entire native capture or eager fork/join submission. Fixed handles survive
named auxiliary-stream resets and retire only with the context; retained graphs
hold that exact context alive. A per-stage lock cannot protect side streams that
remain members of a parent capture. `GPUDeviceProcessLifetime` owns terminal
retirement in the actual main frame, before runtime exit callbacks. Native
contexts and cold discovery hold typed lifecycle borrows; published metadata
reads remain lock-free. This lifecycle authority carries no memory quantities.

The installed ordered reductions compile as wave64 even on native-wave32
hardware. Fixed partitions and shuffle trees retain their numerical order. [ROCmSignedDot4](../../../../src/v2/kernels/rocm/common/ROCmSignedDot4.h)
selects an exact unsaturated native signed-byte instruction using compiler
feature predicates; unsupported targets fail compilation. Adding card names to
a whitelist or spoofing another ISA cannot establish correctness.

## Extensible prefill variants

Future MMA prefill is a separately implemented and qualified program family.
Hardware presence or a legacy precision flag does not install that family.
Bind an executable variant before capture using an explicit contract covering:

- Phase: prefill, ordinary decode or grouped verifier; dense/shared/routed MoE
  ownership, live-row semantics and collective boundaries.
- M/N/K ranges, tails/alignment, batch grouping, sparse ownership and supported
  tensor formats/codebooks; operand, accumulation and output precision.
- Prepared-weight/activation layouts, packing version, strides, tile geometry,
  wave policy, persistent workspace and producer/consumer event requirements.
- Native instruction requirements and loaded function resources: registers, LDS,
  private memory and launch bounds.

Retain generic shape coverage and economical boundaries. Missing instructions,
incompatible layouts or an unqualified mode fail explicitly, without automatic
substitution. Wave32 needs its own program target and arithmetic proof; a host
wave-size integer cannot transform the installed wave64 kernels.

Grouped decode/verifier retains serial-row bytes. Prefill must preserve its
numerical contract and accuracy panel; throughput or cosine-only proxies cannot
justify relaxed tolerances. Share prepared storage/workspaces according to
concurrency. Communicate only live consumer payloads. Live counts and controller
decisions remain on device; variant selection cannot add per-token host probes.

## Program generation and capture identity

Physical facts survive context reset; loaded functions/resource observations
need separate validity. Key executable/resource bindings by device, runtime
generation and exact loaded program identity. Observe resources during
preparation and retire stale bindings without rebuilding hardware metadata.

Capture identity includes program/policy version, target/compiler configuration,
function/variant, device/runtime generation, geometry/wave policy, tensor
precision/layout/codebook, pointers, workspace generation and policy scalars.
Retain transactions before replay; reset data through ordered lifecycle APIs.
Missing bindings or stale generations fail without recapture, eager replay or
host mirrors.

## Current qualification boundary

The complete Unit/preflight build passes for CUDA SM80/86/89/90 and HIP
gfx906/gfx1100. Process retirement, initialization rollback, CUDA projection
capture and all-five-card HIP capture pass 20 repeats with clean driver and
child-retirement evidence. Native runtime unloading no longer races pooled
context destruction. Five test handlers that hid cleanup faults were removed.

The complete frozen revision-28 gate passes: **3,239 pass, zero fail, 29 hardware-
unqualified**, including **844/844 Unit**. The unavailable cases require four/eight
CUDA or eight ROCm devices. Host preflight passes 1,314/1,314, the CUDA lane
272/272, ROCm 378 with two unavailable cases, and the combined lane 431 with
27 unavailable cases. Driver diagnostics are clean and all 516 owned source
files remained unchanged throughout the 88-minute transaction.

The repairs cover typed planning facts, process/capture/mock ownership and
three duplicate native-module consumers. All eight mixed MI50/7900 numerical
failures are fixed: the common MoE sum
helper used different FP32 reduction orders across ISAs, affecting both router
probabilities and shared-expert gating. Reverting only that loop reproduces
the original projection bit differences; restoring it passes all four router
and four projection cases. The new cross-device gate replay, actual-helper
replay and Q4 contribution diagnostic pass with clean drivers and retirement.
The complete repair build passes all shipping targets and spill guards.
[Diagnosis and focused evidence](ROCM_MIXED_ISA_MOE_REDUCTION_DIAGNOSIS.md).
All **29** focused repair registrations also pass with clean drivers and
retirement. This closes the bounded 7900 XTX model-free qualification; work
returns to QFN admission and distributed ngrams. Further XTX tuning and MMA
development are deferred. Whole-model and performance qualification remain
separate obligations.

Cooling/reboot restored all four MI50 links. Evidence under
`/tmp/qfn-admission-integration-r19/`: `metadata-gate-build-r16-result.json`,
`metadata-lifetime-repeat20-r14/`, `metadata-startup-rollback-repeat20-r16/`,
`metadata-cuda-consumers-r16/`, `metadata-rocm-gfx906-consumers-r16/`,
`metadata-rocm-gfx1100-consumers-r16/`, `cuda-centroid-status-r17/`,
`prerequisites-metadata-r16/frozen-results-summary.json`,
`metadata-failure-map-r18.json`, `metadata-numerical-diagnostics-r20-plan.json`
`metadata-numerics-original-comparison-r20.json`, `metadata-router-tree-mixed-r21/`,
`metadata-repair-build-r22-result.json`, `metadata-wave-control-r23.json`,
`metadata-shared-gate-focused-r24/`, `metadata-unit-r26/`,
`metadata-repairs-focused-r26/`, `pcie-recovery-r11.json`,
`prerequisites-metadata-r28/gate/prerequisites.json`,
`prerequisites-metadata-r28/frozen-results-summary.json`,
`prerequisites-metadata-r28/driver-diagnostics.json` and
`metadata-frozen-source-r28-validation.json`.

## Qualification before dispatch installation

1. Use the selected SDK and the complete target set from
   [ROCmArchitectures.cmake](../../../../cmake/ROCmArchitectures.cmake) and
   [rocm-release.env](../../../../scripts/docker/rocm-release.env), including
   dependency images. Build with Ninja, ccache and unrestricted parallelism.
   Authenticate every project-owned shipped code object with the mandatory spill guard;
   local compilation does not qualify other targets.
2. Replay the actual production helper/kernel with recorded precision, geometry,
   ordering and policy. Cover reset, dirty large-to-small reuse, partial/empty
   work and repeated retained capture. Register focused regressions explicitly
   in production preflight. Existing examples are the
   [metadata replay](../../../../tests/v2/cmake/V2ROCmHardwareMetadataTests.cmake)
   and [native signed-dot replay](../../../../tests/v2/cmake/V2ROCmSignedDot4Tests.cmake).
   [CUDA metadata tests](../../../../tests/v2/cmake/V2CUDAHardwareMetadataTests.cmake)
   replay high ordinals and failed publication without devices, then authenticate
   every visible native device and prove repeated public getters issue no new
   discovery calls. Compile-time CUDA attention centroids must also work on the
   first captured launch and after native runtime retirement without host upload.
3. Run affected all-format/backend sweeps and real-model parity through the
   [testing workflow](../../../../.agents/llaminar-testing/SKILL.md), then complete
   required Unit/preflight and integration gates. Absent native hardware remains
   unqualified; compiler success cannot substitute for native execution.
4. Use matched unprofiled Release workloads for performance. Profile one exact
   target/kernel/shape candidate separately; retain loaded ISA, register/LDS/spill,
   occupancy, memory-efficiency and launch evidence beside its timing. Require
   economical prefill and end-to-end behavior, including decode/MTP effects.
   Follow the [ROCm tuning workflow](../../../../.agents/rocm-tuning/SKILL.md).
