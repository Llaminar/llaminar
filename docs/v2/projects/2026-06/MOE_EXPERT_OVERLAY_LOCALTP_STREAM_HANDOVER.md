# MoE Expert Overlay LocalTP Stream Handover

**Date:** 2026-05-10  
**Branch:** `feat/qwen35-moe`  
**Status:** Phase 5E P2P precondition correction in progress; stream-safe LocalTP execution still needs a focused fix.

## Why This Handover Exists

The Phase 8A parity unblock exposed a too-strict hardware gate: Qwen3.5 MoE Expert Overlay LocalTP tests were skipping real ROCm/RCCL execution unless every ROCm participant pair reported direct P2P access. That is not a correctness requirement. RCCL owns transport selection and may use host-staged fallback when peer access is unavailable. P2P should be a performance diagnostic/gate later, not a Phase 5E/8A correctness precondition.

The P2P skip was removed, which allowed the real prepared ROCm LocalTP path to run. That flushed out a separate execution/coherence problem in `MoEExpertOverlayLocalTPExecutor`: shared input tensor residency migrates between ROCm ordinals during the participant loop, and ROCm kernels can still be using device memory asynchronously. A temporary device-wide synchronize was added during debugging, but the user correctly rejected that approach. Do not keep or build on the device-wide sync patch.

## Current User Direction

- Do **not** require all-device ROCm P2P for LocalTP correctness.
- Require that ROCm/RCCL is built and that a usable RCCL `LocalTPContext` can be created.
- Let RCCL handle transport fallback, including host-staged paths when direct P2P is unavailable.
- Avoid full device synchronization. Device-wide sync drains the whole GPU pipeline and is a last resort.
- Avoid default/null streams. Use explicit streams and event-based synchronization consistent with the rest of V2.

## Files Involved

- `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_ExpertOverlay_Parity.cpp`
  - `overlayRuntimeBlockers()` had a strict bidirectional ROCm P2P blocker. It has been removed in the current working tree.
  - It should keep missing registry and real hardware/build/model precondition checks.

- `tests/v2/integration/moe/Test__MoEExpertOverlay_MultiAcceleratorTiers.cpp`
  - `realAcceleratorSkipReason()` had a strict `rocm:0 <-> rocm:1` P2P check. It has been removed in the current working tree.
  - The helper still checks `HAVE_ROCM`, `HAVE_RCCL`, and at least two ROCm devices.

- `src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.cpp`
  - Current prepared path loops over participants and calls `params.input->ensureOnDevice(participant.device)`.
  - That migrates the same tensor between ROCm devices and caused a GPU memory access fault once the P2P skip was removed.
  - A temporary full-device sync helper was added after the down projection; this is wrong and should be removed.

- `src/v2/execution/moe/MoEExpertOverlayLocalTPExecutor.h`
  - `MoEExpertOverlayLocalTPPreparedParticipant` currently has `batch_scratch`, `gate_scratch`, `up_scratch`, `partial_scratch`, and token vectors.
  - It does **not** yet have a participant-local input/hidden scratch, which is likely the right next addition.

- `src/v2/execution/compute_stages/stages/MoEExpertOverlayLocalTPStage.{h,cpp}`
  - Stage construction now owns/preallocates participant scratch and `prepared_partial_views`.
  - This is the right place to allocate any per-participant input/hidden scratch, not inside `execute()`.

## Current Bad Patch To Back Out

Remove the temporary full-device sync path in `MoEExpertOverlayLocalTPExecutor.cpp`:

- `#include "backends/BackendManager.h"`
- `synchronizeParticipantDevice(...)`
- The call to `synchronizeParticipantDevice(..., "prepared_down")` immediately after `fusedSwigluDownPrepared(...)`

That patch was added only while debugging the post-P2P-gate failure. It conflicts with the project pattern of stream/event synchronization and should not ship.

## Failing Evidence After Removing The P2P Gate

Command run:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
```

Observed sequence:

- `createLocalTPContext(..., CollectiveBackendType::RCCL)` succeeds.
- `RCCLCoordinator` initializes with two ROCm GPUs.
- The prepared path reaches `RocmLocalTPTensorParallelExpertsExecutesBothParticipantsAndMatchesReference`.
- First failure before the scratch preallocation fix: `ROCmFloatingPointGemmKernel::multiply_tensor] A and C must be on GPU`.
- After allocating participant scratch in the integration fixture, the next failure is a GPU memory access fault when moving from participant 0 to participant 1.

Important log evidence:

```text
LocalTPContext: Backend RCCL initialized for 2 devices
[MoELocalTP] prepared expert=2 participant=0 device=ROCm:0 tokens=2 gather
[MoELocalTP] prepared expert=2 participant=0 device=ROCm:0 down
TransferEngine::uploadFull Device migration: ROCm:0 -> ROCm:1
[MoELocalTP] prepared expert=2 participant=1 device=ROCm:1 tokens=2 gather
Memory access fault by GPU node-3 ... Page not present or supervisor privilege
```

Interpretation: the strict P2P gate hid a real LocalTP prepared-path bug. RCCL context creation is not blocked by missing direct P2P. The executor’s shared tensor migration and async stream ordering are the problem.

## Recommended Next Fix

1. Back out the temporary device-wide sync patch listed above.

2. Add participant-local input/hidden scratch to `MoEExpertOverlayLocalTPPreparedParticipant`, for example:

```cpp
std::shared_ptr<FP32Tensor> input_scratch;
```

Allocate it in `MoEExpertOverlayLocalTPStage` construction alongside `batch_scratch`, `gate_scratch`, `up_scratch`, and `partial_scratch`. Direct executor tests should allocate it in their fixtures.

3. Replace `params.input->ensureOnDevice(participant.device)` in the participant loop with an explicit per-participant copy/gather strategy that never migrates the shared input tensor while another device may still use it.

Preferred shape:

- Keep a source tensor resident on the continuation/primary device or host-visible compatibility path.
- For each participant, copy/gather into `participant.input_scratch` or directly gather into `participant.batch_scratch` on that participant’s explicit ROCm stream.
- Use stream/event synchronization to order:
  - input copy/gather completion before gate/up GEMMs,
  - gate/up completion before fused SwiGLU/down,
  - down completion before RCCL allreduce consumes `partial_scratch`,
  - allreduce completion before scatter-add on the primary/continuation output.

4. Avoid null/default streams. If any API currently hides a default stream, inspect the relevant kernel/backend implementation and plumb or select an explicit stream consistently with the existing ROCm kernel stream pattern.

Relevant implementation details:

- `ROCmMoEKernel::gatherTokenBatchFromTensors()` reads from `hidden->gpu_data_ptr()` and writes `batch_buffer->gpu_data_ptr()`, then transitions `batch_buffer` to `DEVICE_AUTHORITATIVE`.
- It does not call `ensureOnDevice()` internally and assumes caller has made both tensors resident on the right device.
- `ROCmFloatingPointGemmKernel::multiply_tensor()` requires both A and C to already have GPU pointers.
- `LocalTPContext::allreduce(...)` and RCCL backend paths should be checked for explicit stream/event behavior before adding any synchronization at the executor layer.

## Tests To Run After The Fix

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "V2_Unit_.*(MoEExpertOverlay|ExpertGemmRegistry|Qwen35MoEExpertOverlayGraph)" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_.*MoEExpertOverlay_MultiAcceleratorTiers" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen35MoEExpertOverlay_" --output-on-failure --parallel
```

Acceptance for this handoff continuation:

- No strict ROCm all-pairs P2P skip remains for Phase 5E/8A correctness.
- RCCL context creation failure may still be a valid hardware/build skip.
- The prepared LocalTP test executes both ROCm participants without memory faults.
- No full device synchronization is introduced.
- Stream/event ordering is explicit and documented in code or helper APIs.
- Phase 8A parity is evaluated from real execution or a valid non-P2P hardware/build/model precondition, not the old P2P gate.

## Do Not Forget

The current working tree contains many untracked Phase 5A-8A source and audit files from prior work. Do not assume `git diff` alone shows all relevant changes; include untracked files in review. Use `git status --short` before editing, and do not revert unrelated work.