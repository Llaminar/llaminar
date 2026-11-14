# gfx1100 WMMA for grouped MTP decode

Status: added to the active WMMA work list on 2026-11-14 at the user's request.
This implementation plan extends the [prefill campaign](GFX1100_PREFILL_WMMA.md)
and the [MTP project plan](../2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md).
It records source-audit findings and pending work, not a performance certificate.

## Existing execution and missing coverage

Grouped execution already exists. The missing work is native matrix instruction
producers and measured production selection for grouped decode, with its stricter
serial-row byte contract. Ordinary prefill evidence does not qualify MTP.

| Operation | Current ROCm execution | WMMA work required |
|---|---|---|
| Dense quantized projections | `ROCmQuantisedGemmKernel.cpp` enters `rocmGemv_native_quantised_small_m_*_policy` in the decode-equivalent verifier scope, before ordinary prefill dispatch. `ROCmGemvKernel_native_quantised.hip` uses bounded shared-weight row tiles and ordered dot products. | Implement a native gfx1100 candidate preserving serial arithmetic for every physical decoder and source alias; cover standalone and fused projection APIs. |
| Routed expert gate/up and down | Multi-row MTP enters the fixed-topology grouped transaction in `MoEExpertComputeStage::executeDecodeEquivalentVerifierPrefill`. `ROCmMoEGroupedWmma.hip` already has native projection candidates, but `selectedMoEGroupedPrefillPolicy` maps Automatic to OrderedDot4. Sparse RouteOwned execution remains a distinct Dot4 producer. | Reuse and qualify native leaves for actual verifier routes, including sparse groups; install measured selection. The prefill-named interface alone does not prove MTP coverage. |
| Floating grouped projections | `ROCmFloatingPointGemmKernel.cpp` selects `rocmFp16_batched_projection`, `rocmFp32_small_n_batched_projection`, or `rocmFp32x16_batched_projection`, preserving the serial fixed reduction tree. | Evaluate matrix candidates under each declared arithmetic contract. Half storage does not authorize changing FP32 operands or reduction order. |
| Q8 router | Native WMMA candidates exist, but Automatic admits only whole-row capacity512/K2560/E512. | Qualify verifier row counts and ownership, both router bases and reuse of the router's original Q8 publication. |
| Verifier attention | The installed adaptive WMMA policy admits exact M512 prefill geometries. | Audit the actual verifier dispatch and qualify matrix use separately, preserving serial attention bytes and fixed-query scaling through 64K KV. |

The dense native decoder switch currently contains 18 physical cases. Qualification
must enumerate the repository's full source-format inventory, including normalized
aliases and promoted descriptors; the physical switch count is not source coverage.

CUDA provides a useful existing implementation pattern:
`CUDANativeQuantisedGemvShardImpl.cu.inc` selects
`GeneratedGroupedKernel::TensorCoreMma16` through installed grouped tuning, then
launches `launchTensorCoreSmallMGenerated`. Its `verifierMmaM16N8K32` helper uses
signed INT8 `mma.sync.aligned.m16n8k32`. Preserve the ROCm serial arithmetic
contract when adapting operand staging, integer accumulation and epilogues.

## Implementation work list

- [ ] Freeze a source-owned operation/format/geometry inventory and baseline the
  actual captured ROCm verifier path. Include dense QKV, attention output,
  gate/up, SwiGLU/down, shared experts, GDN projections and selected LM-head rows;
  preserve the existing target/bonus-row ownership of terminal logits.
- [ ] Implement bounded WMMA row tiles for dense NativeQuantised grouped decode,
  including fused APIs. Reuse immutable weight decoding, scale/minimum helpers
  and fixed partition trees where their arithmetic contracts agree. Keep each
  row's FP32 commits in the same order as production serial M=1.
- [ ] Qualify routed MoE WMMA through the actual grouped verifier transaction,
  including static-owner assignment, expert-local live counts, empty and skewed
  participants, compact/promoted descriptors, and sliced down projections.
  Reuse ongoing grouped-prefill kernel work without treating prefill qualification
  as a verifier certificate or introducing prefill LLEP assignment into decode.
- [ ] Evaluate floating grouped projections, router and verifier attention as
  separate candidates. Record a precise unsupported result where an instruction
  cannot satisfy the arithmetic contract; never relax byte equality to admit it.
- [ ] Measure complete captured operations against the current operation-specific
  grouped implementation and serial throughput. Install work-size-aware Dot/WMMA
  selection only from qualified measurements, then verify actual Automatic hits.
  Dot4 remains a first-class selected implementation wherever it wins.
- [ ] Add focused native replay and dispatch regressions, explicitly registered
  in `ProductionTestPreflight` and the applicable grouped-verifier feature suite.
  Run the focused and feature suites requested by the user; this work item does
  not claim full-preflight or model-level certification from component results.

## Required acceptance evidence

1. **Rows and formats.** Use production M=1 as the independent serial oracle;
   sweep every integer M=2..16 and M31 for all supported source formats. Extend
   the contiguous sweep through any larger admitted graph capacity. The current
   single-request controller supports 15 drafts plus the target/anchor input row,
   yielding at most16 logical verifier rows; that is not a universal limit for
   runtime or request-batched row APIs. Exercise mixed
   request lengths, nonzero row origins, zero/partial live work and tile tails.
2. **Arithmetic and lifecycle.** Require native-byte equality against serial
   decode, fixed partition/reduction order, correct fused epilogues and untouched
   inactive rows. Retain captured graphs across dirty replay, full/small/empty
   transitions, reset and regrowth. For integration, verify accepted-state
   publication and continuation under greedy/stochastic and dynamic-depth policy.
3. **Actual production selection.** Enter the common verifier scope and existing
   device-owned row APIs. Authenticate captured symbols, engine counters and
   branch contributions, including deliberate omission controls. Exercise dense
   and MoE participants, TP column slices, replicated outputs and declared PP
   boundaries. No host live-count readback, recapture, allocation or row replay.
4. **Economy.** Time complete retained operations in Release, with canonical
   unprofiled paired samples and unchanged inputs/routes/accepted work. Measure
   live rows as well as capacity, N/K, codebook and route occupancy. Never infer
   a shallow-verifier win from M512 timing or count padded rows as useful TOPS.
   Where a matched RTX3090 comparison applies, report useful TOPS divided by
   each device's relevant theoretical peak; latency remains the dispatch criterion.
5. **Profiles and resources.** Bind each isolated profiler launch to its exact
   timing producer, shape and path. Report ISA, launch geometry, registers,
   occupancy evidence, LDS, memory activity and throughput; require zero memory
   spills/private scratch. Build the complete shipping architecture set, currently
   gfx906 and gfx1100, with the existing spill guards intact.
6. **Total dispatch.** Bind selection to hardware, arithmetic/source identity,
   geometry, ownership, execution phase and capture policy. Handle positive runtime
   M beyond measured points with the established complete policy. Device-owned
   live data may select branches inside a retained kernel; it must not become
   host scheduling state. Authenticate installed Automatic with fresh timing
   and replay evidence instead of relabeling forced-candidate observations.

Current result: the source audit establishes the gap. No new MTP WMMA producer,
installed policy or MTP speedup has been qualified by this work-list update.
