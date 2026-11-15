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
| Routed expert gate/up and down | Multi-row MTP enters the fixed-topology grouped transaction in `MoEExpertComputeStage::executeDecodeEquivalentVerifierPrefill`. Native projection candidates and M512 population policies exist; those policies do not admit ordinary verifier capacities. Sparse RouteOwned execution remains a distinct Dot4 producer. | Reuse and qualify native leaves for actual verifier routes, including sparse groups; install measured selection. The prefill-named interface alone does not prove MTP coverage. |
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

## First implementation slice

Start with device-counted dense NativeQuantised standalone and fused projection
APIs in the actual decode-equivalent verifier scope. The existing
`ROCmNativeQuantisedWmma.hip.inc` already accepts serial-M1 partition counts and
preserves each K32 commit and ordered FP32 fold. The missing authority is row
ownership: `ROCmNativeQuantisedWmmaLaunch.h` has static M/N/K and serial partitions,
but no `DeviceRowRange`/`ProjectionWork`. Reuse that arithmetic while adding the
existing device-owned row contract; launching over allocated capacity would
overwrite inactive rows and does not qualify a verifier producer.

Extend `tests/v2/integration/kernels/rocm/Test__ROCmDeviceCountedVerifierRows.cpp`.
It already captures public verifier APIs, varies prefix counts, compares raw
bytes against independently executed production M1 rows, and checks poisoned
inactive outputs. Cover standalone projection, homogeneous and mixed fused
bundles, and fused SwiGLU/down through one retained capacity31 graph. Exercise
full, small, empty and regrown work, every live row count 2–16, and 17/31. Its
current `deviceCounted(M,count)` fixture does not establish nonzero origins or
request-batched ownership; add those separately. Enumerate the source-format
registry and promoted aliases. Replace Dot4-specific grid certificates with
candidate-specific WMMA identity and contribution checks.

The first economy panel can reuse that fixture's Qwen3.8 IQ4_XS geometries:
N17408/K5120, N5120/K17408 and N10240/K5120, with capacities3/16 and live2/3/16.
Compare complete captured operations, including preparation and epilogues,
against the current grouped producer with31 paired samples. Existing seven-sample
timing is exploratory evidence. The model-loading `V2_Perf_DenseVerifierRows_ROCm`
M2/3/4 gate follows component qualification. Register the focused native replay
explicitly in production preflight before treating this slice as complete.

The standalone verifier epilogue also needs a focused regression. Its current
native grouped admission requires beta zero and unmapped output. Nonzero beta
or mapped output enters a host row loop in a fully active verifier scope;
with an explicit multi-row `DeviceRowRange`, its first one-row quantizer instead
rejects the mismatched physical geometry. Neither behavior certifies grouped
execution. Implement the count-aware grouped combine using persistent workspace
and the original row authority, and test alpha, beta and bias against independent
serial rows while preserving inactive output bytes. Existing M1 alpha/beta tests
and fused per-column bias tests do not cover this defect. New WMMA candidate
admission must never silently enter that row loop.

Implementation seams are the standalone, homogeneous-batched and mixed-batched
`rocmLaunch_native_quantised_small_m_*` functions in
`ROCmGemvKernel_native_quantised.hip`, declared through
`ROCmGroupedVerifierLaunch.h`. Use `projectionWorkForLaunch(M, row_range)` and
the existing compile-time work dispatch. Cursor bounds name original physical
rows, including request holes and nonzero origins; they are not compacted row
counts. Preserve the source-policy serial partition selection already resolved
by those adapters, including original unsliced output widths for fused bundles.

Add a typed verifier candidate scope separate from the prefill override, and
reject incompatible forcing before recording graph nodes. The current WMMA
body uses four waves and `RM=BM/32`; an M16 enum alone would create zero row
fragments. An economical M16 candidate needs an explicit two-wave mapping while
sharing decoder and commit helpers. The existing combine epilogue also needs
the enclosing verifier prefix forwarded explicitly: its current
`dispatchProjectionRows(M, nullptr)` only inherits `ProjectionRowWork`.

The smallest epilogue regression is Q4_0 M3/N17/K32 with nonzero initial output,
alpha, beta and bias, replaying live3,1,0,2 through one captured graph. Add mapped
output, independent production M1 byte oracles, inactive-byte preservation and
actual grouped-symbol/launch-count checks. Register it explicitly in production
preflight; do not infer this coverage from the larger source-format sweep.

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
