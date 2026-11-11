# QFN prerequisite recovery — 2026-11-11

The active target is QFN `PP(EP(4×MI50), EP(4×MI50))`, with ngrams
sharded in VRAM across all eight GPUs on their original MPI owner. The complete
hardware-eligible CTest inventory passed. The user requested checkpointing this
passing slice and skipping the redundant full rerun before proceeding with
real-model numerical diagnosis and performance work.
This is an investigation record, not a tuning dashboard.

## Current qualification boundary

The repaired tree completed every selected test in q39/run-5 with **zero
failures and zero skips**:

| Lane | Passed |
|---|---:|
| Unit | 1,030/1,030 |
| Host preflight | 2,471/2,471 |
| CUDA native preflight | 277/277 |
| ROCm native preflight | 390/390 |
| Exclusive multi-device preflight | 1,088/1,088 |
| **Total executed** | **5,256/5,256** |

The 4,265 registered preflight cases selected 4,226 hardware-eligible cases;
39 unavailable geometries remain explicitly unqualified. The selected cases
include all 24 native row formats on each GPU backend, homogeneous and
cross-vendor transfers, original cross-rank transfers, and four/eight-MI50
layouts. The EP4/EP4 preparation/lifecycle matrix passed all 32 combinations of
ordinal/random ownership, static/dynamic expert balancing and MTP off/1/3/dynamic.
This is model-free evidence; the exact eight-MI50 real-model numerical and
throughput results remain pending.

A chat interruption retired q39's parent runner and driver observer while its
independently owned final CTest continued to completion. Consequently q39 has
complete passing CTest logs/JUnit but **no canonical prerequisite receipt and
no continuous driver certificate for the whole run**. The post-interruption
tail observer closed completely with zero findings; it does not fill the gap.
The earlier focused 56-case native repair batch also retains its complete clean
driver window.

The unchanged-source q40/run-6 rerun found no build work and passed Unit again.
The user then explicitly requested skipping that repeat and checkpointing the
passing slice. Its current CTest was interrupted; the canonical runner retired
normally with return code 254 and its external driver window closed cleanly.
That canceled report is not a passing receipt. No report was reconstructed,
no failure was relabeled, and no hardware or numerical guard was weakened.

The final checkpoint review found a packaging omission: the ROCr installer
required the deferred-SDMA patch and proof directory, but the builder Dockerfile
did not copy them. Both inputs are now copied at the existing installer boundary.
The focused Unit and `ProductionTestPreflight` contract registrations passed
**2/2**, including an original-missing-COPY negative control. No Docker image
rebuild was performed for this correction; inference code is unchanged.

## Faults and repairs

The earlier full transaction executed 5,203 tests: 5,129 passed, 74 failed,
and none skipped. All failures were classified; multiple aliases covered the
same defects. Twelve repair groups addressed the failures:

| Area | Repair and preserved invariant |
|---|---|
| Admission fixtures | Supply complete execution, storage, CPU and directed-peer facts; use exact declared compute-stage buffer bytes. Admission rejection remains strict. |
| Candidate fixtures | Correct original participant identity and synthetic topology geometry. Do not fabricate devices or ownership. |
| Parity and controller fixtures | Supply the synthetic KLD namespace, deferred observed declarations and legacy trace annotation. Numerical limits and controller decisions are unchanged. |
| Source policies | Correct narrow terminal-fence and physical-lane source expectations while retaining negative controls and hot-path bans. |
| CPU MPI request ingress | Distinguish prefill token ingress from decode ingress, retaining original MPI reset, partial-work and terminal checks. |
| Native cohort selection | Resolve the complete ordered backend demand together through observed peer grouping; retain original rank, NUMA and UUID ownership. |
| Retained-prefill ordering | Order fresh request uploads and reusable-bank poison/reset writes by their actual producer event. Each retained bucket stream acquires that event. Existing byte, KV, poison and capture-reuse oracles remain. |
| No-peer transfer coverage | Qualify the no-directed-peer case against actual hardware. Enabled peer access cannot serve as evidence for an absent-peer premise. |
| Controller observations | Borrow the actual bound controller cohort for terminal observations instead of independently selecting another eligible singleton. |
| Planning diagnostic volume | Move eight repeated geometry messages from DEBUG to TRACE. Keep the full inventory, candidate search, measurements and 30-second deadline. |
| Published capacity ownership | Stop reading the moved-from capacity envelope after publishing it. Quotas and later observations borrow the immutable published admission. |
| Native format lifetimes | Give each compiled codec its own original process and price the actual retained graph definitions. Keep all formats, repeated execution, exact bytes, live transfer extents and retirement checks. |

The production capacity defect was in
`OrchestrationRunner::freezeMoEExpertOverlayPlanForLoadedModel`: it moved the
enriched envelope into `moe_expert_overlay_memory_admission_` and then passed
the moved-from value to quota installation. Later footprint and telemetry
reads used that value too. The temporary now ends at publication, and all
downstream readers use the immutable published owner. Actual Hosted
initialize/reset/reuse regressions pass on both GPU backends, alongside the
Native controls. The capacity guard was not weakened.

Follow-up fixture corrections preserve the rank-indexed inventory contract:
reversing the rank roster is a rejection control, while permuting GPU
publication within a rank remains legal. Extending synthetic visibility also
extends the complete synthetic peer matrix.

## Per-format native process registration

Retiring codec graph/context resources does not permit a new independent source
authority within the same original process. Parameterized GTest cases now use
the existing canonical codec list, and each case owns one process. The test
entrypoint reuses the existing production native NUMA bootstrap; external-MPI
controls retain their previous launch path.

Putting 24 cold processes under one aggregate 120-second deadline was too slow.
The shared compiled-test discovery mechanism now emits one guarded CTest case
per compiled codec. It neither duplicates the format table nor extends the
per-process deadline. It rejects missing, duplicate, malformed or asymmetric
backend populations, and keeps the original environment, process guardian,
exclusive scheduling and skip/empty-selection rejection.

Before the native executable is built, a selectable preflight-only sentinel
reports the missing discovery. Unit-only inventory and execution still work.
Successful discovery replaces that sentinel. Build-gate dependencies retain
the native executable, original process helper and required hardware exporter.

The singleton bootstrap cases declare their actual hardware demand: two GPUs
of one backend on one observed NUMA node. They do not declare an external MPI
width. Their runtime stage split remains `{1,1}`. The four/eight-device controls
retain their external MPI, ISA and stage-geometry declarations. A regression
feeds generated CTest metadata through the actual preflight consumer to check
this distinction, rather than checking labels alone.

`V2_Unit_NativeCodecDiscovery` and its explicitly registered preflight counterpart
cover generated commands, original ownership, literal paths, Release owner
selection, missing discovery and canonical hardware requirements. Existing
parity/source-discovery regressions remain green.

## Repetition and driver evidence

ROCm retained-prefill ordering passed **20/20 fresh-process replays**.
The complete ROCm planning case also passed **20/20** with the original
ten-GPU inventory. Every repetition proposed and evaluated 4,915 candidates,
with zero capacity rejections. Times were **28.51–29.69 seconds**, median
**28.865 seconds**, against the unchanged 30-second deadline. Its timing margin
is narrow; these observations do not establish a matched old-code speedup.

The earlier full gate lost continuous kernel-log coverage during a flood of
explicitly enabled amdgpu GPU-VM mapping diagnostics. Seven verbose callsites
were disabled after process retirement, with exact before/after control
records. Fault logging and fail-closed cursor checks were retained. Later clean
windows cannot fill that earlier gap or certify a future gate.

## Evidence and next step

The source checkout is `/tmp/qfn-qualified-checkpoint-r127/source`, branch
`feat/qfn-ngram-continuation`, based on `2948b1e49edc9c834b7d815d9194ce3f11dd1f2a`.
Integration builds retain CUDA 80/86/89/90 and HIP gfx906/gfx1100 qualification.
Local evidence is under `/tmp/qfn-full-preflight-after-cpu-next/`:

- `run-4/`: original full failure collection; not a passing receipt.
- `q25-*`, `q27-*`: retained-prefill and planning repetition evidence.
- `q32-complete-unit.xml`: complete 1,029-test Unit result.
- `q34-*`, `q37-*`: discovery, build ownership and canonical-consumer checks.
- `q38-native-formats.xml` and `q38-native-formats-scope/`: final 56-case native
  result, process retirement and complete clean driver window.
- `q38-native-formats-hardware-*.json`: fresh observed hardware qualification.
- `run-5/{unit,integration-host,integration-cuda-socket,integration-rocm-socket,integration-exclusive}.xml`:
  the five complete passing CTest lanes, 5,256 executed cases and zero skips.
- `q39-full-prerequisites-scope/interrupted.driver-diagnostics.json`: the
  incomplete original driver window, preserved as incomplete.
- `/tmp/qfn-interrupted-q39-recovery/tail-driver-scope/`: complete clean
  post-interruption observation, 208,918 records and zero findings.
- `run-6/` and `q40-full-prerequisites-scope/`: user-canceled repeat and its
  orderly, clean driver closure; not a replacement qualification receipt.
- `/tmp/qfn-green-slice-checkpoint-20261111/packaging-regressions.xml`: the two
  focused packaging/proof contract registrations after the final COPY repair.

The original failure-to-repair map is in
`/tmp/qfn-q17-failure-map-private/failure-to-repair.json`. Detailed patches and
intermediate failure evidence remain task-local rather than becoming dashboard
history. The user accepted the completed passing CTest slice for this checkpoint
and requested proceeding without another full repeat. Next run the exact-cell
numerical diagnosis, paired panel and matched Release benchmark with their
actual evidence boundaries recorded. This checkpoint is not an image or release
certificate, and the absent whole-run receipt must not be invented.
