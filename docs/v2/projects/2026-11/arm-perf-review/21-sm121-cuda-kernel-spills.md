# Idea 21 — CUDA GB10 (sm_120/sm_121) kernel register spills under the zero-spill guard

Status: **In progress** (2026-11-09 — operator GO; Main landing the fixed
sm_121 work on #43 as new commits, .79 compile-only, no demotions,
12.x-guarded, shipped SASS identical, spill-free 121 / 120;121 / 6-arch).
Prior state (2026-11-08 — the spill session fixed the CUDA register-pressure
issue; fix was the stacked upstream PR `Llaminar/llaminar`
**#44** (draft, base `feat/arm64-native-kernels`), linear tip **`09e14d3ae`**,
whose content equals the local merged tree `a5412af30`; spill kernel files
verified unchanged vs `be998fdec`). Remaining then: upstream merge + live SM120/
SM121 and joint-pool validation once the guard was clean. That #44 tip is NOT
in the 46c35f15b lineage; the fold lands on #43 per the operator GO.

## Finding (2026-11-08; evidence: K14-UMA-SM121-SPILLS-20261108)

Tables (mangled + demangled, store/load bytes):
`~/llaminar-spark-results/K14-UMA-SM121-SPILLS-20261108/{sm121,sm120}-spill-table.json`
(18 + 45 rows).

- **sm_121 (GB10):** 18 spilling instantiations (24–276 B).
- **sm_120 is worse:** 45 spilling instantiations (SASS-compatible target).

Canonical kernel classes (from the builder evidence):

| Class | Instance | Spill bytes |
|---|---|---|
| `CUDATinyFP32ProjectionKernels` | `sharedTinyProjectionKernel<bf16,8,float>` | 4 |
| `CUDATensorPrefillKernels` | `tensor_prefill_kernel<>` | 60–88 |
| `CUDAMoEGroupedPrefillKernels` | `groupImmaPairedGateUpSwiGluKernel<13,128>` / `<13,64>` | 44 / 24 |
| `CUDAMoEOverlayActivationPacketKernels` | | 128 / 156 |
| `CUDANativeVNNIPrefillKernels` | | |
| `CUDAPooledImmutableNativeKernels` | | |

Build environment: nvcc 13.0.88, aarch64, CUTLASS v4.2.1, NCCL capture-reentry
2.28.9 — all builder probes pass; the build lane is **closed as blocked by
register pressure**. Fix = CUDA kernel register tuning for sm_121 (60–88 B
tensor-prefill instances being the largest single family).

## Implications

- GB10 GPU support **and the live UMA joint-pool validation (idea 2)** are
  **blocked** on these register-pressure fixes: the `GpuSpillGuard.cmake`
  policy rejects spills in Release/Integration, and the plan requires the
  complete canonical target set plus SM120/SM121 admission before supporting
  the Spark GPU.
- Register-pressure work on the CUDA backend (kernel restructuring,
  launch-geometry/BOM changes), separate from the ARM-CPU items in this
  review.

## Validation

- Compile the complete canonical shipped target set plus SM121 with the guard
  intact; profile the named instantiations; zero spills expected; functional
  SM121 CUDA gates and capture/KV/MTP/state/economy runs after.
- Record the target set with the evidence (spill results are
  architecture-specific).

## Ownership/handover (2026-11-08 — spark-cuda-spil standing down)

- The CUDA spill-fix work is now owned by **Main/ARM** (per operator): he
  folds the sound parts into `feat/arm64-native-kernels` and closes PR #44.
- Canonical trace build SHA: `09e14d3ae` remains **provenance only** for the
  measured numbers; the canonical SHA for future traces = the migrated
  ARM-branch SHA, announced by Main when landed.
- **Tensor-split spec rev 2 (`83c61732d`)**: open design candidate, pending
  sign-off from arm-perf + arm-reviewer (01a1a8d9).

- Tensor-split spec rev 2 (83c61732d): **signed off by arm-perf (2026-11-08)**
  with conditions — §5 exactness proven by the mandated byte-compare +
  boundary sweep (not argument alone); fragment/lane mapping verified against
  mmaFragmentRow/Column before implementation; measured per-target values
  replace every estimate; consumer peak > 170 → stop+report (no silent 4-way
  fallback); add unprofiled before/after Release timing (one barrier per tile
  — report any slowdown with attribution); graph-capture identity includes
  the variant/generation. Implementation + measurements owned by Main/ARM
  after the branch move; arm-reviewer checks the implementation diff.

## Moved into PR #43 (2026-11-08) — most spills fixed

PR #43 = **7e709d60e** (ff from 0868a5573) now contains the CUDA commits from
closed PR #44 (branch deleted):
- a62696317 fix(cuda): sm_120/121 register budgets — SASS identical on the
  shipped archs; 12.x spills before→after: single-row dispatch 276→0, paired
  gate/up 44→0, owner phase 44→0, tiny 4→0, BK64 76→0 (inlining side effect,
  source unchanged), **BK256 328→8**.
- 7e709d60e docs: CUDA_SPILL_TENSOR_SPLIT_SPEC.md rev 2 (with the arm-perf
  conditions).

STILL OPEN on 12.x: BK256 non-ordered variants at 4–8 bytes (per-K A loads
reverted — didn't help), and `tensor_prefill` (the signed-off split spec).
Evidence: `~/llaminar-spark-results/K14-CUDA-SPILL-MIGRATION-a62696317`.

## REOPENED/INCOMPLETE (2026-11-09, arm-sparks) — fold partial for sm_121

- Building Release CUDA arch 121 at 7e709d60e SPILLS: ptxas 'Registers are
  spilled' in CUDANativeVNNIPrefillKernels.cu nativeVnniTC_BK64<...>
  (h0/h4/h24, 256-wide staging) + CUDAFlashAttentionKernels.cu (code=255,
  build stops).
- Evidence: 7e709d60e's CUDANativeVNNIPrefillKernels.cu differs from the
  spill-verified bd9dcd12a (17+/4-; TinyProjectionSharedKernel.cuh 24+/4-) —
  the fold took a DIFFERENT/PARTIAL version; bd9dcd12a's BK64/BK256
  unroll-roll fix is NOT fully present upstream.
- RECOMMENDED: fold bd9dcd12a's kernel content (or remaining rolls) into #43
  (that version probes 0 spills at sm_120+121). Current arm branch
  (302318f1e) contains 7e709d60e → Release CUDA sm_121 builds fail there
  until fixed. Idea-25 trace substitution: cff351901 (523230ab0 + bd9dcd12a;
  U2✓ U5✓) sealed.

## Main's response (2026-11-09) — bd9dcd12a REJECTED as a fix

- bd9dcd12a = the rejected pre-review state: contains exactly the register-array
  DEMOTIONS excluded on purpose (tensor_prefill P/V n-loop unroll 1 over the
  128-float accumulator; BK64 wj-loop unroll 1 indexing acc[wi][wj]; BK256
  non-ordered k-loop unroll 1 indexing A_frag_all/sa_all) — unconditional on
  every architecture. Rolling a loop that indexes a register array moves it to
  LOCAL MEMORY: ptxas reports 'stack frame', not spill stores → the spill guard
  passes while the hottest accumulators live in lmem. '0 spills' on bd9dcd12a
  = that artifact; it would also change non-12.x targets.
- Evidence requested from arm-sparks (on 7e709d60e or 302318f1e):
  1) exact ptxas 'Function properties'/'Used N registers' lines for the
     spilling BK64 instantiations (h0/h4/h24 256-staging) + CUDAFlashAttention
     at sm_121; 2) configure line (build type, CMAKE_CUDA_ARCHITECTURES,
     flags, image). Main's A/B (uma3, 80;86;89;90;120;121) showed 744 BK64
     instantiations at 0 spills on 120/121 → variant set or flags differ;
     CUDAFlashAttentionKernels.cu was NOT in the probed object set.
- Expected remaining 12.x list (fixed WITHOUT register-array demotion):
  BK256 tail + tensor_prefill + FlashAttention (+ BK64 if confirmed).
- Idea-25 qualifier: cff351901 runs the demoted kernels — graph/launch
  STRUCTURE only; kernel timings NOT representative of #43 (sealed that way).

## Discrepancy RESOLVED + open set corrected (2026-11-09, Main)

- Not split on/off: Main's A/B also used --split-compile=8 on
  CUDANativeVNNIPrefillKernels.cu (canonical CMake sets it for that TU;
  FlashAttention is --split-compile=0). REAL difference = ARCHITECTURE SET
  (Main: 80;86;89;90;120;121 vs arm-sparks: 121-only): under split-compile the
  instantiation partition (inlining + per-split-module register allocation)
  depends on that set → BK64 '0 spills' in the A/B was a PARTITION ARTIFACT
  ('module-level inlining'), not a code property. No A/B needed.
- OPEN sm_121 set (real, for GB10 configs 121 or 120;121 — image default
  80;86;89;90 has no 12.x): BK64 (h0/h4/h5/h24, 4–68 B), BK256 (≤16 B),
  FlashAttention flash_attention_2_pipelined (44–108 B @128–255 regs),
  tensor_prefill.
- FIX REQUIREMENT: spill-free under EVERY partition — verified at 121-only,
  120;121, AND the full 6-arch set, with split-compile as configured.
- Fix awaiting the operator's go-ahead (CUDA lane).

## 2026-11-09 — 425002104 has the reviewed spill state (Main)

- a62696317 IS in the head as eb7243a58 (byte-identical diff after token
  map; pos 16). 7e709d60e = tensor-prefill split design doc only (170b8c6af).
  cff351901 (bd9dcd12a demotions) = REJECTED merge, correctly excluded.
- Open set unchanged: BK64 4–68B; BK256 ≤16B; FA2 pipelined 44–108B;
  tensor_prefill per spec rev 2 (5 conditions). Waiting on OPERATOR's
  go-ahead for the fixed lane (no demotion, 12.x-guarded, shipped SASS
  identical, spill-free 121 / 120;121 / 6-arch; new commits on #43).
- Qwen3.8-UD per-bucket trace (idea 25 Dir B) waits on the same decision +
  the GB10 CUDA build; arm-sparks standby (.79 reserved), no probe.

## Root cause found (2026-11-09, Main, spill lane)

- nvcc --split-compile is NONDETERMINISTIC: identical compile of
  CUDANativeQuantisedPrefillKernels.cu for 120;121 gave 4 / 30 / 56 spilling
  functions across runs; each 12.x target lands in 121- or 128-reg
  allocation depending on partition; shipped 80;86;89;90 SASS differs
  between identical split compiles (2 hashes in 3 runs). Explains host-to-
  host BK64 disagreement.
- WITHOUT split-compile: deterministic; BK64 clean on both 12.x targets;
  only the known BK256 pair left. FlashAttention 12.x spills are
  DETERMINISTIC real spills (18 kernels per target without split).
- Fix plan upcoming (direction: avoid split-compile for 12.x or make it
  deterministic). Main confirmed it will take R1 (5d4e3716a) after the
  current spill checkpoint; handover doc =
  /Users/floriankinder/repo/arm-addon-notes/r1-handover.md.

## Fix plan (2026-11-09, Main) — architectural

- REMOVE nvcc --split-compile everywhere; SHARD instead (maxim 34; spill
  guard needs deterministic evidence; preflight test will BAN --split-compile
  on spill-guarded targets). Prefill (d641463bd): 8 codebook-pair shards +
  host-policy header + dispatcher (public ABI + single launch-policy state)
  → deterministic, BK64 spill-free at 120/121/120;121/6-arch. tensor_prefill
  (fbacc8409): 12.x opaque per-step PV address; 255 regs w/o spill on 6
  geometries; shipped SASS byte-identical; producer/consumer split spec
  (170b8c6af) RETIRED. BK256 (5c90b4340): 12.x per-K-step fragment loads
  (pure load schedule; same k/wj/wi order). NEXT: FA (7 shards: prefill/
  context × FP32/FP16/BF16 + decode/utility) then real fix for the 18
  deterministic HD256/1-PV-warp/context-partial spills (255 regs, 16–100 B;
  likely hoisted V-row/score addressing in j-loop). NOTHING pushed until FA
  + policy test land. GB10 runtime gates after (parity/byte, prefill timing,
  tracked CUDA tests incl. RoPEGraphCaptureNoH2D, Qwen3.8 trace, spill
  re-check). Sequencing: ideas-7+25 FF (#43 5d4e3716a → 912d0107) BEFORE the
  CUDA push recommended; CUDA commits stack on the merged tip.

- Sequencing AGREED (Main): ideas-7+25 FIRST (#43 FF 5d4e3716a → 912d0107 on
  WSL (5'') green); CUDA stack rebased locally onto 912d0107 before pushing;
  GB10 gates once on the merged tip. Tensor-split spec retired in the CUDA
  stack (doc removed; history keeps; commit names fbacc8409). SHA sent when
  the full slice is pushed.
