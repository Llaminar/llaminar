# Plan first: native NEON FP16 projection, exact 256-lane contract

2026-11-09 — PERFORMANCE lane, SPARKS/X925 target. DRAFT; no production code
authorized/changed. Source analysis and fresh GCC14 TU compilation use
detached 39c24d8b4. Final implementation base re-confirmed after R2/C4 and
the main-lane guards: 777f7a52a (on f69446e3c). arm-perf confirms FP16
arithmetic unchanged by that WIP, pending final diff verification.

## Findings / scope
No vfmlal/fmlalq source occurrence exists under src/v2 at this checkout.
Fresh production TUs compiled sequentially with existing-container GCC14.2,
one physical CPU, no GPU/network. Flags from build_v2_dgx/build.ninja
(Integration -O3 -g + NEON_I8MM); second diagnostic = -mtune=generic
-march=armv9-a+sve2+i8mm+bf16. Neither emits FMLAL/FMLAL2 in
CPURMSNormKernelT.cpp, CPUSoftmaxKernelT.cpp or KernelFactory.cpp.
Only actual half x half candidate = explicit FP16RNE floating projection
(GDN selective alpha/beta policies), instantiated by KernelFactory.cpp from
cpu/gemm/FloatingPointGemmKernel.h. Exact-tree dispatcher: Scalar/AVX2/AVX512,
NO NEON case; active NEON enters default scalar program. Two FP16RNE
dotRows<4> bodies (FP32 A rounded via binary16; physical-half A) each emit
one scalar FMADD, no vector FMLA, no FCVT/FCVTL: half conversion = integer
bit arithmetic. This is a missing native vector implementation, not just
missed compiler fusion. Fresh armv9 compilation does not resolve it.

Closed non-candidates:
- Attention dot: FP32 query x half key; value accumulate: FP32 prob x half V.
  FMLAL would require narrowing an originally FP32 operand (bytes).
- T5 RMSNorm: t5SquaredSums takes OriginalRows<Storage::FP32,Count>; normalize
  rejects non-FP32 T5FP32; FP16 RMSNorm uses FP64 squaredSums (retain).
- Softmax: FP32 polynomial intermediates after half storage decode.

## Existing exact contract — no changed arithmetic
Canonical: FloatingExpertNumericalContract.h (dotRows, reducePartials);
DeviceFP32NumericalContract.h (fusedMultiplyAdd, add, persistRounded);
FloatingPointGemmKernel.h (roundFloatingOperandHalfRNE, executeHalfOperands,
run_gpu_aligned_expert_fp32_matmul_isa, completeSkinnyMatmulValue).
For each output and row:
1. Init exactly256 logical FP32 lane accumulators to +0.
2. Logical lane l consumes K indices l, l+256, l+512, ... in order; update =
   one FP32 FMA with explicit rounding/dependency edge.
3. Reduce stride128,64,32,16,8,4,2,1: partial[l] = rounded_add(partial[l],
   partial[l+stride]); no alternative horizontal-add order.
4. Preserve alpha/beta/bias completion, read-before-write rules, row
   selection/offset, HalfRNE-specific completion verbatim.
Current scalar already implements this exact contract. New NEON must equal
BOTH an independently constructed contract-ordered oracle AND the current
scalar. No order change for SIMD.

## Variant domain (not permission to narrow)
HalfRNE=false, HalfStorage=false: original unrestricted FP32 -> native
FP32 x widened-FP16 FMA (NOT FMLAL).
HalfRNE=true, HalfStorage=false: original FP32 rounded binary16 RNE, then
widened -> FMLAL candidate after exactly matching that rounding.
HalfRNE=true, HalfStorage=true: original stored binary16 -> FMLAL candidate;
no extra rounding.
HalfRNE=false, HalfStorage=true: original stored binary16 -> half-product
candidate if instantiated; preserve its distinct epilogue.
Other floating weight types stay first-class; do not reinterpret BF16/FP32
as FP16. Explicit-half policy NOT broadened to quantized sources, attention
or RMSNorm.

## Identity proof and limits
For finite original binary16 a,b: <=11 significand bits each; exact product
<=22 bits, fits binary32 precision24; smallest nonzero product 2^-48 is
normal FP32. Thus exact_widen(a)*exact_widen(b) has the SAME exact real
product as a half-widening multiply. With the SAME lane accumulator, one
round-to-nearest FP32 fused add produces identical result bits. Fixed
per-lane K order + fixed pairwise add tree preserve identity inductively
(NO associative reduction claim).
Limits: does NOT authenticate NaN/signaling/payload propagation, Inf x 0,
signed zero, FPCR rounding/flush or half-subnormal treatment. Must match
the admitted runtime contract and oracle. Do NOT: change FPCR globally,
introduce unproved finite-only restriction, canonicalize NaNs, or scalar
retry if the intrinsic differs. Any supported-domain byte mismatch rejects
the candidate until corrected or the documented admission contract proves
the input unsupported.
Activation conversion must match roundFloatingOperandHalfRNE (subnormal
ties + NaN). neon_fp16::narrow NOT interchangeable (truncates subnormals
for FP16Utils); explicit HalfRNE = full RNE. HW FCVTN permissible ONLY
after proving rounding/FPCR/exception words; else implement the existing
RNE bit program in SIMD once per consumed activation fragment. Preserve
source-half words without a second conversion.

## Proposed native schedule (subject to .80 measurement approval)
- Explicit NEON exact-tree implementation (not a special call in the scalar
  default). Preserve shared row selection, team ownership, completion.
- Bounded tile of logical lanes (candidate 16 lanes = 4 FP32 vectors/row);
  visit tile's K indices with stride256 so every accumulator has the
  original sequence. Up to 4 rows = 16 accumulator registers; weight-half
  fragments shared; activation fragments short-lived; no whole-K panels.
- FMLAL/FMLAL2 low/high only where A AND B are both half; mixed native =
  widened B + FP32 FMA.
- Materialize each completed logical lane exactly once into the fixed-256
  reduction storage, then the unchanged explicit binary tree with NEON adds.
  This intentional contract partial bank is distinct from a compiler spill;
  bounded per-invocation/per-worker storage per existing authority. No heap
  allocation/repacking/host shadow.
- Measure lane tiles 8/16 separately if register evidence requires;
  physical tiling must not change the logical tree. K<256 / K%256 tails
  bounded; no out-of-range loads; +0 inactive lanes.
- FMLAL NOT authorized by the current NEON_I8MM envelope. Keep public
  baseline unchanged. Minimal function/TU-local FP16FML target needs native
  ASIMDFHM authentication (+ complete emitted instruction envelope) via
  ARMCPUCapabilities; existing FP16Arithmetic only authenticates
  FPHP+ASIMDHP (insufficient alone). Darwin optional feature discovery needs
  a separately verified OS key before claiming support.
- A fully qualified ordinary NEON widen+FP32-FMA implementation remains an
  independently certified implementation where its envelope applies;
  variant selection is explicit first-class dispatch, never fallback after
  a failed FMLAL launch or correctness check. Runtime-default selection =
  separately gated. No broad armv9 project envelope or M1 baseline change.

## Evidence / acceptance gates
1. Await arm-sparks .80 shipped-object disasm + identity; verify the finding
   there. Use the existing Perf__ production-shaped half projection harness
   (owner discovered; currently NONE exists -> test-only extension of
   Perf__CPUFloatingMoEExpert with explicit FP16RNE instantiation, pending
   R2 confirmation), not a private proxy. .80 scope: GCC14, exact
   envelope/route, M/N/K, threads, persistent input/request policy; candidate
   cycling + fresh-process medians. Current host results are not SPARKS
   evidence.
2. Before production edits: approve contract/variant design + scratch
   target-local candidate. Require repeated latency win beyond noise
   INCLUDING conversion, tree reduction and completion — not intrinsic-only
   peak throughput. Reject uneconomical.
3. Byte oracle: M1..supported grouped; row tiles 1/2/3/4; K 1/3/4/7/8/15/16/
   255/256/257/511/512/513 + real GDN sizes; N tails; row offsets/selection;
   alpha/beta/bias; stored-half + FP32-RNE sources; cancellation, signed
   zero, half subnormals/normals/ties/overflow boundaries, all supported
   exceptional inputs. Each grouped row vs independent serial M1 + fixed-tree
   reference (no tolerance).
4. Instruction census per emitted variant/envelope: FMLAL/FMLAL2 in half
   leaf; ordinary FMA in mixed; no prohibited instructions in baseline
   closure; admission negative controls; complete compiler/flags/source/
   object identity.
5. Register witness: no compiler FP/SIMD spills inside accumulation;
   distinguish typed256-lane partial storage from stack traffic, ABI saves,
   explicit reduction I/O. Cover each row count + lane tile.
6. Reuse V2_Unit_CPUHalfOperands + V2_Integration_CPUHalfOperands_<ISA> +
   production floating-expert format/tree gate. Add focused recorded-input
   production replay (reset, dirty, partial, repeated replay) in
   ProductionTestPreflight. Extend Unit policy/rounding tests + every
   positive worker budget without weakening canonical gate names.
7. After a production slice: complete configured Unit + ProductionTestPreflight,
   then .80 Release correctness/economy + applicable GDN E2E/PerfStats route
   evidence. Preserve optimized dispatch; no deterministic substitution.

## Provenance / hold
Evidence: artifacts/fp16-emission/ (compile manifest, *_production.asm,
*_armv9.asm, projection-inner-*.json, projection-symbols.txt). Whole-TU
counts = census only, not hot-loop estimates. No production code, CMake,
capability API or test registration changed. Implementation HELD pending
final base + .80 evidence + explicit plan approval. Estimate roughly 3-5
focused engineer-days for a fully gated first family, subject to
exceptional-value and existing coverage findings.
