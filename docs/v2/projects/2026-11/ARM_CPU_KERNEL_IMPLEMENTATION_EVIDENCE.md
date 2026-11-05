# ARM CPU kernel implementation evidence

Updated: 2026-11-05. Branch `codex/qfn-ngram-continuation-arm64`, based on
`2cba3ca1d`. This is an in-progress WIP publication; immutable handoffs retain
their own base-plus-overlay identities. This record distinguishes verified
kernel slices from the unfinished objective in
[the implementation plan](ARM_CPU_INFERENCE_KERNEL_PLAN.md).

**Not a serving/performance certificate:** the complete native engine,
production factory/workspace bindings, real models, HTTP/MTP/prefix campaigns,
exact-overlay installation and final `Perf__` scaling have not passed.

## 2026-11-05 RMSNorm integration checkpoint (incomplete)

The new native `NEONRMSNormReduction.h` consumes original noncontiguous
FP32/FP16/BF16/full-width INT32 rows in shared-traversal tiles1..4. Four fixed
FP64 chains/eight lane positions and a fixed fold are independent of M/workers.
T5 is a separate increasing-column FP32 FMA program vectorized across rows.
Independent finite-domain references and actual native M=1 calls cover original
prefix/padding and K fringes through8192. Integer conversion widens before FP64,
retaining low bits above2^24. This reducer proof alone is not RMSNorm integration.

A native production-TU regression now reproduces three original ownership
failures, after adding genuine NEON math seams but before changing ownership:

-1×12288 all-ones, retained3: all12288 values are1.7320247888565063 instead
of0.9999949932098389. Each caller used its private4096 partial, not the12288
original-row summary. This is an owner defect, not last-bit reduction drift.
- A delayed retained small exact-alias call transforms the same row three times
and returns while only1/3 participants have arrived.
- An empty retained invocation also returns at1/3 arrivals.

The frozen TU SHA256 is
`e6a7f1f54bb86f126676665bc2b0e14fa757db5ee7d160bbfecaa051374cfaef`;
raw XML/logs remain outside the repository. It is not an untouched old ARM
engine build. The repair makes summary, inverse and publication one complete
worker-owned row/tile transaction, eliminates FP32 TLS scratch/atomic partials,
protects original full banks/gamma/options before stores, and completes retained
small/empty calls. Unused detached FP32 summary/application entrypoints and the
unused FP32 scratch overload were retired rather than retained as a second path.

`NEONRMSNormFloatingRows.h` shares gamma and directly publishes original encoded
rows with bounded register tails. Staged guarded FP64-mean versus per-row
FP32-mean inverses remain explicit, distinct programs. The old truncating scalar
half publication now uses `FP16Utils`; native lanes use `NEONFP16`, not F16C as a
universal subnormal/payload oracle. The current production-TU floating sweep
checks M1..16/31/33, original prefixes/canaries, independent formulas and actual
M=1 bytes for all three floating formats/T5, including retained workers1..10.
All eight production-TU cases now pass, including explicit half normal-rounding
lanes and distinct inverse programs. The complete fresh five-profile matrix
passes151/151 each; all115 configured preflights pass20 repetitions (2300 raw
passes). Exact current scope and preserved RED runs are recorded below.

Native BF16/FP16 factory source now calls original-storage matrix primitives
instead of allocating/decoding whole FP32 rows. Actual factory-TU C++23 syntax
passes with genuine dependencies and `-Werror=switch`; factory runtime, residual
conversion/workspace paths and whole-engine gates remain unqualified. Do not
infer complete RMSNorm or MTP inference support from these passes.

Static exact-header resource diagnosis found an outlined compile-time visitor
that moved accumulators through memory each K iteration. Direct compile-time
visits remove that call/memory edge: the Rows4 FP32 hot loop has16 native FP64
FMA instructions per8 columns/row,24 SIMD register identities, and no stack
accesses/calls. This is static resource evidence, not PMU or throughput evidence.

Source-only RMSNorm audit also identified open Q8 epsilon/gamma-cache/no-op and
legacy Q24 narrowing defects, INT32 AVX2 low-bit loss, distinct quantizers,
Q16 FP32-square rounding/complete-block/expansion-alias seams, and unadmitted
legacy workspace/row buffers. Those require individual executed regressions,
repairs, explicit preflights and broader qualification; none is closed here.

## Native Spark engine build-only checkpoint

Fresh packet `3bf350a5…a80ba` supersedes the withheld `662c0ac5…ac8e`;662 must
not be executed or patched. The native validator reports exact SDK content
import authentication, real oneDNN3.11.3 install and no OOM, then a first
**configuration** failure: heterogeneous `lscpu` emits multiline socket/core/
thread fields and `V2TestGateFinalization.cmake` cannot evaluate `4 * 1\\n1`.
The application/Unit/preflight build command never started. Native sysfs and
`lscpu` CSV also disagree about core identities; first-row trimming or CSV
deduplication is not an acceptable repair. The returned archive SHA256 is
`15497193eeb13213c06c0d202ac031f59e2a24776b38a9bc25c72a0a57362c67`.
Main independently rechecked its archive digest; detailed raw-record inspection
is still in progress, not replaced by the validator's report. The stopped packet remains immutable; a source-owned
heterogeneous topology fix and a new authenticated successor precede any retry.
No model/GPU/corpus/cluster work or broader test campaign is authorized by this
build-only attempt.

## Linux GCC rounding regression: Spark RED and local repair

The first immutable Spark snapshot (`SHA256SUMS` `c59c5ef2…8152`) built on
both GB10 hosts under both NEON profiles, but each lane passed only112/117:
SwiGLU Unit/Integration/original-row entries and typed-tensor Unit/Integration
failed. Both isolated failing filters reproduced3/3 without parallel CTest.
The frozen123-file source and failures remain intact; Mac passes did not certify
Linux. The returned evidence archive SHA256 is
`b86571e88e682a4e121d43203b9d2751b57a37636b6f3df5e9f79ca447f37b3c`.

A native Linux/arm64 Docker VM on this M1 reproduced the exact first failure
with genuine Ubuntu GCC13.3, original source-owned Release flags and FPCR0.
GCC ISA shows five Newton `fmls` instructions where the expert program requires
separate product/subtraction. All1039 expert mismatches match an independent
explicit fused-Newton candidate: gate bits`41550000`, up`c37f2200` produce old
`c5544733` instead of canonical`c5544734`. The two range-reduction `fmls` are
intentional and remain unchanged. This is missing rounding-edge retention after
inlining, not a changed approximation or a general compiler-bug claim.

Independent volatile-memory typed references also find73 differing checked
blocks, including a defect hidden by native-versus-scalar agreement. At
seed6/block2, Q16+FP32 gives old native/scalar scale`41f3f25b` versus declared
separate-rounding`41f3f25c`. Q16+Q8 native already gives correct`41f3f90b`, while
the contracted scalar diagnostic gives`41f3f90a`; the native result must not be
changed toward that wrong diagnostic. Thresholds, inverses, ties-away, clamps
and signed sums are unchanged.

`NEONFP32.h` now retains native operation results through opaque SIMD-register
edges, matching the existing scalar contract without memory traffic or global
compiler/FP-environment changes. Expert Newton/scale edges and typed decode/add
edges use it. Scalar and bounded legacy decode/add programs retain their
declared separate products through the common scalar register contract.
`CPUTypedTensorArithmetic` independently checks both production dispatch and
the scalar diagnostic. Its two tests and
`NativeSwiGLU.DeviceNewtonRoundingSurvivesInlining` fail against the untouched
old headers/production TU, then pass after repair. The original strict failing
tests remain strict. Explicit focused preflights register the device-contract,
Newton-inlining and typed-add invariants.

The identical isolated GCC expert wrapper has79 instructions before/after,
23→14 SIMD registers, only the two intended range-reduction `fmls` after repair,
five separate Newton `fsub`, and no stack instructions. That is one resource
witness, not PMU, latency, all-kernel economy or serving certification.
Local logs/ISA/negative executables live outside the repository under
`/tmp/llaminar-arm-gcc13-diagnostics`. The successor snapshot with SHA256SUMS
`391c8481…6128` now passes125/125 in all four Spark lanes, with all five focused
cases passing20 repetitions. Evidence archive SHA256 is
`80d41993658a8b8d799528371df4dfb1480a0445abb483034f4c5077e27e3373`.

A supplemental genuine WSL Ryzen7800X3D/GCC14.2 check builds both x86 profiles,
but the new typed native test fails73 AVX2/166 AVX512 assertions; its scalar case
passes. Each distinction reproduces3/3. Keep this separate from the repaired
ARM scope. Source already declares nearest-even for the AVX512 fixed32 Q16
publishers, versus ties-away on AVX2/NEON/scalar: the new generic oracle's
all-ISA ties-away assumption is not valid. Preserve the declared per-ISA
program while distinguishing that oracle issue from possible unretained decode
products using actual native ISA/value evidence. No x86 production fix or
broad x86 certificate is claimed.

### Native x86 value/ISA diagnosis and focused repair

The external native GCC14.2 witness preserves each original publication rule.
AVX2 has73 mismatches against its separate ties-away program; AVX512 also has73
against its separate nearest-even program, rather than the generic oracle's166.
Both split as Q16+Q16=0, Q16+FP32=38, Q16+Q8=35; all73 match the independently
explicit FusedA candidate, with0 unexplained. Initial MXCSR is`1f80`, actual
compiled/runtime routes agree, and named production wrappers contain the FMAs
(the reference candidates' instructions are excluded). Raw archive SHA256:
`7df8431e2ec08e75dd3c2cc0a45bb2c0817e67feef64af3c883ddd096290285a`.
This establishes a helper arithmetic defect, not deployed-model/token impact.

`X86FP32.h` now retains eight-/sixteen-lane product results in vector registers;
only the three fixed32 Q16 add routines' AVX2/AVX512 decode products use it.
Nearest-even versus ties-away, thresholds, scale/inverse trees, clamps and sums
are unchanged. The independent test now names those source-declared programs
through the active runtime ISA rather than assuming one program everywhere.
A signed-half-tie reference check and a halfway-product pair, both operand
orders and disjoint/left/right output reuse, strengthen the regression; the
pair has its own explicit preflight entry. Old-source negatives, repaired
native x86 runtime/ISA/resource checks and broader x86 compatibility remain
required; local cross-code-generation is not execution evidence.

Nine rebuilt local ARM configurations now pass126/126 (five Mac, GNU13 and
GNU14 Linux baseline/DotProd). All94 current preflights pass20 repetitions on
Mac DotProd. The earlier Spark125/125 result remains bound to its older
immutable source; it does not certify this repair until a fresh validation.

A separate genuine native GNU14.2 SDK built from official Ubuntu24.04 ARM
packages compiles, links and runs C++23 `<print>` plus atomic shared-pointer
publication without replacement headers or global FP changes. This closes the
identified GNU13 library prerequisite for subsequent Linux engine builds, not
any unexecuted engine/factory/model gate.

## Implemented numerical and ISA foundations

- `CPUInstructionSet.cmake` admits native `NEON` and separately targeted
  `NEON_DOTPROD` compiler envelopes without x86 flags. Both explicitly reset
  the compiler CPU to `generic`; DotProd adds only that extension to ARMv8.0,
  not an ARMv8.2/LSE instruction envelope that admission does not authenticate.
- `ARMCPUFeatures.h` owns immutable native capability observations;
  `CPUFeatures.h` shares ISA names/dispatch and rejects incompatible explicit
  overrides. A DotProd compiler envelope requires actual DotProd hardware even
  when a narrower diagnostic route is requested.
- `DeviceFP32NumericalContract.h` uses an AArch64 register dependency for
  explicit scalar FP32 rounding, without changing other backend programs.
- `CPUNativeVNNIFP16.h` and `GPUAlignedExpertQ8Primitives.h` use native half,
  maximum, scaling, RNE conversion, byte packing and signed-sum operations.
  Existing bounded partial-block handling retains its numerical program.
- `CPUNativeNEONContributionContract.h` implements widening signed four-column
  dots and explicit ASIMD/SDOT variants. Backend-native FMA and separately
  rounded expert contributions remain distinct. Both instruction variants
  have now executed on this host; the earlier baseline-only limitation is
  superseded.

## Prepared projection slice

Production headers now provide, but the engine does not yet select end to end:

- `CPUNativeVNNIEncoding.h`: the existing source-aware encoding mapping, moved
  out of the owning packer without changing the mapping.
- `CPUNativeVNNIWeightView.h`: checked immutable geometry and one borrowed
  canonical interleaved stream. It owns no allocation, source mirror or
  decoded matrix. Original source payload width and physical execution width
  are separate; both expanded encodings physically contain 32 bytes/column.
- `NEONBlockDecode.h`: native source-order Q4/Q5/nonlinear nibble decode for
  preparation and diagnostics, preserving integer values and exact lane order.
- `CPUNativeNEONMultiScaleOps.h`: four-column native signed dots, exact half/
  opaque-word conversion, source sign/grid lookup and bounded address generation.
  ASIMD and DotProd are separate selected types, not a per-dot runtime branch.
- `CPUNativeVNNIMultiScaleKernels.h`: shared decoder/rounded arithmetic for
  NEON, AVX2 and AVX512, with deduced prepared-reader types and bounded N stores.
  Existing x86 calls retain their default widths/padded contract. The header
  no longer requires the storage-owning packer.
- `CPUNativeVNNIActivationSums.h`: native widening half/quarter signed-byte
  reductions, including -128, without saturation or overread.
- `CPUNativeNEONPreparedProjection.h`: checked synchronous tiles for 1..4
  physical rows and all six encodings. Rows share each decoded vector. Ordinary
  native projections use increasing K with their native FMA program; expert
  projections use the shared ordered partition count/fold. Q6/compact retain
  their separate source-factor block programs. No production serial-row replay,
  heap allocation or per-invocation weight repacking occurs in these leaves.

The focused tests cover all 21 source identities, every physical encoding,
Rows 2/3/4 under both numerical policies, independent production serial M=1
oracles, and independently decoded/memory-rounded source-order references.
Compact geometry additionally covers N=1/3/4/5/63/64/65 and K block counts
1/2/7/16/17/33, including trailing empty arithmetic partitions. Wider logical
M, production row/workspace scheduling, prefill/fused/two-term bindings and
worker scaling remain unverified; a four-row leaf is not that certificate.

## Original-coordinate worksharing and typed input closure

- `NEONFP16.h` now owns the shared native finite-half expansion/nonfinite
  payload correction used by compact operations and typed conversions.
- `NEONTypedTensorOps.h` implements twelve native typed leaves behind the
  existing public dispatch: FP32/BF16/FP16/Q8 residuals, Q8/Q16 additions,
  Q16-to-Q8, Q8 decoding, tensor quantization and full activation quantization.
  Typed residual/tensor quantizers retain their original ties-away programs.
  Full ordinary activation blocks retain SIMD ties-even and division; bounded
  partial blocks retain their existing separate tail program. Expert Q8's
  canonical reciprocal tree is not substituted for either.
- `TypedBlockGeometry.h` is the shared public full-block count check.
  Seven Q8/Q16 APIs reject incomplete counts before mutation on every CPU ISA;
  native private floating-tail leaves remain separately tested. The single
  activation-block API also rejects logical extents outside [0,32].
- `CPUProjectionRowGrid.h` extracts the existing x86 original-coordinate grid.
  `CPUProjectionQ8Publication.h` extracts the existing activation publication
  helpers without changing valid x86 arithmetic, adds precise policy/geometry
  rejection and widens source row-offset multiplication before it can overflow.
- `CPUNativeNEONProjectionBatch.h` binds complete borrowed banks and explicit
  Pair/Four candidates. Original request slices/repeats and inactive prefixes
  retain physical addresses; whole-bank alias proof includes original row
  objects, tables, live/frontier and inherited words. No allocation, weight
  repacking, serial-row replay, live-word shadow or publication is added.
- Batch tests compare actual native M=1 leaves across all 21 sources, both
  arithmetic policies, M=1..16/31 and larger 33/65 capacities, both compiled
  instruction routes and actual 1..10-worker budgets. Standalone and retained
  teams, multi-block ordered partitions, bias, strides, dirty inactive rows and
  data-only reset are covered. Q8 publication additionally covers K=1/31/32/33/
  256/1056 and original request slices.

The canonical entrypoint/owner/workspace layers below are now native. Actual
Tensor factory construction, graph/request-state and slot/migration gates remain
unexecuted; borrowed byte-fixture tests alone are not complete MTP state proofs.
The raw-pointer Q8 publication API retains its caller's valid/disjoint-storage
precondition; it does not inherit the batch binder's whole-bank alias proof.
Ordinary NEON scale conversion uses FP16Utils, like AVX2, not AVX512 F16C's
subnormal-half behavior. No cross-ISA ordinary half-scale byte parity is claimed;
heterogeneous expert publication remains under its separate canonical contract.

### Empty retained-team completion regression

Review found that an empty early return could omit the completion edge of the
nonempty omp-for. A fast member could publish the next live count in a single
region before a late member observed the current empty publication.
`OpenMPUtils::beginWorkshare` now preserves that empty active-team edge without
creating a standalone empty team. Native batch, shared Q8 publication and the
existing NativeVNNI work/rotation entrypoints use it.

`V2_Integration_CPUNativeNEONEmptyWorkshare` runs delayed-observer
empty/publication/nonempty/reset sequences twenty times for publication and
projection separately. Its observer-arrival assertion is essential: a first
numeric-only version also passed the diagnostic no-barrier reconstruction.
The strengthened regression fails that reconstruction (one of three observers
has entered when the next publication begins) and passes production. No actual
hang is claimed. The preflight uses the ordinary 30-second bound. The read-only
reviewer confirmed the source-level completion repair and numerical separation,
not test execution, full-engine correctness or performance qualification.

## Canonical native entrypoints and actual retained owners

- `CPUNativeVNNIExecutionContract.h` owns the shared invocation enums,
  descriptors and original-output scopes. `CPUNativeVNNIGemv.h` selects the
  native architecture implementation rather than importing x86 intrinsics on ARM.
- `CPUNativeVNNIPackedWeights.h` separates the real retained owner from tensor
  preparation. `CPUNativeNEONGemv.h` now implements the canonical serial,
  grouped, ordinary-prefill, FP32, alpha/beta, shared/multi-input fused APIs.
  Native ISA requests reject foreign/scalar routes; Nbc1/2/4/8/16 affect only
  physical N ownership. The current native total policy is not a measured overlay.
- Native fused execution currently retains one OpenMP team with independent
  member workshares. Its correctness is tested, not its economy; combining
  member task grids remains a profiling/tuning question, not a claimed win.
- `CPUNativeNEONResidualCorrection.h` shares four independent primary/residual
  chains for two real rows and folds each pair once. Two-term admission remains
  BackendNative-only under the existing operation contract.
- `NEONHelpers.h` and `CPUQwen4ExpSIMD.h` provide native HC range reduction,
  ties-even selection, the existing explicit FMA polynomial and hardware
  division. Actual HC activation/output bytes are checked against an independent
  memory-rounded reference. This does not certify every remaining SwiGLU family.
- The new execution test populates small **actual retained aligned owners**
  before invocation and exercises canonical entrypoints over all 21 sources,
  both ordinary/expert projection policies, grouped M through 16/31/33,
  original inactive prefixes, bias, strides, fused member/input geometry,
  FP32 beta and ordinary/HC two-term operation. It does not run the tensor
  preparer, actual factory object, model graph or serving lifecycle.

### Phase-write and retained-bank regressions

Review found that projection-only alias proof did not certify the preceding
Q8-writing phase or beta's final destination. `CPUProjectionStorageProof.h`
now authenticates those writes against original FP32 input, weight/owner bytes
and every original work object/table/live/frontier/inherited word. Fused output
also excludes sibling owner/descriptor storage. Before-write negatives are
registered as `V2_Integration_CPUProjectionPublicationStorageAliases`.

The shared two-term path previously rescanned primary scales **after** its
projection completion edge. A fast retained-team member could reuse primary
while a slower member validated the previous bank. Validation reads now finish
before the final projection/failure edge, and only a scoped atomic completion
result is read afterward. `V2_Integration_CPUQ8ResidualRetainedPrimaryReuse`
runs immediate next-workshare reuse. Restoring the old rescan in a diagnostic
temporary header aborts that regression on its first run with a spurious
nonfinite-primary error; production passes.

Actual single-factor owners also returned minima by reading absent secondary
storage. They now return exact zero without that read; the focused owner
preflight runs under ASan. Original IQ4_XS ten reserved coefficient bits are
also rejected by the moved x86 SIMD scale accessors, and the existing explicit
x86 metadata preflight now covers each bit/lane. That x86 extension is
syntax-checked, not executed on this host.

### Native workspace and provenance ABI

`CPUExecutionGeometry` now carries an explicit reduction-storage ABI plus the
participant's actual compiler/runtime ISA. ARM's full-K/canonical register trees
contribute exactly the admitted Q8 bank, not an x86 shared-K slab or fabricated
cache reserve. Unknown cache observations remain zero; cache-dependent native
services still require real implementation/observation. Device-free tests prove
all-format native BOM equality and reject unknown/cross-architecture records.

MPI inventory version 5 and the service observation codec retain these fields;
remote memory declarations no longer borrow root's ISA. MPI source syntax passes
natively. Full service-catalog execution is not certified: its complete native
closure still reaches the unsupported libc++ atomic-shared-pointer specialization
and dependencies normally provisioned after the full configure boundary.

The full-closure probe additionally exposed eager recursive stage comparison
instantiation on Clang/libc++. A dependent equality body fixes it; the focused
`V2_Integration_RecursiveStageSnapshotCompletion` passes, while restoring the
defaulted comparison fails compilation with incomplete Snapshot.

## Native production SwiGLU translation unit

`SwiGLUPrimitives.cpp` now executes native FP32, BF16, FP16 and complete Q8_1
operations, plus the placement-invariant expert program. The native exponential
implements the exact split-ln2/eighth-order FMA program. Ordinary sigmoid uses
hardware division; heterogeneous experts retain five separately rounded Newton
steps. The earlier attempt to reuse the HC channel polynomial failed the new
independent double-exponential accuracy envelope. The implementation was fixed;
the `2e-6` relative envelope was not relaxed. HC's separate source program is
unchanged, and no ordinary ARM/x86 byte-equivalence claim is made.

Bounded tails execute the same four-lane arithmetic as full vectors. Original
serial rows for M=1..16/31/33 agree byte-for-byte with grouped/flattened calls
through the real public APIs, including non-multiple-of-four channels and
256/1024-element task fringes. FP16/BF16/Q8 exact source aliases and retained
teams are exercised, including actual 1..10-worker FP32 budgets. These are
primitive/TU proofs, not actual Tensor-kernel construction or serving evidence.

`NEONFP16::narrow()` retains FP16Utils' normal ties-even/carry, truncated
subnormal, overflow and nonfinite-payload storage program using native integer
lanes. Hardware conversion would choose a different subnormal rounding rule.
The focused proof checks every original binary16 word, all FP32 exponents with
mantissa/rounding boundaries, and 131,072 additional deterministic FP32 words.
BF16 uses its existing bitwise ties-even program. Q8 SwiGLU retains its own
`d=max_abs/127`, `d>1e-10`, reciprocal and ties-even program, not the activation
quantizer's different maximum threshold.

All public storage routes now reject partial source/output aliases and malformed
Q8 extents before mutation. Review also found a pre-existing AVX512 retained-team
hole: each member previously executed every Q8 block. Its unchanged pair/tail
arithmetic now has disjoint workshares and one final completion edge, even for
an empty bank. All-ISA retained Q8 regression and explicit preflight cover
0/1/2/3/5 blocks and both source aliases. The repair is source-reviewed and
syntax-checked; actual x86 execution is still unavailable.

The empty-workshare regression samples atomic observer arrivals **immediately
after return**, before any following barrier. A delayed third member must
already have arrived. Adding a diagnostic Q8 `size==0` early return outside the
repository fails the unchanged current test object on its first Q8 iteration:
one arrival rather than three. Production passes twenty iterations for each of
five workshare entrypoints. Final parallel joins are not used as proof of that
edge. Every discovered geometry/alias/accuracy/completion defect has an explicit
`ProductionTestPreflight` registration.

### Fused SwiGLU/Q8 publication followed by grouped down projection

The actual-owner execution target now links `SwiGLUPrimitives.cpp`, exercising
`swiglu_quantize_activations_to_q8_1()` immediately followed by canonical native
grouped down projection in one retained three-worker team. All 21 sources, both
numerical policies, K=32/64/96/1056 and every M=1..16/31/33 match actual serial
SwiGLU, per-block Q8 publication and native M1 projection, including physical
output padding and trailing canaries. This certifies that fusion seam, not a
complete expert stage/Tensor factory/model transaction.

The shared helper now rejects unknown policy and whole output-vs-either-source
overlap before mutation. Its ceil and physical-row task product use bounded
64-bit arithmetic rather than overflowing positive int geometry. The focused
before-mutation regression is device-free and registered on all CPU ISAs, not
only ARM. This path's output activation quantizer is deliberately distinct from
Q8-input SwiGLU's scale-threshold program.

## Native RoPE floating and in-place quantized slice

`NEONRoPEPrimitives.h` implements separately rounded FP32/FP16/BF16 pairs,
including bounded final fragments. The production RoPE translation unit and
attention/TurboQuant inline entrypoints select those native leaves. Independent
memory-rounded tests cover vector/half-storage boundaries and heads through 514
values. Real serial M1 calls match original grouped M=1..16/31/33, repeated and
restored positions, inactive rows, GQA heads and caller-thread-local angle state.
Darwin Q15 coefficient construction now uses the platform's declared sin/cos API.

`CPUQuantizedRoPECycle.h` owns the original half-block consumption schedule;
`NEONQuantizedRoPE.h` supplies native dequantization, the production quantized
FMA order, RNE requantization and exact integer sums. Even block counts form
independent two-block cycles. For odd count N, the carried block index advances
by `(N-1)/2 mod N`; that stride is coprime to N, so every block retires once,
with anchor block zero last. Both original halves are consumed before replacing
any scale. Only one anchor half, one carried half and one output block are
private, regardless of head length: no head/cache mirror or source-scale bank.

An independent full-head diagnostic oracle checks Q8 and Q16 blocks of
32/64/128 values, single/odd/even counts 1..9/16/17, 32 source/coefficient seeds,
signed extrema and trailing canaries. Native Q8 scale overflow is rejected
before the affected block publishes nonfinite half metadata; already retired
blocks are not rolled back. The additional native conversion/uniform-scale
programs below are now implemented, but this remains **kernel/TU evidence, not
complete RoPE factory/runtime/performance qualification**. Existing x86/scalar
single/odd geometry, widened fixed-scale and half-overflow paths still need
their equivalent implementation/certification; native tests do not certify them.

### Cached and inline ownership/capacity defects

The all-ISA cached-decode regression reproduced a pre-existing error: every
retained-team member rotated the complete shared Q/K banks. A common head
workshare now owns each head once. Review additionally found that the
conditional workshare macro's false branch creates a private one-thread region.
The new call explicitly includes active-team membership in its condition rather
than allowing a small standalone hint to split a retained team. Expanded tests
reproduced wrong bytes and 1/2 observer arrivals instead of 3/10 before this
repair. They now cover total head counts 0..4, null K, actual 1/3/10 workers,
and immediate post-return arrivals before any later barrier.

The inline K path had a separate hard-coded 128-pair coefficient array.
Compiling the **original HEAD header** against the current focused regression
under ASan/UBSan reproduces index 128 and a `stack-buffer-overflow` in `cos_buf`.
The replacement shares fixed 128-pair fragments across original heads, supports
wide/partial heads without a head-sized temporary, validates geometry/extents,
and preserves retained-team membership. A delayed-observer empty test checks its
actual completion edge. All discovered ownership/geometry/capacity invariants
have explicit `ProductionTestPreflight` entries. x86 execution, actual Tensor
factory dispatch and performance remain unproved.

### Native integer conversion and hybrid FP32 programs

`NEONIntegerRoPE.h` now supplies all three Q16 physical block encodings for:

- normalized Q8 conversion: its 1.415 safety factor, positive Q15 bias,
  common/256 physical scale and output-scale return;
- legacy Q8 conversion: no safety factor, the distinct signed bias, the same
  common/256 stored scale but the historical **common-scale** return;
- fixed Q8 conversion: separately rounded FP32 intermediate pairs, ties-away
  quantization and the fixed-cache 16383 clamp, not the normalized integer path;
- fixed/dynamic Q16: original ties-away Q16 ratios, native widened products,
  arithmetic shifts and distinct saturation limits.

Single/odd/even heads are streamed at original logical coordinates. Exact Q16
reuse retains **all original d words** until payload reads complete; output
metadata and a disjoint dynamic scale word publish afterward. Tests exercise
canonical production head dispatchers against independent int64/memory-rounded
oracles, all block widths, six representative counts, sixteen seeds, signed
scales/extrema, exact reuse, canaries, admission failures and scale publication.
These do not replace actual Tensor/stage/graph lifecycle qualification.

Review found one explicit-common-scale pair endpoint where a negative ratio
passed admission but byte -128 overflowed a scaled-int32 intermediate. The
native program now first multiplies byte by Q15 (bounded by 2^22), then widens
multiplication by the admitted signed ratio before sum/bias/shift. No rounding
edge moves, and int64 safely covers every resulting term. The focused test and
preflight include both byte extrema and signed ratio endpoints. A reduced native
reconstruction prints -32767 with the old scaled-int32 operation versus the
mathematical +32767; the repaired program prints +32767. This is a reproduced
native pair-domain defect, not an observed whole-model inference failure.

Public Q8-to-FP32 hybrid conversion now streams native four-lane original halves
rather than using the former `dequant[256]` head mirror. The existing x86 scalar
program is likewise streamed without changing its numerical selection. Complete
output/input/output/position-word extents are checked before writes; the existing
reciprocal-power coefficient program and padding-position semantics remain.
All-ISA tests compare real serial M1 with wide grouped original rows and retained
1/3/10-worker teams, plus immediate empty completion and alias admission. A
separate independent native pair oracle proves the arithmetic through 1024-value
heads. Per-call coefficient-vector ownership and full graph/arena integration
remain part of the unfinished runtime qualification.

## Native anchored AQ8 attention-key codec

`CPUAttentionKeyQ8.h` now selects native NEON for the existing anchored key
format, with no decoded-cache/head mirror or new storage/publication authority.
The complete finite-residual/maximum pass finishes before any output byte.
Two independent four-lane searches overlap all seven exact rational decisions;
one bounded eight-byte store retains original code order. Decoder groups widen
eight signed bytes and retain one immutable scale, preserving the declared
`(scale*q)*abs(q)+anchor` order without FMA. Foreign/uncompiled explicit ISAs
fail; the compiler envelope is authenticated even for the ASIMD-only leaf.

The existing CPU conformance source is now shared by the full Unit/preflight
and focused native build through `V2CPUAttentionKeyQ8Tests.cmake`. It proves all
three physical widths, every rational boundary and neighbouring float, every
finite FP32 exponent, every legal signed code, real positive worker counts,
nonzero anchors and exact zero-residual branches. Poisoned destinations prove
positive-zero scale and complete zero-code publication. Signed-zero decoder
expectations use the declared addition program rather than copying anchor signs.
Every-coordinate invalid anchors and finite subtraction overflow reject before
block publication. Complete-span aliases protect source/compressed storage and
immutable anchors, including an attempted exact anchor overwrite. Focused alias,
invalid-anchor, boundary and zero-residual preflights are explicit.

### Bounded AQ8 tuning evidence, not final inference qualification

The canonical `Perf__CPUAttentionKeyQ8.cpp` target now compiles unchanged in the
focused build, reports actual native compiler/operation ISA and rejects foreign
routes. No generic training or corpus download is involved. On this M1 Max,
three paired fresh-process comparisons of byte-authenticated Release kernels
(128 heads, seven internal samples, one worker) measured these changes:

| Native change | D64 | D128 | D256 |
|---|---:|---:|---:|
| Independent paired encoder searches, throughput ratio | 1.165x | 1.186x | 1.184x |
| Eight-byte decoder/retained scale, throughput ratio | 1.284x | 1.209x | 1.617x |

A simple outer unroll did not improve timing and was rejected. Final native
body inspection covers all widths/operations: D128/D256 encoders use 18 SIMD
registers, decoders five; D64 encoding's unrolled validation uses 30 registers
with six ABI-preserved lower SIMD lanes. No arithmetic-loop spills were found.
The eight-byte decoder removes the earlier per-head stack rematerialization of
its four-byte expansion constant. ABI entry/exit saves are not arithmetic spills.

A separate 1..10-worker Release sweep ran both original and selected native
binaries for each width/operation. Selected-kernel best speedups were 5.5–7.5x;
ten-worker speedups were 4.9–6.0x, not monotonic scaling or P-only placement.
Native Xcode CPU Counters recordings succeeded for isolated encoder candidates,
separately from timing. Samples include P and E cores; raw event identities and
exact hot-phase attribution are **not yet a qualified IPC/cache receipt**.
These measurements certify neither full attention nor cache/request lifetime,
full-model inference, final dispatch economy, or the goal's final `Perf__` round.
Local timings, traces, assembly and candidate binaries remain outside the tree.

## Native compressed-centroid attention math

`CPUTQAttentionOperands.h` binds original synchronous cache/query/numerator
borrows before ISA dispatch. Its private constructors admit only the existing
64/128/256-coordinate TQ tensor ABI (plus explicit zero work), the selected
source or fitted reconstruction norm, finite scalar weight/product, and disjoint
numerator/cache/codebook spans. These are geometry/metadata checks, **not proof
of actual storage capacity or a cache lease**; enclosing owners still provide
complete live storage and retain the publication lifetime.

`NEONTQAttentionPrimitives.h` consumes those checked types. TQ4 reconstructs
original 3+1-bit indices and uses TBL4 over the original sixteen FP32 codebook
words. TQ8 performs indexed native lane loads because ASIMD has no gather, then
uses native SIMD arithmetic. Two four-lane FMA chains, the fixed lo+hi/pair fold
and one source-norm multiply define native scores; numerator updates use one
separately rounded weight/fitted-radius product and per-coordinate FMA. TQ8
instantiations do not load the unrelated TQ4 table. No decoded head/cache or
source-scale bank is constructed. Eight unavailable-x86-ISA fallback stubs were
removed; existing x86 arithmetic bodies and explicit scalar diagnostics remain.

Independent ordered-FMA/memory-rounded and byte-domain oracles cover the native
score, numerator, all centroid codes/bit positions, distinct metadata roles,
threshold fringes, canaries and before-write geometry/storage/metadata failures.
Actual serial public calls match original-head worker grouping for M1..16/31/33,
three heads per row and every positive worker count in the real host budget.
Unit excludes the worker sweep; shared focused preflights cover the invariants.
This is **correctness-only leaf evidence**: full FA2 scheduling/KV lifetime,
rotations, actual factory/graph binding and TQ resource/economy remain unfinished.

## Native TurboQuant forward and transpose rotations

The production `TurboQuantRotation.h` entrypoints now bind checked original
matrix/vector/result borrows before dispatch. Square extents and row addressing
are widened; output cannot overlap the input, complete matrix allocation
(including unused vector capacity) or original owner descriptor. Explicit empty calls are noncollective. Native forward rotation
reuses production `vec_dot_four_rows` to share query loads across four matrix
rows, with the identical serial primitive for at most three residual output rows.
Native transpose reuses native AXPY in increasing original-row FMA order. Neither
operation allocates, creates a team, transposes a matrix or mirrors a head.

Four unavailable-ISA substitutes were retired. Existing x86 aligned arithmetic
is retained; AVX2 transpose initialization and AVX-512 forward/transpose loops
now bound partial-vector accesses and use bounded tails. Native tests cover every
dimension 0..33 and representative 63/64/65, 127/128/129 and 255/256/257 fringes,
independent exact forward/transpose oracles, retained canaries and original-head
M1..16/31/33 with actual positive worker budgets. Matrix/descriptor/vector aliases
and malformed geometry reject before output writes, with explicit preflights.
Review additionally found that the original cold orthogonality verifier returned
zero error for a NaN matrix: a NaN comparison never updated its maximum. The
repaired verifier rejects nonfinite observations. A reduced original-source
negative prints `error=0` for a 1x1 NaN matrix; current code throws a precise
nonfinite diagnostic. The focused regression/preflight covers NaN/infinities and
finite-overflow observations across representative dimensions/rows, without a
hot-path matrix prescan. Equivalent x86 registrations/syntax pass; **x86 runtime tail safety is not yet
certified on this host**.

Cold generation retains its existing RNG/FP program but rejects nonpositive or
unrepresentable dimensions and widens element/row indexing. A diagnostic built
from the original HEAD generator matched current matrix bytes for three real
head dimensions and four seeds under the same compiler/standard library.
Orthogonality/round-trip tests exercise real generation. Seed equality is **not a
cross-standard-library matrix-content certificate**; mixed-ABI publication and
identity must be proven in the unfinished runtime work. Rotation resource/economy
and complete quantize/attention/cache lifetime remain open.

## Native CPU softmax and original-coordinate publication

`CPUSoftmaxOperands.h` supplies checked synchronous row/batch borrows and a
storage-independent `CausalPrefix`. Native FP32/BF16/FP16/Q8 softmax performs a
complete maximum pass, fixed four-chain FP64 exponential summary, then bounded
in-place publication. `NEONSoftmaxPrimitives.h` uses a native split-ln2/FMA
exponential with explicit subnormal scaling, not the expert SwiGLU -87 cutoff or
per-coordinate scalar libm substitution. Independent memory-rounded program bytes
and double-exp probability/accuracy gates cover original prefixes, signed/zero
multipliers, wide/tail geometries and all four storages. Original active NaN or
positive-infinite maxima reject before row writes; fully masked/-Inf rows author
zeros. These borrows prove neither actual capacity nor an arena/cache lease.

Several existing cross-backend defects were repaired alongside native closure:

- Q8 softmax interpreted/stored `Q8_1Block::d` as private BF16 despite its FP16
  schema; fixtures concealed that mismatch. Original-source execution reconstructs
  probability **87.3745** under the real FP16 reader instead of 1/32; repaired
  production returns **0.0312481**. Scalar/compiled x86 metadata and fixtures now
  consume the canonical FP16 ABI.
- The scalar maximum discarded legal all-128 rows and integer extrema ignored
  coefficient sign. A metadata-only original reconstruction returns code **0**
  for an all-128 row; current code returns **127**. Sign-correct original extrema
  use the same rounded coefficient as the exponential pass.
- FP16 output scales need an explicit representable minimum. The quantizer now
  uses `max(max_probability/127, 2^-24)` on every backend; a maximum that cannot
  produce any nonzero RNE code rejects before stores. A uniform 262144-column
  regression retains half scale **1** and codes **64**, while the metadata-only
  original reconstructs half scale **0**. This is the declared quantizer's
  physical range rule, not an ISA/precision fallback.
- The x86 exponential polynomial lacked its second constant/linear Horner stage;
  compiled-polynomial accuracy has a focused preflight. Inactive Q8 exponential
  lanes are explicitly zeroed where the vector polynomial is consumed. All
  existing Q8 coordinates and vector-end predicates were widened/bounded.
- Review found one remaining AVX2 sum predicate narrowing column 2^31 back to
  int. It now consumes the checked original prefix. A CPU-local predicate witness
  demonstrates original `-2147483648/active=1` versus current `active=0` without
  fictitious multi-GiB capacity; this is not x86-kernel runtime evidence.
- Review also found an unaligned local F16C probability array passed to an aligned
  load. A typed fixed-span helper now uses an unaligned load, with deliberately
  unaligned public FP16 and compiled-x86 fragment regressions. No observed old
  stack-alignment crash is claimed.
- Row-major APIs now cooperate with an active team even for a false standalone
  hint. Empty calls complete before release. Twenty original-source empty calls
  observe only **1..2 of 3** arrivals before the enclosing join; all repaired
  calls observe **3**. Actual serial public rows match M1..16/31/33 batches across
  every positive host worker count, including immediate empty observers.

Every defect has a focused `ProductionTestPreflight` entry. Actual x86 execution,
legacy unavailable-ISA raw-stub/compiled BF16-scalar/F16C cleanup, the typed kernel
copy/factory boundary, FA2/KV/MTP state and softmax resource/economy remain open.
The real CPUFlash class now resolves all four softmax and both rotation entries,
and the current partial dispatch wiring advances past the bare x86 dot bodies.
Actual class instantiation still fails at orphan OpenMP timing reductions,
x86 prefetch and the excluded Q16 prefill body; native FA2 policy/full admission
remain unfinished.

The enlarged sanitized suite exposed redundant immutable preparation/reference
work in the existing projection worker sweep. Reusing each original fixture,
activation bank and actual instruction-specific serial oracle across all worker
budgets retains every case and cuts its isolated sanitizer run **22.530→9.085s**.
The ordinary 30-second timeout and unrestricted CTest parallelism are unchanged;
the complete sanitized 111-test suite passes in **18.50s**. This is feedback-loop
qualification, not production kernel throughput evidence.

## Original-cache native attention math

`NEONAttentionMath.h` now supplies genuine one/two/four-row shared-query register
programs for original FP32/FP16/BF16/I16 cache storage, increasing-row shared-output
FMA accumulation, widened exact integer dots and the original threshold/divide/
inverse/RNE query quantizer. Noncontiguous rows remain at original addresses;
no cache/head repack or decoded mirror is created. Original geometry, finite
weights and numerator/cache/metadata aliases reject before output writes. Exact
integer results outside the declared int32 domain reject rather than wrap.

Actual native M1 and independent memory-rounded references cover every width
0..33 plus representative63/64/65,127/128/129,255/256/257 fringes. Integer signed
extrema/cancellation, query thresholds/limits/tails, canaries and before-write
admission are exercised by shared Unit and explicit preflight registrations.
The original FP32 dot also matches production `vec_dot` bytes.

Partial production CPUFlash dispatch wiring is now present. After the frozen
Spark successor, all four orphan timing reductions were converted to the
lexical `CPUWorkshareTiming` authority; small retained teams no longer enter
nested standalone regions. Native L1 prefetch and shared Q16 scale/normalization
routing remove the remaining bare x86 references in actual FP32 C++23 class
instantiation. That class now parses with a still-unhandled NEON launch-policy
warning; it is not runtime-qualified. The timing scope's registered standalone/
retained/lifecycle/release tests pass, but those do not exercise all new class
callsites. These later edits are not covered by the frozen Spark result, nor
are leaf/scope/syntax checks FA2 scheduler/KV/cache-lease/request/factory or
resource/economy evidence. See the dated Linux Spark handoff for the
first frozen117-test RED campaign and the Linux ARM64/CUDA13/MXFP4/cross-host
qualification targets.

## Standalone ownership-header regressions

A stronger C++23 instantiation of the actual FP32 CPUFlash class exposed an
include-order defect: public inline `MPIContext` constructors instantiated
exception cleanup of `unique_ptr<MPITopology>` before that type was complete.
Both borrowed constructors now live in `MPIContext.cpp`, preserving all original
rank/communicator assignments and the existing destructor order. The original
HEAD header fails the standalone regression at its constructor; the repaired
header passes. Actual native optimized `MPIContext.cpp` code generation emits
both public overloads. This does not execute an MPI lifecycle or qualify ORTE.

The same closure exposed `AlignedVector.h` relying on transitive declarations of
`std::terminate`. It now includes `<exception>` directly. Standalone production
C++23 compilation fails with the original header on NEON and x86, but passes with
the repaired header; C++20 had hidden the missing prerequisite. Explicit Unit
and preflight regressions allocate no storage and initialize no MPI/backend.
The MPI header test uses the real external SDK; the focused project requires
that SDK and CMake's C++23 feature support, not fake topology/MPI declarations.
Strict Debug additionally exposed one pre-existing signed/unsigned slice compare;
its explicit cast preserves the preceding implicit unsigned comparison semantics.

The real CPUFlash instantiation remains **FAIL**, not a kernel-integration pass:
all four native softmax and both rotation entries now resolve, and partial
native dispatch advances past the bare x86 bodies. The class now fails at orphan
OpenMP timer reductions, x86 prefetch and the excluded Q16 prefill path. Native
FA2 launch policy and complete helper/runtime closure remain unfinished. The diagnostic uses
the canonical libfort v0.4.2 source and vendored JSON, not dependency stubs.
These failures identify the next production-closure work; they do not justify
selecting scalar attention or weakening the full engine gate.

## Review and focused regressions

A separate reviewer session inspected the prepared-view and numerical slices
read-only. It did not build, execute, profile or certify serving.

1. **Real owner payload widths:** slot/demotion ExpandedInt8 owners publish 32
   while direct preparation can retain the source codec width. The view now
   separates these identities and its owner adapter checks both legitimate
   metadata forms without changing storage or existing x86 producers.
2. **Zero secondary factor:** symmetric/source-single-factor and explicit
   rotated views return exact zero, after validating all coordinates. They do
   not read a nonexistent secondary vector.
3. **Reserved coefficient bits:** the IQ2_S vector path previously masked high
   bits into a plausible coefficient. A focused regression reproduced failure
   before the fix. Native/AVX2/AVX512 operations now reject reserved words;
   IQ3_S/IQ4_XS native checks retain their own exact domains. Every vector lane,
   including later vectors, is covered by the negative tests.
4. **Failure wording:** coefficient rejection is fatal, not output rollback.
   Earlier valid vectors may already be authored; the enclosing execution owner
   may publish only after successful completion. Tests preserve this distinction
   rather than claiming atomic writes or adding a hot full-matrix pre-scan.

Explicit preflight entries include the native source-coefficient regression,
its x86 counterpart, and `V2_Integration_CPUPreparedNativeReadPublication` over
actual catalog-wide CPU slot/demotion owners. The latter and the x86 execution
case are registered but **not yet run** on this Mac's incomplete full engine.
Standalone byte fixtures do not replace them.

## Compiler-envelope regression and evidence lifetime

Real-object inspection found AppleClang emitting 22 auto-vectorized SDOT
instructions in the earlier baseline prepared-test object, despite baseline
frontend feature macros. `-march=armv8-a+simd` alone retained an implicit
`apple-m1` backend CPU. A small optimized byte-dot motif reproduced two SDOT
instructions with those old flags. Explicit `-mcpu=generic` fixes the boundary;
the rebuilt baseline object now has zero SDOT, while the DotProd object
contains 2002. Earlier value-test passes on M1 did not prove the old baseline
instruction envelope.

Build-time evidence now compiles that motif using the actual ordered command
of the representative CPU ISA target from CMake's compilation database,
including compiler arg1, SDK/architecture/configuration and late target/TU
options. Only source/object/dependency operands are replaced; optimization, sanitizer
removal and native non-LTO emission are explicit diagnostic controls. Empty,
LLVM-IR-only and GCC-LTO payloads cannot pass as zero-instruction witnesses. Runtime verification uses
retained source/configuration/assembly evidence and the effective command,
without executing or requiring a compiler SDK. The compilation database, recursively referenced response files and small
final-target manifest are required evidence, not optional substitutes. Nested
response changes, missing files and cycles are explicitly covered.
This is one compiler-envelope witness, not a certificate of every project TU.

A real separate negative build adding late `-mcpu=apple-m1` to the target is
rejected for SDOT leakage. Failed rebuilds retire JSON/assembly success before
any database, version or compiler work. Device-free lifecycle tests reject
stale source/policy/checker/configuration, source-level command changes,
missing/modified assembly and wider/unexercised instruction claims. The failed
rebuild regression has its own explicit `ProductionTestPreflight` entry.

## Native FA2 policy and real dependency closure

The actual FP32 CPUFlash class now parses under native C++23 with
`-Werror=switch` and no unhandled NEON identity. Production and the existing tile
tournament consume one compiler/runtime mapping and cache-observation seam.
The initial ARM policy chooses a canonical256-row span without importing x86
measurements or fabricating private cache/line capacity. Existing x86 measured
bands remain unchanged. Device-free tests cover all12storage pairs, native and
foreign identities, exact overrides, unknown cache fields, footprint overflow
and widened parallel geometry. Genuine x86 CPUID observations now reject
unproven/shared private caches and wrapped sizes instead of borrowing legacy
convenience defaults; inventory and FA2 consume that same record. The shared
policy target also runs both Q16 reduction-range tests. This is policy/syntax
proof, not an executed FA2/KV/cache-lease/request/factory certificate.

A genuine GNU14.2/OpenMPI4.1.6/NUMA/OpenSSL CPU-only Linux configure now builds
and installs canonical oneDNN v3.11.3, commit
74d04752d9eaefff6a9ff62466c4d20b155e5bca. Earlier unrestricted compilations
exhausted the VM; isolated exact failed TUs with unchanged production flags
measured maximum child RSS2214664–2231712KiB, about2.1GiB. Those diagnoses
are not full-build passes. The new phase owner retains configured objects after
failure, publishes only complete installed-manifest closure, binds compiler
ARG1/toolchain identity and exposes both command streams. Source-origin checks
reject without deleting existing work; the discarded old probe status/stderr
do not establish its underlying cause retrospectively. Seventeen real
CMake/Ninja/Git lifecycle/serialization cases pass without a counterfeit SDK,
including successful same-size/same-mtime repair and exact manifested-library
selection despite an unlisted higher-priority shadow. Invalid owned install
surfaces retire before rebuilding, retaining compilation objects; mandatory
compiler ARG1 belongs to both child arguments and cache identity. Retirement
intent now persists across receipt removal, rejected symlinks and partial
cleanup; every configure/install attempt checks surface ownership. Build and
install are separate actual phases so a benign failed compilation retains its
configured work. New two-invocation and interrupted-retirement regressions fail
the unchanged staged662 owner with actual outside-prefix writes and changed
same-mtime header bytes, then pass the repaired owner. That staged packet was
explicitly withheld from execution; it is not silently patched or certified.
Nested destination directory links are also rejected before every producer,
including ordinary failed-compile retries without an install receipt. A third
old-owner regression observes real writes through `include/oneapi`; the repaired
owner leaves the outside tree untouched. Only existing in-surface library-file
symlinks are admitted. Lifecycle schema3 migrates ambiguous old states once.

A separately preserved genuine SDK adds Ubuntu GMock1.14 to GNU14.2/OpenMPI4.1.6;
whose local OCI index is
`c8131fc0939590e3cf27dca5cf461e95e16210941c28050816e28abf2650a02d`
compiles/links/runs real C++23 print, atomic shared-pointer and GMock facilities.
The same exported Linux/arm64 manifest is
`b6644c2d668a5ec7441d34430560184a148bba2ce94b7b644033e1d2592e8af7`,
with runtime config
`fe2a76833b40124566c7e9fa1e83cc097821a675a50abe8f411b6479cf9e0038`.
Archive/index/config/all seven layers were independently hashed; classic
Docker may expose that config as its image ID instead of the local OCI index.
This is an explicit content-identity binding, not permission to change SDKs.
Full engine configuration/generation now succeeds. Its first actual build
exposed C-only OpenMPI AArch64 internal atomic headers in the C++ native
component. A thin genuine C11 ORTE adapter now owns those headers/layout asserts;
the unchanged typed environment/readiness policy stays C++17 behind a narrow C
ABI. No SDK macro or atomic emulation was introduced. Component/helper/probe
builds and four actual Linux preflights (native C ABI, live environment, source
policy and native daemon success) pass. The ABI test reads MPI state through the
exact CMake-selected MPI DSO, not an assumed transitive dependency. A prior
Mac-bind runtime-root UID mismatch remains RED; an explicitly owned native
Linux POSIX session volume passes without relaxing UID/inode checks. Full
application/core, complete canonical gates and serving remain open. Its next
ordinary full core build exhausted the8GiB VM on ComputeStageFactory.cpp and
FusedQKVGEMMStage.cpp; no semantic compile or executable success is inferred.
The genuine SDK archive was SHA-verified on WSL; an adequate-memory Spark
build-only attempt requires a fresh authenticated successor and capacity window,
not a smaller worker budget or a changed source branch.

## Latest functional verification

The completed five-Mac checkpoint below rebuilt151 focused registrations;
Linux126 and Spark125 rows deliberately retain older source scopes. These are
native CPU-only leaf/build checks, not the complete CMake-owned Unit/preflight
inventory of the engine.

| Build | Result |
|---|---|
| AppleClang Release, baseline NEON | PASS,151/151;4.01s |
| AppleClang Release, NEON_DOTPROD | PASS,151/151;4.06s; explicit ASIMD/SDOT execute |
| AppleClang Debug, strict warnings (`-Wall -Wextra -Wpedantic -Werror`) | PASS,151/151;10.61s |
| AppleClang Integration + ASan/UBSan, NEON_DOTPROD | PASS,151/151;19.86s |
| Homebrew LLVM Release, baseline NEON | PASS,151/151;3.98s |
| GCC13.3 Release, baseline/DotProd, native Linux/arm64 VM on this M1 | PASS,126/126 each; not a Spark rerun |
| GCC14.2 Release, baseline/DotProd, same native Linux VM | PASS,126/126 each; genuine C++23 SDK |
| GB10 Spark hosts×NEON/DotProd, immutable successor `391c8481…6128` | PASS,125/125 in each of four lanes;93 preflights included |
| Shared compact 8/16-column templates and the actual focused metadata-test block under x86 AVX2/AVX512 flags | PASS, syntax only for those projection templates |
| Historical WSL GCC14.2 typed target, old391c source | RED native73/166 assertions; scalar PASS; preserved diagnosis |
| Repaired WSL GCC14.2 complete focused projects, `eb3120f8…69a37` | PASS AVX2 80/80, AVX51282/82;55preflights each,0fail/skip |
| Canonical pre-commit with real OpenMP configured | FAIL at the explicit native-macOS memory/placement implementation gate; full Unit/preflight not run |
| `git diff --check` | PASS |

All115 current configured preflights pass20 repetitions; raw output
contains2300passes, with no fail/skip. Raw current logs/XML are outside the
repository at `/tmp/llaminar-arm-policy-matrix-gYPR1Adn`; every nested testcase
and the2300 raw passed records were independently parsed. The first expanded
checkpoint attempt remains RED at strict compilation of unused legacy RMS
configuration/integer-root helpers; those dead helpers were removed, not
warning-suppressed. Historical139/105 and135/101 checkpoints also passed20.
The intervening
136-test sanitizer run remains RED:135/136 pass, with
`V2_Integration_CPUNativeNEONProjectionBatch` hitting the ordinary30-second
boundary during concurrent vendor compilation. Contention is context, not an
established cause. No byte assertion failed in that log. Its unchanged binary
was copied and hashed before/after20 isolated-window repetitions: all20 pass
in226.74s without changing workers, coverage or the timeout. The subsequent
complete139-test five-Mac matrix/105-entry stress and current151/115 pass do
not erase or retrospectively explain the timeout. Earlier94-entry Mac and
93-entry native Linux GNU13.3 passes retain their narrower historical scopes. Neither receipt is a claim of full engine preflight:
native actual-owner execution, batch/leaf arithmetic, empty-team completion,
original rows, phase-write aliases, retained primary reuse, owner minima,
workspace/recursive-type ABI, typed arithmetic/geometry, coefficient rejection,
all SwiGLU numerical/storage/retained-team regressions, all-source fused
SwiGLU/Q8/down transactions, native RoPE pairs/cycles, cached/inline RoPE
ownership/capacity regressions, integer conversion/metadata/bias/hybrid/ratio
proofs, native anchored AQ8/alias/zero/boundary/invalid-anchor checks and
compressed-centroid math/borrow checks, rotation arithmetic/ownership/fringes/finite verification,
softmax arithmetic/accuracy/FP16 metadata/range/coordinates/alignment/worksharing,
original-cache floating/integer/query-quantization attention arithmetic/admission,
standalone ownership headers, lexical timing-authority release/lifecycle,
GCC-sensitive separate Newton/typed-add rounding, operand-order/exact-reuse
product boundaries, native FA2 policy/geometry and dependency lifecycle, and
compiler-envelope/nested-response/native-assembly checks. Repeat stress after further implementation changes; these remain
CPU kernel/build checks, not engine certificates. The first transparency draft
is an explicitly unqualified WIP checkpoint using the repository-documented
WIP commit procedure; neither canonical pre-commit nor readiness is claimed.
The periodic checkpoint again ran the registered canonical hook and got the
same explicit Darwin memory/placement failure before complete gates. Its
repository-documented WIP commit bypass is disclosed, not a passing gate.

The complete NativeVNNI refresh Python file was also attempted: 118 tests ran,
63 failed; the shell-driver fixtures encounter unported Linux-only `lscpu`
discovery on this host. The complete file is not green.
This required tooling work remains open; no fake command or skip was installed.
The two touched generated-policy/prefill source-architecture tests pass
independently after following the extracted grid's canonical source.

Additional value-domain checks include all 65,536 binary16 conversions,
low/high half words inside packed metadata, opaque-word widening, complete
Q4 nibble/Q5 high-bit domains, full signed-byte pair extrema, original Q6
signed coefficient pairs, and IQ2_S/IQ3_S/IQ4_XS coefficient domains. Existing
reciprocal/Q8/subnormal/cancellation proofs remain in the shared focused suite.

### Repaired native x86 evidence

Returned archive SHA256
`212303df8052203ca6b0fb375d97c87e6e070c0575fef24d9e3e986186d85b61`
was independently verified. Raw native JUnit confirms80/80 AVX2 and82/82
AVX512 with55preflights each; four typed cases each pass20/20 outside CTest.
New tests against unchanged old391c headers fail73 native and3 alias assertions
per ISA, while scalar and signed-program controls pass. The unchanged external
probe reports declaredSeparate0 and per-op0/0/0 on both ISAs; all16 included
probe/binary hashes verify. The retained AVX512 nearest-even program correctly
has95 generic-away disagreements, not95 new failures. Named wrapper FMA sites
fall from4/4/4 and6/2/6 to zero; AVX2 retains its existing temporary bank/frame,
and the inspected AVX512 wrappers have no stack accesses. Static96-element
witnesses do not establish PMU throughput or a whole-engine spill certificate.

## Preliminary ISA tuning, not timing certification

One isolated AppleClang Release Q6_K / Rows4 / ASIMD / expert specialization
was inspected from the focused production-header object. Full eight-group
unrolling hoisted activation groups and introduced FP/SIMD stack traffic.
Bounding the group loop, then replacing the final-column `memcpy` with bounded
native lane stores, removed accumulator address escape. The initially inspected
Apple-M1-target function then had only eight ABI FP/SIMD saves/restores. After
correcting the compiler CPU to generic, reinspection finds those eight ABI
accesses plus four vector accesses at ordered partition boundaries; none are
in the per-K-block traversal. The earlier eight-access count is not the
current generic-target resource result.

This does not establish all-specialization resource coverage, native PMU/cache
behavior, canonical latency, grouped economics, physical-core scaling or the
code generation of the final production binary. Those remain required in the
existing `Perf__` owners. No profiler-contaminated result is used as a timing
label and no performance speedup is claimed here.

## Reproduce focused checks

```bash
cmake -S tests/v2/portability -B build_arm_portability -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM:FILEPATH="$(command -v ninja)"
cmake --build build_arm_portability --parallel
ctest --test-dir build_arm_portability --output-on-failure --parallel
```

On AppleClang, use the README's [explicit real OpenMP discovery arguments](../../../../README.md#aarch64-portability-checks-not-inference);
the short command above assumes OpenMP is already discoverable by CMake.

Use a separate build directory with
`-DLLAMINAR_CPU_ISA=NEON_DOTPROD` for the admitted SDOT envelope.
Local trees also include `build_arm_portability_debug`,
`build_arm_portability_dotprod_sanitized`, and `build_arm_portability_llvm`.
They are ignored working artifacts, not published corpus/certification payloads.

## Required continuation

- Complete actual Tensor-factory construction, preparation/slot/demotion,
  graph/workspace publication and model/request-state qualification of the now
  native entrypoints. Header parsing and small-owner execution do not certify
  the full engine path.
- Finish floating/nonlinear/RoPE, attention/KV/TurboQuant, GDN/QFN, MoE/sampling
  and stage/transport SIMD coverage from the source inventory. Native FA2
  tail/rescale exponentials remain to be unified and runtime-qualified; retire
  the existing AQ8 `key[D]` decoded-head scratch in favor of original-code
  streaming arithmetic rather than blessing it as a bounded cache mirror.
- Complete Darwin memory/topology/cache observation, loader/process/MPI
  lifecycle and robust native serving. Full configuration still rejects the
  Linux-only memory backend; that is implementation work, not an external impasse.
- Implement authenticated native exact-overlay collection/installation over a
  total ARM policy. Do not reuse x86 timing labels or invoke generic retraining.
- Pass complete engine Unit/preflight, real source/slot/demotion and x86 gates,
  model/HTTP/MTP/prefix correctness, then the final Release `Perf__` round with
  isolated profiling, grouped economics and actual 1..10-worker scaling here.

No generic prefill/decode ML retraining, model/corpus download, exact-overlay
installation or full-objective completion has occurred.
