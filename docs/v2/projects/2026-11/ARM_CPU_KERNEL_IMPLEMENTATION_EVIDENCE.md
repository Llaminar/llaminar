# ARM CPU kernel implementation evidence

Updated: 2026-11-10. Branch `codex/qfn-ngram-continuation-arm64`, currently
based on `2948b1e49`. Historical entries and immutable handoffs retain their own
source, toolchain and qualification boundaries. See
[the implementation plan](ARM_CPU_INFERENCE_KERNEL_PLAN.md) for the current frontier.

**Not a serving/performance certificate:** focused compiler, kernel, metadata
and lifecycle evidence does not replace full Release model, HTTP/MTP/prefix or
`Perf__` qualification at the final installed source.

## Flash Next planning maintenance — 2026-11-10

The real four-split Qwen3.8-Flash-Next-UD source at `e8866ec04` fails automatic
planning at context 4096 with `Unclassified request work: output_hc_down.weight`.
The focused native baseline regression reproduces all nine missing Flash Next
semantic-role classifications, not only that first encountered tensor.

`58dcdaff5` uses the existing source-role authority for HC/indexer/PLE projections
and scalar parameters. Its original assumption that terminal HC inputs consume
one compact final row was subsequently rejected by source review (#177): the
actual graph executes HC/normalization before vocabulary-head row selection.
The original immutable table instead becomes a typed native-row gather on
its declared decoder-layer consumer; neither a whole-table GEMM nor allocation
capacity is charged as live lookup traffic. Unknown roles remain fatal, and PMA
placement/admission is unchanged. Lookup pricing explicitly remains a source-byte
and FP32 proxy without fitted backing-tier/cache latency.

`0d6f09919` completes the shared sparse metadata fixture's terminal HC, indexer
gain and PLE inputs and passes explicit closed original-rank service evidence
through admission. This repairs the first test attempt's incomplete inputs; it
does not bypass production service admission. Native GNU14 NEON Integration
build and `V2_Integration_PlanningFlashNextWeightWorkContract` pass (two GTests,
including synthetic CPU/CUDA/ROCm observations; no accelerator/payload loaded).
The broader `V2_Unit_MemoryPlanner` run initially passed 415/424 cases; nine ngram
fixtures declared foreign or unknown CPU/GPU NUMA ownership and failed before
request classification. `a847db328` repairs those observations without weakening
ownership checks and adds PP terminal/lookup stage-locality coverage. The native
planning Unit family passes all 424 cases; the expanded three-GTest preflight
passes 20 consecutive repetitions. These tests did not yet discriminate the
incorrect terminal HC row-count assumption.

The #177 corrective successor separates typed invocation-row policy from stage
ownership. HC and final normalization retain all active forward rows; only the
vocabulary projection compacts to one row. Request pricing consumes the checked
row API and reports the actual prefill/decode terminal counts it uses; focused
assertions discriminate multirow prefill from one-row decode, including PP and
legacy output normalization. Corrected-source validation remains pending.

The actual `.79` planning proof is held before launch at the superseded
`0d6f09919` until the corrected estimator pin is supplied, with the same
4096/8192 commands and canonical non-attached driver-diagnostic window.
The 8192 `Higher-priority PMA plan must fit before ngram storage selection`
failure remains an independent unverified admission issue, not an assumed
consequence of weight classification. No inference or attached profiling is
admitted by this diagnostic window.

## 2026-11-06 immutable4e56 successor audit

Main independently verified the x86 return archive SHA256
`46aa1c53a25217c217b60bc5b759b13cb26012228aa352b542f4358d21af58ca`
(4432870B), all157 raw hashes and archive/extracted bytes. Actual GNU14 Release
AVX2 and AVX512 each pass24 registered cells and12 individually counted
focused cells×20=240 raw passes, without skips. Only after both pass, actual
AVX512-built binaries under runtime AVX2 pass15 AQ8/value,8 lookup and1
separately compiled C/C++ configuration GTests. Explicit dot-ISA sweeps do not
prove that every dot in that suite uses the environment-selected ISA. Three
canonical auto-ISA dot invocations independently authenticate byte-exact output
and identities AVX2/AVX2,AVX512/AVX512,AVX512/AVX2; their timings are not economy.

Ten native ELF machine62 binaries, actual built-TU commands, raw XML/repeat
counts, source-authored TIMEOUT30 and before/after default-FP probes were
checked. Source owns GNU++20 for AQ8/Perf/ISA/compiler witnesses, GNU++23 for
lookup and a real separate C compiler. A packaging helper's incorrect demand
for C++23 everywhere remains disclosed; metadata correction did not rebuild or
rerun the producer. Exact SDKf492 config/36 layers, all3989 installed-prefix
records and selected tools match the independently qualified payload. A new
owned ext4 build/cache volume was admitted separately from the nearly full
source/evidence filesystem; no other owner's storage was reclaimed.

The separate ARM return SHA256 is
`3ab1ac742962ae5111e5086193dea300d933dd707f8239c3054899329558a667`
(1952164B), all111 raw hashes verified. Normal Integration NEON/GNU14 configures
in96.419s; unrestricted app+both gate-target build exits1 after216.702s. The
first actual failed command is `CPUQwenSparseAttention.cpp:22`, missing
`immintrin.h` (build.log854–856). Source additionally contains an explicit
AVX-only dot compile error and NEON-rejecting ISA admission; these are source
observations, not extra compiler diagnostics. Actual immutable-lookup object
891136B/SHA`abf09cf2b152db89ce4bcb328bc945c2bb3bb25b9396ed6230ef669eeda6fa92`
and KV append now have successful Ninja records. Neither implies an executable.
Parent inventory is2145 commands/3446 SHOW-ONLY tests/3145 genuine SIGTERM36
pairs; **zero tests execute**. Actual O3/g/NDEBUG/custom-assertions/ASIMD flags
and the archive-bound43-member SDK closure were checked.

Both complete7076-entry before/after maps equal SHA256
`12700915c91245ea7c6192595772ff8dbe4c0934abb4c1d3b52b95d9252158db`.
Capture starts81.036ms(x86) and223.991ms(ARM) after terminal exit, not removal.
Actual target/ancestor cgroups and host capacity were inspected; missing ARM
root limits stay absent. Private peaks are2186571776B and36692869120B, with
zero OOM/throttling; Docker terminal states are PID0/OOMfalse. Owned producers
and collectors were retired and both windows released. Immutable4e56 scopes do
not cover the later INT32 changes, complete engine/FA2/PMA/serving or performance.

## Original INT32 summary/publication vertical work

A full-width INT32 summary must not pass original integers through FP32.
Eight copies of16777217 require2251800082120712, not2251799813685248. AVX2 now
widens directly into eight FP64 chains; NEON retains its original full-width
eight-chain program and AVX512 its distinct existing program. Explicit scalar
intent is increasing-column FP64 FMA. Typed complete-bank borrows reject
source/output/options/inverse/gamma intersections and wrapping geometry before
stores. Empty and small retained calls use lexical collective completion.

Changing summary ownership alone exposed the old fused caller's private TLS
summary/inverse banks. The repair therefore replaces staged/fused publication
together: original integers and gamma are streamed twice through native
register fragments; `(float(x)*inverse)*gamma` has separate FP32 edges. Complete
NEON/AVX2 eight-coordinate or AVX512 sixteen-coordinate fragments retain RNE;
scalar tails and explicit scalar diagnostics retain ties-away. Original scale
is `A>0?A/127:1`, without a foreign quantizer floor. Inverse0 publishes zero
codes/scale1 without reading gamma values. Nonfinite/unrepresentable rows are
fatal before that row stores; earlier retired rows are not rolled back. Fused
ARM summaries share1..4 original rows and keep their FP64-mean inverse and
publication under one owner. There is no decoded row, TLS bank or scratch
resizing; the replaced scratch type/overload and helper declarations are retired.

Actual unchanged old publication-region bytes (SHA256
`8980f84558ff80c788f91f9b39747d848c2955c4025d3a93bdf7338db2bd1737`)
reproduce the missing native half-tie program and staged empty early returns.
Review found a live private-helper caller in the actual legacy precision TU;
its complete target reproduces an undefined scalar-normalization symbol before
retirement is completed. The repaired case uses the public admitted publisher
with independent FP32 products and poisoned code/scale output. A core-free
selection compiles the **entire actual TU**; the canonical whole precision Unit
registration and existing numerical thresholds remain unchanged. Strict Debug
then exposed three otherwise-vacuous scalar reference cases when no x86
comparison is compiled: they now check independent original arithmetic, not an
unused-variable suppression. The unused fixture EPSILON constant was removed;
no case's authored epsilon changed. The actual strict whole TU and all four
affected cases pass. Numeric fixtures reject a Scalar/foreign inventory rather
than skipping it.

Before this caller/test closure, the scoped five-profile matrix passes195
registrations per profile and152 preflights×20=3040 independently counted raw
passes. Genuine GNU14 ARM passes37 actual RMS cases,9 Q16 legacy cases,27
registrations,10 INT32 preflights×20 and2 compiler-policy cells with actual
O3/g/NDEBUG commands; its15663-entry source maps match. These are explicitly
**pre-review source scopes**, not later-source or full-engine certificates.
The first local producer helper had a Python parse error before any admission
or container creation; the separate corrected producer ran. A metadata verifier
also initially expected the wrong armv8.2 envelope; actual source correctly
uses armv8-a+simd+dotprod. Both helper failures remain preserved, not relabeled
kernel failures. These pre-review snapshots do not supply current native-x86,
CPU PMU/economy or full-integration proof. No older4e56 x86 receipt supplies it.

### Published344 native-x86 INT32 qualification

Main independently authenticates successor archive SHA256
`329f00ad9500bc33189c05de0f5ea7ea7d69a5783ad40882e79ee7dec1525ff6`
(10057181B), all320 covered archive/extracted hashes and the unchanged original
7079-entry source/archive reconstruction. AVX2 and AVX512 each pass36 actual
registrations and11 individually counted INT32 cells×20=220 raw passes. Only
after both pass, actual AVX512-built/runtime-AVX2 binaries execute37 workshare,
4 real precision-caller,9 Q16 and1 separately compiled C/C++ configuration cases.
Five explicit Scalar filters select13 cases per ISA, all failing specifically
at native admission, with no skips. Each ISA's exact outside-source finite-check
countercopy has one parent failure: the early native-vector `gamma[1]` NaN at
original line252 fails to die; the four other death assertions remain correct.
Genuine original test objects/libraries and exact compile/link inputs are bound
to both controls; ordinary source, objects, libraries and executables remain
unchanged. All57 subprocess phases return their expected codes (45 zero and12
explicit diagnostic exit1). Source-owned C compiler defaults and GNU++20/23
remain distinct; no uniform dialect flag is invented.

The first attempt remains **INCOMPLETE_HELPER_RED**, archive
`d3bdfb9ee42b5e1d8b05aa0bce30ab1b2c97dc7f09c400ab77234b24c7394a84`
(7274650B/all192 hashes). It completed both native profiles but a validation
helper's `inventory` callable was overwritten by a CTest JSON variable;37 mixed
names were listed, not executed. A distinct fresh-window successor changed only
the two variable references, retaining source/SDK/flags and the old failure.
Its first collector then used the wrong ownership label key; a separately
approved metadata-only collector copied retained artifacts and retired the
already-stopped producer, without executing it again. The misleading earlier
host-log filename is explicitly PRE-retirement. An optional summary verifier's
incorrect demand for an explicit C dialect remains preserved and unrerun.
Main's own first verifier also confused30 total descriptors with original link
inputs: the actual22 original inputs plus8 diagnostic descriptors were verified
without changing evidence or rerunning source.

SDKf492/36 layers, all3989 prefix records, nine actual tools, real compiler
commands,26 original objects and10 ELF62 binaries are authenticated. Default-FP
probes retain XCR0=231/MXCSR8064/rounding0; these are not per-thread observations.
All7079 source records before/after equal SHA256
`f6aa4710bd909a4a32c5351bb7bf1db3a344845e9f3803d696e3e98b4650df7a`;
immediate capture starts61.730ms after exit. Actual target/ancestor cgroups,
new owned executable POSIX storage and separate host-evidence capacity were
admitted. Peak3404128256B, zero OOM/throttling, exit0/PID0/OOMfalse. Actual owned
retirement is07:06:03.741922UTC, not terminal exit; the fresh host receipt and
window release are retained. Both old/new volumes remain owned and preserved.
This completes the published INT32 **focused leaf-correctness manifest only**;
it is not full engine, factory/PMA, serving, current sparse source or economy.

## Current native sparse-attention implementation (not a full-engine certificate)

**Sanitizer qualification correction:** the dedicated tester found a recovering
UBSan null-score pointer-offset diagnostic in c759/K2 whole-caller logs despite
GTest exit0. Historical sanitizer case-pass counts below are assertion receipts,
not proof of UB-free execution. The working fix avoids forming an absent score
address and makes common core-free CTest UBSan diagnostics fatal; clean reruns
are pending. No model/engine/economy qualification is implied by old pass counts.

`QwenSparseNativeBankContract.h` supplies one checked original byte-layout
program to static validation and CPU readers. It rejects stride/address overflow,
wrong roles/layouts/Q16 blocks and unaligned metadata before consumption;
`CPUQwenSparseNativeCodec.h` decodes genuine native register fragments from all
eight existing codecs without staging a selected head. TQ tables are the supplied
original words, not substituted globals. QSA's own eight-/sixteen-chain score
butterflies remain distinct from generic attention folds. The value program
rounds decoding before probability FMA; its bounded eight-coordinate inverse
rotation retains increasing original-coordinate FMAs without a256-word mirror.

The actual production CPU TU now compiles and links with the complete existing
QSA caller TU and genuine StateIO/replay/TP/address dependencies. Initial missing
TP/address symbols and strict interface warnings are preserved; the real
implementations and explicit overrides/unused unsupported-parameter consumption
close them, without stubs or warning suppression. Baseline/strict actual native
all-codec/Q16/ring, inactive-output/traffic and grouped/serial byte cases pass.
The initial enabled legacy-plus-lifetime run executed20 cases without failures.
After real terminal-counter and consumed-indexer-numeric regressions were added,
the pre-review72 GNU14 snapshot executed27 enabled cases; its separate explicitly
selected historical `DISABLED_RankingOnly...` case also ran successfully (GTest's
disabled-name counter remains1 despite actual run/completed status; this is not
relabeled ordinary Unit coverage). These snapshots precede the following replay
admission repair and independent exact-basis oracle.

New public-call regressions cover original-bank/gate/metadata writer exclusions,
exact same-head query/output reuse, M1..16/31/33 with all positive caller workers,
and immediate empty/nonempty retained completion. The first gate-exclusion fixture
incorrectly treated an unbound array as consumed input; the preserved failure
was repaired by binding the actual gate, not by rejecting legitimate unused
storage. Original partial-tail early-return and one-edge attention-nowait copies
produce the expected two immediate-completion failures each. Geometry fixtures
also falsify removed Q16 guards and incorrect2/4-byte alignment policies using
genuine otherwise-admissible storage; both legal TQ roles are exercised.

Review71's two fixture gaps are closed: fresh TQ4/TQ8 FLTMAX/book2 and
fitted-subnormal-word8/book16 controls distinguish radial multiplication order;
high-lane7 NaN/Inf and positive-source/negative-fitted Value deaths cover the
previously omitted finite/role guards. Counter overflow now aborts in its owner,
not through OpenMP unwinding. Indexer dot corruption is rejected before ReLU can
mask NaN as zero; exact outside-source copies that remove this guard or restore
counter throws fail only the intended death assertions.

A complete pre-review72 snapshot passed218 registrations on each of five local
profiles and170 preflights×20 (3400 raw passes). Fresh genuine GNU14 executed17
actual TUs/six ELF183 binaries,22 registrations,17 focused cells×20, bank8/codec3/
FP323/27 enabled callers and the explicit ranking case. All15671 source records
matched before/after and Main at audit; capture started29.295ms after exit.
The named4GiB compiler/test BOM was admitted against5385936896B available;
actual target/docker/root ancestry retains absent root limits, affinity0–9,
private peak3016704000B, no OOM/throttling, exit0/PID0/OOMfalse and owned removal.
`/tmp/llaminar-qsa-pre-replay-P2-independent-audit.json` records this **pre-fix**
scope, not a complete replay-admission certificate.

Review72 then found an actual P2: replay was copied before its interval check.
One shared checked original-range guard now precedes the first replay load or
member-address formation in all five public CPU entries. Actual public calls
reject aligned wrapping and misaligned roots via the specific address diagnostic;
an exact separate-header missing-Core-guard copy produces two failed subprocess
exit assertions (wrapping SIG11, misaligned late-error exit18). No source was
replaced by a diagnostic copy.

Review72 also identified same-path/symmetric-basis oracle coverage, not a proven
transpose implementation defect. `Test__CPUQwenSparseExactBasis.cpp` now independently
unpacks original TQ4/TQ8 codes/tables and role radii, executes declared FP32 edges/
eight-or-sixteen-chain scores/softmax/selected FMAs and increasing-coordinate
transpose FMAs. Distinct nonsymmetric orthogonal bases, wrapped physical history,
rows1/3/16 and both disjoint/exact-query-alias calls compare every output/guard word.
Wrong reverse-within-eight and transpose-index references must differ; actual
outside-source production-header copies that reverse the coordinate sequence or
skip the value inverse each fail12 exact-word assertions in the one parent case.
These controls were refreshed after adding complete input immutability/traces;
first pass logs remain, while the control directories contain the final-fixture
artifacts. Fresh focused Unit/preflight registrations cover both review findings.
All eight independently filtered public numeric/lifetime cases reject Scalar at
native admission, with one selected/failed case and no skips each.

Baseline/strict closure cases pass. Complete current enabled callers also pass
29/29 in both strict Debug and optimized DotProd ASan+UBSan; all ten actual TUs
have GNU++23, last-O3/g/NDEBUG, both sanitizers and source-owned DotProd flags.
The final source-only closure review73 closes both review72 findings without
claiming execution. A fresh post-review matrix now passes220 registrations on
each of five local profiles and all172 preflights×20=3440 independently counted
raw passes. Fresh GNU14 passes24 registrations,19 focused cells×20=380 raw passes,
8 bank/3 codec/3 FP32/29 enabled actual callers and the separately opted-in ranking
case. Actual18 compiled TUs and six ELF183 binaries are authenticated:14 GNU++23,
three GNU++20, one genuine default-C probe; all retain source-owned O3/g/NDEBUG.
Optimized DotProd sanitizer commands cover all14 bank/codec/complete-caller TUs.
All15672 source records remain unchanged before/after and matched Main at audit,
map SHA256 `9e1d05834a0fc6695b59e06bd3eca2dc046f06613a5487f6f514ad03fb7ee6f0`;
the after-map starts31.245ms after terminal exit. Actual private peak3557351424B
fits the named4GiB compiler/test BOM, no OOM/throttling, exit0/PID0/OOMfalse and
owned removal. Typed `--mount` read-only source and private target/docker/root
ancestry are verified. The auditor's first `HostConfig.Binds` assumption and
second Docker diagnostic-capitalization assumption both remain preserved RED;
only metadata interpretation changed, never source/helper/test execution.
`/tmp/llaminar-qsa-post-review72-independent-audit/audit.json` binds these records;
the older218/170 and9o4gieek receipts remain pre-fix scope.

Separate actual GNU14 object disassemblies isolate TQ4/TQ8 value specializations.
Their selected loops contain93/94 instructions,24/10 SIMD register identities and
two native FMA instructions, with no hot stack accesses or calls. All eight
inverse-coordinate loops likewise have two FMA instructions and no hot stack/
calls; ordinary16/48-byte GPR prologues are not hot spills. The first manual
object-path probe used a host path instead of actual `/work/source` command
coordinates; actual compile-command routing corrected the probe, not the kernel.
These are static ISA observations only, not PMU/economy or all-codec resource
certification. Native-x86/current full-engine and economical grouped execution
still need their own fresh evidence; no generic training is required.
Current complete-engine Unit/preflight/model/HTTP/PMA/economy and the final
all-family `Perf__` round remain unfinished. The current canonical pre-commit
was executed locally and remains RED at `CMakeLists.txt:173`: native macOS CPU
memory/placement still requires an implemented native authority, not a libnuma
stub. Its complete Unit/preflight dependencies never built and no such tests
ran. A disclosed `--no-verify` draft checkpoint is not a gate waiver. The
immutable4e56 engine build stays RED; no old ARM build token authorizes a producer.

### Uncommitted native indexer preparation after a669 (not part of sealed packets)

The working tree now dispatches indexer preparation once per representation into
native full-IEEE RNE casts, a register-only128-square butterfly,
normalization and partial split-half RoPE. Separate FP32 edges are register-sealed.
The original rounded-input/square128 buffers are retired; normalized128 remains
the **produced reference cast-boundary operand**, completed before output stores
for original same-head semantics, not a decoded KV/weight mirror. At most one
native fragment's remaining rotary pairs use scalar fringes; tables load native fragments,
while the allowed64-wide analytic source retains original FP32 frequency/trig.

New independent tests use FP64 binary grids rather than production conversion
code, all65536 half/BF16 words, half midpoint neighbors, deterministic FP32 words
and explicit signed half/BF16 finite-to-Inf midpoint/neighbor pins. Public query/
pool bytes cover all64 even rotary widths, M1..16/31/33, all three representations,
tables plus the supported64-wide analytic policy, exact query reuse and guards.
Invariant diagnostic normalization is memoized only after original input equality
checks; every production geometry still executes. The first oracle fixture used
unrounded pool inputs and last-token position; its RED is retained, and only the
reference was corrected to original operand/average casts and block-start position.
An actual separate original-a669 **Clang** CPP bridge passes the expanded oracle.
The genuine GNU14 old-object/current-oracle bridge instead fails4250 FP32 cells:
GNU's default cross-statement contraction fused the second RoPE product into its
add, while Clang did not. FP16/BF16 bridge cells pass; current separated native
programs match the independent oracle. This is a compiler-dependent production
defect repaired by explicit rounding boundaries, not preservation of old GNU14
FP32 bytes. A second mean/epsilon FMA is also unsafe at allowed subnormal means:
summary word64 divided128 rounds to0, then epsilon word1 yields variance word1;
one FMA yields word2. New focused public regression pins that and a legitimate
45-degree equal-product RoPE cancellation, including exact query reuse.

A separate exact-header counter selects the distinct truncated-subnormal storage
program: one parent fails the intended input0x33000001/half-minimum exact word,
with original test/link inputs authenticated. The first full native31-case run
failed only three precise fatal-stage strings; the implementation now retains
original RMS/normalized/rotary diagnostics without weakening assertions. Current
baseline/strict31 assertion runs passed; optimized31 has the recovering UB report
above and is not sanitizer-clean. Reviews74–76 close only their bounded source
slices; they confer no execution credit. Current c759 source passed222 registrations
on five native profiles and174 preflights×20, plus genuine GNU14 nineteen TUs,
26 registrations/21 cells×20/31 enabled callers. That source is isolated by the
user-designated tester at treec759a800e911eed9f4d65e992e58e5a9fba56d72;
Main now implements while the tester owns all long checks on immutable copies.
K2 adds native four-original-operand pooling and passes31 assertions on baseline,
strict and optimized-SAN, but inherits the same UB report; its compiler-resource
classification is not inherited from c759.

Static actual c759 ISA proves real vector spills in GNU14 FP16 summary/norm/table
RoPE, Clang FP16 summary/table RoPE and Clang BF16 table RoPE; these are not the
produced normalized128 buffer or analytic call saves. They disqualify the current
compute-bound candidate. Tester-sealed K3, treee1810129463732a8d2a9c313ef469858b83dfcd0,
uses native four-coordinate ASIMD registers (x86 remains eight), direct-FP32 full-
half casts and opaque bounded copies. Actual baseline/strict/fatal-UBSan profiles
pass32 enabled callers, explicit ranking1 and31 focused registrations each, with
no sanitizer diagnostics. Genuine GNU14 K3 passes the same31 registrations,
23 focused cells×20 and32 enabled callers; target peak3.72GB stays below its
4GiB BOM, source15676 records are unchanged and its private producer is removed.
Main inspected the raw source/terminal/admission records and32-case sanitized
XML; artifact scopes remain K3, never later live Main. The exact old GNU14 object
still fails the focused mean/RoPE test4 times, and the exact unguarded-score CPP
counter aborts under fatal UBSan with offset2080. These are deliberate outside-
source diagnostic controls, not repaired positive producers.

K3 actual per-symbol code eliminates every inspected fused FMA and tail memcpy,
but still spills: GNU14 FP16 summary has12 vector stores/reloads; Clang FP16
TablePlanes reloads7 invariant vector slots per iteration. Other inspected
FP32/BF16 summary/norm/table paths and produced pool phases are clean. K3 is
therefore **not zero-spill/economy qualified**. Source review77 additionally
finds the new common sanitizer decoration breaks legal perf-only omissions,
and the focused cancellation only exercised a scalar rotary pair. The next
**untested K4** passes failure policy atomically through the directory registrar,
adds a config-only wrapped-omission regression, and pins rotary2/16/18 (scalar,
fully native and mixed) cancellation bytes. Its pressure hypothesis sequences
right-subtree original roots after left summaries and places half-cast register
entry/exit dependencies to prevent interleaved integer live sets; neither changes
machine arithmetic, input ownership or the original butterfly. The existing GNU
`always_inline` lane helper is explicitly inline, not warning-suppressed.
No FP environment, tolerance, source timeout or global contraction flag changes
are used. Source-only ACK78 closes77P2/P3 without execution credit. Sealed K4 tree
2d70cc8c4110d75e19d3af6188f44614587fa493 then passes32 callers and31 registrations
on each of the three native profiles. Genuine GNU14 adds23 focused cells×20;
all three
IndexerProgram filters independently select/run1, and individually forced Scalar
runs select1/fail native admission1/no skip on each profile. The old GNU14 object
now fails12 cells, covering both stages/aliases and rotary2/16/18. The old K3
registration module fails only the new perf-only omission case; current3/3 pass.
The actual original Clang CPP object is independently byte-reproduced and passes
both mean/RoPE stages and all three widths, not just the earlier ordinary oracle.

K4 vector-pressure spills are zero in all inspected prepare/pool variants on both
compilers, with no fused operations or memcpy in those specializations. Scalar
pressure is **not** zero: GCC FP16 reloads position once per native table fragment;
Clang reloads position in all three representations, plus output+half in FP32/
BF16. All six variants reread source/dimension and branch per fragment. Separate
fringe-loop pointer/position spills remain; these are not analytic call saves or
normalized-buffer reads. K4 still emits17 GNU always-inline warnings after the
lane-only fix. This is static evidence, not PMU/economy or whole-engine support.

K5 binds table roots once at the original position, or borrows
original setup-owned analytic frequencies. A typed source/value geometry drives
the complete native/mixed rotation without rereading output-aliasable p fields.
No original frequency array is copied or recomputed; the two supported indexer
sources remain TablePlanes and AnalyticTheta, while the core-only engine source
is still rejected by indexer admission. The produced normalized128 stage and
original arithmetic/cast/fringe/copy orders remain. Remaining score/transpose
helpers are explicitly inline, never warning-suppressed. Sealed K5 tree
97f376d2d3ac939d29180073a9efc3c1cc032d2c passes baseline/strict32, but optimized-
sanitizer is **RED31/32** with42 FP32 analytic64 assertions (28 query,14 pool),
starting at query word2066 and pool word271; there are no sanitizer diagnostics.
The tester stops before GNU14, resources or controls. Source-only ACK79 cannot
establish compiled libm call equivalence or override that actual numerical RED.

The new code shape let Clang merge production cosf/sinf into Apple's paired
`__sincosf_stret`, while the optimized-sanitizer oracle kept the public unary
calls. A separately labeled native diagnostic over1,383,688 angles finds8691
cosine and14556 sine word differences, including angle0x3a83ef22. Its private
paired call is diagnostic only, never installed production. Thus earlier
Clang oracle bridges do not prove a compiler-independent elementary-function
contract; their mean/FMA/table scopes remain distinct. No expectation/tolerance
is relaxed, and no failing source producer is retried silently.

**Untested K6** defines the actual host program as the two original public
cosf/sinf functions. A native register identity between completed cosine and
its unchanged sine argument prevents pairing after inlining without a global
flag, FP-mode change or private ABI. Both the common host rotary helper and the
bound analytic source consume that primitive; CUDA/HIP device policies are
unchanged, but their new-header compilation remains unverified. The independent
oracle uses test-only volatile public-function pointers, not the production
factor algorithm. A new explicit Unit/preflight public regression isolates
unary words through genuine theta10000 setup invariants at positions1/4/8/16/32,
with inverse1, paired normalized1/0, all guards/aliases and original-frequency
immutability; it also pins the divergent primitive angle against the live SDK.
Sealed K6 tree8e8fd965ff0309005f1ed5ffe98060074c8d6c9f passes the existing32
cases on all three native profiles, including the formerly failing broad
analytic oracle, but its new case is **RED** before the alias call. At position1,
legitimate high-pair cosf values round to1, equal to original input1, violating
the fixture's all-word poison precondition. The primitive/disjoint-position1
checks pass; that does not certify remaining aliases/positions. No sanitizer
reports occur, and no libm-symbol/GNU14/resource/control work follows the RED.
The initial sealer also preserves a missing-K4-Git-alternate metadata failure;
its corrected full-chain tree seal precedes source execution, not a retroactive
claim that the first metadata attempt passed.

Untested K7 repairs only the fixture: original query2, mean2 and epsilon2 give
exact inverse1/2 and the same normalized1/0 elementary-word isolation. Every
original active word now differs from its expected output, even cosf==1; all
poison/guard checks remain, not skipped or weakened. Production K6 code is
unchanged. Sealed K7 treeb06114c9b1919654f7f6561bce86987b6cfc7f5a passes33
callers/32 registrations on three native profiles and genuine GNU14, plus24 GNU
focused cells×20. All four numeric filters select/pass1 individually and select/
fail native admission1 under explicit Scalar. All positive logs are sanitizer-
clean; GNU warnings are0. The base tree/blob/symlink/gitlink closure is now
resolved before patching, with the K6 metadata first-RED retained separately.

Actual K7 production objects import only public cosf/sinf (not a paired ABI),
and all18 native-profile executables have no sincos import. Current-object
controls pass; the actual K5 object fails the new public regression4 times and
the broad independent oracle42 times on both Release and optimized-SAN. Mach-O
control relinks have identical text/symbols/other bytes except fresh linker UUID
and ad-hoc signature hashes; they are not mislabeled byte-identical ELF links.
Source-only ACK80 closes the K6 fixture/source review without execution credit.

K7 complete actual caller/helper analysis proves zero stack traffic/calls/policy
loads in every table fragment/fringe/copy loop. Scalar resource debt remains:
GNU FP16 analytic suffix copies reload normalized/output pointers per iteration,
GNU BF16 reloads output; Clang-inlined FP32/BF16 pool loops reload four raw-root
words per coordinate. Analytic library-call saves remain separately classified;
its scalar fringe is unreachable under the admitted64-wide policy. No broad
matrix or economy certificate follows those unresolved pressure findings.

K8 seals the already-admitted analytic half32 in the typed rotation,
failing fatally if that post-admission geometry is lost rather than silently
substituting a width. This removes unreachable analytic fringes and exposes the
fixed suffix. One outlined whole128-coordinate pool producer binds four original
roots once, bounding register liveness outside the large OpenMP request lambda;
it is not a scalar/whole-row replay or operand mirror. Arithmetic, original
normalized/pooled stages and all K7 tests remain unchanged. Source-only ACK81
finds no new ownership/order/geometry issue. Sealed K8 tree
3eb767b4f8747c022ce08e40b5fd03d3ec6b1638 passes all33 callers/32 registrations
on three native Mac profiles and GNU14; GNU repeats24 focused registrations20
times with zero warnings, and fatal sanitizer logs have no diagnostics. The
actual27-function prepare/pool/rotate closure has no vector OR GPR stack access
on any call-free loop cycle on either compiler/all three representations.
Native analytic loops keep four public cosf/sinf pairs and only separately
classified AAPCS call saves; table loops remain free of stack/policy reloads.
The current-link old paired-call object still fails4 new/42 broad byte asserts;
the current control passes. These are not whole-QSA zero-spill/economy claims.

The same inspection discovers unchanged GNU14 scoring-loop GPR debt outside
preparation: fifteen unrolled key-fragment offsets pin registers, spilling the
ISA reference/head bound per head and row/start/block/bound per block while
reloading original policy roots. Its32 native dot FMAs are intentional, not a
preparation-contraction defect. Broad qualification is held, not used to hide
this old caller frontier. Untested K9 binds the original query/pool/score roots,
visible count and ISA value to one outlined whole score phase per ranking row;
no per-head/block call or complete M1 indexer replay is added. The original
four-head native dot/fold/finite/ReLU/add/scale and OpenMP row/selection ownership
remain unchanged. Its new public-indexer regression independently checks live
scores and exact selected/count output plus every inactive/unused guard around
the512-pool threshold, all three representations and grouped M1/3/16/33. The
produced query/pool stages are that score oracle's inputs, not a replacement for
the independent preparation oracle. Unit and explicit ProductionTestPreflight
registration are added. Source-only ACK82 finds no new production issue, while
correctly limiting the initial score oracle's claims: its produced query/pool
inputs do not certify preparation, and explicit cutoff-tie/reversed-head
falsification was not yet demonstrated.

Sealed K9 treef559d3ed78bae5fcf33a71a310f72908623221c0 passes34 callers/33
registrations on all three Mac profiles and GNU14, whose25 focused cells each
pass20 times with zero warnings. Fatal sanitizer diagnostics and paired libm
imports are absent. Actual prepare/pool/rotate text remains identical to K8,
and both scoring callers have no pressure stack accesses on call-free cycles.
Clang scorePools is clean. GNU14 now hoists all128 pool floats across four heads,
using all32 vector registers and spilling seven q registers:112B stores/block,
448B reloads/block, plus finite-check/scale constant reloads. This is a real
register-pressure RED, not an ABI save; green bytes do not waive it. The new
score test passes the old K5 object because its arithmetic was unchanged; that
is positive math coverage, not a structural spill discriminator. Broad testing
remains held.

K10 ties the exact original pool root to each preceding head sum (zero
before head0) with an inlined native-register identity. It changes no pointer
bits, operands, arithmetic, policy, allocation or collective edge; it prevents
compiler lifetime hoisting, not runtime memory ordering or lease/publication.
The score fixture now includes explicit independently asserted head-order and
cutoff-tie witnesses in all three representations: orthogonal query heads give
X, X*2^-24, X*2^-24,0 against512 identical old pools, while the new basis pool
wins and forces an ordinal tie at the cutoff. Reversing head addition must
change the final scaled word. Source-only ACK83 verifies the unchanged root
and formal witness construction, not emitted scheduling. Sealed K10 tree
f8cb644d570a2434e2adfc04529237848764bcc7 passes34 callers/33 registrations on
three native Mac profiles and GNU14;25 GNU focused cells each pass20 times.
Warnings, sanitizer diagnostics and paired libm imports are absent. All29
profiled preparation/scoring caller/helper functions have zero stack accesses
on call-free loop cycles on both compilers; the20 preparation symbols retain
K8 text. GNU scorePools falls from32 to14 vector registers, Clang uses9;
intentional32 dot FMAs remain. ABI call-loop/semantic storage is distinguished;
this scoped static result is not whole-engine or economy qualification.

Exact copied-root controls reproduce the current object/links before mutation.
Removing only the identity call preserves34/34 mathematical results but restores
GNU's seven-q/constant spills; that is structural compiler falsification only.
Reversing head order fails the score test with scaled0x3f7e5f8e versus0x3f7e5f90,
but its fatal first-word assertion stops at FP32: actual FP16/BF16 failures are
not observed. Reversing the installed tie comparator fails all24 score cells
and the existing tie/replay regression across the tested native profiles. The
current-link old paired-libm object still fails4/42 assertions while score
math passes. All these controls remain distinct from positive-source tests.

K11 changes only score-test reporting: every original word is still bit-compared
and zero mismatches is required, but one nonfatal aggregate per cell lets later
representations execute. No production arithmetic, expected bytes, guards,
geometry, test name or registration changes. Source-only ACK84 finds the count
logically equivalent and bounded. Sealed tree
df5f28d4618f1c60846f4f186dd85bb09368e75a passes34 callers/33 registrations on
three native Mac profiles and GNU14 (25 focused cells×20). Its optimized Apple
and GNU production objects are byte-identical to K10, not newly relabeled ISA
proof. Rebound head reversal now actually fails all24 cells: three
representations×two demands×M1/3/16/33, on baseline, optimized sanitizer and GNU.
Tie reversal fails the same24 cells and the existing dirty-tie regression;
identity removal still passes math and restores GNU spills. Current copied-root
link controls and separate paired-libm controls retain their bounded scopes.

The complete configured core-free matrix on that exact immutable K11 tree
builds all74 TUs and passes all229 registrations on each of five profiles:
Apple ASIMD Release, DotProd Release, strict Debug, optimized DotProd ASan/UBSan
Integration, and LLVM22 ASIMD Release. All179 registered preflight names pass20
times per profile:17,900 actual raw pass records with exact inventory matches.
Every test retains30-second timeout and fatal UBSan environment; all25 individual
native numeric filters run one positive case, and all25 Scalar controls reject
native admission without skips. Four configure-only CMake metadata cells are
not native-execution proof. Before/after15676-entry manifests are byte-equal;
only the separately declared read-only mode freeze distinguishes them from the
original seal. Main independently audits the raw data and source relation, not
newly executes these tests.

Actual LLVM22 and Apple DotProd Release objects are separately reproduced and
profiled:16 preparation/scoring caller/helper functions each have zero stack
accesses on call-free cycles, scorePools uses9 vector registers/117 instructions
with32 intentional dot FMAs, preparation has no FMA, and public analytic unary
calls retain their ABI saves. This is scoped static compiler evidence, not
whole-engine/economy or all-family certification. The110-file first data seal
omitted build/configuration logs; its over-broad wording is preserved and a
separate216-file supplement authenticates those logs, all74 actual compilation
commands per profile, objects and resource records.

Twelve RMSNorm omp-simd transform warnings remain OPEN in the optimized
sanitizer build (four per TU in three targets), not hidden by the passing tests.
They concern the explicit scalar summary/generic FP32 apply paths after native
ARM dispatch, not a demonstrated QSA or native-fallback defect. The other four
profiles have zero warnings. See [GitHub issue36](https://github.com/Llaminar/llaminar/issues/36);
no sanitizer, warning or arithmetic policy is weakened. Complete indexer
M/positive-worker totality remains open: the existing all-worker QSA lifetime
sweep covers attention, while legacy indexer grouping uses M15/three workers
and this independent score regression uses four representative M values with
one caller. Neither the configured matrix nor these timings certify serving,
current native-x86, PMA, model/HTTP behavior, or the final Perf__ round.

K12 introduced twelve real format/tail integration cells for that indexer
coverage gap. Each independently derives prepared queries, new pools, scores,
selection and counts from original raw/tail/gamma/table inputs, once per geometry
rather than once per worker. It sweeps M1..16/31/33, all prefix remainders0..3,
nonzero position origin, both query-demand policies, inactive capacity and every
caller-admitted positive OpenMP team size. Persistent test banks are re-poisoned
without rebinding. Standalone and retained calls authenticate every output and
source guard; retained thread0 checks immediately after return before the outer
join, with no test barrier that could repair a missing kernel completion edge.
Each parameter has its own unchanged30-second timeout. A named CPU-team resource
serializes only these whole-team cells against one another, never caps workers;
the common directory registrar owns that resource with fatal sanitizer policy
and perf-only omission. Its real LANGUAGES NONE metadata fixture checks both
registration contexts. Production source and arithmetic are unchanged. No K12
build, numeric, timing or totality success was observed. Its actual baseline
configure and metadata-only policy checks passed:241 registrations,191 preflight,
all timeouts30 and fatal UBSan environments, exactly twelve team-resource locks,
and both policy filters executing three LANGUAGES NONE metadata cases. The tester
then held before any native compilation or GNU admission. A slow shell-based
source-manifest collection hit the tester's tool timeout; a separate Python
collection authenticated unchanged source. This was not a native/test failure.

Source85 found a test-authentication gap: vector<float> equality permits signed
zero changes, so K12's source-byte claim was unproved. Unverified K13 replaces
all seven source comparisons with exact word authentication. Supplied finite
dyadic rotary planes now vary by position and pair rather than hiding address
errors behind identity coefficients. Synthetic parameter/filter prefix
OriginalInputs replaces OriginalReplay, and the fixture comment describes each
unchanged timeout budget without claiming an unmeasured bound. The existing
explicitly registered lifetime regression additionally calls the real indexer
and selector with delayed participants at short/ranking prefixes, all three
representations and live/empty frontiers. This is an arrival check before the
outer join, not yet an executed missing-barrier countercontrol. Production code
remained unchanged. Subsequently, the sealed old-base K13 Mac slice passed all
three native profiles:46 enabled cases,45 scoped registrations and17 individually
filtered positives plus17 real Scalar-admission negatives per profile. Its
largest new-cell CTest duration was4.31 seconds in strict Debug; this is not an
economy result. Source guards matched and the scoped production objects retained
their prior byte/text identities. No K13 GNU or proposed countercontrols ran.

An operator-requested local rebase replayed all nine ARM commits onto upstream
f48c856105f6d1495a4c342ed2cc0c62b4d11273 without conflicts. Ordered added/removed
patch payloads were identical, and all seven dirty K13 files were restored byte
and mode exactly. The local committed head isfe4d37d19482c3161d7cdb0698a96027d9e0d68e;
the remote PR was not force-pushed. Exact workspace treea97be2f7a4a77c1531c4735ea293ab5f7f854dc7
was separately sealed as K14:15786 blob/link records plus one uninitialized corpus
gitlink. Every raw input/tree/snapshot payload and mode relation matched. Source87
found no narrow compatibility issue in the eight overlapping files and direct
native-collective headers; this does not audit every upstream change. A fresh
K14 baseline configure, metadata/policy checks, actual attention target compile,
FP32Tail0 sweep and Completion case passed. The other profiles, GNU, controls and
complete engine gates remain unqualified on that base. No old whole-tree result
is relabelled as K14.

A separate unverified scalar-RMS successor declares increasing-column FP64 FMA
for original FP32 words. Finite FP32 squares are exact in FP64; the existing
OpenMP SIMD reduction nevertheless allowed an unspecified lane fold. The new
source removes that reduction directive in favor of an explicit arithmetic
chain, consistent with the existing scalar per-row program; native ISA trees
are unchanged. The existing detail seam receives an independent volatile
product/add byte oracle, a2^27-plus-sixteen-ones discriminator, varied anchor
positions, widths/tails and finite boundary words. The focused
V2_Integration_CPURMSNormScalarFP32Summary entry joins ProductionTestPreflight.
K15 subsequently passed its three Mac RMS profiles:34 actual registrations and
whole binaries9/15/39/3 cases per profile, including the focused scalar case both
native and explicitly Scalar-forced. The current object was reproduced byte
exactly. Rebinding the old K14 object into current links failed291 assertions in
Release, but correctly passed O0 and optimized sanitizer builds where the old
reduction was serial. Those profile-dependent controls retain their actual
outcomes. Optimized sanitizer compilation reduced the original twelve warnings
to six: only the unchanged gamma/no-gamma apply loops remain across three RMS
translation units. Source guards matched; GNU/x86/economy remain unqualified.
Mixed finite-boundary fixtures contain a large anchor and do not independently
certify tiny-only arithmetic. The shared scalar leaf also reached inherited
uncompiled-ISA stubs; K15 alone did not certify admission/fallback closure.

Unverified K16 replaces those remaining generic forced-vectorizer application
loops with compiled scalar/AVX2/AVX512 row publishers, selected once before the
worker region. Native ARM keeps its existing shared-row publisher. Compiled
fragments and bounded tails preserve the separate input*inverse then gamma
products, including identity gamma and exact input/output aliases, without
scratch or a new admission authority. Unsupported generic row ISAs fail rather
than substituting scalar; the two uncompiled FP32-summary alias stubs are retired.
A new public-primitive regression uses exact mean9, inverse1/sqrt(12), subnormal
and signed-zero gamma words to discriminate premature inverse*gamma rounding;
it covers M1..16/31/33, widths1..65 and representative tails, inactive/outer guards,
exact aliases and representative retained calls checked before the outer join.
Its focused FP32Application preflight is explicit. K16's first baseline build
and focused application case passed; all three optimized-sanitizer RMS translation
units compiled with zero warnings (12→6→0 in that exact closure). Baseline ARM
scalar publishers retained separate vector/scalar products without hot helper
stack traffic; the caller has one indirect phase call and two ABI-boundary stack
loads per row. Instrumented frames are not a performance certificate, and x86
publishers have not run. Source89 found no production math/bounds issue but two
new-test authentication gaps: native intent lacked a self-admitted native pair,
and requested3-worker teams were not checked. Unverified test-only K17 adds both
checks and scoped dynamic-policy restoration. Its Scalar-forced application
filter must now reject rather than certify a native subcase; the separate scalar
summary filter remains a legitimate Scalar positive. Collective numeric-exception
handling, broader resources/economy, GNU/x86 and complete engine gates remain
open. Later K17 qualification passed all three RMS profiles with zero warnings,
35 registrations and9/15/40/3 whole-binary cases per profile. Its native admission,
Scalar-positive summary, Scalar-negative application and exact-team controls all
executed as intended. The one-line premature inverse*gamma mutant failed exactly
2808 weighted explicit-Scalar cells in each profile while native ARM-tile and
identity-gamma cells remained correct; current-link controls passed. Full K17
five-profile core-free qualification subsequently passed:243 tests once and193
preflight cells20 times per profile (19,300 raw passes),74 translation-unit
compilations from63 distinct sources and114 build steps/profile, zero
warnings/errors/sanitizer hits, clean ten JUnit
files, and unchanged15,761 regular files plus25 links. Main independently checked
all95 artifacts under seal `a216a2965b08f47cb27f6757f83b296abdd5fec4edf15cd4c19cebc5fb39a6d9`,
actual per-name repetitions, registration environments and raw RMS disassembly.
The eight-instruction scalar FP64 FMA chain and31/44-instruction pointwise
publishers are call/stack free; the caller retains one indirect call and two
per-row ABI stack loads (28 total Apple/29 LLVM stack accesses). These are static
resources, not economy measurements. Strict Debug's representation case peaked
at25.07 seconds without timeout; separate data inspection identified missing
PROCESSORS accounting for whole-team tests, not an excuse to raise the budget.
Standalone issues36/37
were consolidated into the active implementation work at the operator's request,
not closed as proof of resolution. The isolated arm-helper lane now owns the
remaining systemic numeric/lifecycle/half-policy/identity-gamma repair; Main owns
integration and actual engine qualification.

The native Spark SDK now has an independently checked mount-primitive PASS in a
separate SDK-only transaction. The first B4 attempt stopped before its admission
token and ran no SDK CMake: its physical-root capacity assumption and collector
finally-name shadow were preserved as separate defects. Functional mock regressions
reproduced the old cleanup failure and proved the typed B5 successor retires its
owned target even with logging errors. B5 then passed on the actual pinned SDK:
physical root absent-controller fields use fresh MemAvailable, controlled ancestors
retain their real limits, lock/removal/source-read-only guards pass, and the
mounted output root legitimately persists after its payload is removed. Actual
exit0/no-OOM, source guards, CID removal and absence are recorded. This is not
project source compilation or an engine/serving certificate. The complete-engine
BUILD-ONLY source/dependency guard and lifecycle protocol still require review.

The first canonical-hook binding attempt stops before execution: this immutable
snapshot is a plain file tree, so git root discovery fails128. It is not a hook
PASS or the expected macOS/libnuma failure. A separately owned exact-tree Git
checkout then binds that same source using read-tree/checkout-index, no commit,
no Main refs and no optional corpus initialization. The unmodified hook runs
once with a fresh outside build directory and fails1 at CMakeLists.txt:143:
`Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)`.
The clean environment supplies no libomp discovery hints. No Unit/preflight or
app build follows. This differs from earlier attempts with genuine OpenMP
already configured which reached the native-macOS/libnuma guard; neither
outcome is a passed gate. No
retry, SDK stub or altered hook is used. The scoped checkpoint preserves all
first failures, warning debt and a669's original sealed packet identities;
publication does not confer execution authorization or end-to-end completion.

The operator explicitly transferred all Spark/related WSL remote coordination
to Mac arm-sparks01a1b12f. It independently verified the Main three-file handoff
and outgoing26-file DATA transfer; no execution/window/observer grant moved.
Outgoing cleanup of owned WSL artifacts/images is a report, not an independent
observation or authority from the receiver's receipt ACK. Historical intake/SDK
paths may now be absent and need fresh authorized restoration; original seals
remain unchanged. Main retains implementation, local tester retains immutable
local qualification, and task39's joint UMA capacity gap still blocks target15.

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

## Original-Q16 RMSNorm slice (current worktree, not full integration)

The unchanged publice981 production TU fails all three newly focused cases:
incomplete32-value geometry silently floors rows and writes partial outputs,
FP32 expansion overwrites original compressed/gamma banks, and a retained empty
invocation returns after1/3 arrivals. Exact original source/hash and negative
logs/XML are retained outside the repository. The new checked expansion borrow
protects complete original source/output/gamma/options spans before worksharing;
it neither invents physical capacity nor extends a tensor/arena lease.

Native ARM now consumes1..4 original Q16 rows directly. Decode products and
squares round to FP32 **before** widening into eight fixed FP64 chains. This is
not floating RMSNorm's FP64-square program. Every physical M shares the same
K-chain/fold and original FP32 mean/sqrt/divide inverse; gamma loads are shared.
No decoded row, hot allocation or shadow coefficient bank is introduced.
Actual M1 bytes cover M1..16/31/33, original prefixes/canaries, K32/64/96/1056
and workers1..10. An independent256-shape program runs both default and explicit
`force_scalar` policies, retaining volatile FP32 edges and distinguishing widened
FP64-square results; stored `sum_qs` is not substituted for squared energy.
Reference reduction is selected from declared active ISA and scalar intent,
never observed output: serial scalar, eight-chain NEON/AVX2, sixteen-chain AVX512.
The separate K128 RN discriminator produces mean words`3c000000` and`3c000001`
for the respective eight/sixteen programs and checks their distinct inverses and
actual public bytes. Production AVX512 arithmetic has not been changed. Invalid policies reject, and numeric corruption inside an
OpenMP collective is fatally diagnosed rather than repaired to zero.

AVX2 now has a genuine vector Q16 body with the same explicit FP32-square
program. A named unavailable AVX512 entry rejects instead of silently calling
scalar; existing compiled AVX512 arithmetic remains distinct. Both actual x86
compiler envelopes first passed syntax; the separately authenticated native x86
correctness result below now closes that bounded runtime scope, not economy.
Current GNU14/native ARM passes15 production-TU,9 actual legacy-TU and3 reduction
cases. Final Mac five-profile170/170 and130preflights×20=2600 raw passes were
independently parsed from actual XML/raw records, without failures or skips.
The earlier167/128 checkpoint remains historical. Factory/runtime/workspace and
the remaining Q8/INT32/residual programs are unfinished.

Retiring the unavailable named AVX512 scalar substitution exposed six old Unit
calls: the unchanged actual legacy TU passes2/8 and fails6/8 with the precise
unavailable-envelope diagnostic. Those original negatives remain preserved.
ISA-neutral accuracy/edge cases now invoke public native dispatch, with the
original numerical thresholds; a ninth case proves real named AVX512 output or
precise rejection and untouched output. Available-entry assertions check every
finite output against the full reference with the unchanged strict`1e-5` bound,
not a NaN-blind maximum-difference summary. The canonical Unit name is preserved
in shared core-free registration using the actual production TU; whole-legacy
and reduction-identity regressions have explicit preflight entries. Source review
closed the caller/oracle findings, not x86 runtime or whole-engine readiness.

One exact baseline four-row assembly candidate initially loaded span pointers
from the hot stack and retained summaries around libc sqrt branches. Checked
contiguous original strides and SIMD-across-row hardware sqrt/divide remove
those hot stack loads/calls while preserving the independent byte program;
inactive inverse lanes are1, not0/0. The bounded optimized hot loops use28 SIMD
register IDs under AppleClang, with no SP access/calls; four d8..d11 ABI saves
are not hot spills. A separately compiled genuine native GNU14 baseline candidate
uses23 SIMD register identities across its bounded reduction/publication loops,
with no hot SP access/calls; its32-byte frame is GPR/cold-exception storage, not
hot SIMD spill traffic. These static resources are not achieved IPC/cache/
throughput, PMU evidence or a final `Perf__` certificate.

Native x86 correctness is now independently verified for frozen public73da,
not the later attention worktree. Old CMake/CTest3.28.3 could not satisfy the
real4.x metadata prerequisite. The original BuildKit attempt failed before any
RUN/install when registry metadata resolution ignored the intended local-only
boundary. Archive`445761da8119c28f4f7fc680f1d32b7df3cf953ba3979f57b3be74c6253b3ea8`
retains that RED and its unchanged35-layer base. The explicit exact-local-ID,
`docker create --pull=never`, authenticated offline-wheel successor first exposed
an ignored empty-entrypoint reset; imageac7a remains unqualified. An explicit
metadata-only successor sets the actual`/usr/bin/env` entrypoint and retains
exactly the same36 rootfs IDs, first35 identical to the base. Final SDKf492's
248 raw hashes,3989 installed records, official wheel/tool bytes, genuine
CMake/CTest4.0.3, LANGUAGES NONE SIGTERM/36, C++23/OpenMP/GTest/MPI-link witnesses
were independently authenticated. MPI linkage is not MPI initialization.

The first source73da/d18 native WSL Ryzen7800X3D/GNU14.2 producer then passed
20 actual registered cells per AVX2/AVX512 profile, including ISA/cache
preflights. Eight distinct Q16/legacy cells each ran20 times:160 raw passes per
ISA. Only after both profiles passed, the actual AVX512 binaries with explicit
runtime AVX2 passed14 RMS/workshare+9 legacy GTests, no failures/disabled/skips.
The x86 RMS inventory is14, not ARM's15. Source-owned TIMEOUT30 stays unchanged;
the SDK SIGTERM/36 witness is separate. Actual production-TU flags, six native
AMD64 ELF binaries and mixed-lane environment/commands are bound in the return.
XCR0=231, MXCSR=8064/`1f80` and rounding0 remain unchanged in the before/after
probes; this is not claimed as in-process sampling from every test thread.

Main independently verified archive`cb1c48f7a0cbaff24a9ce587d46ba5e98d1e4f53e6d44a1a1b94c737160a48e5`
(4832866B), all178 raw member hashes, actual XML and each repeated raw cell
count. The native private producer finished01:07:09.793293235UTC, exit0/PID0/
OOMfalse; peak1615630336B. A host terminal guard began the complete7063-entry
source-after map75.379ms after **terminal exit**, before container removal.
Before/after maps are byte-identical with SHA256
`42819719509e5a59870c8a3c03a047d0c0a6fdac6a336fa082dbfb0adae8c08f`.
The separately approved read-only/nonroot/network-none host-cgroup-namespace
observer captured actual target→docker→root ancestry; the source producer
retained private namespaces. Parent historical OOM counts are not this run's
OOM events. Owned containers were retired and the W4A4 capacity window released.

The earlier d072 private-namespace cgroup helper stopped before compilation or
source execution and stays an admission-helper RED. A later cancelled scheduled
observer reported exit2 with cause **uninvestigated**, supplying no qualification
evidence; actual terminal/raw tests/guard receipts determine the source result.
The relay's initial Mac verifier also lacked`hashlib.file_digest`; its pre-check
error is preserved and an explicit bounded SHA256-stream verifier succeeded
without changing the archive. None of these focused results certifies the full
engine/Unit/preflight, later AQ8 changes, model/GPU/cluster or performance.

The actual canonical hook was attempted again for this slice. Default AppleClang
OpenMP discovery failed; explicit genuine Homebrew libomp then reached the same
fatal`CMakeLists.txt:173` native-macOS memory/placement diagnostic. Neither full
Unit nor production-preflight target build/test ran. A published draft checkpoint
therefore uses the documented explicit hook bypass, disclosed as incomplete gate
evidence rather than a readiness waiver. Hooks remain registered and unchanged.

## 2026-11-06 original AQ8 attention operands (partial integration)

`CPUAttentionKeyQ8Dot.h` binds complete original query heads, immutable anchors
and an authenticated native ISA in a non-default-constructible immutable tile.
D64/128/256 and query widths1..4 share register-only decoded fragments, preserving
the original NEON two-chain, AVX2 eight-lane and AVX512 sixteen-lane dot programs.
No key head, decoded cache bank or coefficient mirror is materialized. Existing
NEON dot helpers are the component oracle; x86 references transcribe the class
loops, rather than claiming execution of private class methods. No universal
cross-ISA byte oracle is implied.

Independent RN controls distinguish separate decode words`36000000` from
reassociation`35800000` and an outer FMA`35c00000`; actual native quantizer
publication also supplies non-power-of-two scales. Original33-row strided query
banks,1..4-query tiles, all head widths, seeds/exponents, M1/component bytes,
readonly roots, malformed geometry and foreign ISA rejection are covered.
Each separately filtered numeric receipt rejects an empty native ISA inventory.

The initial four-query resource candidate repeatedly loaded query-root pointers
from its hot stack. Passing only validated roots by value and storing immutable
bindings removes those loads, without copying query/key/coefficient data. Isolated
D128/Rows4 loops use16 AppleClang or19 GNU14 SIMD register identities with no hot
SP access/calls. The actual canonical Release `Perf__` D128/Rows4 worker was also
disassembled separately: eight native FMA instructions in its K loop, no hot
stack traffic. These are bounded resource observations, not PMU certificates.

`CPUQ8AttentionValue.h` supplies explicit native NEON/AVX2/AVX512 original32-code
value accumulation. The FP16 scale is decoded canonically, weight*scale rounds
separately, and original signed codes enter intentional numerator FMAs. The
source/block intersection and complete32-value output checks precede stores;
only explicit scalar diagnostic intent selects its scalar program. All signed
codes, seven half scales, four weights, canaries/readonly metadata and an
independent memory-rounded combined-weight/FMA reference distinguish premature
code scaling. This replaces the class's plain ARM/AVX2 V loop as well as the
existing AVX512 leaf, so **symmetric Q8 attention also needs full-core validation**;
AQ8 dot resources alone do not qualify the value kernel. Separate native32-code
value candidates show eight intentional FMAs,5 AppleClang/11 GNU14 SIMD identities
and no hot SP access/calls; cold dispatch/validation and ABI saves remain. This
bounded ISA inspection is not measured value-kernel economy or full FA2 proof.

Source review exposed a real integration boundary: both AQ8 entries and the
adapter include were still x86-only. Only AQ8 was separated from those Q8/TQ
sections; other guards were not opened indiscriminately. Genuine native GNU14
preprocessing finds zero AQ8 definitions/calls in the frozen73da class and one
actual definition/three occurrences after repair. Explicit FP32 production-class
and new regression-TU syntax now compile the native adapter and its Q8/TQ
value/rotation dependencies. **Syntax is not linked/executed FA2.** Query widths2–4
remain leaf-only: the canonical adapter currently binds width1.

`V2_Integration_CPUFlashAttentionAQ8NativeEntries` is an explicit full-core
preflight. It uses the real ring factory, physical NUMA endpoint registration,
setup-owned workspace manager, native tensor/verifier/two-request APIs,
D64/128/256, Q8/TQ4/TQ8 values, M3 and worker budgets1/3/10. Every publication bank,
including serial oracles, is NaN-poisoned before its independently asserted call;
finite and strict full-byte checks cannot inherit a previous writer's result.
This is origin0/capacity17/window7 with one canonical partition, not wrap,
context-summary lifecycle or production PMA admission. The test has passed actual
native GNU14 syntax only; **full-core linking/execution remains unqualified**.

The final focused worktree checkpoint is176/176 in each of five Mac profiles,
136 preflights×20=2720 independently counted raw passes, without fail/skip.
Native GNU14 runs all15 codec/dot/value/authentication cases. Earlier172/132,
173/133 and175/135 checkpoints retain their narrower source scopes. The complete
native engine/gate inventory is not represented by those counts.

The canonical `Perf__CPUAttentionKeyQ8.cpp` now cold-binds exactly one native dot
D/query-width/ISA candidate per process. All authentication precedes warmup and
seven timed samples; no head reconstruction, binding or allocation is timed.
Complete NaN poisoning and shared nonempty/finite/bitwise authentication reject
no-op, partial, identical-NaN and empty banks, including the admitted all-zero
heads1 fixture. Raw dot samples are retained in CSV. See the stable focused-build
and benchmark procedure in `tests/v2/portability/README.md`.

One bounded unpinned M1 Max Release experiment, three fresh processes per shape,
records median per-query widths1→4 of10.205→5.356ns (D64),21.462→9.961ns (D128)
and39.127→19.046ns (D256):1.906x/2.155x/2.054x register-fragment reuse. Raw samples,
binary digest and exact-symbol ISA evidence remain outside the repository.
**This is neither placement/PMU/scaling nor FA2/MTP/server throughput evidence**;
the width1 production callback cannot inherit the width4 helper's economics.
Remaining attention-exp tails/rescaling, grouped-query scheduler wiring, actual
workspace/cache/request lifetimes and final all-family `Perf__` still apply.

The actual canonical hook was repeated for this attention checkpoint: default
AppleClang OpenMP discovery fails; after explicit genuine Homebrew libomp setup,
it again fails at`CMakeLists.txt:173` because native macOS memory/placement is not
implemented. Full Unit/preflight never build or execute. The explicit documented
draft-checkpoint bypass remains disclosed, not a gate/readiness waiver; neither
the focused176/136 matrix nor the old73da x86 pass replaces this missing gate.

## 2026-11-06 immutable-lookup engine frontier and native row repair

The fresh, separately admitted73da/5adf ARM engine attempt consumed its one
BUILD-ONLY authorization. Exact SDK4e8a/indexd211/configa78, GNU14, baseline
ASIMD, normal Integration and unrestricted Ninja configure successfully in
93.262s; the app+two complete gate-target build fails after222.436s at
`CPUImmutableLookupKernels.cpp:17`:`immintrin.h` unavailable. This is a **new
immutable compiler RED**, not the old c76 KV append header and not the newer
attention source. Actual2138-command parent metadata and3430 CTest SHOW-ONLY
entries/3129 genuine SIGTERM/36 pairs exist; zero tests execute and no application
executable exists. The prior KV append repair now has a successful Ninja object
record; a later read-only collector reports its7970392B/hash, not an app pass.

Main independently verifies return archive
`984a29d6cec4b0f8311922008cd9bc5fdd78f9259b517d6d1285826a2722c863`
(2677349B), all196 raw member hashes, actual failed command/log, configuration,
terminal and source maps. Complete7063-entry before/after maps are byte-identical
SHA`42819719509e5a59870c8a3c03a047d0c0a6fdac6a336fa082dbfb0adae8c08f`;
the host guard starts201.801ms after terminal exit. Peak37560524800B, OOM0 and
throttle0 expose no resource-shortfall explanation. Fresh owner/cgroup/tool
admission, physical package36/20-core topology and dependency ownership remain
recorded. Owned producer/collector retire and the lab window releases; **no
patch/retry, CTest, model or GPU work follows**. The b9d metadata-helper RED and
corrected43-entry SDK-chain witness remain separate observations.

The exact original CPU TU hashes
`43443a51b43dd5118cfa3fdbeabc7eea8245a8b24f7e4a633dc93d6dfc438385`.
Actual native focused compilation reproduces its architecture-header failure.
A diagnostic copy minus exactly that include still fails at unconditional SSE/
SSSE3 IQ4 setup: header removal alone cannot implement the native operation.
No x86 emulation/header stub or whole-operation scalar replacement is installed.

The actual production TU now widens FP16 with canonical `NEONFP16`, shifts
original BF16 words exactly, and uses register table lookups/integer widening/
one FP32 product per IQ4 coordinate. Source half bits and the canonical signed
IQ4 grid remain authoritative. No intermediate decoded row is constructed;
the requested live FP32 publication is the operation's output, not a cache or
weight mirror. Original generic per-coordinate arithmetic for the other20
quantized formats is unchanged, not newly certified SIMD/economy.

`NativeEmbeddingLayout` admits incomplete logical rows with complete physical
last blocks. Old `count*width/32` flattening could publish nothing or interpret
padding as another row. The replacement owns every original ceil-block slot,
clips its live output and preserves original row coordinates, including width1.
AVX2/AVX512 also receive bounded masked fringe stores without changing their
full-block products; mixed512-codegen/AVX2-runtime stays admitted. Complete extent,
overlap and geometry checks precede `beginWorkshare`; valid zero work completes
a retained team without a standalone team, and no source scan follows release.
These synchronous checks neither prove capacity nor extend service/PMA leases.

The actual-TU Unit and four explicit preflights cover all24 source representations,
original partial widths, poisoned independent/actual-M1 outputs, readonly source,
canaries, aliases/wrapping geometry, workers1..10 and retained empty/nonempty
publication/reuse. Independent integer controls exhaust all65536 half/BF16 words;
separate format-specific tiny tapes put actual NaNs/subnormals/signed zero into
bounded fringes and assert that coverage. Every guard word is checked, with
adjacent/interior corruption negatives. Independently filtered numeric receipts
require an admitted active non-Scalar ISA; an actual Scalar NativePayloads
negative rejects all three selected cases, without skips. Source-only reviews
64–66 closed the native-admission/guard/fringe fixture holes; no reviewer ran the
implementation or qualified runtime from inspection.

The final source passes185 registrations in each of five native Mac profiles
and142 preflights×20=2840 independently counted raw passes. Genuine GNU14 with
actual O3/g/NDEBUG commands runs8 kernel cases,5 registered lookup cells,4
preflights×20 and the separate2 compiler-policy cells. The older6-case GNU14
unoptimized witness and intermediate185 matrices keep their narrower scopes.

Diagnostic countercontrols retain native arithmetic and restore **only** the old
coordinate or empty-completion policy in copies of the repaired TU. Width1 with
three original rows changes from0 wrong words to3 untouched poison words;
empty retained calls change from0 to2 early completions. These are labelled
isolated-policy controls, not execution of the unmodified frozen TU, whose
native compilation remains RED. Separate actual optimized IQ4 worker inspection
finds two native table lookups/eight vector products, six SIMD identities and
no vector spill in those math regions. AppleClang still reloads the OpenMP
iteration bound from one GPR stack slot between blocks; this is not described
as zero hot-stack traffic. Static resources are not PMU, timing or scaling.

A fresh immutable successor must prove full app/gate compilation/linking and
actual AQ8/symmetric-Q8/core/runtime tests; isolated lookup repair does not
qualify service I/O, PMA admission, graph/request lifetime or inference economy.

## Focused Integration compiler-policy correction

Inspection of actual GNU14 commands, rather than the configuration label,
exposed another defect: the focused project set custom Integration CACHE flags
**after** `project()` had initialized them empty. The first6-case GNU14 lookup
witness and old sanitized Integration profile were unoptimized diagnostics.
Historical numerical/sanitizer passes remain recorded, but are not reclassified
as optimized evidence. Main engine declarations already occur before `project()`;
the actual73da engine command retains O3/g/NDEBUG/custom assertions unchanged.

`V2PortabilityBuildTypes.cmake` now seeds C/CXX Integration initializers before
language enablement, only when undefined. Both normal/pre-include and CLI-cache
`_INIT` values, and authored final flags, retain their exact identity. Two focused
real-CMake negatives reproduce the initial initializer-clobber mistake before
repair; a literal-newline fixture parse error is preserved separately. Existing
empty final caches fail closed instead of being forced into another policy.
The old sanitizer tree correctly fails that admission in matrixxNle22cE. The
explicit new `build_arm_portability_dotprod_sanitized_optimized` tree has actual
O3/g/NDEBUG plus ASan/UBSan for both languages; the old tree remains untouched.

Five device-free tests call genuine CMake per-config initialization with
LANGUAGES NONE; they prove cache lifecycle, not compiler execution. A separate
ordinary target compiles real C and C++ witnesses and executes their
optimization-present/NDEBUG bits in Unit/preflight, including installed runners.
That binary does not prove exact O3/g, custom assertions or every TU; actual
compile-command evidence supplies that narrower proof. No runtime compiler SDK
or executable temporary mount is required. An earlier compiler-in-/tmp fixture
failed on the real noexec mount and remains preserved; it was retired rather
than relaxing mount security or silently skipping execution. The local helper's
invalid explicit private-PID spelling likewise failed before container creation;
its command/helper remains separate from the corrected default-private run.

The actual canonical hook was repeated after both repairs, with genuine cached
libomp: it still fails at`CMakeLists.txt:173`, the explicit missing native-macOS
memory/placement backend. No complete Unit/preflight target build/test follows.
The disclosed draft-checkpoint bypass neither changes the hook nor waives its
failure. Fresh source-bound native x86 and ARM engine validation are still needed.

## Configure-host topology and evidence retirement

The returned native Spark archive was independently SHA-checked together with
all100 raw file digests. Original sysfs records and inherited affinity describe
package36,20 singleton sibling groups and clusters56/1144. The actual `lscpu`
CSV instead aliases20 CPUs to10 socket/core pairs; neither presentation labels
nor cluster count are physical package authorities. A4.2KiB source-pinned,
device-free recorded fixture retains these inputs and their archive provenance.

`CPUHostTopology.py` now captures Linux inherited affinity and original kernel
sibling sets once, validates IDs/self-membership/reciprocity/nonoverlap and
checks affinity again before returning. Sparse interval parsing never expands
large ranges. `V2TestHostTopology.cmake` binds this immutable observation, not
unchecked shell pipelines or missing-field defaults. Existing shared rank
budgets admit only a genuinely uniform accessible package/SMT projection;
asymmetric vectors remain explicit and fail with a precise unsupported-policy
diagnostic rather than using a minimum, maximum or invented one-core geometry.
This configure-host scope is not runtime/rank topology or asymmetric scheduling.

Admission retires the previous receipt and interrupted temporary output before
HostOS, observation or projection can fail, then atomically publishes complete
JSON by same-directory rename. The focused two-attempt owner regression repeats
failed projection/observer admission and proves both files absent. On Linux the
observer-negative reaches the deliberately missing interpreter; on Darwin it
stops at the genuine HostOS guard. The registered recorded CMake regression
builds the real early/late Unit and preflight marker-target inventory and checks
all three PROCESSORS values. Nine metadata/actual-CMake cases pass on Mac and
the native Linux SDK; a separate native observer/gate fixture authenticates the
VM's actual1-package/10-core geometry, not the Spark's20-worker inference.

## Genuine CTest ownership-metadata closure

The old genuine SDK's CMake/CTest3.28.3 supports runtime timeout signals but
omits `TIMEOUT_SIGNAL_NAME` and `TIMEOUT_SIGNAL_GRACE_PERIOD` from actual JSON.
The same compiler-free production-shaped probe exposes SIGTERM/36 under
CMake/CTest4.0.3. The new focused test fails on unchanged3.28 and passes on
native4.0.3 and Mac4.3.1; missing fields are never manufactured. Official3.31
serializer source lacks these recognized properties;4.0 source includes them.
This interface proof does not replace real MPI timeout/retirement execution.

The exact new native SDK runs SourceAdmissionDiscovery19/19, QFN source closure
4/4 and production-parity **Unit metadata**130/130 with real original-helper
bytes. These are not model campaigns. It also compiles/links/runs genuine
C++23 print/atomic-shared-pointer/GMock and current native RMSNorm8+3 cases.
Actual full factory/engine/runtime qualification remains open.

The common image installer pins genuine Kitware4.0.3 next to Ninja; test builds
reject older tools before heavy dependencies, and installed runners retain the
builder's exact entrypoints plus native/package data. The replaced devcontainer
latest-Kitware producer and fixed ordinary-build concurrency environment are
retired; structural controls complement actual tool-output tests. A newer tool
requirement retains the original compiler-policy envelope. The first LLVM
attempt accidentally opted into module scanning and failed on an absent module
response file; its RED remains, and response authentication was not weakened.

New explicit SDK archive SHA256:
`4e8a5adc654a7f6d571f3718248bae4a59266d2f871ab9b9176b73897b78e434`
(320776402 bytes). OCI index`d211c0e0…68bf0`, native runtime manifest
`06a78c8b…f42bda`, config`a78c51c7…039d1`: all9 encoded layers and uncompressed
rootfs diff IDs were verified, retaining all7 exact oldc813 base layers.
Old SDK/source artifacts remain unchanged. New identity must be explicitly
admitted for a new immutable build-only successor; export is not remote import,
retry authorization, CMake4 vendor/full-image certification or a serving proof.

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
Main independently rechecked its archive digest/all100 raw record hashes and
inspected the original sysfs/affinity, actual command/compiler and cgroup failure
scope. Configure alone failed; app/gate compilation never began. The stopped packet remains immutable; a source-owned
heterogeneous topology fix and a new authenticated successor precede any retry.
No model/GPU/corpus/cluster work or broader test campaign is authorized by this
build-only attempt.

## Genuine native engine configure successor and compiler frontier

New immutable packetc76b306f…72f74/publice981 with explicit SDK4e8a/indexd211
received one fresh lab/capacity-admitted CPU-only build window. Main independently
checks returned archive`37a73cb48f14a1b12775350d28a7e9deab2570ac7c78bf79fe8e5e6d6dad7014`
(2308502 bytes), all124 raw hashes, all7060 original source-before records,
actual native config/rootfs identity, real compiler command and topology fields.
Actual parent configure/generate **passes97.265s**:1 accessible package,20
eligible cores,1 available thread/core and OMP20, matching original package36,
core/sibling/affinity0..19 records. CTest4 show-only exposes3420 entries and
3119 actual SIGTERM/36 pairs; **no tests execute**.

The ordinary unrestricted app/complete-gate target build then **fails90.352s**
at the unused unconditional `<immintrin.h>` in KVCacheAppendStage.cpp:31.
Private cgroup peak19,521,458,176 bytes, OOM0; terminal PID0/exit1, no hidden
success. Genuine parent2136-command database now exists. The source-after
comparison is recorded as a successful7060-entry verifier summary, not a retained
immediate per-file map. A read-only supplement later captures the unchanged owned
source at22:25:44.222002–22:25:44.602449UTC. Main independently verifies archive
`8db22132…66726`, all10 raw hashes and exact equality of all7060 per-entry
before/after records (bytes/modes/UID/GID/kinds/link targets). This later capture
is explicitly not retrodated or reconstructed into immediate evidence. No new
container/build/package/config activity runs. Prior packet/SDK/results stay
immutable and the lab window was released.

Main removes that unused header without adding an ISA branch, scalar substitute
or model-stage arithmetic. A focused Unit/explicit preflight keeps direct ISA
headers out of this publication stage; native codecs remain authorities.
The **actual current stage TU** passes genuine GNU14 syntax with the exactc76
production command and genuine NUMA/oneDNN/dependency closure. The corresponding
Mac syntax attempt remains RED on unavailable real `numa.h`; no stub is supplied.
Syntax is not an app/gate build. A new source successor and fresh authorization
must precede any additional native full build; never patch/retryc76/3bf/662.

The owner-requested commit-message cleanup changes no trees: publicf5b3ab56e
is tree-identical to originale981, and tag`arm-cpu-checkpoint-e9811f0a2` retains
all original authenticated ancestors. Frozen packet identities remain unchanged.

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
- `CPUNativeQuantisedFP16.h` and `GPUAlignedExpertQ8Primitives.h` use native half,
  maximum, scaling, RNE conversion, byte packing and signed-sum operations.
  Existing bounded partial-block handling retains its numerical program.
- `CPUNativeNEONContributionContract.h` implements widening signed four-column
  dots and explicit ASIMD/SDOT variants. Backend-native FMA and separately
  rounded expert contributions remain distinct. Both instruction variants
  have now executed on this host; the earlier baseline-only limitation is
  superseded.

## Prepared projection slice

Production headers now provide, but the engine does not yet select end to end:

- `CPUNativeQuantisedEncoding.h`: the existing source-aware encoding mapping, moved
  out of the owning packer without changing the mapping.
- `CPUNativeQuantisedWeightView.h`: checked immutable geometry and one borrowed
  canonical interleaved stream. It owns no allocation, source mirror or
  decoded matrix. Original source payload width and physical execution width
  are separate; both expanded encodings physically contain 32 bytes/column.
- `NEONBlockDecode.h`: native source-order Q4/Q5/nonlinear nibble decode for
  preparation and diagnostics, preserving integer values and exact lane order.
- `CPUNativeNEONMultiScaleOps.h`: four-column native signed dots, exact half/
  opaque-word conversion, source sign/grid lookup and bounded address generation.
  ASIMD and DotProd are separate selected types, not a per-dot runtime branch.
- `CPUNativeQuantisedMultiScaleKernels.h`: shared decoder/rounded arithmetic for
  NEON, AVX2 and AVX512, with deduced prepared-reader types and bounded N stores.
  Existing x86 calls retain their default widths/padded contract. The header
  no longer requires the storage-owning packer.
- `CPUNativeQuantisedActivationSums.h`: native widening half/quarter signed-byte
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
existing NativeQuantised work/rotation entrypoints use it.

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

- `CPUNativeQuantisedExecutionContract.h` owns the shared invocation enums,
  descriptors and original-output scopes. `CPUNativeQuantisedGemv.h` selects the
  native architecture implementation rather than importing x86 intrinsics on ARM.
- `CPUNativeQuantisedPackedWeights.h` separates the real retained owner from tensor
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

## Historical159-registration checkpoint and native SDK verification

The completed five-Mac checkpoint below rebuilt159 focused registrations;
Linux126 and Spark125 rows deliberately retain older source scopes. These are
native CPU-only leaf/build checks, not the complete CMake-owned Unit/preflight
inventory of the engine.

| Build | Result |
|---|---|
| AppleClang Release, baseline NEON | PASS,159/159;4.18s |
| AppleClang Release, NEON_DOTPROD | PASS,159/159;4.20s; explicit ASIMD/SDOT execute |
| AppleClang Debug, strict warnings (`-Wall -Wextra -Wpedantic -Werror`) | PASS,159/159;10.46s |
| AppleClang Integration + ASan/UBSan, NEON_DOTPROD | PASS,159/159;20.30s |
| Homebrew LLVM Release, baseline NEON | PASS,159/159;4.38s |
| Current GNU14/CMake4 native Linux RMSNorm targets | PASS actual production-TU8/8 and reduction3/3; not a full focused-engine build |
| Current native SDK4 topology/tool/production-pipeline Unit scripts | PASS9/9,3/3,243/243; read-only source checkout |
| Current SDK4 actual MPI-helper source/QFN/parity Unit metadata | PASS19/19,4/4,130/130; helper/module and7 built-source identities checked |
| Mac selected image-contract/fixture cases | PASS14/14; source contracts, not a Docker image build |
| Broader Mac production-pipeline Unit script | RED243 cases,38failures+5errors in the recorded broader run; Darwin fixture/runtime closure remains open |
| GCC13.3 Release, baseline/DotProd, native Linux/arm64 VM on this M1 | PASS,126/126 each; not a Spark rerun |
| GCC14.2 Release, baseline/DotProd, same native Linux VM | PASS,126/126 each; genuine C++23 SDK |
| GB10 Spark hosts×NEON/DotProd, immutable successor `391c8481…6128` | PASS,125/125 in each of four lanes;93 preflights included |
| Shared compact 8/16-column templates and the actual focused metadata-test block under x86 AVX2/AVX512 flags | PASS, syntax only for those projection templates |
| Historical WSL GCC14.2 typed target, old391c source | RED native73/166 assertions; scalar PASS; preserved diagnosis |
| Repaired WSL GCC14.2 complete focused projects, `eb3120f8…69a37` | PASS AVX2 80/80, AVX51282/82;55preflights each,0fail/skip |
| Canonical pre-commit with real OpenMP configured | FAIL at the explicit native-macOS memory/placement implementation gate; full Unit/preflight not run |
| `git diff --check` | PASS |

All121 current configured preflights pass20 repetitions; raw output contains
2420passes, with no fail/skip. Current logs/XML are outside the repository at
`/tmp/llaminar-arm-policy-matrix-f2nydp46`; every nested testcase and all2420 raw
passed records were independently parsed. Historical151/115 (`gYPR1Adn`),
155/118 (`AaIrfBXH`) and157/119 (`bt5rQaPv`) remain distinct narrower scopes.
The first expanded RMS checkpoint remains RED at strict compilation of unused
legacy configuration/integer-root helpers; dead helpers were removed, not
warning-suppressed. The intermediate LLVM module-response failure (`hlqTwROx`)
remains RED; retaining the original policy envelope repaired it without
ignoring missing responses. Native read-only-source pipeline RED exposed a
fixture writing into checkout; the fixture now owns lexical temporary storage,
with explicit preflight and unchanged production pipeline code. Broader Mac
path-fixture failures remain open. Historical139/105 and135/101 also passed20.
The intervening
136-test sanitizer run remains RED:135/136 pass, with
`V2_Integration_CPUNativeNEONProjectionBatch` hitting the ordinary30-second
boundary during concurrent vendor compilation. Contention is context, not an
established cause. No byte assertion failed in that log. Its unchanged binary
was copied and hashed before/after20 isolated-window repetitions: all20 pass
in226.74s without changing workers, coverage or the timeout. The subsequent
complete139-test five-Mac matrix/105-entry stress and current159/121 pass do
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
same explicit Darwin memory/placement failure before complete gates. The fresh
default cache first failed OpenMP autodiscovery; the real Homebrew libomp was
configured explicitly before the repeated canonical hook reached that Darwin
fatal. Both logs are retained. The repository-documented WIP commit bypass is
disclosed, not a passing gate.

The complete NativeQuantised refresh Python file was also attempted: 118 tests ran,
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

### x86 CPU-only gate at `be6658c38` / `05930d91a`

arm-wsl ran an Integration build on a Ryzen 7800X3D (8 cores / 16 threads,
one NUMA node) inside a WSL2 container.

- **Configuration:** CPU-only, shared core, AVX512 envelope, non-root.
  Configure omitted exactly 291 GPU-only registrations.
- **Results:** Unit 942/950, preflight 2234/2244.
- **Byte identity:** an `LLAMINAR_ISA_LEVEL=avx2` rerun of the NativeQuantised,
  FusedVerifier, ExplicitRounding, RMSNorm, attention, softmax, RoPE and
  SwiGLU families failed only the profiler-surrogate tests that also fail
  under AVX512. Every byte-exact kernel suite passes on both ISAs.

One branch defect was found and fixed in `05930d91a`. Compact multi-scale
`rows` had rejected empty trailing fixed-count K tiles (for example
`[26, 24)`), which aborted the GPU-aligned expert and expert-prefill
all-format suites on both ISAs. The new regression
`CompactEmptyTrailingKTilePublishesWithoutReading` is in preflight, and 4/4
pass on both ISAs after the fix.

The remaining failures are classified as follows.

| Failure | Count | Class | Evidence |
|---|---|---|---|
| Profiler surrogate fitting | 7 | Needs a CUDA device | x86 retraining tooling |
| `ModelLoaderRowSlice` | 1 | No models on the host | — |
| Release/pooled target selection | 3 | Known base-branch failure | `v2_test_memory_planner` is referenced before it exists in `V2PooledNgramPerformanceTests.cmake` |
| `CPUJointBufferedSource` | 2 | WSL2 limitation | `/proc/self/io` is absent |
| `CPUStartupAffinity`, `MoEOwnedSharedBackingNativeStartup` | 2 | WSL2 limitation | Open MPI `--cpu-set` fails silently there for any list, even `mpirun -np 1 --cpu-set 0-15 echo hi` exits 1 with no output; the unpinned selector passes, and the multi-rank probe fails fast with its exact-affinity diagnostic |

### Cross-architecture greedy text identity

Qwen3-0.6B Q8_0 (sha256 `9465e63a…bb031`), CPU only, `--no-mtp`, `-c 4096`,
greedy (`--temperature 0 --seed 42`), 64 tokens, raw prompt through
`llaminar2 oneshot`:

| Host | ISA | Commit | Result |
|---|---|---|---|
| Ryzen 7800X3D (WSL2) | AVX512 | `05930d91a` | rc=0 |
| Ryzen 7800X3D (WSL2) | AVX2 (`LLAMINAR_ISA_LEVEL=avx2`) | `05930d91a` | rc=0, byte-identical to AVX512 |
| DGX Spark GB10, 20 cores | NEON dotprod | `be6658c38` | rc=0, response byte-identical to x86 (346 characters) |

The two commits differ only in the compact multi-scale empty-tile admission,
which this Q8_0 model does not reach. Seal:
`K14-CROSSARCH-PROBE-GB10-20261107`.

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

## Final qualification (2026-11-08)

The earlier continuation list is superseded by the following verified state.
Seals live in the Spark results archive named below.

| Area | Result | Seal / commit |
|---|---|---|
| GB10 CPU-only Integration gate | Unit 951/958, preflight 2250/2252; 3 known upstream target-selection, 6 CUDA-device profiler surrogates | `K14-GATES-A8D380CD0-20261108` |
| M1 Max Docker VM gate | Unit 938/958, preflight 2194/2252; failure set identical to `7b655e015` (NUMA-less host, upstream, CUDA surrogate, VM slowness) | `M1-gate-a8d380cd0` |
| x86 CPU-only gate (AVX512 and AVX2) | byte-exact suites pass on both ISAs; one branch defect fixed | `05930d91a` |
| Cross-architecture greedy text | Qwen3-0.6B byte-identical on GB10 NEON dotprod and x86 AVX512/AVX2 | `K14-CROSSARCH-PROBE-GB10-20261107` |
| Serving A/B/C (GB10, mode-asserted) | 35B MTP dynamic 18.18 -> 27.19 tok/s with the pooled bundle; text byte-identical across commits | `K14-ABC-*` |
| Exact overlays | byte-neutral, end-to-end regression; retired | `K14-OVERLAY-AB-HEAD-20261108`, `264970e14` |
| Final Release `Perf__` round | thread scaling all formats on mixed/X925/A725 placements, byte-authenticated; fused verifier 45/45; FA2 regret and GDN grouped economics handed to follow-up | `K14-PERF-FINAL-264970E14-20261108` |
| GB10 integrated GPU (with `codex/cuda-sm121-spills`) | zero-spill sm_121 build, joint-pool admission and over-subscription rejection, 35B 132/97 tok/s, deterministic across fresh processes | `K14-UMA-MERGED-BEF53EDB6-FINAL-20261108`, `98e427c7c` |
| Refresh-suite CTest timeout | CTest PROCESSOR_AFFINITY grants at most the parallel level; not a defect | `K14-REFRESH-CONFIRM-A8D380CD0`, `a8d380cd0` |

Out of scope for this branch and tracked on the separate
`codex/arm-perf-review` branch: I8MM grouped rows and 27B
dense MTP economics, GDN grouped verifier, FA2 policy regret, region/barrier
count, LM head, placement by core class, in-model overlay collection, GB10 CUDA
register pressure (separate branch), NUMA-less hosts and a native macOS engine
port. No generic prefill/decode retraining was performed.

## GB10 CUDA maintenance evidence — 2026-11-10

This section supersedes the earlier separate-branch disposition for the CUDA
register-pressure work, not the historical CPU qualification results above.

- `e5ee7f672`: the rectangular expert-tier CUDA/ROCm fixtures use the writable
  unsealed publication span. GB10's stream-fixed cases pass at this revision.
- `18b455d62`: the ops fixture now includes the production residual declaration
  instead of a five-argument C redeclaration of the six-argument ABI. The crash
  was reproduced before the fix. Explicit stream-ordered uploads, launches and
  readbacks cover FP32/BF16/FP16. LM-head workspace binding and KV append input
  preparation also pass. Both complete test files and the two new focused
  `ProductionTestPreflight` invocation-contract aliases passed on GB10.
- `371f5ab35`: the all-format captured prefill bucket-M invariant gate passes
  with its added long-K/full-wave geometry. All six automatic tile obligations
  remain; the formerly missing tile 5 is exercised without forced dispatch.

### Narrow grouped IMMA scheduling bound

The 12.x 32-column IQ4/IQ2S projection compiler bound admits at least ten CTAs
per SM. No register array, shared-storage layout, arithmetic operation, stream,
launch grid or public execution mode changes. The resource regression retains
its 48-register ceiling and ten-active-CTA floor, and now applies that same
strong contract to source-scale IQ4 on 12.x as well.

| Execution codebook, 32 columns | Previous registers | Candidate registers | Candidate local bytes |
|---|---:|---:|---:|
| IQ4 nonlinear (4) | 62 | 48 | 0 |
| IQ2_S (13) | 56 | 40 | 0 |
| Source-scale IQ4_XS (24) | 56 | 45 | 0 |

The changed translation unit passes the fatal spill guard for
80/86/89/90/120/121. Shipped 80/86/89/90 instruction and encoding lines are
identical, preserving function/template identity while normalizing only
nvcc's source-digest salt in internal symbol names. This is instruction-byte
identity, not identity of complete object files or debug metadata.
The kernel source SHA-256 is
`65e4940c33928afc84a34e5bc6d8e807d95d8932f80460f5cf10e17b38f9014f`.

Native `CUDAMoEPrefillCompilerResourcesAllCodebooks`,
`CUDAMoEPairedGateUpRegisterLifetimeAllFormats` and
`CUDA_MoEPrefillStagedMetadataByteInvariant` gates pass. They exercise actual
production dispatch and captured full-buffer serial-row byte comparisons.

### Timing and driver-safety limits

A matched synthetic production-shaped Integration/O3 trainer diagnostic
(h2048/r512/e256/top8, IQ2_S gate/up, IQ4_XS down, M512, power-law routing)
produced 24 candidate records per side with zero bit mismatches. For
`imma_paired_g32__d32`, the 15-sample median changed from 3336.51 to 3315.53 us;
its down projection changed from 56 registers/9 active CTAs to 45/10. The
approximately 0.6% timing difference is within the observed noise and is not a
Release performance or end-to-end speedup claim.

NCU emitted a baseline report, but its owned driver window caught new NVRM
`refcntRequestReference_IMPL` warnings. Further NCU/NSYS attachment was stopped;
that report is not a clean qualification receipt. Subsequent non-attached
baseline and candidate workload windows both passed driver diagnostics.
The initial profiler attempt failed before launch because its session root did
not exist; its incomplete driver-window record is retained separately.

Raw commands, objects, disassembly, resource logs, CSV samples and driver
reports remain on `.79` under `/home/factory/idea25/slot-probes/`, with the
attached report under `slot-profiles/` and driver-window files alongside those
directories. `candidate2-runtime.log` records the passing native checks and
clean non-attached driver window. These diagnostics do not certify the full
CUDA gate, a Release model benchmark or the pending Qwen3.8 trace.
