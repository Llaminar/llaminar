# gfx1100 NativeQuantised WMMA design and qualification

Target: Radeon RX 7900 XTX, native gfx1100 (RDNA 3). Worktree:
`/workspaces/llaminar-gfx1100-wmma`, branch
`feat/gfx1100-native-quantised-wmma`, based on develop `ee1a82d0b`.
Acceptance was revised by the user: every codebook must match the RTX 3090's
achieved fraction of dense INT8 theoretical peak on the same workload. This
replaces the blanket 65% requirement. Both M512 and M1024 pass all 21 source
formats against independently confirmed RTX 3090 references. Both profiler
manifests are complete. The generated policy now installs 34 gfx1100 physical
keys across the two row counts, and installed all-format replay passes. The user
limited this feature checkpoint to focused regressions and the feature suite,
and authorized committing and pushing with `--no-verify`. Complete preflight
and model qualification are not claimed.

The declared performance/installation surface is every one of the 21 source
formats at `Qwen36MoE_GDN_QKVProjection` (N8192/K2048), M512 and M1024.
M512 matches the row-count anchor of the canonical 512-token prompt workload;
M1024 supplies the larger prefill anchor. Functional replay separately covers dispatch boundaries,
small and tail geometries. Unmeasured shapes retain their existing policy.

## Numerical and ownership contract

Every registered source format uses its existing compact decoder, including
IQ4_XS physical24 source factors, dual-scale formats, asymmetric minima and
IQ1_M signed corrections. INT8 WMMA computes exact INT32 half-block dots;
the shared FP32 contribution helpers and public serial-M1 K partitions retain
the existing arithmetic order. No dequantized weight copy, new persistent
workspace, atomics or host inference state is introduced.

The current family uses four wave64 waves per workgroup with 32x32, 32x64,
64x64 and 64x128 output tiles. A wave32 experiment was slower across the harder
codebooks; its source and measurements remain local evidence. The dedicated
translation unit keeps the wave contract explicit. LDS publishes decoded operands and metadata before WMMA,
then joins consumers before overwrite. Each output has one writer. Partial
tiles mask only the final execution-granularity remainder. The WMMA symbol is
compiled only for gfx1100, and host admission requires actual gfx1100 metadata.
gfx906 retains its existing implementation in the same shipping binary.

This follows the CUDA native-prefill pattern in
`CUDANativeQuantisedPrefillDevice.cuh`: compact decode, integer MMA, and exact
block contributions. WMMA fragment layout, wave size and resource limits are
AMD-specific and independently verified.

## Tuning and exact overlays

The existing ROCm sweep owns preparation, complete projection capture,
byte comparisons and raw event samples. Explicit `native-quantised`
preparation now admits all source formats with authenticated architecture,
UUID and PCI identity. The older `native-quantised-q8` collection scope remains
Q8-only. Source-default historical cells keep their identities.

New WMMA records use producer `NativeWmma` and remain architecture-qualified.
The common exact emitter preserves unrelated rows, including historical
unobserved rows at the same geometry. Physical24 has its own new measurement
key; historical source4 observations remain historical. The common
`native_prefill_profiler` binds winner and actual Auto baseline requests to
sealed complete-operation samples for every format.

Diagnostic entry (after selecting the actual gfx1100 device):

```sh
HIP_VISIBLE_DEVICES=5 LLAMINAR_SKIP_CUDA_STARTUP=1 \
  /tmp/llaminar-gfx1100-wmma-build/tests/v2/v2_perf_native_quantised_sweep \
  --gtest_filter=NativeQuantisedSweepTest.WmmaCapturedReplayAllFormats
```

The physical ordinal is machine-specific. Current observed identity is
`GPU-da4b8b5228e8147a`, PCI `0000:b5:00.0`; never infer identity from that ordinal
in a published corpus. Remove ISA overrides before discovery or qualification.

## Acceptance contract

1. Native all-format byte checks through retained production capture, reset and
   repeated dirty replay, partial M/N tiles and representative K partitions.
2. Complete shipping-target spill audit (gfx906 and gfx1100), per-candidate ISA,
   occupancy and isolated hardware counters.
3. Stable interleaved event timings for every required codebook and the declared
   production prefill geometries, with complete-operation and leaf scopes kept
   distinct. Retain unsuccessful candidates and actual clocks.
4. Common sweep validation, isolated profiling, emitted exact entries and
   actual installed Auto replay. Keep unmeasured geometry dispatch total.
5. Focused regressions and the feature suite must pass before the authorized
   feature-branch commit and push. Model numerical panels, matched unprofiled
   Release benchmarks and complete preflight are outside this checkpoint's
   user-approved scope; their absence prevents a model/release qualification claim.
6. For every codebook, an achieved fraction of dense INT8 peak at least as high
   as the RTX 3090 reference on matched shapes, M and full-projection scope.

[AMD's WMMA reference](https://gpuopen.com/learn/wmma_on_rdna3/) specifies
512 INT8 operations/CU/clock. At the locally reported 96 CUs and maximum
2.482 GHz, the denominator is 121.995264 TOPS.
AMD SMI authenticates 96 physical CUs. HIP reports 48 multiprocessors in WGP
mode; the selected SDK documents two CUs per WGP. Preserve both raw facts and
normalize their unit before computing peak: 512 operations per CU is 1024 per
reported WGP, not 512 per reported WGP.
Use useful `2*M*N*K` operations, never padded or duplicated work. Format
decoding, scale arithmetic and required producers remain visible costs;
neither a pure-instruction microbenchmark nor profiler timing proves this goal.

Both local RTX 3090s report 82 SMs and native CUDA peak clock 1.860 GHz.
[NVIDIA's GA102 whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)
owns the dense rate of 2048 INT8 tensor operations/SM/clock. Their denominator
is 312.36096 TOPS; NVIDIA-SMI's separate 2130 MHz maximum clock bin does not
replace the native clock authority. Equal fractions therefore require
`AMD_TOPS >= RTX3090_TOPS * 0.390558615263572`. Compare useful operations and
the same full retained projection on both cards. Retain both card identities
and use the stronger measured reference rather than selecting the slower card.
The harness's former sm86 rate of 2304 was incorrect. A device-free replay now
also checks the published Turing, A100, Ada and Hopper rate configurations.

Local development evidence is outside source under
`/tmp/llaminar-gfx1100-wmma-evidence` and corresponding build/test logs.
Native captured replay now passes all 21 source formats across aligned and
partial M/N tiles and K256/512/768/1280. A focused whole-row regression first
reproduced negative-zero loss, then passed with the single-partition direct
publication fix; it also covers K32/96/160 and partition counts 1/3/5. Both
replays are explicitly registered in production preflight. The common Python
suite has 71 passing tests, the profiler suite 154, and the spill/input-identity
suite 45. Separate host admission and evidence-identity preflight checks pass.
Expanded captured replay covers every M1–129, larger M256/512/1024 anchors,
partial N tiles, nonuniform serial partitions and supported short K tails.

Development timing on Q4_0, M1024/N8192/K2048, full retained projection:

| Candidate revision | Best WMMA TOPS | Same-run Auto TOPS |
|---|---:|---:|
| One K32 staging slot | 33.86 | 47.13 |
| Four K32 slots | 45.83 | 46.93 |
| Vector A loads and retained row scales | 54.38 | 47.32 |
| Reused correction sums and chained single-scale dots | 58.00 | 47.15 |
| Wave32, two slots, coalesced staging, grouped traversal | 54.49 | 47.73 |
| Wave64, vector LDS with row/stage permutation | 55.72 | 47.26 |

These are minimum native-event observations from 31 samples after 10 warmups,
not installed/certified overlay measurements. The historical best rate is 47.5%
of the 121.995 TOPS denominator. These historical experiments are superseded by
the matched all-format collection and profiling below. Broader production
qualification is outside this feature checkpoint. The full gfx906/gfx1100 compile passes the spill
guard. Final ISA for each revision must stay attached to its own measurements.

Sol 6.1 review found and prompted focused fixes for runner candidate filtering,
qualified installed replay, physical24 streaming admission, K-dependent streaming
LDS bounds, shared-source alias completeness, and signed-zero arithmetic. The common collector now selects gfx1100-supported counters and owns scoped
STABLE_STD power policy. Its actual winner/Auto proof completed two independent
requests, each including activation quantization and projection. Both report
positive activity/waves and zero scratch; every scope verified restoration to
AUTO. The exact emitter accepted that authenticated Q4_0 cell and retained the
base rows in a temporary output; the later complete two-anchor installation
uses separately authenticated all-format evidence.

A real common-collector failure exposed sudo/PAM reintroducing
HSA_OVERRIDE_GFX_VERSION from /etc/environment after an outer environment
cleanup. The trainer then skipped with no device. Failed attempts remain raw
diagnostics. The corrected collector separately elevates AMD SMI mutations,
keeps rocprof unprivileged, and clears ISA overrides after sudo when the
profiler itself is explicitly elevated. Focused environment, restoration and
legacy-receipt regressions belong to production preflight.

An earlier same-projection diagnostic sweep covered all 21 source formats with
zero byte mismatches. Every best WMMA candidate beats same-run Auto: Q4_0
58.00 TOPS, Q4_1 46.41, Q2_K 37.98, IQ1_M 27.63, Q8_0 56.55. These are useful
full-projection rates, not pure instruction throughput. The revised RTX 3090
relative targets now pass. Installed all-format replay also passes; model-level
qualification has not run.

## Device-input and profiler identity repairs

A purported wave32 K2 rebuild retained the K4 code object. Its timing record
is excluded as K2 evidence. Controlled private-cache reproduction showed that
host-only HIP preprocessing can return an old preprocessed-cache entry after
a device body changes, or even a direct-cache hit for a device-only include.
Depend mode alone misses the latter. The compiler launcher now hashes each
target's actual device-preprocessed input into ccache identity and merges all
device dependencies, including system headers, into Ninja's object rule.
Both gfx906 and gfx1100 pass cold/warm/header-change/reset replay, native LDS
inspection, failed-header object retirement and all original spill controls.
Unchanged source retains actual cache hits. These proofs have explicit
ProductionTestPreflight registrations and need no accelerator at verification.

The initial common Q4 profile is historical v9 evidence. The v10 collector
additionally requires the original executable and loader-resolved core before
and after collection; identical tile dimensions cannot authenticate a changed
wave or code-generation policy. New wave64 vector Q4 timing and independent
winner/Auto collection completed under v10 with positive counters and verified
power restoration. Raw closure receipts are sealed with the profiler evidence.

## Matched-reference measurement repairs

The CUDA inventory used a source-codebook payload map to admit physical
staging kernels. IQ4_XS executes codebook24, so Python expected 13 candidates
where the actual compiler admitted 43. The candidate reader now derives its
payload map from physical execution identities, preserving all 30 staged
IQ4_XS candidates. Canonical collection also clears inherited tile filters and
diagnostic peak overrides. Focused regressions belong to evidence preflight.

Both timing tournaments previously warmed valid output before their final
certificate. They now poison the live output and consumed activation/scale
banks, replay the same retained executable, and compare every output byte
before recording timing events. CUDA additionally poisons asymmetric sums and
active canonical K-partials; expanded ROCm INT8 poisons its epilogue input.
Three distinct patterns exercise reset and repeated replay. Producer-only
diagnostics preserve authenticated activation inputs. Explicit CUDA decode
diagnostics retain their output-only replay contract without consulting stale
prefill metadata. Native negative controls disable real captured quantizers
and projection nodes; CUDA also leaves the reducer enabled while disabling
its producer. Both new controls are registered in production preflight.

The paired-fragment compiler-fence experiment was retired after an all-format
run showed no useful gain (Q4_0 55.04, Q4_1 45.64, Q2_K 38.26, IQ1_M 28.00
TOPS). ISA showed unchanged LDS work and additional waits/register pressure.
The original vector consumer is restored. Historical experiment records remain
under the local evidence root; none supplies an installed dispatch claim.

## Confirmed M1024 utilization comparison

Both 3090s completed all-format tournaments with 10 warmups and 31 samples,
each sample replaying 16 complete projections. The second card won every
format's screening comparison. Its chosen candidates were frozen, then rerun
in fresh processes with the same sample count. The table uses these independent
confirmation minima, not pooled minima from additional trials. AMD rates use
the common emitter's single candidate per physical key, retaining independent
source-alias rows. Every candidate passed poisoned retained replay and the
complete-output byte certificate. Confirmation latency moved by at most 3.64%.

All 21 source formats pass; the minimum ratio is 1.9973x and maximum 2.80x.
The complete rows and raw native-event samples are retained under
`matched-utilization-confirmed.csv`, `matched-gfx1100-all21`, the two
`matched-3090-device*-all21` screening roots and
`matched-3090-device1-confirmation-all21` in the local evidence directory.
These timing records certify the declared M1024 measurement point. Installed
dispatch has separate replay evidence below; no end-to-end model speedup is claimed.


## Confirmed M512 utilization comparison

The same 21-format scope passes at M512, with utilization ratios from 2.0303x
to 2.8055x. Independent confirmations retain the strongest screened card and
candidate: device0 for IQ4_NL, device1 for the other 20 source formats.
Confirmation latency moved by at most 4.59%. Q4_0 achieves 43.21% of native
peak versus 21.28% on the selected 3090; IQ1_M achieves 22.16% versus 8.38%.
The complete comparison is `matched-utilization-m512-confirmed.csv`; original
samples, frozen reference selection and both confirmation roots remain under
the evidence directory. These are complete projection measurements, including
quantization and required reduction, with the same replay/certificate contract.

The complete normalized comparison is below. Each percentage uses that card's
own dense INT8 peak; the ratio compares those percentages, not raw TOPS.
N8192/K2048 and complete-projection timing are common to every row.

| Format | M512 gfx1100 / 3090 % peak | Ratio | M1024 gfx1100 / 3090 % peak | Ratio |
|---|---:|---:|---:|---:|
| IQ1_M | 22.16 / 8.38 | 2.643x | 22.92 / 8.79 | 2.609x |
| IQ1_S | 35.27 / 14.10 | 2.502x | 36.75 / 15.19 | 2.420x |
| IQ2_S | 35.32 / 13.11 | 2.694x | 36.59 / 14.06 | 2.603x |
| IQ2_XS | 37.83 / 14.13 | 2.678x | 38.97 / 15.22 | 2.561x |
| IQ2_XXS | 41.81 / 18.68 | 2.238x | 43.99 / 20.08 | 2.191x |
| IQ3_S | 39.25 / 16.92 | 2.320x | 40.51 / 18.08 | 2.240x |
| IQ3_XXS | 41.39 / 18.39 | 2.251x | 43.30 / 19.86 | 2.181x |
| IQ4_NL | 42.52 / 20.04 | 2.122x | 44.49 / 21.59 | 2.060x |
| IQ4_XS | 39.96 / 19.56 | 2.044x | 41.36 / 19.92 | 2.076x |
| Q2_K | 30.65 / 11.48 | 2.670x | 31.46 / 12.02 | 2.616x |
| Q3_K | 38.76 / 14.84 | 2.613x | 39.89 / 15.84 | 2.519x |
| Q4_0 | 43.21 / 21.28 | 2.030x | 45.32 / 22.69 | 1.997x |
| Q4_1 | 36.03 / 16.12 | 2.235x | 37.35 / 17.27 | 2.163x |
| Q4_K | 36.13 / 16.28 | 2.220x | 37.39 / 17.22 | 2.171x |
| Q5_0 | 41.78 / 19.73 | 2.117x | 43.30 / 21.23 | 2.039x |
| Q5_1 | 34.96 / 15.56 | 2.247x | 36.18 / 16.66 | 2.171x |
| Q5_K | 34.99 / 15.46 | 2.263x | 36.16 / 16.65 | 2.172x |
| Q6_K | 33.90 / 12.09 | 2.806x | 34.97 / 12.50 | 2.799x |
| Q8_0 | 43.48 / 18.88 | 2.303x | 44.58 / 20.41 | 2.184x |
| Q8_1 | 43.42 / 18.89 | 2.298x | 44.69 / 20.39 | 2.192x |
| Q8_K | 43.46 / 18.78 | 2.314x | 45.28 / 20.39 | 2.221x |

## Installed policy and qualification scope

The baseline is isolated in `/workspaces/llaminar-gfx1100-wmma-baseline` before
installing generated entries. Its separate Integration build owns the diagnostic
prerequisite attempts below; no model observations were collected. The measured Release
trainer and loaded core are preserved with the completed profiler evidence.
The two additive emitter receipts authenticate 582 retained base rows plus
17 physical keys at each row count, for 616 total rows. The installed include
SHA256 is `02417ccdcce02371de75cbc8c649f33770136815fd3929902bd2a3c6156f17d8`.
The independent baseline include remains unchanged.

A focused mixed-architecture replay now uses the same ordering authority as
production lookup and preserves legacy rows beside qualified gfx906/gfx1100
entries. New PerfStats producer tags authenticate physical producer, tile and
launch resources at dispatch/capture publication; they do not count graph
replay multiplicity. Both invariants have explicit production preflight entries.
The two WMMA replay gates select gfx1100 from the immutable visible inventory,
record native participant identity, and declare optional native-ISA requirements
so absent hardware remains unqualified rather than silently passing a skip.
The latest replay changes compile in the baseline Integration build. Its
unmasked mixed-inventory run passed producer counters, all-format captured
replay and whole-row/partial-K arithmetic replay (three tests, 81.324 seconds).
Each recorded gfx1100 at ordinal 5 with the expected UUID. Candidate installed
runtime replay passed all 12 source-format groups in 295.262 seconds. It
independently derives 42 expected WMMA cases across all 21 source aliases
and two row counts and observed all 42. The XML and authenticated summary are
`candidate-installed-all-formats.xml` and
`candidate-installed-all-formats-summary.json` in the evidence directory.


## Completed profiler joins

Both M512 and M1024 completed all 42 isolated winner/actual-Auto requests,
eight passes each, with no failed, missing or unsupported requests. At M1024 every
projection reports zero scratch. Every request verified AUTO → STABLE_STD → AUTO
power restoration. WMMA physical allocations are 100–164 VGPRs, 17,920–32,768 LDS
bytes per workgroup, with 256 threads. Measured L2 hit rate is 75.12–83.26%; LDS
bank conflicts are 1.33–5.26%. These are diagnostic counters, not timing labels.
At M512 the 42 projections also have zero scratch and verified power restoration.
Its WMMA allocations are 100–164 VGPRs and 18,432–32,768 LDS bytes; L2 hits are
74.17–82.06% and LDS conflicts 1.33–5.26%. The complete separate summary is
`matched-gfx1100-m512-all21/profile-summary.json`.
The selected SDK does not expose achieved occupancy or VALU utilization; native
function occupancy and logical register counts remain in the original timing
records. Physical descriptor allocation must not be confused with logical VGPRs.

The common emitter authenticated all source aliases and produced 17 gfx1100
physical keys per row count, retaining all 582 installed base keys.
The proof is `generated-m1024-retention.json`, with per-key policy comparison in
`generated-m1024-summary.csv` and profiler summary under
`matched-gfx1100-all21/profile-summary.json`. The second additive receipt is
`generated-m512-retention.json`; its combined output is now installed in the
candidate source. Sol 6.1 independently verified the complete ordering, both
receipt chains, all source aliases and byte preservation of the base rows.

Full integration dependency preparation requires fresh NCCL/RCCL artifacts for
this checkout's exact completed-broadcast patch generation. The shared artifacts
belong to a different revision and correctly fail current proof admission. The
HIP cache-identity change also requires a fresh linker/compiler proof; that
original/repaired proof passed for gfx906 and gfx1100. Dependency rebuilds use
private paths and the canonical installers; no proof is transplanted and no
shared installation is modified. Both private dependencies have now built;
their complete canonical gate owners remain part of each full preflight build.


The private NCCL installer exposed a dash-prefixed architecture-argument bug.
Both proof invocations now bind `--architecture=value` as one quoted token.
The device-free replay executes the actual Bash argument expressions through
both real parsers, covers default/single/multiple SM selections, and reproduces
the original single-SM failure. All 19 packaging tests pass; the focused replay
has its own ProductionTestPreflight entry. A fresh native sm86 NCCL artifact
passes GIN, thread-lifetime, shared-file and completed-broadcast builder proofs.

Compiler-bound proof commands use actual GCC 14 with ccache as a separate
launcher. Resolving a compiler-name ccache shim erases its driver argv[0]; direct
`/usr/bin/g++-14` preserves the same compiler and OpenMP ABI while giving the
proof an executable compiler identity.

Model prompts are frozen in `model-panel-authored.json` (SHA256
`92ed6b0af99406fd7389411c78f1092b1f1eb19a5d359fd366b53ba558d9b6c6`):
61, 512 and 1024 tokens, three decode observations each. Only the tokenizer was
loaded during preparation. Full prerequisite and model runs will retain
`HIP_VISIBLE_DEVICES=5,0,1,2,3,4`, with competing mask authorities and ISA spoofing
removed, so the existing ROCm0 cell uses gfx1100 while all AMD participants
remain available. Changed visibility cannot reuse the same hardware receipt.

## Full-gate repair status

The first complete baseline prerequisite attempt is not passing. Its 1,034-test
Unit phase found missing compile-database configuration, three absent synthetic
CSV fixtures left by an earlier rename, stale test expectations inherited from
the starting revision, and timeouts during concurrent builds. Its independent
host integration phase was interrupted after confirming these defects; CUDA,
ROCm and exclusive phases did not run. Original partial results and the failed
receipt remain in `preflight-baseline-r1` with the separate stop record.
No model has been admitted through this failed run.

The restored ROCm fixture payloads match their original small source fixtures;
all four existing CSV/generator checks pass. Their current names now have a
narrow ignore exception and explicit preflight replay registrations. The
trainer source policy still bans allocation globally and preserves the original
prepared-execution transfer/wait budgets. Its separate arithmetic diagnostic
has an exact two-copy/two-join boundary and injected-leak negative controls;
all six policy checks pass.

Inherited contract repairs keep production behavior intact: cross-host tests
author their own eligibility tags, workspace source checks follow the typed
transaction-binding authority, and numerical CI tests exercise the current
accuracy phase and owner retirement. Cold DEBUG BOM diagnostics remain visible
while graph-construction geometry remains excluded. Python checks and all four
repaired C++ test owners pass; quiescent full prerequisites remain outstanding. Stale
overlay eligibility tests now exercise all 24 diagnostic cells per definition,
retaining declared placement while rejecting release eligibility. Dense release
coverage follows the code-owned topology inventory.
The plain candidate Release and both full Integration builds are complete,
including the reviewed fixture repairs. Independent project-owned CUDA compilation
for sm80/sm86/sm89/sm90 passed the mandatory spill guard, and its compute-peak unit
passed. This compilation does not claim runtime qualification on absent cards.

The quiescent second baseline Unit run passed 1,033 of 1,034 checks in
194.106 seconds, including CPU all-format GEMM parity in 38.60 seconds and
residency publication in 1.07 seconds. Only the dense-prefill Python suite
exceeded its 30-second timeout. The following host phase was stopped cleanly
after 53 passing checks; `preflight-baseline-r2` remains failed and incomplete.
Profiling identified repeated construction of immutable canonical fixture
inventories. The two common fixture helpers now reuse those declarations;
all 71 checks pass standalone. Four disjoint Unit shards retain all 71 checks
and the original timeout, while explicit integration selectors remain intact.
`dense-prefill-shard-membership.json` authenticates their exact union and
malformed-shard rejection. These full-suite attempts remain diagnostic records.

The third baseline transaction passed all 1,037 Unit registrations in
154.403 seconds. Its four dense-prefill shards passed in 2.02–10.99 seconds.
The host lane passed 2,566 of 2,567 checks; CUDA passed all 279. The ROCm lane
passed its 401 available checks, with two eight-device scopes explicitly
unqualified on this host. The exclusive lane completed all 1,047 selected tests
with four failures. The complete transaction retired with return code 8 after
13,129.18 seconds; its failed receipt remains diagnostic evidence only.
The focused concurrent shard/evidence check also passed all five registrations
in 11.40 seconds after build I/O settled, retaining the earlier timeout evidence.
An independent sample/receipt audit reproduced all 42 utilization comparisons,
the physical-CU/WGP conversion, both peak denominators and the independently
confirmed strongest RTX 3090 references. The minimum normalized ratio remains
1.997317 across the complete declared comparison surface.

The third host lane subsequently exposed one inherited test configuration defect:
`ImmutableLookupAdmittedSetupEvidence` used a wildcard suffix outside PerfStats'
literal domain/name-prefix grammar and collected no owner records. Its fixture
now selects the whole setup domain and proves that warmup data is collected,
then discarded while the original owner identities/counts/values survive reset.
Production benchmark filters already select the supported exact family and are
unchanged. The existing preflight entry passed its 20 fresh-process repetitions.
The third transaction has retired and cannot admit models.

The exclusive lane exposed a mapped-activation fixture race: the independent
publication recipe checked both shared timelines after joining only its local
GPU. Local terminal observation now completes through the existing rank
consensus before a separate checkpoint validates both signals. Both original
directions, all signal assertions and the two-node/97-node captured graphs
remain intact. The existing `MappedActivationAdmission_MPI` preflight case
passed 20 fresh process lifetimes in 85.32 seconds after rebuilding. An earlier
repeat passed three times, then timed out during device setup under unrelated
compiler contention; that failed attempt and its pressure evidence remain
retained. The successful repeat keeps the original 30-second timeout. The original failure
is retained in `r3-mapped-activation-mpi-failure.log`.

One CUDA FP32 pipeline test passed its GTest body and device cleanup, then its
MPI launcher reported missing rank deregistration. Static review found the
expected initialization/finalization owner; attribution remains unresolved.
`r3-cuda2-fp32-mpi-retirement-failure.log` preserves the failure. The original
registered case subsequently passed 20 fresh-process repetitions in 119 seconds
with identical launch policy. No lifecycle change is justified by the current
evidence; the original failure remains part of the qualification record.

Both CUDA LocalTP context-storage cases retained an obsolete two-graph BOM
expectation. The production timing family owns four simultaneous executables:
body, start marker, stop marker and parent. Its single reservation is reused
across sequential phases. The fixture now uses the existing
`PlanningExecutionMeasurement::kPreparedGraphCount` authority while retaining
zero-reservation retirement, context-storage survival, repeated-family plateau
and successor-context checks. Both original preflight cases pass after
rebuilding; production memory accounting is unchanged. Together with the
immutable-owner check, the focused repaired-owner set passed in 26.18 seconds.

## Feature-checkpoint validation

The user subsequently limited completion to focused regressions and the feature
suite and authorized a feature-branch commit/push with `--no-verify`. The fourth
full baseline preflight was stopped by SIGINT to its original CTest process
group. All original driver processes retired. Its receipt records return code
-2 after 247.446 seconds and remains incomplete, not certification evidence.
No model reference, numerical panel or end-to-end benchmark was run.

The final candidate device-free feature selection passed all 40 registrations
in 17.87 seconds. It covers the full dense-prefill/profiler Python shards,
exact-overlay and architecture admission, peak arithmetic, trainer fixtures and
boundaries, compiler device-input identity, and both CUDA/HIP spill proofs.
`candidate-final-feature-host.log` and its JUnit XML retain the exact selection.

The final gfx1100 selection passed all 17 native ROCm registrations in
325.49 seconds with no skips: captured all-format WMMA, whole-row/partial-K
arithmetic, producer identity/PerfStats, omitted-work dirty-replay controls and
all 12 installed-overlay groups. This run exposed only the native gfx1100 card;
the earlier mixed-inventory replay remains the separate gfx906/gfx1100 proof.
The CUDA omitted-work dirty-replay check also passed in 2.48 seconds. Logs and
JUnit XML are `candidate-final-feature-rocm.*` and
`candidate-final-feature-cuda.*`. Both Sol 6.1 reviewers found no remaining
code or coverage blocker, and the complete comparison table was checked against
both original confirmed CSVs.

The candidate CUDA LocalTP prefill check initially failed its unchanged native
stack-footprint restoration probe; a repeat passed 11 times before failing
again. A separate test transaction was concurrently starting and retiring CUDA
contexts on those same devices. Once that transaction finished its CUDA phase,
the original check passed 20 consecutive fresh-process runs in 124.20 seconds.
This is consistent with interference in driver-global free-memory observations,
but the original exception did not identify the failed scalar predicate, so the
precise failure remains unattributed. Production admission checks and timeouts
were unchanged. Both failed runs, the concurrent process identity and the
successful repetition remain in `candidate-final-focused.log`,
`candidate-final-localtp-repeat.log`,
`candidate-final-shared-gpu-contention.json` and
`candidate-final-localtp-repeat-r2.log`.

The final combined focused selection then passed all five registrations in
36.26 seconds: immutable-owner setup/reset, mapped-activation MPI, both CUDA
LocalTP context-storage lifetimes and the original CUDA FP32 pipeline case.
Together with the 58 feature checks above, all 63 selected registrations passed
without skips. `candidate-final-focused-r2.*` retains the combined run, and
`candidate-final-focused-feature-summary.json` binds all four result XML hashes.
