# Idea 14 A deliverable — c007 owner-test/self-check checkpoint (2026-11-10)

Provenance: station perf lane packet, relayed verbatim by arm-perf; base
`c007a59a2` (no lane commits; source unchanged). Evidence class:
**owner-test/self-check ONLY** — not canonical CTest, complete
Unit/preflight, production binary, GB10 economy or historical CPU/dense
model certification. Local evidence root:
`/home/tdsynnex/repo/llaminar-perf-idea14/artifacts/idea14/`.

## Verified results

IDEA14_OWNER_SELFCHECK_RESULT configure=0 build=0 trace=0 controller=0 timing=0
29/29 trace-owner tests PASS (24ms); 71/71 controller-owner tests PASS;
1/1 bookkeeping-owner test PASS. No test selection/omission or stubs; all
unchanged source owners were run. All 15 source/fixture/wrapper hashes and
5 binary/build hashes rechecked. Monitor 745c521af exit 0; container retired.

Host/SDK: Grace station, CPU affinity 70, one CPU quota, 8 GiB, network
none, Ubuntu GCC 14.2.0, CMake 4.0.3, Ninja, Release -O3 -DNDEBUG.
Compiled ISA target NEON_I8MM = code-generation identity only. No model or
GPU work occurred.

Unchanged Perf__MTPDepthController composite harness (10,000 warmups +
5,000,000 timed ops/row, steady_clock wall-ns/op):
- dynamic_window16_record_step: **14.609 ns**
- dynamic_window1_record_step: **55.546 ns**
- dynamic_depth0_bypass_step: **9.013 ns**
Three aggregate means from the synthetic composite owner harness — not
isolated model transactions, per-call p95, cycles, a CPU cost profile,
canonical GB10 timings or an A/B policy win. PMU unavailable; no perf
policy changed.

## Readiness result

Actual controller tests demonstrate full windows, early demotion, repeated
cadence, partial observations after completed windows, clipping exclusions
and reset. Terminal WindowNotReady is NOT evidence that no earlier window
evaluated; global 43.4% draft acceptance cannot be assigned to the final
depth-10 window. The current CPU/dense controller has NO timing authority,
so acceptance alone cannot establish economic loss. The historical C-era
trajectory remains unproven WITHOUT its actual inputs.

Observed-terminal → supported-conclusion table:

| Observed terminal information | Supported conclusion |
|---|---|
| last reason WindowNotReady alone | last eligible observation deferred, or fresh/reset state; earlier history not established |
| WindowNotReady plus windows>0 | earlier windows DID evaluate; a partial tail can explain the last reason |
| windows=0 plus complete ordered trace | examine eligible-count, clipping/stop, bounds, min_samples, cadence, cooldown; find first blocking predicate |
| current depth10 + request acceptance 43.4% | final selector + whole-request aggregate, not a depth-10-window sample |
| updates=0 | selector unchanged; windows may still have evaluated and held |

Zero-accept rate (Z/R, default 0.30) and draft-token acceptance
(accepted/attempted, demotion threshold 0.55) are DIFFERENT statistics.
Profitability depends on complete transaction cost vs actual newly emitted
output Y — not acceptance alone.

## Trace semantics

Authored CPU/native v3 golden cases are not model captures; genuine
recorded v3 confirmation paths pass their declared baseline contract;
historical v2 and survival fixtures keep their explicit
incompatible/frontier contracts. None become CPU/dense/GB10 evidence.
Official 19-source identity generator used unchanged (full hash table in
the raw packet below).

## Historical data

Inventory PENDING arm-sparks read-only sweep — not a declared universal
absence. B remains coordinator-owned after idea-31 census → B9 census →
Bundle-1 bound; fresh bounded v3 trace/timing observations in that booked
wave resolve the decision gap. No CPU profile/backoff before a negative
current paired result.

## Failed attempts preserved (not relabeled as passing gates)

1. Full canonical configure: configure=1, tests NOT_RUN — read-only
   `external/onednn/build-neon_i8mm.llaminar.lock` creation failed.
   Result SHA ad62aa354dabbd563de72cf049c80aefec2b8b35a1cfa9fb57f3b9888d17f878;
   configure log SHA bda991945c762b55e8acd9c135580288a5b822fc77497dd20db2b41e3911aff0.
2. Direct owner-v1: build=1 — controller fixtures needed real
   GlobalDeviceAddress::rocm + ModelMemoryProfile::mainLayerCount.
   Compile-commands SHA eaaee21a43c1aecf66766c9eeb126c3a88745060d9d4754d161969313466687c;
   build log SHA 3e25ddecaf7e25e6dd3580acec738dfba22d500eb57fb2069a7371e9589a6448.
3. Direct owner-v2: build=1 — model metadata TU required the MPI SDK;
   resolved with actual MPI::MPI_CXX, no synthetic functions/headers.
   Build log SHA 8393b1a4a19f276f0ad0d21736eefb80b5cf678c6a3eb3ae36158594f0c57c7b;
   compile-commands SHA d53110fd993e289a94097301c7fb72b04285a8906355ce3aa4d30d3cd0f73074.
4. Owner-v3 passes above. Both metadata TUs = unmodified c007 files,
   fully compiled (-ffunction-sections -fdata-sections); ordinary
   -Wl,--gc-sections applies ONLY to the controller-test link (drops
   unrelated loader entrypoints; no tests/algorithms removed). MPI types
   are metadata; no MPI_Init/accelerator init. Trace/perf ELFs need no MPI.

## Evidence digests

Result SHA 0516c4a4ac45b67ba084b8ce97fea9c339f93c1986e5e773997d1aa1cebb0a89;
trace.log f6be0e0da2f5f6af2c2c070b4cf979b54c6ec2074606460ae99fa00b984c3571;
controller.log b92de71c427d327513eeca68d853afe80e456e410f1d507b513e0a744deb4077;
bookkeeping.log 5eb1309672fbed1492d46b6c4d4808f9d3cb7743601331423b3e2206eb414d96;
exact-commands.txt 13c43cbdabfaf62d2db72c914239e4455fdb26333250366e31fbd647eeb45d12;
elf-needed.txt f4c398209eec329aedc317b2bd3d6740ace3d735f4e207b11e0fc3b222b975d1.

## Compiled source SHA256s (all unmodified c007 files)

327cc5ddc8cde66590da3cb6d728e820c4c3610a48e4e1a71a91c389e601a6d4  src/v2/execution/mtp/MTPDepthController.cpp
ea5f754666a269740e9c15de49b823bc9ff5e7e6baa206c73b4a6123d0ade019  src/v2/backends/GlobalDeviceAddress.cpp
b0f646f60d0239f56e8f644c96f15a06f4bb9821b5285363827d59d96afa59e7  src/v2/planning/ModelMemoryProfile.cpp
dc4502eee456d27a4171890a68aaae6e868f920bc02a6be99cb1970aff082098  tests/v2/unit/mtp/MTPDepthTraceReplay.cpp
d1ba6898bbda08935a9baa08214ab4b4ae25cda03d15641e09d7f1289a9e804e  tests/v2/unit/mtp/MTPDepthTraceReplayMain.cpp
2b133b660577354b203ced81822dd2979efa393e963ce21e2e3621cfc84ba4df  tests/v2/unit/mtp/Test__MTPDepthTraceReplay.cpp
6857880bd0bf294e5973193e3ebcab05e15c97e4de0b8bba805e58e00b742ef4  tests/v2/unit/mtp/Test__MTPDepthController.cpp
eaa2d3c469f11f93697c8054fbcc4b1896efc9e121ab6db89982cb63592dfe2a  tests/v2/performance/Perf__MTPDepthController.cpp
8ae0ac83c77aae1b5b2b53ef36aec127b168661059540c570eeb96ba13070e8b  cmake/MTPDepthReplayIdentity.cmake
be90a58ac34d83e8251cb7807cac060ce39b0b9686cfc61efad5474b1774d739  artifacts/idea14/owner-selfcheck/CMakeLists.txt
9dae26bd2275bd0a0c9366d37d05c3b5cced9a226ec059e3effd332fa4e0871c  tests/v2/unit/mtp/fixtures/MTPDepthTraceConfirmationRecorded.h
2a36bd4a7ad45b60681fc9a856e42dbcb1196e47c8b016c7c1a7939bab429355  tests/v2/unit/mtp/fixtures/MTPDepthTraceDirectionalRecorded.h
4f0aa7538b5796dcae54d1cb79598558f61f9e5171fd519e2a2c1d426637a824  tests/v2/unit/mtp/fixtures/MTPDepthTraceGolden.h
8cf50b25aca7f6b7a74acfa7a0d75c250841256d6e7dab59c2e400f9a05c5847  tests/v2/unit/mtp/fixtures/MTPDepthTraceRecorded.h
f4aad2276921f60e0b286a6bd1806ea953462b52d8213c77020965d84de4c809  tests/v2/unit/mtp/fixtures/MTPDepthTraceSurvivalRecorded.h

Binaries:
1f3064cba0051feda1a15215a8ae322e03747da6c4d1c94838d1143ef7cadeb9  idea14_owner_trace
9fd6db4449f91cd4049ce9d7c8d1980a08063a67dcc35915f3d8d363967593cd  idea14_owner_controller
82b5b35dade5da77c36d584e849d4149c22f105a37ded2d8d6ad2ae86c69646a  idea14_owner_bookkeeping
da6038b5675d56070c28ea5a6d6143ffaa600187d5bfd265ddab30416e7ed6e2  compile_commands.json
4df834021de52b9d4a1e3d305adc2d0de019c9a02644f1a6c4df451c5034e00e  generated MTPDepthReplayBuildIdentity.h

## Source audit highlights (full narrative in the lane packet, relayed to reviewer)

- Host MTPDepthController constructed lazily (OrchestrationRunner.cpp:7464-7482);
  validated init (MTPDepthController.cpp:249-346); reset = request boundary
  (:361-377); publication witness from actual publisher O/L/P/stop counts
  (OrchestrationRunner.cpp:14134-14171); recordMTPDepthObservation emits
  evaluated-window records only for evaluated decisions (:7924-8033).
- Native GPU generation has a different live authority (device_terminal_ledger,
  OrchestrationRunner.cpp:17624-17640) — this A job tests host behavior only.
- windowReady (:403-489): pending measured trial needs min samples before
  parent cadence; R >= max(window_size,min_samples) = full window (evaluation
  can still hold); measured-cost path can act at min_samples post-cooldown;
  the ordinary CPU/dense path has NO measured-cost authority.
- At evaluation the consumed window clears (:850-851, 982-984) whether or not
  depth changed; a later partial observation returns evaluated=false/
  WindowNotReady and overwrites last_decision (:841-846).
- Scalar summary takes lastDecision().reason + independently computes
  accepted_tokens/draft_steps request-wide (OrchestrationRunner.cpp:17635-
  17651) — NOT a synchronized last-window tuple.
- MTPDepthDefaults.h: no CPU hardware profile. MTPDepthCostProfile.h: ROCm/
  QFN/MoE TP4 scope only. MTPDepthPolicyScopeProjection.h:45-46 returns
  cpu_architecture_unscoped for a CPU primary. A future measured CPU cost
  profile needs a typed, complete admission identity.
- Existing owning tests cover early demotion (DynamicEarlyDemotes...),
  healthy partial-window waits, learned-hold cadence (LearnedHold...: three
  16-run windows, clipped interleaved, WindowNotReady on partials, zero
  updates — synthetic CPU/MoE contract, not recorded GB10/dense), reset,
  fixed/observe, budget exclusion, probes, cooldown, bounds, measured
  confirmation.
- MTPDepthTraceGolden.h = AUTHORED v3 data (13 host records; windows at
  transactions 5/9/13; clipped transaction 2; Y34/V33 separate publication
  accounting) — not a model capture.

## Disposition

No controller/profile/default/kernel or production inventory change proposed.
A measured CPU policy remains conditional on a negative new paired economy
result, with separate Main-owned exact-pin review and a ProductionTestPreflight
replay regression for any production change. Canonical .80 owner-test run
only if review requires. Full literal compile/archive/link commands are in
the lane packet (retained on the station; relayable on request).
