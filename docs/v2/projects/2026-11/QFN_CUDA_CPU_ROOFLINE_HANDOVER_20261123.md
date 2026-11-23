# QFN CUDA and CPU roofline handover

This is the November 23, 2026 checkpoint for the paused goal to reach at
least 80% of the hardware rooflines in 2×CUDA/2×CPU QFN inference. The
working branch is `feat/qfn-cuda-fused-inference`, based on develop
`9594337225e73973f29f9f9897f0c35471e979ac`. Its latest code checkpoint is
`b5d8b2c4`. [Draft PR #70](https://github.com/Llaminar/llaminar/pull/70)

Resume on this branch when the goal is resumed. The two immediate pieces of
work are final Release qualification of the CUDA resident route-prefix/FFN
split and native qualification of the grouped verifier across all source
formats. The CUDA change with retired polling passed native replay on both
cards. The earlier split prototype was faster end to end with identical model
outputs. The final cleaned source still needs its own end-to-end evidence.

## Measurement status

| Scope | Latest qualified measurement | Remaining work |
| --- | --- | --- |
| CUDA original F320/S320 resident aggregate | Roughly 29.9–33.4% of the selected reference across measured cases | Recompute a route-matched phase floor for the final captured graph and measure the full service |
| CUDA split F320/S320 Q4_K/Q5_1 suffix | 477.3–485.3 GB/s against 886.09 GB/s at `160×256`, or 54–55% of the launch-matched DRAM reference | Attribute the gap by live route count, prepared weight traffic, instruction mix and active warps |
| CUDA split Q4_K/Q5_1 HC and route prefix | 252.2–252.8 GB/s at `164×256`, 96 registers/thread, 33.1% achieved occupancy and zero spills | Profile the faster quad-normalization candidate and score the actual prefix phase mix |
| CPU rank-one complete local routed single-row service | 61.83 GB/s service proxy against 110 GB/s measured with both sockets active, or 56.2% | Close the service gap on each socket and qualify grouped and other codebooks at their own matched geometries |
| CPU isolated prepared FFN arithmetic | Gate/Up plus Down reached 82.3% of the socket reference in the isolated case | Keep this distinct from complete service and hybrid throughput |

The experiment host has two RTX 3090s at PCI `0000:3f:00.0` and
`0000:42:00.0`, two Xeon Gold 6238R sockets with 28 physical cores each,
and native ROCm devices for `gfx906` and `gfx1100`. The AMD overlap model
used the two rank-local MI50s on NUMA node 1. Use the actual topology and
assignment at run time. The baseline is the four-shard Qwen3.8-Flash-Next
`UD-Q4_K_XL` source with the shared `Q8_0` MTP sidecar, a recorded
595-token prompt, a 4,096-token context, and 112 GPU-resident experts per
layer. `F320/S320` denotes the 320 local routed and 320 local shared FFN
columns in the frozen tensor-sharded placement.

The CUDA hardware calibration covered the geometry needed for the 15 ROCm
staged-family references, with 828 probe processes over 46 PCI/grid/width
combinations. The maxima were 886.47 GB/s DRAM, 2.627 TB/s L2, 9.449 T DP4A
instructions/s and 12.549 T FFMA instructions/s, each from its recorded
geometry. At `160×256` the two cards reached 886.09 GB/s DRAM, 8.742/8.879
T DP4A instructions/s, 11.775/11.991 T FFMA instructions/s and approximately
2.5 TB/s L2. Use the phase scorer and exact card, grid, block width, source
format and route count. Compute the dominant memory or arithmetic time per
ordered phase. Add phases with sequential dependencies and measure concurrent
service and waits on the captured critical path. Use physical DRAM counters
for memory-bandwidth efficiency and label logical weight/service rates as
proxies. Keep the route-matched byte counts, arithmetic counts and scorer
outputs with the timing. Vendor peak rates and hot-L2 fixtures are poor
comparators for the production resident stream.

Some isolated CUDA vocabulary M6–16 cells are in the 93–99% range. Earlier
matched wide Gate/Up and Down results included 14 of 21 source formats at
80% or above on both roles. The roofline documents preserve the matrices,
formats and dispatch qualifications. The current 80% goal still applies to
the production families and complete CPU/GPU service, with coverage across
quantizations.

- [CUDA probes, phase scorer and recorded measurements](../../../../tests/v2/performance/kernels/cuda/roofline/README.md)
- [CPU simultaneous-socket, ISA and complete-service references](../../../../tests/v2/performance/kernels/cpu/roofline/README.md)
- [MI50 resident HC/FFN recurrence, phases and arithmetic](QFN_MI50_RETAINED_HC_FFN.md)
- [Peer latency, sharding and CUDA/MI50 overlap](QFN_CUDA_DECODE_PLACEMENT_AND_PEER.md)

## Sharding, transfer and model correctness

- The 10 KiB CUDA peer-store copy took 2.27 µs in both directions over PXB,
  versus 2.85–3.05 µs for peer DMA. Direct rooted reduction at 10 KiB took
  3.62–3.65 µs versus 9.86–10.08 µs for NCCL Reduce. Both adapters support
  directed peer reads/writes and report no native peer atomics. Production
  adoption of the event-ordered prototype requires exact captured edges from
  both input producers and retention through the last peer access. Keep the
  cross-device ordering and actual live transfer extents visible in the graph.
- Tensor-sharded and apportioned placements produced different model histories
  at token 10. Each placement matched its own ordinary and fixed-depth history.
  Same-input captured Down arithmetic reproduced different FP32 bytes between
  the full-width reduction and two half-width fragments on both GPUs. A
  frozen-history model diagnostic already found different full-vocabulary
  logits immediately after prefill. The precise model-stage source of the
  prefill difference is open. The 51.82 versus 44.12 tok/s comparison used
  different generated histories and verifier work. Retain F320/S320 for A/B
  tuning until the placement comparison has a matched numerical contract.
  `QFN_CUDA_DECODE_PLACEMENT_AND_PEER.md` records the token IDs, logit witness
  and placement change together with the peer and overlap results.
- The exact 112-to-144 change preserved all 256 tokens and MTP work. Its
  rate advantage changed sign with run order, and it used about 686 MB more
  prepared weight storage per GPU. Keep 112 as the tuning control.
- CUDA and MI50 mapped-packet overlap are already installed for their
  qualified placements (`9bc5f312`, `68be59bc`). This publishes the route
  packet before independent shared FFN, allowing CPU dispatch to overlap
  with GPU computation. The existing gfx906 retained HC/FFN arithmetic and
  the pending CUDA route-prefix/FFN split are distinct programs. Both use the
  same principle of finishing setup and immutable routing metadata before the
  arithmetic that consumes them.
- For every prospective dispatch change, compare full generated token IDs and
  per-iteration MTP work on the same Release binary across both run orders.
  Prove the production path with counters or actual graph inventory. If tokens
  diverge, reduce the first differing token, row and stage with frozen input
  and original serial production arithmetic. Register a focused
  `ProductionTestPreflight` regression for the defect. Grouped verifier rows
  require serial byte identity, including when they participate in MTP.

## Current slice and shipped direct publication

**CUDA polling retirement.** The cleaned candidate passed six retained native
CTest entries on each RTX 3090, plus twelve additional stress invocations per
card. Each recorded frontier case checks 384 captured replays, including the
data and the native graph edges. Driver warning/error logs matched after
excluding the unrelated kernel audit suppression counter. The cleanup and its
preflight registration are in a local candidate patch. Release builds of the
cleaned source and actual seeded model tests remain pending.

**CPU Q8_1 row publication.** A 3×33 input reproduced byte differences in
mutable conversion and an immutable conversion error of 144 produced versus
216 required bytes. The original loop grouped a flat FP32 input stream into
blocks, while native Q8_1 storage pads each row independently. The fix runs the
same quantizer per native block, writes terminal blocks through the owning
row, and lets immutable construction adopt the final aligned allocation.
The test checks three distinct rows against the original block quantizer,
both mutable and immutable entry points, and a poisoned fourth capacity row.
Both `V2_Unit_Q8_1Tensor` and `V2_Integration_CPUQ8RowPaddedPublication`
passed on AVX2 and AVX-512. The tested code is in `b5d8b2c4`. See
`branch-qualification/cpu-q8-row-publication/` for the failing original
probe, the fixed probes and the configured test logs.

**Installed CPU work.** `8a43eead`, `451e0b13`, `ba59e9bf`, `75ff2308`,
`437971fd` and `17380d93` cover ordered K folds and serial, compact, paired,
wide and grouped tail stores into live output columns. The latest CMake
qualification covered one- and multi-block and prefill paths in both ISA
modes. The ordered fixture exercises all 21 sources, seven tail widths,
three K shapes and M7, with exact output and padding guards. The grouped MoE
Down inverse-ownership proof in `d1b3a668` moved a 2,544-call rank-one
validation interval from roughly 1,438 ms with the original quadratic scan
to 3.54 ms using the existing reciprocal route mapping. The latest direct
CPU FP16 change `da5eceeb`
converts FP32 into the owning CPU ring slots in both HEAD_MAJOR and
POSITION_MAJOR layouts. Native tests distinguish ordinary and verifier
advancement for an input spanning more than the ring and check raw bytes
through wraparound.

The isolated packed CPU Down change was faster across 216 measured
format/socket/route/order cells. Its crossed QFN Release comparison reproduced
all tokens and MTP work, with ordinary decode +4.61–5.70% and fixed-depth-seven
decode +2.22–3.04% against the staged control. A later exact comparison of the
combined CPU changes against the already direct packed control shifted sign
with run order. Ordinary combined throughput was −2.03% / +0.42% and fixed
depth seven was −1.34% / +0.76%. The measured effect varies with order.
See the CPU roofline README and the paired
summaries under `cpu-direct-canonical-down/model-comparison/` and
`direct-publication-release/final-source/model-comparison/`.

Other shipped audit work includes direct floating-point probabilities,
direct canonical route scatter, and native GPU aliased short convolution
(`ef6e0d81`, `f39123ad`). Captured CUDA shortconv at 10,240 channels and
M2–4 took roughly 2.0–2.55 µs versus 3.9–5.2 µs for staging. MI50 was
36–40% faster and gfx1100 30–33% faster on matched captures. The ROCm INT8
beta publication fix (`a0cae367`) also corrected the partial-CTA shared-load
barrier contract. The actual odd-N defect is covered by a source-based
integer oracle in the native preflight regression.

## Resume the CUDA resident path

1. The known-safe packet boundary is native completion of one `164×256`
   HC/route prefix. CPU ticket/publication and the `160×256` FFN suffix branch
   from that completion and join before dependent work. Every suffix CTA has
   16 live shared Down columns, even with zero routed experts. Both RTX 3090s
   reproduced the stall when an independent CUDA reader spun alongside the
   fully resident cooperative grid. Adding one, two or three reserved CTAs
   did not fix it. A real model run hit the driver watchdog. Preserve the
   native completion dependency and the separate legitimate inter-device wait.
2. The original split prototype, built from `75ff2308` plus its patch, passed
   the same-Release dual-order QFN comparison. Across both orders, ordinary
   decode gained +0.64–2.80% and fixed seven gained +1.28–3.85%. Complete
   throughput gained +0.59–2.70% and +1.11–3.42%, respectively. All 24 measured
   256-token histories and MTP work counts matched. The selected native path
   was authenticated by seven fewer captured compute stages per layer on both
   GPUs. Capture structure is available when the benchmark PerfStats filter
   omits the counter. Prefill variations are session variation for this
   decode-only change.
3. Build the cleaned source in Release and run the saved seeded stochastic
   comparison. The runner uses seed 424242, temperature 0.65, top-k 40 and
   top-p 0.95 for ordinary, fixed seven and dynamic MTP. It checks two
   measured 96-token iterations in each cell with full token IDs, MTP work,
   selected graph structure and driver status. Then repeat the quiet crossed
   256-token ordinary/fixed-seven campaign for the cleaned source, rewriting
   the timing wrapper's source/build paths and output directory so the original
   run remains intact. The current prototype selector is
   `LLAMINAR_QFN_CUDA_MAPPED_RETAINED_M1_HC_FFN_CANDIDATE=0/1`.
4. The independently timed ROCm-style adjacent-four prefix norm keeps the
   original square-sum and Q8 reductions. It passed the full raw capture
   oracle and cut sharded capture latency 3.4–5.5% across all three tested
   format pairs, both cards and both run orders. Counted cases improved
   0.9–4.2%. The initial shared scratch union grew from 4 KiB to 11 KiB,
   including the suffix where the storage is unused. Give the suffix a
   phase-specific scratch type, retain the exact prefix folds, and recheck
   shared memory, registers, spills and graph bytes. Use the saved exact
   Nsight single-phase/single-device wrapper as the template for profiling
   this candidate. Then measure it in the real model. The separate Down
   warp-distribution experiment gave only a 0.24–1.58% sharded improvement
   and slowed some counted cases. The distinct routed-Down warp patch is
   untested.
5. Tune the suffix independently using hot-path time, achieved active warps,
   IDP.4A/FFMA/PRMT mix and actual prepared weight traffic. Standard SM86
   Q4_K/Q5_1 split profiles showed 80 suffix registers, 23.8% achieved
   occupancy and zero spills. Treat each profiler run as a single exact
   kernel/ISA/shape/M measurement, separate from canonical event timing.
   Compare with MI50's straight-line mathematical body and keep pointer,
   format, launch-limit and validity setup out of the inner loop.

## Resume the grouped verifier and drafters

The CUDA all-source retained packet candidate replaces its small hardcoded
Gate/Up and Down tables with the authoritative source catalog. All 21 original
source formats are represented through 18 base physical layouts plus the
lossless moved codebook. NVCC compiled 114 entries (38 Gate/Up and 76 Down),
with zero stack or local memory reported by the resource scan. The fixture
compiled against the actual target flags. The native all-source CTest is
pending. It compares production prepared packets with independent staged
table/one-route arithmetic at F320 and F640 for Routed and Shared programs,
M1–16, empty and large-to-small replay, poisoned inactive rows, Q8 bytes,
scales and FP32 contributions. Run
`V2_Integration_CUDA_RetainedGroupedFFNAllSources` and
`V2_Integration_CUDA_RetainedGroupedFFNPacketReplay` on each RTX 3090. The
current production grouped selector uses the staged graph. Prior retained
grouped and scalar drafter experiments had neutral or negative real model
results, so they stayed experimental.

Once that baseline passes, apply the shorter F320 packed unroll to its core
three CUDA files. Use the latest expanded all-source fixture. The standalone
NVCC results reduced registers in 11 formats with no reported spills, for
example IQ3_S 186→88, IQ3_XXS 150→80, IQ2_S 162→86 and Q5_0 108→63. F640
was unchanged. Prove the exact ordered K fold on both cards, then time the
actual captured packet roles using independently prepared expert weights
with a representative working set beyond L2. The existing shared-weight
fixture is suitable for correctness. Establish timing and roofline by
format, M and live route mix. The prior Down weight-reuse experiment already
slowed down, with evidence under `m4-grouped-weight-reuse/`. Retain the
separate Q8/logit, packet and real-model probes for verifier and drafter
attribution. Before selecting a new fused path, run ordinary, fixed and
dynamic MTP on the real workload with exact histories, exercised paths and
repeatable throughput.

The ROCm grouped beta/bias candidate writes its terminal result directly
into the owned destination. The dynamic version passed three native MI50
and four native gfx1100 tests, including the latter's WMMA path. Paired
capture timings improved beta latency 3.87–6.92% on MI50 and 11.57–24.20%
on gfx1100. The raw MI50 path slowed by up to 3.59%. The prepared static
variant selects the fold before capture for Dot4, the ordered split-K
reducer and WMMA. Run the saved static wrapper, then adapt the matched timing
runner to its new DSO and separate output directory. Compare both beta and
raw modes on both devices before adoption. Keep the partial K producers and
their fixed tree, and remove the obsolete full-result copy only when the
terminal publisher is authoritative.

## Continue the publication audit

The remaining known CPU KV sites are Q8_1 ring conversion, legacy TQ4/TQ8
and FP32 paths, and FP32 source slices on the CPU batch path. The corrected
generic Q8_1 tensor conversion is the prerequisite for the Q8_1 KV change.
Bind the final cache row or head slots before each publication, derive its
per-row native extent and handle partial terminal blocks for odd head depth.
Check both ring layouts, wraparound, partial and oversize batches, untouched
slots, and independent serial-verifier cursor advancement. Use the FP16 ring
test and owning-cache writer as the basis. Some legacy flat block/scatter
loops assume alignment, so add a reproducer for any odd geometry defect.

Start the remaining audit at actual CPU/CUDA/ROCm stage entry points. Follow
the scratch buffer to the first consumer and bind the final exact extent where
the arithmetic and ownership allow it. Grouped affine oneDNN output still
needs one scatter into sparse canonical route slots. A second post-scatter
copy was removed. Parallel K partials, explicit cross-device transfers and
unavoidable alias hazards have independent lifetimes. Keep those lifetimes,
the regular reduction order and the exact stream/event publication intact.
Retire full-output buffers and launch nodes when direct publication actually
removes their purpose. Check guards, partial rows and capture replay in the
preflight entry for the changed real path.

## Recover the work and reproduce it

All paths below assume `/workspaces/llaminar`. Historical receipts, exact
patches and local runners are under `benchmark_results/qfn-cuda-2x2-95943372/`.
These are workspace-local artifacts. The handover bundle is designed to
survive loss of the detached `/tmp` worktrees. The source branch and this
document are the portable Git state. Preserve the local evidence directory
when moving this work to another machine.

```bash
cd /workspaces/llaminar
results="$PWD/benchmark_results/qfn-cuda-2x2-95943372"
handover="$results/handover-20261123"
(cd "$handover" && sha256sum -c SHA256SUMS)
column -ts $'\t' "$handover/patches/bases.tsv"
```

`patches/bases.tsv` gives the exact origin commit and old worktree for each
complete candidate. Restore a missing candidate to a detached worktree at
its recorded base and use `git apply --check` before `git apply`. Files named
`.increment.patch` require the named predecessor. The grouped shorter
unroll full patch also contains an earlier catalogue fixture snapshot. For
the timing comparison carry over its three core files, using the current
all-source fixture in both arms. `scripts/` preserves the actual wrappers,
including draft-model campaigns and profiler selectors. `config/` holds the
baseline campaign, tensor-sharded plan, and observed Integration and Release
CMake caches. The older scratch checkpoint is included for historical context.

These commands recover an editable copy of the cleaned CUDA candidate.

```bash
clean_src=/tmp/qfn-cuda-native-packet-clean-src
clean_patch="$handover/patches/cuda-native-route-completion-clean.patch"
git worktree add --detach "$clean_src" 437971fd
git -C "$clean_src" apply --check "$clean_patch"
git -C "$clean_src" apply "$clean_patch"
```

The two Release qualification runners use the recorded native receipts and
create their own independent Release source and build trees.

```bash
bash "$handover/scripts/qfn-build-native-packet-clean-release.sh"
bash "$handover/scripts/qfn-prove-clean-packet-stochastic-model.sh"
```

The reusable dual-backend build is
`/tmp/qfn-roofline-dual-integration-gcc14-isolated`. Its private source tree
`/tmp/qfn-dual-roofline-qualification-src` started at `82a043c6` and has
accumulated several overlays. It is separate from the feature branch. Use the
`patches/dual-integration-shared-qualification.snapshot.patch` file in the
bundle if that tree disappears. The most recent successful
CUDA cleanup and Q8_1 builds shared this tree. Never run two applying wrappers
against it concurrently. Inspect scripts before rerunning them because some
expect their patch to be absent and will stop if it is already present.
The CUDA grouped and static ROCm wrappers were deferred before they started.
Both pending patches passed `git apply --check` on the preserved snapshot.
The Release wrapper for the CUDA cleanup and its seeded campaign were likewise
deferred before any build or GPU work began.

The qualified build used GCC 14, Ninja, ccache, CUDA SM86, ROCm 10.1 with
`gfx906;gfx1100`, the existing NCCL provider and locally built RCCL. The
archived configuration wrapper records exact flags and dependency paths. If
the caches are rebuilt, use the current CMake and backend skill instructions
with authenticated source/build paths. Existing Integration build objects and
Release reference caches may be reused only with their consuming source and
configuration. Explicitly run native AMD work with
`env -u HSA_OVERRIDE_GFX_VERSION`; the durable devcontainer and
`/etc/environment` injection is gone, but the existing container PID 1
inherited the old `9.0.6` override. Recheck enumerated native architecture and
device ownership after reboot. The repository hook was changed in `811c26ca`
to run Unit only. Full preflight remains a separate gate. The recorded full
Unit run passed 1,107 tests on an earlier qualified snapshot. Recent changes
have their focused receipts.

The following paths are relative to `$results`.

| Evidence | Location |
| --- | --- |
| Phase calibration and all-family source | `cuda-cta-width-roofline/` |
| CUDA liveness failure and native-edge witness | `branch-qualification/cuda-independent-packet-readers/isolated-native-scheduler/results.log` |
| Original CUDA native, Nsight and model crossover | `branch-qualification/cuda-native-packet-frontier/`, especially `native-ncu/`, `suffix-roofline/` and `release/model-crossover/summary.txt` |
| Clean CUDA native results and driver snapshots | `branch-qualification/cuda-native-packet-clean/device-[01].log`, per-card repeat logs and `driver-*-stable.log` |
| CUDA norm event timings and static resources | `branch-qualification/cuda-frontier-arithmetic/summary.txt` and per-arm `resources.log` |
| Grouped CUDA catalog and shorter unroll resources | `branch-qualification/cuda-grouped-catalog/native-resources.log` and `branch-qualification/cuda-grouped-short-unroll/resource-comparison.csv` |
| ROCm beta paired timing and isolated native profiles | `branch-qualification/rocm-grouped-direct-epilogue/paired-timing/summary.txt` and `native-profiles/summary.txt` |
| CPU native Q8_1 and FP16 receipts | `branch-qualification/cpu-q8-row-publication/` and `branch-qualification/native-fp16-kv-publication/` |
| CPU validation attribution | `direct-publication-release/grouped-linear-isolation/profile/summary.txt` |
| Earlier full Unit qualification | `branch-qualification/actual-gcc-isolated/full-unit.log` |
