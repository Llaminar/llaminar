# Linux ARM64 / DGX Spark validation handoff

Date:2026-11-05. This is a dated coordination/evidence record, **not support or
serving certification**. The comprehensive ARM plan remains authoritative for
unfinished implementation and final qualification.

## Source ownership and immutable handoff

Mac implementation owner: session `01a1a8bf`, worktree
`/Users/floriankinder/repo/llaminar`, branch
`codex/qfn-ngram-continuation-arm64`, base/HEAD
`2cba3ca1d5b7a901b349d37b0dff0c46589a04af` at capture time. Both handoffs were
base-plus-uncommitted-overlay snapshots; that base HEAD alone does not contain
the port. Later WIP publication does not change those identities. WSL's active
owner worktrees are not modified.
The Spark session creates its own isolated validation directory/worktree and
performs no duplicate ARM implementation.

The requesting WSL session is
`01a1abb4-4ac5-733e-8ec6-48fa0f2a91de`. Its runtime alias did not exact-match
native cross-machine discovery; fresh `herdr --machine wsl agent list` supplies
the full UUID. Native asynchronous intercom to that full UUID at `@wsl` delivered
successfully. Do not send only to the short-lived Mac CLI relay.

Frozen source handoff on WSL:
`/home/fank/llaminar-work/arm-handoff/20261105T130110Z-nj1_ou9s`.

| Artifact | SHA256 |
|---|---|
| `SHA256SUMS` | `c59c5ef28d856d49dfb865367d2acdd829c38f7c8bac407e6e6ee1d4178c8152` |
| `base-source.tar.gz` | `547cdb07d131ee246dcb4a02cf58131527d38449436a482c5bea5779ccbad0c2` |
| `overlay.tar.gz` | `3ade625c30970194d93d09c5ca5544f1d94d4d40f32c04ec61e86ca2b7d177d7` |
| `manifest.json` | `148e9d58c036048254cf2d04815e0e2515628eb21d24f35c7d7afd72867d5452` |

The base object was absent on WSL, so the34.6MB source-only base archive is
provided alongside the1.16MB overlay. Historical result/model/corpus directories
are excluded explicitly; no LFS/submodule/model fetch ran. The manifest binds123
changed files, modes and symlink targets. `HANDOFF.md`, `verify-overlay.py` and
`SHA256SUMS` supply exact reconstruction and baseline/DotProd configure commands.
Remote SHA verification passed after transfer. Independent extraction into a
new Mac directory built and passed117/117 for **both** frozen Release ISA trees
(3.47s/3.44s), without relying on the moving checkout or Git metadata.

The earlier five-tree checkpoint was117/117 plus all87 configured model-free
preflights repeated20 times. The captured CPUFlash dispatch/timing work remains
WIP and is not part of a whole-engine certificate. In particular, actual class
instantiation still exposes timer reduction/prefetch/Q16 guard/native FA2 policy
and lifecycle work. The handoff includes that disclosed WIP for reproducibility,
not as an invitation to run production models.

## First campaign: RED, with immutable evidence

Both hosts×both NEON profiles configured and built, then each complete focused
CTest run passed112/117, failed5 and skipped0. Every configured87 preflight entry
ran and no test timed out. Identical failing registrations were CPUSwiGLU
Unit/Integration, NEONSwiGLUOriginalRowTails and NEONTypedTensorOps
Unit/Integration. The two isolated failing GTest filters each failed3/3 outside
parallel CTest. Source123-file verification remained exact before/after; no
patch, compiler/arithmetic override, thread cap or timeout increase was applied.

Mac evidence is under
`/Users/floriankinder/llaminar-spark-results/20261105T123901Z`;
`arm-owner-evidence.tar.gz` has verified SHA256
`b86571e88e682a4e121d43203b9d2751b57a37636b6f3df5e9f79ca447f37b3c`.
The remote sources, binaries, commands, compiler databases, ISA receipts,
JUnit/logs and inventories remain unchanged. Containers retired without model,
CUDA, MXFP4 or cluster execution.

Native Linux/arm64 GCC13.3 on the Mac now reproduces the exact failures with
ordinary source-owned flags. Expert ISA contains five contracted Newton steps;
all1039 mismatching outputs equal an explicit fused-Newton diagnostic. Typed
volatile-memory references also find a hidden shared native/scalar Q16+FP32
error, while the observed Q16+Q8 mismatch is the scalar diagnostic's error.
`NEONFP32.h` and explicit scalar decode-product edges now retain the declared
separate rounding without changing thresholds/inverses/ties/clamps/sums,
tolerances, global flags or floating-point environment.

New focused independent tests fail the frozen code and pass after repair.
The repaired checkout passes125/125 in five Mac trees plus native Linux GCC
baseline/DotProd; all93 configured preflights pass20 times on Mac DotProd and
Linux baseline. The timing scope's registered tests are also included, but its
production class integration remains unfinished. These local results do not
rewrite the frozen Spark RED evidence. Deliver a **new** source/hash manifest
for a fresh four-lane run; do not patch or rerun the old Spark source to erase
its failures. Successor identities belong to its external `SHA256SUMS` and
manifest, not a self-referential source-document hash.

## Successor focused ARM PASS; supplemental x86 FAIL

The fresh126-record successor at
`/home/fank/llaminar-work/arm-handoff/20261105T135752Z-mbvtfsvp` has SHA256SUMS
`391c8481b095a1ff1b2a1b1efe1fa6615af387d6c0710ec2d1d8fefba6946128`.
Both Sparks×NEON/DotProd now pass125/125, including93 preflights, with0fail/skip.
All117 original registrations remain and all five old failures now pass.
The requested three selections contain five GTests; each passes20/20 outside
parallel CTest. Baseline receipts have0SDOT/UDOT/I8MM, DotProd2SDOT/noI8MM.
Before/after126-file manifests remain exact; the original RED artifacts remain
unchanged. Independently reconstructed Mac archives also pass125/125 under
both Release profiles.

The already provisioned native x86 WSL host is a Ryzen7800X3D,8 physical/16
logical CPUs, with real CPUID/XCR0=231 admission. Existing genuine SDK
`3ffbc94b…b47f` supplies GCC14.2/GTest/OpenMPI; no package or GPU operation ran.
Both isolated typed targets build, but native AVX2/AVX512 tests fail73/166
assertions while scalar passes; unchanged binaries reproduce that distinction
3/3. The new generic oracle assumes ties-away, whereas source explicitly
retains nearest-even for AVX512 fixed32 Q16 publication. Preserve those
programs and diagnose contraction separately; these runs alone do not establish
an x86 production defect or model impact.

Returned raw logs/JUnit/flags/SDK/loader/hardware/coverage evidence is under
`/Users/floriankinder/llaminar-spark-results/20261105T135752Z-rounding`.
`owner-evidence.tar.gz` independently verifies SHA256
`80d41993658a8b8d799528371df4dfb1480a0445abb483034f4c5077e27e3373`.
Later main-worktree CPUFlash timing/prefetch integration is outside this frozen
certificate. No full engine, models, CUDA, MXFP4, cluster or economy gate ran.

## Reported lab inventory and initial scope

The remote session reports two Linux ARM64 GB10/SM121 hosts with CUDA13.0.88,
20 physical cores/no SMT/one NUMA node each: ten Cortex-A725 and ten Cortex-X925.
Raw sysfs identifies20 unique package/core pairs and singleton siblings; `lscpu
-e` misleadingly renumbers across CPU models. Reported private L2 is512KiB/A725
and2MiB/X925, L1D/L1I64KiB each, with8MiB and16MiB L3 sharing domains respectively.
These are remote inventory observations, not Mac kernel execution evidence.

Both native g++13.3 ASIMD/DotProd compile-only controls passed before the first
campaign; the subsequent real focused kernel run is RED as recorded above.
Both fail a C++23 `<print>`
probe. A real isolated CPU-only Ubuntu24.04 GTest/OpenMP/OpenMPI/NUMA SDK with
Ninja1.13.0 is reported identically on both hosts:
`sha256:5285d390f52aa2909c36e243f2dd153b7578d7f891234ad6ca02c695d9ea58ce`.
No GPU mounts or host package/service changes were made. Full production needs a
compiler/standard-library closure that actually supplies its C++23 APIs; do not
substitute a fake `<print>` header or hide a failed compiler gate.

Initial authorization is frozen **focused CPU** baseline+DotProd builds, CTest,
ISA-envelope receipts and actual inventory/worker evidence only. No full-engine,
model, CUDA or cluster campaign starts from this WIP. No source patches, skipped
failures, arbitrary thread caps, enlarged timeouts, x86 measurement relabeling,
fake NUMA/MPI or fetch fixtures are allowed. The current fixed projection sweep
covers1..10 workers, matching the Mac; it must be extended/qualified for all20
physical Spark workers before claiming whole-host coverage or final scaling.

Remote baseline records are under
`/home/fank/llaminar-work/spark-validation/20261105T123901Z`, with per-host
`arm-cpu-inventory.log`, `cpu-sysfs.json` and `cpu-sdk-inventory.log`. Payloads are
not copied into this source repository.

## Required follow-on integration

1. **MXFP4 source/ABI:** authoritative owner sources are
   `https://github.com/fank/llaminar.git`, PR25 `feat/mxfp4-weights` at
   `fbee08c68` (18B/32 prepared) and PR27 `feat/mxfp4-e8m0-scales` at
   `281358a1f` (17B/32 native E8M0). Physical codebook25/arithmetic policy4 and
   sealed prepared/wire/archive versions must agree on all ranks. The Mac
   baseline predates them. Port the canonical nibble-LUT/source decode/scale
   widening and NativeQuantised GEMV/GEMM directly; do not reinterpret E8M0 as half,
   invoke a generic retrainer or infer support from other formats' tests.
2. **Physical memory:** GB10 CPU/GPU UMA needs one physical-capacity authority.
   Host available RAM and GPU memory availability are not two dedicated pools to
   add/subtract independently. Coherent access does not permit host-owned GPU
   sampler/verifier/KV/request state or extra per-token host reads.
3. **CPU policy/economy:** retain heterogeneous core identities and real cache
   sharing, not a fabricated uniform2MiB private cache. Discover the exact
   cpulists from sysfs, exercise every positive worker count through20, and
   separately measure A725/X925/mixed placement. Exact overlays require actual
   native timing/resource/PMU evidence, never x86 provenance or generic retraining.
4. **CUDA13/SM121:** the owner reports MXFP4 source is architecture-generic but
   qualification exists only for SM80/86/89/90. Compile the complete canonical
   shipped target set plus the admitted Spark target with the spill guard intact;
   run real SM121 all-format/factory/capture/KV/MTP/state/economy gates. Required
   NCCL/native runtime dependencies remain canonical; do not substitute a vendor
   DSO or advertise missing capture support.
5. **Cross-host runtime:** use real Linux NUMA/OpenMPI with the required native
   process/session ABI, matching source/SDK/images and sealed MXFP4 metadata.
   Validate inventory/ISA/workspace wire codecs, per-host rank/device identity,
   device-owned controller publication and ordinary30-second collective failure
   boundaries before serving. Model paths and binaries must exist on both peers.
6. **Models/production:** only synthetic DSV4 MXFP4 models are reported resident.
   They cannot certify real-weight inference. Discover canonical tests/campaigns,
   obtain required model authorization separately, preserve expected token
   payloads, then run reviewed HTTP/prefix/MTP/state and final Release Perf__
   scaling. No model download or expected-token rebaseline is implied by this
   handoff.
