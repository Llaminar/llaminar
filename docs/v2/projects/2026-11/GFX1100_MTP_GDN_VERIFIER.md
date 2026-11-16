# gfx1100 complete GDN verifier qualification

Work-size-aware GDN Automatic dispatch qualifies161 measured keys. Both
all-source Release timing passes retain every selected median win across
SingleQ8 and TwoTermQ8. All16 exact-producer profiles and six focused host gates
pass. Eight affected GPU gates (128 cases) and both CUDA/ROCm capture gates
(12 cases) also pass. The earlier forced-candidate
baseline passed43 Release cases and three registered Integration gates.

## Original operation and numerical contract

`Test__ROCmGDNVerifierTransaction.cpp` captures actual `GDNProjectionStage`
execution with K2560 and QKV/Z/alpha/beta widths10240/6144/48/48, matching the
existing four-source GDN component. Every quantized source in the canonical
21-format registry appears independently as QKV and Z. A separate Q4_0/Q8_0
mixed-source control retains both arithmetic modes. Original FP32 alpha/beta
are materialized under the established FP16RNE preparation policy.

Dot4, forced WMMA and serial M1 have separate prepared stores, mutable kernels,
workspaces and outputs. Within each owner, all four distinct original sources
belong to one prepared store. The M1 oracle executes the actual complete stage
with the same activation precision; it does not borrow grouped quantization,
weights from another candidate's prepared owner or grouped output snapshots.

SingleQ8 applies native candidates to QKV and Z. TwoTermQ8 retains the existing
corrected QKV producer and applies the candidate only to Z. Both modes include
all four projections, publication and preparation-independent execution in the
complete-stage timing interval. This work does not replace floating kernels or
rocBLAS, and does not claim serving-topology or model-level MTP speedups.

Every physical M2–16 covers full/one/empty/regrown/repeated work and every live
prefix. Outputs and transient scratch start dirty; active rows must equal
independent serial bytes and inactive capacity must remain poisoned. Reset
retains the captured topology. Exact native arguments authenticate original
input/Q8/scale/sum banks, rows, source descriptors, outputs, publication mode,
partition span, and bounded nonoverlapping partial planes within each packet.
Sequential subgroup packets may reuse an admitted bank.

Correction publication and projection authenticate their full prepared QKV
source, quantizer, output, row authority and four distinct primary/residual
banks. Omission controls independently remove the full activation-publisher
family, native projections or corrected projection, checking both the expected
failure and surviving sibling outputs. TwoTerm already publishes primary Q8
before Z; removing only Z's later quantizer would not reliably remove the
activation producer. Native inventory rejects private scratch.

The fixture explicitly initializes immutable hardware metadata and finds an
actual visible gfx1100 participant. It does not require that card to be the
host's first physical GPU.

## Forced candidate economy

Each sample contains eight complete stages between retained internal GPU events.
Setup, poison, observation and independent M1 execution are outside the timing
interval. There are31 alternating Dot4/WMMA pairs per full physical extent;
all prefix correctness checks precede timing. Power remains ordinary `AUTO`.

| Mode | Cells | Native median wins | All31-pair wins | All31 wins plus ≥5% median gain | Geometric gain of that stronger set |
|---|---:|---:|---:|---:|---:|
| SingleQ8 |315|161|122|117, across all21 sources|1.17242×|
| TwoTermQ8 |315|158|84|44, across12 sources|1.06417×|

Unconditional WMMA is uneconomical: geometric ratios across every measured M
are0.94493× for SingleQ8 and0.98488× for TwoTermQ8. SingleQ8's first stronger
winner varies from M9 to M15 by source. At M16 its gains span1.07099–1.33814×.
TwoTerm's smaller gains reflect the unchanged corrected QKV cost; IQ1_S/M16
still loses0.23%. Selection must retain the original arithmetic mode and source
identities, with Dot4 at losing or unmeasured work sizes. These observations do
not yet qualify actual Automatic execution or prescribe a monotonic threshold.

## Frozen forced-candidate evidence

Root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Full Release replay:
  `mtp-gdn-verifier-release-replay-1794789637128590058/receipt.json`.
- Full Release timing and per-mode analysis:
  `mtp-gdn-verifier-release-timing-1794789732667064905/{receipt,analysis}.json`.
- Registered Integration gates:
  `mtp-gdn-verifier-feature-1794789844938488017/receipt.json`,92.20s CTest wall time.
- Both builds: `gdn-verifier-{release,integration}-final3-build.log`.
- Frozen producer: `mtp-gdn-verifier-seed-snapshot-1794790099422775362/snapshot.json`,
  four records/42 files, SHA256
  `923640b37a7562e7c4e74833761c266986d060d4dd373ea99256f8e173b9da34`.

## Automatic implementation and qualification

The complete four-source binding is installed before corrected QKV can publish.
Copied subgroups retain exact copied storage and ordered roles against that
original operation. Native admission has a separate GDN type: named QKV/Z
members, typed prepared alpha/beta controls, original device/input/rows/K,
correction mode and row identity, and all three Q8 publication banks. QKV's
existing original-span contract is unchanged. The public fused-projection verifier API
installs both native scope types, with an empty binding for the unrelated operation.

SingleQ8 admits QKV/Z; TwoTerm admits only Z while retaining the original QKV
source in its performance key. Standalone Z supplies its original packet before
row/scratch tiling. Controls must be physical FP16 with FP16RNE activation
arithmetic, N48/K2560, and distinct FP32 outputs. Input and all four outputs must
be FP32. The neutral GDN boundary prevents an unqualified bundle from borrowing
scalar eligibility. Exact hosted extent remains mandatory; mixed sources,
guarded capacity, unknown shapes and unmeasured widths keep Dot4.

Common provenance preserves CPU's counted correction rows beside fully active
siblings. Exact correction/sibling identity equality is a ROCm native-admission
requirement. SingleQ8 continues to accept unused correction-row metadata. The
actual tiny CPU stage and original model-row resolver cover these distinctions;
malformed-binding controls explicitly declare their separate shared authority.
Device-free tests cover original/copy/role identities, count reset, nested and
concurrent recording, source/control/precision rejection and M1–33 totality.

The initial mask selects117 SingleQ8 and44 TwoTerm keys from the frozen seed's
all31-pair wins with at least5% median gain. Actual Automatic replay passed43
cases, all21 sources and both modes plus guarded/mixed exclusions. Its captured
kernel inventory matches all161 independently derived mask keys; all active
outputs equal independent serial-M1 bytes across dirty prefix/reset replay.

First actual-Automatic timing passed630 cells/19,530 pairs. All161 selected
medians win: SingleQ8's117 native keys have geometric gain1.18868×,
range1.04842–1.37337×; TwoTerm's44 have geometric gain1.06391×,
range1.03576–1.10068×. All31 samples win for110/117 SingleQ8 keys and44/44
TwoTerm keys. These are complete-stage gains;469 Dot4 controls remain explicit.
The second630-cell/19,530-pair Automatic pass again wins all161 selected
medians. SingleQ8's117 keys have geometric gain1.18518×,range1.03272–1.36177×,
with112 all31-pair wins; TwoTerm's44 have geometric gain1.06039×,
range1.03516–1.08771×,and all44 win every pair. Across both modes the second
native geometric gain is1.14968×. The5% seed margin is a qualification filter,
not a guaranteed future gain; actual confirmation retains a minimum3.27%.

All16 separate gfx1100 profiles pass, eight counters for each Q8_0/M16 packet.
SingleQ8 combines QKV/Z in a256×16×1 grid; TwoTerm launches Z at96×16×1.
Both use128 threads,80 logical VGPRs,7,040 logical LDS bytes, wave64 and no
private storage or memory spills. Loaded ELF SHA256 remains
`3cb215279203a6e5781dac965660e7fca1e5824f1ffaa926201cb4a4d84c896a`,
matching the earlier all-source verifier and QKV producer. Each counter run
restores AUTO and retains the exact source/core/executable and grid joins.

| Counter | SingleQ8 | TwoTermQ8 |
|---|---:|---:|
| GPUBusy |100%|100%|
| Memory-unit busy |68.2540%|57.1899%|
| L2 hit rate |22.2084%|23.6757%|
| FETCH_SIZE |43,576.375 KiB|16,370.625 KiB|
| Wavefronts |8,192|3,072|
| LDS conflict cycle ratio |2.85036%|2.85036%|
| Average issued VALU instructions/wave |475|475|
| Occupancy |40.6705%|32.6527%|

LDS conflict ratio uses the SDK's indexed-LDS active-cycle denominator; it is
not a percentage of total kernel time. These diagnostic counters do not replace
unprofiled complete-stage timing. Six registered host gates/32 GTests, eight
affected GPU gates/128 GTests, and both CUDA/ROCm shared-stage capture
registrations/12 GTests are green with no skips.

Current evidence beneath the same root:

- Actual-Automatic Release replay:
  `mtp-gdn-verifier-automatic-release-replay-1794791608656033249/receipt.json`.
- First actual-Automatic Release timing:
  `mtp-gdn-verifier-automatic-release-timing-1794791810558835174/{receipt,analysis}.json`.
- Second actual-Automatic Release timing:
  `mtp-gdn-verifier-automatic-release-timing-1794792422806485165/{receipt,analysis}.json`.
- Exact uninstrumented profile owner:
  `mtp-gdn-verifier-automatic-release-replay-1794792029614595702/receipt.json`.
- Six final host registrations/32 GTests:
  `mtp-gdn-verifier-automatic-host-1794792179229527206/receipt.json`.
- Eight affected GDN/QKV feature registrations/128 GTests:
  `mtp-gdn-verifier-automatic-feature-1794792609617490536/receipt.json`.
- CUDA/ROCm shared-stage capture registrations/12 GTests:
  `mtp-gdn-verifier-symmetry-1794793171616413517/receipt.json`.
- Independent final Automatic audit (profiles, replay, timings, host and feature gates):
  `gdn-automatic-qualification-audit.json`, SHA256
  `9da44a1b4a83114b4f42c0921ade9f4864dbd2fd641bb41cd911206730a3e6e3`.
- All16 profiler records:
  `gdn-automatic-profile-batch-1794792180383471791/progress.json`.
- Frozen final producer and qualification records:
  `mtp-gdn-verifier-automatic-qualified-snapshot-1794793588829245658/snapshot.json`,
  SHA256 `ce9c4dca418b12b9c787e3a4420b82d0edec65462104fba47323469e05ff55f3`.
  This preserves23 final records, all16 profiles and four historical green seed records.
- Production builds retain gfx906/gfx1100 plus CUDA sm86, with the spill guard.
  `gdn-auto-release-build-r3.log`, `gdn-auto-release-build-final.log`,
  `gdn-auto-integration-build-r3.log` record the compiled source.

An early query-resolver observation overlapped an in-progress Integration
relink and is not qualification evidence. The same stable final artifacts then
passed20 repeated query-gate runs (`gdn-auto-query-repeat.{log,xml}`). The first
expanded host run also exposed obsolete shared-count assumptions in the tiny
CPU fixture's negative controls; the final six-gate/32-GTest host receipt
qualifies their corrected declarations without a production arithmetic change.
