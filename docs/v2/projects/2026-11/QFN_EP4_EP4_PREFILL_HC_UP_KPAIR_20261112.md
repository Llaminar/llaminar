# QFN HC-up prefill: native gfx906 K-pair staging

## Result and qualification scope

Matched Release prefill improves **793.724→817.653 tokens/s (+3.015%)** on
PP(EP(4×MI50), EP(4×MI50)), with fixed-head ngrams sharded across all eight
cards. This is **11.265%** above the original734.870 tokens/s before the
[publication](QFN_EP4_EP4_PREFILL_PUBLICATION_TUNING_20261112.md) and
[HC-down tile](QFN_EP4_EP4_PREFILL_HC_DOWN_TILES_20261112.md) changes.

The three original Q8 source-alias replay cells pass separately on gfx906 and
gfx1100, including byte equality, dirty reset, partial/empty work and aliasing.
The installed MI50 program wins across M32/128/512/1024/2048; gfx1100 remains
unchanged. This is a focused kernel and matched-prefill checkpoint, not full
preflight, release certification, a new KL panel or an OpenCode result.

One existing broader ROCm case passes its arithmetic but still loses the
continuous driver-log cursor in both q226 and q230. Those health observations
remain failed; they are not superseded by the six complete new cell windows.
The next project gate is accuracy panels and OpenCode on PP(EP4, EP4) and
TP4 MI50 with hybrid experts. Further tuning is deferred to that qualification.

## Matched full-model measurement

q229 uses the original four-shard Qwen3.8-Flash-Next-UD-Q4_K_XL file, exact
2560-byte `" test" * 512` prompt,512 prefill tokens, one emitted token and no
after-prefill decode. Context4096/M512, one warmup/three measured requests,
seed42, temperature0/top-k1/top-p1, Static/Ordinal expert placement, MTP off
and prefix cache off. FP32 activations, FP16 KV/router operands with FP32
router accumulation, FP16 MoE alpha/beta, and HC two-term Q8 are unchanged.

Discovery rank1 on NUMA1 owns stage0 devices7/6/8/9 (layers0–23) and tail
1/3/4/2 (layers24–47). Original UUID/PCI ancestry, MPI2 rankfile, eight VRAM
ngram owners, source files and policy are retained. There are no visibility
masks, profiles, stage clocks or overlapping builds/native jobs. The benchmark's
prefill clock ends at durable inference completion, before first-token emission.

| Observation | q183 previous tiles | q229 K-pair |
|---|---:|---:|
| Request1 prefill, ms |646.835414|626.617517|
| Request2 prefill, ms |644.336694|626.080789|
| Request3 prefill, ms |645.060744|626.182857|
| Median prefill, tok/s |793.723699|817.652534|
| First output ID, each request |1228|1228|

All eight terminal ngram live-ID, payload, call and retirement tuples match.
The driver window is complete with407,380 records/35 snapshots and no findings.
Native processes/descriptors retire and all eight GPUs return to10,866,688B
idle VRAM. Twelve sparse whole-lifetime samples show individual native process
VmHWM750,751,744B and21,946,552,320B, no swap, and selected VRAM peaks
20.931–21.327GB/card. These are not per-phase peaks or a long-duration leak proof.

Full-model artifacts: `/tmp/qfn-ep4ep4-prefill-next-20261112/` retains the
benchmark JSON, original command/configuration pins, source diff, native
ownership, memory observations, driver journal, validators and receipts.
`/tmp/qfn-ep4ep4-prefill-next-prep-20261112/` retains the launcher and q183
comparison. No decode or long-context speed is claimed from this512/1 workload.

Component artifact paths below are relative to
`/tmp/qfn-hc-up-kpair-20261112/` unless an absolute path is given.

## Implemented policy

The existing ROCm cooperative codebook19 two-term HC-up projection retains
32x64 output tiles and 256 threads. Its compiled gfx906 production body stages
two successive K32 groups through separate 64-thread producer cohorts and
retires each pair together. Each output retains its original increasing-K
FP32 folds for primary and residual, then one final sum. Odd final groups
retain their original fold without adding a synthetic zero contribution.
There are no new weight/output copies, global partials, host count readbacks,
runtime admission rules or alternate inference paths.

`projectGrouped<0>` is production: gfx906 selects pair staging at device
compilation; gfx1100 retains original single-group staging. The original
single-group `<1>` and named reference functions remain diagnostic numerical
oracles. Candidate `<2>`, Row64 entrypoints and their trial implementations
were retired. CUDA behavior is unchanged. Native inputs, live-row authority,
workspace/publication ownership and numerical contracts remain the same.

Files: `src/v2/kernels/rocm/gemm/ROCmQ8ResidualGrouped.cuh`,
`src/v2/kernels/common/GPUQ8ResidualCorrection.cuh`,
`src/v2/kernels/common/Q8ResidualGroupedDiagnostic.h`, the existing
`Test__Q8ResidualCorrectionDevice.cpp` fixture and canonical test registration.

## Completed build/body proof

Release q217 passed in 83.220 seconds. Root's q228 offline audit passed all
four entry/target comparisons in `static-q217/audit.json`: normalized full
instruction text and raw resources of gfx906 production0 equal measured q203
pair2; gfx1100 production0 and both original reference1 bodies equal q203
single1. All four body diffs are empty; no retired diagnostic2 remains.

| Target/body | Raw VGPR | Raw SGPR | LDS bytes | Private/spills |
|---|---:|---:|---:|---|
| gfx906 production0 |60|42|9216|0/0|
| gfx906 original reference1 |57|40|4608|0/0|
| gfx1100 production0/reference1 |57|38|4608|0/0|

All are wave64/256 threads with no dynamic stack or scratch instructions.
Integration q216 also passes. Both builds retain the mandatory spill guard
for CUDA sm80/86/89/90 and HIP gfx906/gfx1100. Runtime evidence here covers
MI50/gfx906,7900 XTX/gfx1100 and the unchanged RTX3090/sm86 path; absent
architectures are compilation-qualified only.

## Unprofiled complete-chain economics

q209 (MI50) and q213 (XTX) each retained 31 pairs per original source. Q8_0
and Q8_1 use M512/H2560/R320 (HC-up N10240/K320); Q8_K is a separate legal
M512/H64/R256 fixture. These are component-chain timings, not model throughput.

| Source | MI50 single→pair median us | Ratio of medians | XTX single→pair median us | Median paired ratio |
|---|---:|---:|---:|---:|
| Q8_0 |1527.12→1374.72|1.110859|1037.52→1036.14|1.000148|
| Q8_1 |1367.96→1172.20|1.167002|997.405→972.844|1.025230|
| Q8_K |197.16→178.28|1.105901|593.482→586.822|1.009420|

The statistics remain explicitly distinct. XTX Q8_0 is flat, supporting
gfx906-only promotion. Source receipts: `q209-paired-summary.json` and
`q213-paired-summary.json` in this evidence directory.

After promotion, q225 compares actual production against the original reference
with31 balanced ABBA/BAAB pairs for each original Q8_0 shape:

| M, N10240/K320 | Production median, us | Reference median, us | Median paired speedup |
|---|---:|---:|---:|
|32|479.479|497.559|1.03821×|
|128|624.759|680.359|1.08903×|
|512|1170.36|1368.04|1.16893×|
|1024|2066.60|2468.68|1.19050×|
|2048|3989.72|4795.80|1.20221×|

The policy uses the existing cooperative geometry for every supported row count;
it does not select the win only at M512. q225 also records Q8_1 M512 at
1.11285× and the separate legal Q8_K geometry at1.10510×. These fresh paired
ratios, not cross-run absolute clocks, support the row/format result. Raw pairs
and source labels are in `q225-paired-summary.json` under the evidence root.

Both rejected Row64 trials are retained outside source. Row64/256 had raw
VGPR 116/110 on gfx906/gfx1100 and slowed MI50 complete-chain medians for all
three sources (single/candidate ratios 0.913/0.920/0.927). Row64/512 restored
four rows/thread but also lost all three MI50 comparisons
(ratios 0.983/0.983/0.948). Neither was installed. Evidence:
`../qfn-hc-up-row64-20261112/row64-256-measurements.json` and
`../qfn-hc-up-row64-20261112/row64-512/paired-summary.json`.

## Isolated profiles and limits

q214/q215 passed source/native selection and clean retirement. All 28
independent processes contain 39 chosen grouped dispatches (8 warmup/31
observed), zero opposite grouped, and 39 each of the other five complete-chain
nodes. Profiles cover Q8_0 only. MI50 is native ordinal1/PCI8c:00.0,
CSV Agent 3/GPU 5593; missing-counter warnings name unused XTX Agent 7.

MI50 trace-only postwarm grouped medians are 713.279→514.559us while unchanged
HC-down GEMM is 368.320→368.319us. Selected MI50 counters report
VALUBusy 34.10→48.23%, MemUnitBusy 47.82→62.24%, launched waves 10240 both,
and LDSBankConflict 0 both. Pair staging halves dynamic joins at K320
(20→10) while doubling LDS. Counter fractions support improved utilization;
they do not establish a unique cause, achieved occupancy or HBM-byte saving.

Profiler resources are separate from raw metadata, particularly gfx906
SGPR 48→96 versus compiled 40→42; no simple-rounding explanation is asserted.
XTX misses SALUBusy/MemUnitStalled/VALUUtilization/FlatVMemInsts; its reported
zero busy/cache/instruction values and saturated LDSBankConflict 100 remain
unvalidated and unusable for attribution. Profiler durations are diagnostic,
never canonical timings or summed wall-time estimates. The full-model result
comes from q229 above. Detailed counts/resources/counter validity and reducer:
`profile-reduction/REPORT.md`, `profile-reduction/q214.json`,
`profile-reduction/q215.json`, `reduce_profiles.py`.

## Driver observer repair and remaining limit

The observer now drains nonempty snapshots without a fixed one-second idle
wait, polls buffered close nonblocking, and still waits one second when idle.
Accepted closure still requires a fresh final read. Frozen primitive fields
serialize directly with identical JSONL bytes; timestamp equality only
prefilters candidate anchors before full exact ordered cursor comparison.
No records, cursor rules, classifications, host logs or driver settings were
removed or relaxed.

All 25 device-free tests pass; original-body controls fail the new polling
contract as expected. q197 Unit + BusyLogDrain + PublicationOrder CTests pass.
Fresh q196 retained 814,825 records/10 snapshots completely with zero findings.
However q188 and later q205 **remain failed continuity proofs**; q205 lost its
cursor after 609,552 records/9 snapshots although arithmetic passed. The repair
is a throughput improvement, not universal noisy-window closure. Each format
is subsequently scoped as its own registered cell with full startup/teardown
checks. q214/q215 driver windows passed with 97,171/96,107 records and zero
findings. Full receipts/limits:
`/tmp/qfn-driver-observer-throughput-20261112/REPORT.md`.

## Canonical registration and focused qualification

Explicit `ProductionTestPreflight` aliases:

- `V2_Integration_ROCmHCUpKPairCapturedReplayQ8_0`
- `V2_Integration_ROCmHCUpKPairCapturedReplayQ8_1`
- `V2_Integration_ROCmHCUpKPairCapturedReplayQ8_K`
- `V2_Integration_HTTPDriverBusyLogDrain`
- Existing `V2_Integration_HTTPDriverPublicationOrder`

The native owner is `v2_test_qwen4exp_device_primitives`; driver tests are
device-free Python. Performance owner
`v2_perf_q8_residual_prepared_projection` registers each source's
`V2_Perf_ROCmHCUpKPairM512PairedChain*`, exact separate
`...ProfileProduction*`/`...ProfileReference*`, and
`V2_Perf_ROCmHCUpKPairRowSpectrumQ8_0`. These performance aliases do not
replace the correctness gate. `V2_Unit_GPUDriverDiagnostics` remains registered.

Completed focused receipts under `/tmp/qfn-full-preflight-after-cpu-next/`:

- q216/q217: Integration/Release builds; q228: all four promoted body/resource
  identities. Frozen evidence is under
  `/tmp/qfn-hc-up-kpair-20261112/static-q217/`.
- q218: five selected CTests, including grouped/short-K geometry and the
  complete25-test driver fixture plus its explicit preflight regressions.
- q219–q224: each Q8_0/Q8_1/Q8_K cell on MI50 and XTX passes with complete clean
  driver windows. Cases cover M32/33/64/65/511/512/513,20 replay histories,
  both alias modes, K1/K3 tails where legal, unchanged banks/source identities,
  and prepublication rejection. The actual model shape is included.
- q225: seven separate shape/source event observations; clean driver interval.
- q227: existing CUDA grouped-projection CTest passes with a complete driver record.
- q229: matched full-model measurement above, complete native/driver retirement.

q226 and its unchanged q230 rerun pass the existing broader ROCm arithmetic
CTest, but fail the enclosing driver observer after88,931/228,956 records.
Their missing intervals remain unqualified. No host logging setting, timeout,
classifier or cursor check was weakened. This known burst-throughput limitation
requires separate observer work; no full preflight pass is asserted.

`AGENTS.md` also now requires tensor cores/equivalent matrix instructions for
GPU prefill matrix work on supporting architectures. Exceptions require matched
architecture/precision/geometry evidence, with the numerical contract preserved.
This MI50 change keeps native dot4 arithmetic; no gfx1100 matrix implementation
or unmeasured exception is claimed.
