# gfx1100 selected LM-head component

The actual device row selector followed by the actual quantized LM-head stage
has a measured Dot4/WMMA crossover at M5. The complete forced M2–16 panel
qualifies seeded Q6_K at the original full-vocabulary geometries: N248320 with
K5120 (dense) or K2048 (MoE). Dot4 wins all31 pairs at M2–4; WMMA wins all31
at every M5–16. These are seeded weights, not original model-weight bytes.
Narrow Automatic replay, two timing panels and16 native profiles now pass.
TP ownership, other formats
and controller integration remain outside this component's measured domain.

## Operation and independent oracle

`Test__ROCmSelectedLMHeadTransaction.cpp` executes real
`HiddenStateRowsSelectStage` with `WorkspaceBoundDeviceIndices`, the producer-
owned `VERIFIER_LOGIT_ROWS` workspace name and `ClampToFirst`. The fixture supplies
the index input; the MTP metadata producer is outside this boundary. The actual
`LMHeadStage` consumes the selected tensor through the capture ledger's internal
producer edge, computes all positions and uses decode-equivalent arithmetic.
No device count or exact-hosted extent is manufactured. Every selected row is
materialized; negative indices select row0 and are not empty work.

Two forced candidates, actual Automatic and an independent M1 owner retain separate prepared
payloads, activations, arenas and outputs. The original Q6_K tensor is generated
once per geometry in243 deterministic1024-row chunks (last512 rows), using
physical CPU workers. The tensor adopts its aligned byte storage without a
second full copy. The production loader prepares four distinct handles in one
batch, retained across the measured row widths. This is seeded component data.

M1 receives rows copied directly from the original input, independently of the
candidate selector. Its logits form the byte oracle. Replay changes source
activations, permutes and duplicates indices, exercises both clamp boundaries,
resets stages, poisons compact/output/temporary banks and repeats the original
capture. The authoritative index bank has `SerialGraphFamily` lifetime and is
excluded from scratch poisoning. Independent omission of selector, quantizer
and matrix producers must break output; the other candidate must remain correct,
and restoring the original producer must restore every byte.

Captured arguments authenticate the source/compact/index addresses, row-copy
policy, uncounted Q8 quantizer, genuine `fullyActive(M)` range, original prepared
source8/superblock/physical8 identity, partial/output banks and ascending reducer.
The native grid is3880×16×1 with128 threads: P16, K32 span10 or4. The original
single-device serial partition-width policy is zero. Captured inventory reports
zero private storage: selector14 registers, quantizer19, ordered matrix34,
native matrix80 with7040 LDS bytes, and reducer8.

## Bounded Release result

Each sample times eight complete selector-plus-head operations in a retained
internal-event interval. Setup, input publication, poison and observation stay
outside that interval.31 alternating Dot4/WMMA pairs per cell run under ordinary
AUTO power; every sample still checks the serial-byte oracle.

| Geometry | M | Dot4 µs | WMMA µs | Dot4/WMMA | WMMA pair wins |
|---|---:|---:|---:|---:|---:|
| Dense |2|1442.615|1726.142|0.83575×|0/31|
| Dense |5|2928.960|1879.208|1.55861×|31/31|
| Dense |9|4486.021|2059.999|2.17768×|31/31|
| Dense |16|6242.329|2259.591|2.76259×|31/31|
| MoE |2|661.580|790.116|0.83732×|0/31|
| MoE |5|1372.305|931.132|1.47380×|31/31|
| MoE |9|2111.160|1035.267|2.03924×|31/31|
| MoE |16|3008.937|1305.859|2.30418×|31/31|

Both focused Integration registrations pass and are explicitly registered in
production preflight: `V2_Integration_ROCmSelectedLMHeadDenseReplay` and
`V2_Integration_ROCmSelectedLMHeadMoEReplay`. This was a focused run, not the full
preflight suite. Build configuration retains the shipping gfx906/gfx1100 and
sm86 targets and the existing spill guard. Native replay here uses gfx1100.

Evidence is under `/tmp/llaminar-gfx1100-wmma-prefill-evidence`:

- Timing: `selected-lm-head-timing-1794808398069226890/receipt.json` (8 cells/248 pairs).
- Analysis: `selected-head-bounded-timing-summary.json`.
- Integration: `selected-head-bounded-integration.{log,xml}` (2 registrations).
- Release snapshot: `selected-head-bounded-release-snapshot-1794808684348345127/snapshot.json`,
  SHA256 `307ad0c65b880511cbaf39aba4f7e76ecd85188937da61dc18ef36456ff06e66`.

## Complete forced M2–16 panel

The complete panel contains30 geometry/width cells and930 pairs. Across M5–16,
Dot4/WMMA geometric-mean speedup is2.10366× for dense (minimum1.53058×) and
1.90819× for MoE (minimum1.47448×). Both focused Integration registrations pass
again (15.37s). The complete forced baseline is preserved before any Automatic
source edits:

- Analysis: `selected-head-full-width-timing-summary.json`.
- Integration: `selected-head-full-width-integration.{log,xml}`.
- Source/binary/evidence snapshot:
  `selected-head-full-width-forced-snapshot-1794809356013805645/snapshot.json`,
  SHA256 `65abff9b47172656c37fa7445546d2c9ada4da8ae02c367765bed5d80d88bdd8`.

## Qualified narrow Automatic selection

The measured rule selects Q6_K native WMMA only at M5–16 for these two heads. The actual LM-head stage
binds a typed materialized-operation identity using the existing graph input
ownership; a new selector-issued lifetime token is unnecessary. Keep this
identity separate from ordinary scalar projections and counted identity-layout
heads. Materialized selection explicitly binds full rows, shadowing and restoring any
enclosing source prefix. The native fixture records under a differently sized
M19 prefix whose device count is zero. Host replays cover counted/sliced/repeated
row rejection, request-scope rejection, restoration, all21 formats and widths
M1–65. All focused binding/policy replays pass on the updated build.

Initial admission must retain exact source identity, original input/output and
prepared ownership, unbiased FP32 output, N248320, K5120/2048, canonical serial
partition policy, native gfx1100 and only measured M values. Local-vocabulary
shards, differing mirrored serial widths, mapped outputs and unmeasured formats
remain outside this measured domain. Actual-Automatic capture, byte replay,
timing and separate profiles qualify this bounded installation.


The final candidate retains four independent prepared owners: Dot4, Automatic,
forced WMMA and M1. Automatic's capture is recorded inside a deliberately empty
M19 source-prefix scope; the head must materialize and process its own full rows,
then restore the enclosing scope. This also proves that an absent optional row
range does not accidentally inherit the original source count. The singleton
native packet is exposed only for this head kind (and the existing GDN path),
so QKV/KV/GateUp scalar-subgroup dispatch remains unchanged.

Each timing panel contains30 cells and930 balanced triples (31 per cell), with
eight complete selector/head operations per internal event. All24 WMMA-selected
cells win every pair against Dot4 in both panels. The six retained-Dot4 controls
preserve their original captured kernel inventories.

| Actual Automatic panel | Selected geometric gain | Selected minimum | Selected maximum | Dot4 / Automatic at M2–4 |
|---|---:|---:|---:|---:|
| Initial |2.001636×|1.478587×|2.784646×|0.999280–1.001796×|
| Confirmation |1.997859×|1.475335×|2.787031×|0.998671–1.000574×|

The final Release core SHA256 is
`e255c49ab937462171da4014d2c1651bbb0c5230fbaa95d411af1f56b8462b56`.
Both panels and both one-operation profile replays authenticate the same source,
binary and core. Sixteen separate counter launches profile native M16, one metric
per original geometry per launch, with selected-region tracing and timing events
disabled. They join the exact captured dispatch to the loaded code object,
descriptor and ISA. Both use wave64,80 logical/allocated VGPRs,7040 logical LDS
bytes (7168 allocated), zero private storage and zero VGPR/SGPR memory spills.
Loaded native code SHA256 is
`3cb215279203a6e5781dac965660e7fca1e5824f1ffaa926201cb4a4d84c896a`.

| Native M16 profile | Dense K5120 | MoE K2048 |
|---|---:|---:|
| Occupancy % |48.9616|44.3487|
| Memory unit busy % |73.7773|76.0879|
| L2 hit % |15.9947|26.7128|
| LDS bank conflict metric |2.312139|2.312139|
| Wavefronts |124160|124160|

Profiles use STABLE_STD with verified restoration to AUTO. Only the separate
unprofiled AUTO-power panels supply latency/speedup. These counters describe
M16 at the two declared geometries; they are not whole-model throughput.

Focused validation passes seven CTest registrations: four host binding/policy
registrations (34 GTest executions), both native head registrations, and the
existing device-free prefill/stage boundary suite. Release and Integration
builds retain gfx906/gfx1100 and sm86 with mandatory spill checks. Two Sol6.1
read-only reviews covered admission/packet topology and independent evidence.
No full preflight campaign was run.

Final local evidence:

- `selected-lm-head-timing-1794810759752090063/receipt.json`.
- `selected-lm-head-timing-1794810899877635465/receipt.json`.
- `selected-head-automatic-timing-summary.json`.
- `selected-head-automatic-profile-index.json` and `selected-head-automatic-profile-summary.json`.
- `selected-head-automatic-host.{log,xml}`, `selected-head-automatic-integration.{log,xml}`,
  and `selected-head-stage-boundary.{log,xml}`.

This closes the measured Q6_K selected-head component. It does not certify other
head codebooks, TP/source-slice ownership, MTP controller index production,
original model-weight bytes, or complete-model MTP economy.

Qualified snapshot:
`selected-head-automatic-qualified-snapshot-1794811716955552070/snapshot.json`,
SHA256 `5881d55092d41afa2fe9eddec4f93a772c473ab44d222d8c39f9d12669d6d1de`.
It preserves60 timing cells/1860 triples,16 profiles, seven focused registrations,
exact Release/Integration producers and source/build evidence before subsequent work.
