# gfx1100 shared-expert verifier transaction

The complete captured router → shared FFN → gate/add operation now has a
serial-byte oracle and an all-format forced-candidate timing panel. Automatic
selection is not installed yet. This is component evidence; the routed residual
is explicitly injected and does not represent routed-expert execution.

## Operation and replay contract

`ROCmSharedVerifierTransaction.h`, included by the existing MoE verifier
performance fixture, uses the actual `SharedExpertFFNStage` and
`SharedExpertGateStage`. The router produces Q8 activations through
`routeVerifierRowsDecodeEquivalent`; the shared FFN borrows that publication.
Each Dot4/N32/N64 candidate retains separate stage objects, workspace, output
and graph owners. All use one persistent device count and the original worker
stream. No count readback chooses execution.

The geometry is H2048/I512 with a 256-expert/top8 router and one shared expert.
The 24 cases comprise all 21 canonical source formats at physical 16, plus
IQ2_S gate/up with IQ4_XS down at physical 8/9/16. Every live prefix 0–capacity
is checked: 393 source/capacity/live keys. Independent original M1 table decode
and serial gate/add define raw FFN, gated FFN and combined-output bytes.
Inactive physical suffixes must be positive zero.

Retained replay covers full→8→9→empty→full, repeated full work, all positive
prefixes and reset without rewriting an unpublished count. Consumed scratch,
router Q8 and outputs are poisoned. The private input-Q8 bank remains untouched.
Actual parent-node omission separately proves the router producer, final gate,
and active native writers. Physical 8 has no native topology; physical 9/16 can
retain native nodes while live ≤8 executes RouteOwned. Restoring each omitted
family must recover serial bytes immediately.

`V2_Integration_ROCmSharedVerifierTransactionReplay` is explicitly registered
in production preflight. Only focused tests run for this campaign. The canonical
preflight append now follows the list's initialization; existing attention tests
already retained their explicit preflight labels before that bookkeeping repair.

## Release candidate timing

The final collector records 31 alternating candidate-order triples per key,
eight complete operations per retained timeline interval: 12,183 triples total.
Every sampled launch independently overwrites poison and passes the byte oracle.
Preparation, poisoning and observation stay outside the internal event interval.

| Live work | Cells | N32 median wins over Dot4 | Dot4/N32 range | Geometric mean |
|---|---:|---:|---:|---:|
| 9–16 rows |177|177|1.21873–1.87860×|1.43535×|
| 0–8 rows |216|173|0.98650–1.09125×|1.03269×|

All 177 native-active cells win all 31 paired comparisons and prefer N32 over
N64. Sparse medians remain mixed; the worst N32 loss is approximately 1.37%.
Physical 8 is a RouteOwned control, not a WMMA speedup. These forced results
do not certify actual Automatic dispatch.

The profile mode records one complete operation without internal timing events.
Poison and its completion precede the selected profiler region; output download
and serial-byte verification follow the region. Each counter launch selects one
exact GU/down producer and binds its loaded code object and launch geometry to
the unprofiled timing record. All 18 diagnostic passes succeed: eight metrics
for each mixed-format physical 16 producer, plus active GU/down controls at
physical 9. Both producers use 128 threads and 3,520 LDS bytes, with zero private
memory, VGPR spills and LDS bank conflicts.

| Physical 16 producer | Logical VGPRs | Occupancy | Memory-unit busy | L2 hit | FETCH_SIZE (KB) |
|---|---:|---:|---:|---:|---:|
| IQ2_S gate/up |55|1.584%|34.307%|51.434%|875.500|
| IQ4_XS down |52|2.630%|27.476%|34.592%|653.875|

The small single-expert grids have low measured occupancy despite the complete
operation's relative speedup. These two source producers do not establish
per-source profile coverage for all 21 cases. Actual Automatic installation and
its all-format qualification remain open.

Evidence root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Timing: `mtp-shared-verifier-timing-1794771952010301836/receipt.json`.
- Replay: `mtp-shared-verifier-replay-1794772153703450281/receipt.json` (24 cases, 1,179 triples).
- Profiles: `mtp-shared-verifier-profile-batch-1794772242471031552/progress.json`.
- Analysis: `shared-verifier-forced-summary.json`.
- Registration discovery: `shared-attention-preflight-discovery.json`.
- Build logs: `mtp-shared-transaction-profile-proof-{release,integration}-build.log`.

The frozen `mtp-shared-verifier-forced-snapshot/snapshot.json` contains20 evidence
records and211 files, including the exact binaries, source, observations, loaded
code objects and counter data. Manifest SHA256:
`8122877b7072029b4866f49bdba0d9478b2a4ac6f0b5de077c38cb5118a84507`.

## Remaining installation work

Admit a new explicit shared-verifier operation identity rather than the existing
`Unmeasured` grouping provenance. Only complete, full-width TokenRows work with
OriginalTokens and authenticated PublishedQ8 belongs to this measured operation.
Generic shared grouping, routed/selected routes, internal quantization, split
phases and down slices remain outside this evidence.

Retain exact original gate/up/down source identities, including the superblock
bit. Current arithmetic-policy masks combine source aliases and cannot prove
that a source tuple was measured. Initially qualify same-source tuples only at
physical 16, and the measured IQ2_S/IQ4_XS tuple at physical 9/16. Existing device
route admission owns the live ≤8 boundary; no host scheduling state is needed.

After installation, rerun actual Automatic replay and timing, profile the selected
source producers, and retain explicit neighboring-key and source-alias exclusions.
Small-work costs remain part of acceptance, rather than being hidden by full-row
speedups.
