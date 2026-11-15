# gfx1100 shared-expert verifier transaction

Automatic now selects N32 for the qualified complete captured router → shared
FFN → gate/add operation. All 28 replay cases pass, covering 453 original-source,
physical-capacity and live-prefix keys. Every one of the 177 native-active timing
cells wins all 31 paired comparisons, by 1.23091–1.88403× (geometric mean 1.43667×).
All six affected CUDA/ROCm feature registrations and four focused host
registrations pass. Per-source Automatic profiling is in progress. This is
component evidence: the explicitly injected residual does not represent routed
expert execution or a complete model speedup.

## Operation and replay contract

`ROCmSharedVerifierTransaction.h` uses the actual `SharedExpertFFNStage` and
`SharedExpertGateStage`. The router produces Q8 activations through
`routeVerifierRowsDecodeEquivalent`; the shared FFN borrows that publication.
Dot4, N32, N64 and Automatic retain separate stage objects, workspace, output,
publication and graph owners. All use one persistent device count and the
original worker stream. Replay never reads that count on the host to select work.

The geometry is H2048/I512 with a 256-expert/top8 router and one shared expert.
The roster contains all 21 canonical original sources at physical 16, mixed
IQ2_S gate/up with IQ4_XS down at physical 8/9/16, and four neighboring exclusions:
Q4_0 physical 9, IQ4_XS physical 15, IQ2_S/IQ4_NL physical 16 and Q8_0/Q8_1
physical 16. Every live prefix 0–capacity is checked. Independent original M1
table decode and serial gate/add define raw FFN, gated FFN and combined bytes;
inactive physical suffixes must be positive zero.

Retained replay covers full→8→9→empty→full, repeated full work, all positive
prefixes and reset without consuming an unpublished count. Consumed scratch,
router Q8 and outputs are poisoned. Private input-Q8 storage remains untouched.
Actual parent-node omission proves router, final gate/add and active native
writers separately. Native topology remains present while live ≤8 selects the
existing RouteOwned producer. Restoring omitted nodes must recover serial bytes.

The submission counters independently authenticate Dot4/N32/N64 selection,
router-Q8 reuse and physical capacity during warmup/capture. Their values must
remain unchanged through retained replay, reset, timing and profiling: they count
host submissions, not device executions.

## Exact Automatic admission

The typed `DecodeEquivalentVerifier` shared-grouping purpose establishes
`SharedVerifierCountedOriginalRows` provenance only for original device-counted
rows with matching physical/count capacity, zero origin and one repetition.
Preparation revokes old provenance immediately; failed or abandoned preparation
cannot publish it. The retained `DeviceRowRange::CaptureIdentity` binds the
subsequent projection's physical rows and top1 geometry. Tests reject laundering
prepared 32-row or 16-row grouping into another admitted capacity.

Only complete, full-width TokenRows work with OriginalTokens and authenticated
PublishedQ8 qualifies as `CompleteFullWidthSharedVerifierTokenRows`. Ordinary
shared preparation, fixed/sliced/repeated row authorities, split phases,
down slices, runtime descriptor tables and other publication layouts retain
their existing policy.

The selector consumes exact original gate/up/down identities from the existing
immutable singleton descriptor tables. Original codebook and superblock bits
remain distinct from physical decoder and normalized arithmetic masks. Both
masks must agree with the source identity; promoted representations are excluded.
The admitted tuples are all 21 same-source triplets at physical 16 and the
IQ2_S/IQ4_XS tuple at physical 9 or 16. Actual gfx1100 resources must also admit
N32. The existing device route policy selects RouteOwned for live ≤8 and native
ExpertTiled above it; no new host count, live decision word or classifier kernel
is introduced. CUDA consumes the same checked provenance without a new native
selection rule.

The policy replay exhausts 21³ source tuples across eight capacities, with
independent exclusions for operation, geometry, ownership, preparation, masks,
missing identity and source aliases. `V2_Integration_ROCmSharedVerifierPolicyReplay`,
`V2_Integration_MoESharedVerifierGroupingOriginReplay` and the GPU transaction
replay have explicit ProductionTestPreflight registrations. This campaign runs
those focused gates, not the full preflight suite.

## Actual Automatic Release timing

The collector records 31 alternating four-path samples per key, eight complete
operations per retained event interval: 14,043 quadruples across 453 keys.
Each sample overwrites poison and passes the serial-byte oracle. Preparation,
poisoning and observation stay outside the internal event interval.

| Automatic domain | Cells | Median wins over Dot4 | Dot4/Automatic range | Geometric mean |
|---|---:|---:|---:|---:|
| Admitted capacity, live 9–16 |177|177|1.23091–1.88403×|1.43667×|
| Admitted capacity, live 0–8 |207|191|0.98858–1.09266×|1.03576×|
| Excluded capacity/source controls |69|21|0.97685–1.00987×|0.99506×|

All native-active cells win all 31 pairs. Sparse admitted cells include the cost
of retaining dormant native topology; the worst median latency increase is
approximately 1.16%. Excluded controls authenticate actual Dot4 topology, so their
separate-allocation timing spread is not a WMMA win or loss. No full-row result
is used to hide sparse costs.

The earlier forced panel preferred N32 over N64 in all 177 native-active cells.
Its immutable archive remains `mtp-shared-verifier-forced-snapshot/snapshot.json`,
SHA256 `8122877b7072029b4866f49bdba0d9478b2a4ac6f0b5de077c38cb5118a84507`.

## Profiling and evidence

Profile mode executes one complete operation without internal timing events.
Poison and completion precede the selected profiler region; output observation
and serial-byte verification follow it. Each counter launch selects one exact
Automatic GU/down producer and binds its loaded code object, source, physical
capacity, native device and resources to the unprofiled timing record. The first
mixed M16 Automatic GU pass reports 55 logical VGPRs, 56 allocated VGPRs,
128 threads, 3,520 LDS bytes, zero private storage/spills and 100% GPUBusy.
The all-source sweep covers 23 admitted cases, two roles and eight metrics;
its progress record is authoritative until all 368 passes finish.

Prior forced mixed-source diagnostics measured low occupancy (GU 1.584%, down
2.630%) despite the complete operation's speedup. Those 18 diagnostics do not
substitute for the current Automatic per-source sweep.

Evidence root: `/tmp/llaminar-gfx1100-wmma-prefill-evidence`.

- Actual timing: `mtp-shared-verifier-automatic-timing-1794774686055257080/receipt.json`.
- Current Integration replay: `mtp-shared-verifier-automatic-replay-1794774517408655746/receipt.json` (28 cases, 1,359 quadruples).
- Analysis: `shared-verifier-automatic-summary.json`.
- Four host registrations: `shared-verifier-auto-host-receipt-r2.json`.
- Six backend feature registrations: `mtp-shared-verifier-automatic-feature-1794775018952372789/receipt.json`.
- Automatic profiles: `mtp-shared-verifier-automatic-profile-batch-1794775093977237140/progress.json`.
- Release build: `shared-verifier-auto-release-build-r2.log`.
- Integration builds: `shared-verifier-auto-integration-build.log` and `shared-verifier-auto-feature-integration-build.log`.

Release and Integration compile the full shipped ROCm gfx906/gfx1100 target set
and CUDA sm86 with the spill guards enabled. Native runtime performance evidence
belongs to gfx1100. Remaining checkpoint work is to finish the Automatic profiles,
freeze their exact evidence and publish the feature changes. Routed MTP and the
broader grouped-stage bundles remain separate open work.
