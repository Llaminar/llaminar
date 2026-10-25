# CPU GatedDeltaNet epoch tuning results — 2026-10-25

The completed exact-develop comparison improves short-context median paired CPU
decode throughput by **1.86%, 5.49% and 5.88%** at fixed draft depths 2, 3 and 4.
Dynamic depth improves **38.53%**, with substantially less speculative work;
that result includes a changed adaptive workload and is not isolated kernel speedup.

Both comparison arms use a common losslessly adapted model fixture: one BF16
MTP sidecar router is represented as FP32 because unmodified develop does not
support that original router format. The production cores remain unchanged.
The comparison therefore measures the same compatible fixture on exact develop
and feature, rather than claiming original-artifact support on develop.

The fixed-depth kernel campaign passes byte equivalence and economy in all 672
measured cells. Depth 1 is **not final economy-qualified** by this campaign.
Short-context prefill throughput regresses approximately 5–7% by paired medians.
These are CPU-scoped results, not GPU or release certification.

## Scope and implementation

This report records the completed CPU experiment on 2026-10-25.
All raw evidence paths below are relative to `parity-results/gdn-epoch-tuning/`
and remain ignored local artifacts. No GPU implementation, GPU performance or
release-certification claim is included.

The production CPU recurrence retains the fixed P=8 epoch arithmetic contract.
Each head's canonical state contains phase, dense checkpoint and eight raw updates.
Grouped queries preserve serial-row floating-point order and accepted endpoints.

The implementation replaces full candidate-state snapshots with immutable epoch
checkpoints, one shared raw-update log and typed candidate prefix identities.
The first projection retains the initial checkpoint while reading its values;
there is no separate initial matrix-copy pass or mutable complete-state clone.
At an epoch boundary, the first ordered update reads the old checkpoint and writes
the next retained checkpoint; subsequent updates use that new checkpoint in place.
Query contraction shares reads across rows without reassociating reductions.
Mapped-head Q/K preparation is reused and redundant output clearing is removed.

Retained initial state is owned by the arena after the call, including when the
caller overwrites its initial input or unrelated graph/backend work executes.
A synchronous initial-readout lifecycle forbids early publication and leaves
failed callbacks unpublished. Copy/move of the local lifecycle authority is disabled.
Generation tokens, typed logical capacity and validated request geometry separate
live rows from admitted storage. Header and per-head payload use 64-byte alignment;
alignment padding cannot silently advertise additional candidate rows.

Accepted-state publication is a validated direct scatter from immutable retained
slots to canonical live state. Copy, inactive-history zero-fill and phase-word
operations run in the existing persistent parallel publisher.
All participants' plans and full source/destination spans are validated before
any live write; there is no serial materialization into an intermediate full bank.
Dense materialization remains an explicit observation/serialization boundary.
PhysicalMemoryAuthority and arena planning remain the allocation authority.

The change also repairs accepted verifier terminal-logits publication: the selected
row follows the accepted hidden-state endpoint and publishes through the canonical
host transfer authority. Reset/session epochs and producer reuse are tested.
This repair is necessary for continued production sampling after grouped acceptance.

## Correctness and production-path proof

The final prerequisite receipts cover 1,362 passing tests: 737 Unit and 625
ProductionTestPreflight entries (`prerequisites-r5/prerequisites.json`).
These gates include focused lifecycle, malformed-plan/alias rejection, all accepted
prefixes, every phase, tails, ragged request geometry and large→small→large replay.
The retained state and subsequent continuation are compared byte-for-byte.
They also cover publication ordering, selected terminal logits and memory accounting.

The original-model campaign completed all 13 selected model cells, including
the positive dynamic-acceptance witness (`final-qualification-runs-r2.json`).
These production model proofs are separate from the common-fixture develop timing
comparison and do not certify the original model on an unsupported develop router.

`supplemental-converted-greedy-proof-r1/summary.json` records four additional
128-token candidate proofs on the common converted fixture: MTP off and fixed
2/3/4. Every generated token matches the serial greedy reference.

`instrumented-develop-fp32-router-depth3-route-proof-r1/summary.json` is a separate
instrumented route proof, with exact prompt, output tokens and MTP work matching
its unprofiled candidate request. It proves fused initial retention and derived
checkpoint writes execute, with zero explicit checkpoint-copy passes/bytes and
zero mutable-bank-copy bytes. It is not a timing sample.
Instrumentation is kept out of canonical model timing.

## Exact develop baseline and common model fixture

The primary baseline is clean, unmodified develop
`ef1f54b0962d63531a9b33e06ca676e5124d05cc`, not a prior feature/P8 snapshot.
The feature is `c46a99f9e8c08962cd83cd93a3f6de86441fd770` plus the source diff
authenticated by each completed campaign's source inventory. Frozen Release
core SHA256 values are develop
`365486e9dd2fadc7311ec47d8bde24b2fc63047f8750ab190d656015c0429f10`
and feature `4de70d59dae07c608c90b5351ba9602866e4965c037efef8de2f801edec12d0f`.
`develop-baseline-ef1f54b09/runtime-identity.json` records both loader closures.
Both implementations use the same model file and matched request settings.
The original Qwen3.6-35B-A3B IQ3_S artifact has a BF16 MTP sidecar router,
`blk.40.ffn_gate_inp.weight`. Pure develop's CPU router supports FP32/FP16 there
and fails at `MTP0_moe_routing` before any draft step on this original artifact.
This is an existing format-support limitation, not evidence of GDN numerical failure.
The failed original-artifact baseline remains in `develop-comparison-moe-final-r1/`.

To compare the unchanged production cores, a separate common GGUF fixture changes
only that router from BF16 to FP32 using exact bit expansion:
`uint32_le = uint16_le << 16`. Every BF16 value is represented exactly.
All metadata, tensor order and all other tensor payload bytes are preserved;
tensor offsets naturally change. The original weights are untouched.
The converter independently rereads the output and proves inverse bit recovery,
all other tensor hashes, metadata preservation and complete source/output hashes.
See `compatible-model/router-f32-manifest.json` for the 753-tensor proof.

Common fixture: `compatible-model/Qwen3.6-35B-A3B-UD-IQ3_S-mtp-router-f32.gguf`.
Its SHA256 is `ef378a5bfeb30116d0c0df4b88acab1ef1e48c10b669a7aff3efd9840e0f1bdc`.
This is a lossless representation adaptation for baseline compatibility, not a
claim that pure develop supports the original BF16 router or that the model files
are byte-identical. Both comparison arms consume the same converted file.

## Completed short-context reference cohort

`develop-comparison-moe-fp32-router-r1/summary.json` records 24/24 valid runs,
three runs per implementation at each fixed depth and dynamic policy.
Each request has 595 prompt tokens and 384 generated tokens, temperature 0.8,
seed 42 and 28 physical workers. Receipt errors/background observations are empty.
Both builds are Release AVX512 on a Xeon Gold 6238R, pinned to CPUs 0–27;
the controller uses CPU 55 on the other socket. Context capacity is 4,096 and
MTP graph capacity is 15 draft tokens. Fresh processes run in ABBAAB order,
with one warmup and one measurement each. Every measured request performs its
full prefill; model loading, warmup and instrumentation are outside timing.
Paired ratios use adjacent baseline/candidate runs, not a ratio of separate medians.

| Policy | Develop median decode TPS | Feature median decode TPS | Paired median C/B | Paired range |
|---|---:|---:|---:|---:|
| Fixed 2 | 20.104 | 20.464 | 1.01856 | 1.01630–1.06226 |
| Fixed 3 | 19.647 | 20.726 | 1.05492 | 1.00977–1.06918 |
| Fixed 4 | 16.371 | 17.557 | 1.05877 | 1.05603–1.07247 |
| Dynamic | 14.303 | 20.752 | 1.38527 | 1.36725–1.45435 |

| Policy | Accepted tokens Δ | Draft steps Δ | Verifier runs Δ | Verifier token count Δ |
|---|---:|---:|---:|---:|
| Fixed 2 | +1 | −2 | −1 | −3 |
| Fixed 3 | 0 | 0 | 0 | 0 |
| Fixed 4 | −3 | +11 | +2 | +13 |
| Dynamic | −12 | −333 | +12 | −321 |

Deltas are feature minus develop. Each implementation repeats its exact tokens
and MTP work across all three runs. Implementations first differ at token index 83.
These are matched-request version deltas, not identical-output kernel attribution.
The dynamic gain includes substantially reduced adaptive draft/verifier work.
Secondary paired prefill throughput ratios are 0.94950, 0.93023, 0.94848 and
0.93460 respectively: prefill regresses in this short cohort.
No decode gain should conceal that secondary observation.

## Kernel economy and isolated counters

`final-depth-kernel-analysis-r1.json` contains 672 byte-exact, economical cells.
The physical-worker sweep covers 1/2/4/7/14/28, draft depths 2/3/4 (M=3/4/5),
all eight epoch phases, representative value-head geometries and retained capacity.
ISA regimes include AVX512 Release, true AVX2 Integration at -O3, and the mixed
AVX512-build/forced-AVX2 dispatch regime. True AVX2 is not inferred from an env flag.
The ordinary serial comparator does not pay speculative retention costs.
The grouped timing includes production checkpoint/raw-log retention. Accepted
endpoint materialization is checked separately; full-model timing includes the
production publisher and continuation. This CPU kernel test is not GPU capture proof.

`final-counter-analysis-r2.json` authenticates 48/48 isolated profiles with all
requested events scheduled 100%. Cycles/instructions/IPC and LLC observations
use separate launches from canonical unprofiled timing.
Summed 28-worker cycles are not elapsed latency; profile spans retain bounded
FIFO control work. Phase 0 and phase 7 are separate shapes, not statistical replicates.
AVX2's actual core is Integration -O3 with symbols; AVX512 is Release.
LLC evidence exists only for AVX512 and does not justify a general LLC-reduction
claim. Small miss counts depend on residual cache state. These counters support
local execution analysis and cannot explain the whole develop-to-feature model delta.

| ISA | Workers | Median grouped/serial counted cycles | Grouped IPC | Serial IPC |
|---|---:|---:|---:|---:|
| AVX512 | 1 | 0.743 | 1.279 | 0.755 |
| AVX512 | 28 | 0.830 | 0.162 | 0.141 |
| AVX2 | 1 | 0.772 | 1.958 | 1.085 |
| AVX2 | 28 | 0.864 | 0.208 | 0.173 |

These medians cover M=3/5 and phases 0/7. All 16 paired counted-cycle ratios
favor grouped execution (0.609–0.923); these ratios are not elapsed speedups.

## Memory interpretation against pure develop

For 16 retained rows, 32 value heads and dk=dv=128, dense develop candidate slots
require 33,554,432 bytes per layer. Aligned compact feature slots require
7,053,312 bytes per layer: a 78.9795% reduction in this retained-slot component.
Canonical CPU state grows from 2,097,152 to 2,361,472 bytes per layer (+264,320).
Across 30 main GDN layers that canonical-state growth is 7,929,600 bytes.

Pure develop had no separate CPU mutable work bank to remove; old shared-work
naming/accounting applies to GPU workspace. Do not claim CPU work-bank savings.
Role/layer stable names merge capacities across mutually exclusive graph shapes;
concurrent layer state remains exclusive. Physical arena peaks also depend on
lifetime overlays and floors, so per-layer logical savings are not process RSS.
For exactly one 16-row slot family plus one canonical bank across 30 layers,
the logical net reduction is 787,104,000 bytes. This excludes weights, KV cache,
short convolution, prefix archives, additional request banks and allocator effects.

## Canceled long-context experiment

The user explicitly canceled the 16k/64k campaign because it took too long.
There is no completed paired long-context cohort, no qualified long-context
speedup and no 64k run. Partial decode TPS is not reported as a result.
Native 16k baseline cache warmups worked, but do not establish a version comparison.

R1/R2 port-harness failures and the R3 explicit cancellation remain preserved in
the ignored long-context artifacts. Cleanup stopped only the owned controller
and server, required no forced kill and left no owned workload process running.
`perf_event_paranoid` was restored to 4. No incomplete timing row is reused in
the completed short-context or kernel results above.
