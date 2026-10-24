# QFN fixed scalar column publication 180

## Result and scope

Checkpoint 180 measures **819.594 prefill / 41.030 decode tok/s**, against
frozen 179 controls at **819.241 / 40.797**. The extended A/B/A/B/A comparison
retains all 15 measured requests: **+0.572% decode**, with effectively unchanged
prefill (+0.043%). The first bracket has overlapping decode ranges; the second
favors the candidate by 1.011% with non-overlapping ranges. This is a small,
workload-specific gain, not a claim that every request improves by that amount.
Castagna's **42.959 decode tok/s** target remains open.

All 185 canonical final-logit rows across 26 files are byte-identical to both
179 and 151. All 15 natural 256-token sequences also match. The complete
**823-entry Unit/affected gate** passes, as do the seven focused native tests.
This is scoped qualification; complete ProductionTestPreflight remains the
historical G60 result. The explicit FP16 attention experiment is unchanged and
is not promoted to a new default by this change.

## Removed work and ownership

The fixed M1 projection-distributed MoE path previously gathered columns into
a temporary rank-major bank, then copied them to the final row-major bank.
For one row, these addresses are identical:

```text
source      = (rank * tokens + row) * local_columns + local_column
destination = row * participants * local_columns + rank * local_columns + local_column
tokens = 1, row = 0  =>  both = rank * local_columns + local_column
```

`appendMoEProjectionPipeline` now selects direct publication for
`rows == 1 && !live_rows`, independently of Prefill/Decode labels. Its existing
native allgather writes the original final tensor and BufferId, retaining the
original `projection_assemble` terminal identity. The local
`ProjectionColumnGatherStage` specializes only diagnostic metadata; native
transport, validation, exact streams, publication, coherence and capture
identity remain inherited from `NativeAllGatherStage`. The retained original
binding owns prepared descriptor and route-evidence leases through retirement.
The actual stage type is `NATIVE_ALLGATHER`, while snapshots keep their original
complete-value output keys.

Shared-column finalization redirects that same native receive to its declared
combined-output bank. Existing shared packing, reduce-scatter, gate arithmetic
and event/collective order remain intact. Its legitimate packing transpose is
retained. No new GPU kernel, precision policy, transport or allocation is added;
existing arena capacity still supports wider rows.

Device-counted M1 retains its original banks. An empty native receive leaves
inactive bytes untouched, while the original assembler copies its receive bank
to the output. Redirecting an empty receive would change those inactive bytes.
There is no host count readback. Fixed M1 with invalid routes or empty expert
owners still gathers the original explicit zero column contribution. Wider
rows retain their original gather and assembly.

For the measured four-participant model, each final gather still sends exactly
640 FP32 columns (2,560 bytes) per participant and receives 10,240 bytes. Those
extents are checked by the native fixture and unchanged collective implementation;
they are not inferred from allocation capacity or profiler grid dimensions.

## Evidence for the target

The retained 175 trace records 48 final identity copies per decode step per GPU.
Median same-GPU exclusive service is 238.051 microseconds/step, with individual
GPU observations of 235.215–239.878. This is instrumented service, not removable
unprofiled latency; no parallel GPU times are added. The later 177 and 179
changes do not modify this copy. The source-bound refinement is retained in
`/tmp/qfn180-next-target/retained175-opportunity.json`, using the original
`indexer175-model-join-r1` and `indexer175-model-intervals-r2` artifacts.

A separate short production run uses the existing native graph inventory,
without a profiler: 512 prompt tokens, four emitted tokens, one warmup and one
measured request. All four GPUs have resolved native names and **zero final-copy
observations**. There are two materialized executables per GPU, each with 3,078
nodes in the family records. Kernel inventory counters aggregate 6,156 kernel
and 96 grouping observations across those two publications. The grouping
counter is not claimed as a separately measured 48 nodes per executable, and
these setup observations are not execution counts.

The first diagnostic wrapper incorrectly compared file access time before and
after reading its inventory. Its failed receipt remains unchanged. The separate
`direct180-model-path-audit-r1` reconciliation passes stable metadata checks,
actual model/config/token validation, runtime/source/loader/driver checks and
clean raw process exit. It explicitly retains the original wrapper's failed
derived exit verdict. Six offline regressions cover stable-read identity,
changed-file rejection and the actual counter format. No model rerun was needed
to repair this postprocessing defect.

## Native regressions and builds

The explicit preflight registrations are
`V2_Integration_MoEProjectionScalarPublication_{CUDA2,ROCm2,ROCm4}`. They exercise
both semantic phases, native/counted intermediate transport, wider-row to M1
reuse, 20 full/empty/partial/changed-owner replays, request reset and hard
retirement. The affected all-format fixture also verifies final tensor/BufferId
ownership, exact byte extent, original snapshot keys and `CompleteValue`.
An actual captured original assembly supplies its function identity; the
production capture contains zero final copies for M1 and one for wider rows.
Shared-column completion retains its separate packing call.

A construction-only check preserves counted-row declarations. Real native
counted receive behavior is covered separately by
`V2_Integration_NativeAllGatherArenaCountM1_{CUDA,ROCm}` and
`V2_Integration_NativeAllGatherArenaCountOverlapM1_{CUDA,ROCm}`, including
retained 1-to-0 transitions, poisoned tails and useful communication extents.

The initial starter exposed stale fixture assumptions about the removed node's
count and name. Correcting those assertions changed no production code. The
original failure is retained; the repaired starter passes **7/7 in 35.50 s**.
The final gate passes **778 Unit + 41 projection + 4 native M1 registrations**
in **530.02 s**. No CTest entry or required integration case is skipped; the
95 existing, source-authenticated internal Unit limitations remain explicit.

Integration and Release builds bind the same final source. CUDA targets remain
80/86/89/90 and HIP remains gfx906, with normal spill guards. The Release rebuild
after the fixture-only correction is a no-work build. No new GPU kernel resource
claim is required for deletion of an existing launch.

## Canonical numerical result

These are equal-prompt means of final-logit KL against the same-GGUF independent
reference, with frozen teacher histories. Required groups remain separate.

| Group | Phase | Mean KL, nats | Top-1 agreement | Top-5 overlap |
|---|---|---:|---:|---:|
| Main, 16 prompts | Prefill | 0.025614 | 93.75% | 92.50% |
| Main, 16 prompts | Decode | 0.028276 | 95.83% | 92.29% |
| Held-out, 8 prompts | Prefill | 0.014911 | 100% | 95.00% |
| Held-out, 8 prompts | Decode | 0.004553 | 100% | 91.67% |

The threshold group remains diagnostic: mean KL is 0.000168 prefill and
0.000210 decode. Reset and threshold logits are included in the 185-row byte
comparison. This selected-topology panel is not full-corpus or image certification.

## Matched Release workload and samples

Both binaries use the same Qwen3.8-Flash-Next UD-Q4_K_XL GGUF shards, four MI50s
in `TensorParallel + GateUpOwnedDownColumns`, sharded vocabulary, RAM ngrams,
28 physical NUMA1 workers (CPUs 28–55), context capacity 4096, a 512-token
prompt aligned to the 512-row bucket, and 256 emitted tokens/255 decode forwards.
KV is native post-RoPE FP16. Sampling is greedy with seed 42; MTP and prefix
caching are off. Each arm has
one warmup and three measured requests. No profiler runs during timing.

```text
LLAMINAR_QFN_FA_OUTPUT_ACTIVATION=single_q8
LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION=single_q8
LLAMINAR_GDN_PROJECTION_ARITHMETIC=fp16_moe_alpha_beta
LLAMINAR_QFN_GDN_QKV_ACTIVATION=single_q8
LLAMINAR_QFN_HC_UP_ACTIVATION=two_term_q8
LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16
LLAMINAR_QFN_SPARSE_ATTENTION=fp16_tile
LLAMINAR_QFN_SPARSE_ATTENTION_PHASE_SCHEDULE=prefill8_decode33
```

| Arm | Binary | Prefill median tok/s | Decode median tok/s | Decode sample range |
|---|---|---:|---:|---:|
| A1 | Frozen 179 | 819.198 | 40.709 | 40.677–40.808 |
| B1 | Candidate 180 | 819.353 | 40.934 | 40.824–41.013 |
| A2 | Frozen 179 | 820.297 | 40.801 | 40.797–40.926 |
| B2 | Candidate 180 | 819.831 | 41.212 | 41.047–41.268 |
| A3 | Frozen 179 | 819.150 | 40.732 | 40.721–40.810 |

The first A1/B1/A2 summary is preserved: +0.330% decode with overlapping ranges.
The additional A2/B2/A3 bracket measures +1.011% with non-overlapping decode
ranges. The combined score uses all nine control and six candidate samples,
without counting shared A2 twice. All runtime/source/driver/resource checks and
normal shutdown checks pass. The reported +0.572% is the combined median ratio;
the full combined ranges still overlap and remain visible in the artifacts.

## Reproducibility

Source at qualification is HEAD `1309a0fe50fab3b329e39f98f1e4b9df826193f0` plus
patch SHA256 `7279d85c550c27fcbf18cef06b7b8c502a4147ab45f8a898a05751dc459e45c5`.
Model/native-library identity uses retained metadata and runtime observations,
not whole-model/core payload hashes. Raw local artifacts stay out of Git.

- Builds: `parity-results/qwen-flash-next/direct180-{integration,release}-build-r3/`.
- Tests: `direct180-starter-r{1,2}/` and `direct180-unit-affected-r1/` under that result root.
- Accuracy: `direct180-model-fp16-r1/`, including both 185-row comparisons.
- Timing: `direct180-model-throughput-{a1,b1,a2,b2,a3}-r1/`.
- Native inventory: `direct180-model-path-r1/`; separate reconciliation: `direct180-model-path-audit-r1/`.
- Frozen binaries: `/tmp/qfn-{smallgroup179,direct180}-final-runtime-r1/`.
- Original benchmark and extended summaries: `/tmp/qfn-direct180-model-benchmark-r1/summary.json`
  and `/tmp/qfn-direct180-model-extension-private/summary-five-arms.json`.
- Exact owned commands and source pins are retained with those packets and
  `/tmp/qfn-direct180-qualification-r2/`.
