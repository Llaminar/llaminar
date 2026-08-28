# Reading parity CSV evidence

## Find the evidence set

Each exact cell writes beneath:

```text
tests/v2/integration/parity/results/<git-short-hash>/<sanitized-test-name>/
```

The directory is generated debris. Preserve it while diagnosing, upload it on
CI failure, and do not commit it.

A complete mathematical cell contains:

- `prefill_layers.csv`
- `prefill_summary.csv`
- `prefill_stages.csv`
- `decode_steps.csv`
- `decode_layers.csv`
- `decode_stages.csv`
- `prefix_restore.csv`
- `production_path.csv`

MTP cells additionally require `mtp_transactions.csv`. Specialized cells can
add router, rank-fragment, or other diagnostics, but they cannot replace the
canonical files. Pipeline-parallel tests merge
rank-local fragments into the canonical set; missing owned layers, embedding,
final norm, or LM-head evidence is a failure.

ExpertOverlay cells additionally retain:

- `expert_residency_diagnostics.csv` for rank zero and
  `expert_residency_diagnostics_rank_<rank>.csv` for every follower, containing
  the controller transaction and physical-residency publication counters even
  when the cell passes;
- `promoted_expert_execution.csv`, which ties moved expert identities to
  independently routed native/reference numerical rows.

## Read in this order

### 1. Production path

`production_path.csv` answers whether the claimed implementation ran. Inspect
`execution_path`, inner-forward and complete graph capture/replay fields,
generation-controller authority, segmentation fields, `model_context_reused`,
cell elapsed time, budget, and pass status.

For homogeneous GPU cells, require a complete prefill capture or replay and a
complete decode capture or replay with no segmentation. Treat numerical green
on eager or segmented homogeneous execution as uncertified.

For MTP, `forward_full_graph_capture` and `forward_full_graph_replay` describe
captured transaction fragments only. The complete `full_graph_capture` and
`full_graph_replay` fields remain false unless PerfStats also proves one native
device-generation parent. CUDA requires
`device_generation_controller=true`,
`generation_execution_policy=native_conditional_parent`, and
`native_generation_parent=true`. ROCm instead requires the authenticated
`host_scheduled_captured_transactions` policy: the device controller owns all
mutable state and the host may copy only the exact 48-byte immutable ticket
that selects a complete retained transaction graph. Its ticket provenance,
participant submission ledger, and controller transaction count must agree.
`unclassified`, `inconsistent`, a mutable state payload, or hosted eager
execution is an architectural failure even when every checkpoint passes.

Unexpected `model_context_reused=false` in compatible adjacent cells usually
means the immutable context key includes runner-owned policy. Unexpected
`true` can mean the key omitted a physical weight/topology identity. In either
case inspect the key; never patch around it with a host copy or extra model
cache.

### 2. Output summaries

`prefill_summary.csv` contains LM-head cosine and KL divergence, Top-1/Top-5
overlap, whether the reference Top-1 is in the native Top-K, layer pass counts,
and the overall result.

`prefix_restore.csv` proves fresh seed, complete hit, and partial hit through
the production cache surface. Complete hits require exact persistent-state
identity. For a partial hit, `main_kv_policy` remains `exact_bytes` unless the
snapshots themselves prove that ExpertOverlay changed placement between the
serial oracle and replay. The placement-aware policy still hashes the entire
restored prefix byte-for-byte; only the explicitly retained recomputed suffix
may use full-value FP16/BF16/FP32 comparison. Require positive
`main_kv_exact_prefix_segments` and `main_kv_numerical_suffix_payloads`, a
passing numerical flag, and a minimum cosine at or above the cell's typed
threshold whenever that policy is reported. Missing suffix bytes or any cached
prefix hash drift is a hard failure.

`decode_steps.csv` contains the same distribution evidence per incremental
step plus native/reference token IDs and exact/Top-3/Top-5 token matches. Find
the first bad decode step. A later-only failure often implicates KV indexing,
position/rope state, cache reset, sampling state, or MTP advancement.

For an MTP cell, inspect `mtp_transactions.csv` and, when the specialized
long-horizon campaign emits it, `mtp_sidecar_token_trace.csv`. Every speculative
transaction must have an equally sized `production_verifier_draft_tokens`
vector, and `verifier_identity_transaction_count` /
`verifier_identity_depth` must agree with its committed device-controller
transaction. These fields come from the persistent identity copied by the fused
response/state commit, not from reusable proposal or verifier-input scratch.
They also identify the exact forced-branch Hugging Face oracle when quantized
predictor argmax leaves the canonical reference branch. A later row containing
the prior transaction's draft vector is stale publication; an empty vector on a
speculative row is missing publication. A terminal absorbing row may retain the
last committed identity while selecting depth zero because it commits no new
verifier transaction.

For a dynamic-depth row, also require
`dynamic_policy_witness_executed=true`,
`dynamic_policy_witness_serial_token_exact=true`, a positive
`dynamic_policy_witness_window_delta`, positive attempted-draft and verifier
counts, and identical emitted/oracle token vectors. The witness deliberately
runs after any initially due ExpertOverlay maintenance boundary. If the
physical depth-15 checkpoint transaction passes but this witness remains zero,
inspect generation-budget clipping and maintenance cadence before inspecting
sidecar arithmetic.

### 3. Layer rollups

`prefill_layers.csv` and `decode_layers.csv` identify the first failing layer,
its average/minimum cosine, worst stage, largest cosine drop, compared-stage
count, and pass result. Prefill also records maximum kurtosis and its stage.

Check stage counts before scores. Missing rows can produce deceptively clean
averages and indicate a hook/publication/merge defect rather than numerical
accuracy.

### 4. Stage rows

`prefill_stages.csv` and `decode_stages.csv` are the primary root-cause data.
They include:

- backend, optional decode step, layer, and stage identity;
- cosine, cosine drop, relative L2, maximum absolute difference, SNR, RMSE,
  error entropy, element count, threshold, and pass status;
- native and PyTorch min/max/mean/stddev/L1/L2/kurtosis plus NaN/Inf counts;
- router-specific top-expert/set/weight evidence when the stage is sparse.

Core vector metrics use double-precision accumulation even though published
snapshots are floats. A stage passes its typed cosine threshold only when the
native tensor also contains no NaN or Inf.

## Interpret common signatures

- First failure at embedding: tokenizer identity, vocabulary mapping, weight
  loading/dequantization, or embedding sharding.
- Q/K projection first: fused projection offsets, quantized matrix layout,
  bias, head partitioning, or capture-time pointer/workspace identity.
- RoPE first: positions, rotary layout, scaling, or decode offset.
- attention scores/softmax first: mask, GQA head mapping, KV precision/layout,
  reduction order, or cache length.
- attention output/residual first: output projection, collective order, or
  residual ownership.
- FFN/router first: gate/up transform, activation, routed expert ownership,
  top-K normalization, expert weight layout, or movement publication.
- shared/combined MoE first: shared gate ordering, TP reduction, or combining
  routed and shared outputs.
- final norm first: earlier residual drift or norm epsilon/reduction.
- only LM head/KL/Top-K fails: LM-head weight layout, last-token selection,
  vocabulary partition/gather, or meaningful accumulated drift.
- prefill green and decode step 0 bad: request reset, KV publication, or the
  definition of the terminal-prefill observation.
- deeper decode progressively degrades: KV write/read index, precision,
  position state, or stale graph scalar/pointer.
- MTP sidecar stage first: sidecar weights, depth-specific state binding,
  recursive hidden-state input, or dynamic-controller selection.
- MTP checkpoints pass but the committed verifier vector is missing, delayed,
  or has the wrong depth: response/controller/diagnostic publication ordering,
  arena-event identity, or mirrored-rank aggregation—not sidecar arithmetic.
- routing set differs while dense stages match: router normalization/top-K tie
  policy or expert index mapping, not expert GEMM arithmetic.
- `promoted_expert_execution.csv` reports `exact_zero_route_rows`: the native
  and reference expert outputs were independently routed and both bytewise
  zero, which is exact equality (quantized experts with all-zero scales can
  legitimately do this). `one_sided_zero_route_rows` is a real mismatch and
  must be zero for the witness to pass.
- `reference_lineage=diverged_by_prior_routing` with
  `proof_disposition=inconclusive`: an earlier top-k branch changed the hidden
  state, so this later per-expert HF comparison used different inputs. It is
  diagnostic, cannot certify movement, and is not evidence of a kernel defect.
  Confirm that the same destination has another `certified` witness and that
  the ordinary downstream stage/LM-head/KL gates pass. A `failed` disposition,
  one-sided zero, invalid evidence, or a canonical-lineage numerical mismatch
  still fails closed.
- metrics look good but counts/stages differ: snapshot publication or PP merge
  completeness defect; do not accept the run.

High kurtosis warns that a few outliers can dominate max error while cosine
stays high. Low error entropy suggests a systematic layout/scale/bias problem;
broad entropy with gradual drift more often suggests accumulated arithmetic
differences. Use these as diagnostic clues, not replacement pass criteria.

## Correlate authority state and PerfStats

Numerical CSV mathematics cannot prove expert movement or capture selection by
itself. For ExpertOverlay, `expert_movement.csv` is a direct serialization of
`IOrchestrationRunner::moeOptimizationMovementLedger()` and is the movement
authority. Read `movement_axis` as logical intent (`tier_residency`,
`participant_placement`, or `combined`) independently from `direction`
(`promotion`, `demotion`, or `same_priority`). A combined promotion is valid
proof of both Dynamic axes even when there is no same-priority edge. PerfStats
must corroborate lifecycle, physical transport, economy, and publication, but
must never be promoted from optional observability into placement authority.
Correlate the exact cell log and request-local domains:

- `forward_graph`: full capture/replay and segmentation evidence;
- `moe_placement`: requested ordinal/random physical owner mapping;
- `moe_rebalance`: host-authoritative planning, transport bytes,
  ownership/replica change, destination apply, and Static zero-movement proof;
- `moe_overlay_controller`: device-authoritative accepted movement
  transactions and physical byte counts;
- `moe_overlay_residency`: device-side staging and atomic epoch publication of
  the new physical placement;
- MTP domains used by the fixture: requested/effective depth, verifier work,
  and dynamic-controller decisions.

Positive movement requires both an authority-ledger semantic change and payload
evidence. A proposal or plan entry without a durable ledger edge and
transferred/applied bytes is not movement. Static must assert an empty ledger
and that the whole physical movement family remains zero rather than checking
one convenient counter.
Use the per-rank `expert_residency_diagnostics*.csv` files to reconcile a
cross-rank failure before treating a missing rank-zero counter as missing
movement.

## From evidence to regression

Reduce to the earliest bad stage and one exact configuration. Preserve the
model, prompt, reference identity, backend, precision, topology, graph setting,
and policy. Add a focused regression at the narrowest invariant that can catch
the bug without loading a model when possible; GPU behavior belongs in a CUDA
or ROCm integration test.

After the fix:

1. Run the focused regression.
2. Run the exact production campaign containing the failed cell.
3. Confirm every canonical CSV exists and the first-bad stage is now green.
4. Confirm path and movement PerfStats still prove the optimized behavior.
5. Run the complete unfiltered matrix and retain its JSON timing report.

Never fix a CSV failure by deleting the row, reducing checkpoint coverage,
loosening a threshold without mathematical justification, disabling capture,
changing the configured precision, or regenerating against different model or
prompt bytes.
