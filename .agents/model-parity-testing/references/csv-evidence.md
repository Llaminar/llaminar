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
- `production_path.csv`

Specialized cells can add router, MTP, rank-fragment, or other diagnostics, but
they cannot replace the six canonical files. Pipeline-parallel tests merge
rank-local fragments into the canonical set; missing owned layers, embedding,
final norm, or LM-head evidence is a failure.

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
device-generation parent. Require `device_generation_controller=true`,
`generation_execution_policy=native_conditional_parent`, and
`native_generation_parent=true`. A
`host_scheduled_captured_transactions`, `unclassified`, or `inconsistent`
policy is an architectural failure even when every numerical checkpoint passes.

Unexpected `model_context_reused=false` in compatible adjacent cells usually
means the immutable context key includes runner-owned policy. Unexpected
`true` can mean the key omitted a physical weight/topology identity. In either
case inspect the key; never patch around it with a host copy or extra model
cache.

### 2. Output summaries

`prefill_summary.csv` contains LM-head cosine and KL divergence, Top-1/Top-5
overlap, whether the reference Top-1 is in the native Top-K, layer pass counts,
and the overall result.

`decode_steps.csv` contains the same distribution evidence per incremental
step plus native/reference token IDs and exact/Top-3/Top-5 token matches. Find
the first bad decode step. A later-only failure often implicates KV indexing,
position/rope state, cache reset, sampling state, or MTP advancement.

For an MTP cell, inspect `mtp_sidecar_token_trace.csv` at the same time. Every
call with positive `selected_depth` must have an equally sized
`production_verifier_draft_tokens` vector, and
`verifier_identity_transaction_count` / `verifier_identity_depth` must agree
with that call's committed device-controller transaction. These fields come
from the persistent identity copied by the fused response/state commit, not
from reusable proposal or verifier-input scratch. A later row containing the
prior transaction's draft vector is stale publication; an empty vector on a
speculative row is missing publication. A terminal absorbing row may retain the
last committed identity while selecting depth zero because it commits no new
verifier transaction.

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
- metrics look good but counts/stages differ: snapshot publication or PP merge
  completeness defect; do not accept the run.

High kurtosis warns that a few outliers can dominate max error while cosine
stays high. Low error entropy suggests a systematic layout/scale/bias problem;
broad entropy with gradual drift more often suggests accumulated arithmetic
differences. Use these as diagnostic clues, not replacement pass criteria.

## Correlate PerfStats

CSV mathematics cannot prove expert movement or capture selection by itself.
Correlate the exact cell log and request-local PerfStats domains:

- `forward_graph`: full capture/replay and segmentation evidence;
- `moe_placement`: requested ordinal/random physical owner mapping;
- `moe_rebalance`: Dynamic/LLEP planning, transport bytes, ownership/replica
  change, destination apply, and Static zero-movement proof;
- MTP domains used by the fixture: requested/effective depth, verifier work,
  and dynamic-controller decisions.

Positive movement requires both a semantic change and payload evidence. A plan
entry without transferred/applied bytes is not movement. Static must assert the
whole movement family remains zero rather than checking one convenient counter.

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
