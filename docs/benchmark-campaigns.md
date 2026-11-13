# Benchmark campaigns

`benchmark --benchmark-campaign PATH` measures an explicit list of request
policies under one prepared production runtime. It uses the same
`BenchmarkRunner` warmup, timing, token generation, counters and result schema
as a single benchmark. Weights, topology, retained capacity and placement stay
owned by the admitted runner. Each cell resets request state, publishes its
MTP policy through `configureMTPRequestPolicy`, and executes ordinary benchmark
requests. No second inference harness or model reload is involved.

A campaign contains one immutable sampling cohort. Its temperature, top-k,
top-p and positive seed must match the startup command exactly. Run different
sampling cohorts in separate processes. Production sampling specialization
still applies: the default `speculative-sampling` capability serves a greedy
request through its greedy specialization. A greedy-only capability rejects
a stochastic MTP cohort. Off cells use the same sampling law as their controls.

For example, with a production configuration that admits a predictor and
retains depth 15:

```bash
LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
LLAMINAR_BENCHMARK_ITERATIONS=3 \
llaminar2 benchmark --config serving.json --context-length 4096 \
  --n-predict 256 --temperature 0 --top-k 1 --top-p 1 --seed 42 \
  --mtp-graph-capacity-draft-tokens 15 \
  --benchmark-campaign campaign.json
```

The configuration selects model and hardware placement normally. The campaign
cannot override either. `--prompt`, `--prompt-file` and
`--benchmark-json-output` conflict with `--benchmark-campaign`, because the
campaign owns prompt inputs and incremental output paths.

## Input schema

Every object has exactly the fields shown below. Unknown fields, duplicate
keys, duplicate IDs, invalid types, unsupported policy values, output collisions
and ambiguous train/holdout identities fail before the first cell.

```json
{
  "schema": "llaminar.benchmark-campaign.v1",
  "output_directory": "/absolute/fresh/results",
  "sampling": {"temperature": 0, "top_k": 1, "top_p": 1, "seed": 42},
  "prompts": [
    {"id": "review-1", "partition": "train", "text": "Review this implementation carefully.\n"},
    {"id": "design-1", "partition": "holdout", "text": "Explain the ownership of a bounded queue.\n"}
  ],
  "cells": [
    {"id": "review-off", "prompt_id": "review-1", "mtp": {"mode": "off"}},
    {"id": "review-fixed3", "prompt_id": "review-1", "mtp": {"mode": "fixed", "depth": 3}},
    {"id": "review-observe5", "prompt_id": "review-1", "mtp": {
      "mode": "observe", "minimum_depth": 1, "maximum_depth": 15, "initial_depth": 5
    }},
    {"id": "design-dynamic", "prompt_id": "design-1", "mtp": {
      "mode": "dynamic", "minimum_depth": 1, "maximum_depth": 15, "initial_depth": 0
    }}
  ]
}
```

Cells execute in authored order; there is no implicit matrix expansion. Fixed
depths are literal positive values from 1 through 15. Fixed depth zero and
automatic fixed selection are rejected. Observe and Dynamic share the explicit
`minimum_depth`, `maximum_depth` and `initial_depth` fields; `initial_depth: 0`
requests the canonical runtime's automatic initial choice for that mode. Every
enabled policy must fit the startup retained capacity.

Observe uses `MTPDepthPolicyMode::Observe`: it records controller observations
and recommendations while retaining its initial active depth. Dynamic may apply
those recommendations. Both inherit controller geometry from startup, including
`--mtp-depth-window` and `--mtp-depth-min-samples`; campaign cells cannot override
those settings. For an aggregate observation that stays below the evaluation
threshold throughout a 256-token response, set both startup values to 1024 and
use an explicit Observe initial depth. Observe emits the same ordinary benchmark
timing and counter fields.
The parent command's positive `--n-predict` applies to every cell. Global
benchmark warmup/measurement settings apply unchanged to every cell.

Each prompt table entry is tokenized once, without extra BOS/EOS, before the
first request. Natural token IDs are retained unchanged. Every prompt plus the
decode budget must fit the admitted context; truncation is forbidden. The same
native token history cannot appear in both `train` and `holdout`, even under
different prompt IDs or text spellings. Partitions belong to immutable prompt
identities, never to a policy variant or repetition.

## Results and failure preservation

The output directory must be a fresh absolute path under an existing parent.
Each cell atomically publishes `<cell-id>.json` using the existing
`llaminar.benchmark.v1` schema. Campaign results additionally contain:

- `prompt.token_ids`, the exact once-tokenized input, beside its byte SHA-256;
- `campaign`, binding the raw campaign document digest, prompt ID, partition,
  cell ID and zero-based cell index;
- `request_policy`, with effective Off/fixed/observe/dynamic bounds, verification
  specialization, authored initial intent and canonical resolved initial depth;
- `device`, `model_class` and `sampling_mode` from native runtime authorities;
- the typed `training_scope`, including verifier-width policy, actual TP/PP,
  retained capacity, model geometry and hardware architecture pair;
- `runtime_configuration` and `runtime_configuration_sha256`, preserving the
  resolved immutable startup policy separately from effective request fields.

`config.sampling` retains the complete cohort. Its MTP execution fields reflect
the current request, while `mtp_graph_capacity_draft_tokens` retains the
original physical capacity. Off execution has effective depth zero; this does
not release retained predictor graphs or weights. A valid generic training
scope is emitted with `training_scope_generic_reason`; it remains a valid
benchmark result. A trainer requiring a particular measured domain must reject
that generic scope rather than infer geometry from a model filename or device
label. External runtime/source custody still belongs to the invoking workflow;
a configuration digest does not identify a binary or grant certification.

`summary.json` is atomically updated after each cell. It uses
`llaminar.benchmark-campaign-result.v1` with `complete`, `success`, `cell_count`,
`completed_cells`, `failure_reason`, `document_sha256`, and ordered
`cells: [{id, file, success}]`. Files are relative cell filenames.
`completed_cells` counts successful cells. A terminal failed campaign has
`complete: true`, `success: false`; its failed cell is retained when publication
succeeds, and later cells do not run. Earlier result files are never replaced
or deleted. An I/O failure preserves the successful prefix and reports its
concrete cell/path error; an interrupted process may leave an incomplete
summary or owned `.partial` file. There is no automatic retry or result adoption.

The model is prepared once. Per-cell warmup remains outside measured timing;
first-use request-policy preparation follows the existing production runtime.
Request resets preserve prepared model and graph owners. These local results
are diagnostic data, not image, numerical or public HTTP qualification.

## Draft acceptance statistics

`mtp.acceptance_rate`, each measured iteration's `mtp.acceptance_rate`, and
`mtp.request.acceptance_rate` report accepted draft tokens divided by actual
attempted draft tokens (`accepted_tokens / draft_steps`), or zero without
attempts. At depth above one, a rejected transaction can discard several draft
tokens. The legacy `rejected_tokens` field counts rejected transactions on the
captured path and cannot serve as the missing-token denominator.

The separate `stochastic_acceptance_rate` remains accepted stochastic tests
divided by performed stochastic tests. The benchmark JSON retains schema
`llaminar.benchmark.v1`: this corrects a derived statistic without changing
fields, raw counters, sampling, execution or timing. Historical results retain
their original published values; re-analysis or trainer inputs should derive
token acceptance from retained `accepted_tokens` and `draft_steps` explicitly.

## Controller observation evidence

Physical draft efficiency and the controller's acceptance feature have different
denominators. Dynamic and Observe omit transactions whose commit budget clips
the verifier outcome; physical work counters include their attempted drafts.
Consumed verifier comparisons are a third quantity and cannot replace either
denominator.

When passive PerfStats collection is enabled, benchmark JSON retains the exact
`mtp.device_generation_terminal_depth_policy_state` family. Its terminal tags
include eligible pending runs, attempted and accepted tokens, rejected tokens,
full/zero accepts and rollbacks. It adds no live device probe. Normal measured
reset removes warmup observations. These are residual window counts; they are
whole-request eligible totals only when the authored Observe window never fills
or evaluates. Such a proof must check the actual native policy and all participant
counts, rather than infer it from a filename or requested settings.

Paired trainer summaries may retain the original unprofiled fixed-depth costs
and physical `acceptance_pct` while supplying explicit `controller_*` evidence.
The trainer requires the complete `eligible_unflushed_observe_v1` counter set,
its source artifact and measurement count. It validates attempted tokens against
the observed depth/run count, acceptance/full/zero/rollback algebra and consistent
feature semantics within each trained domain, including holdouts. Partial or
mixed evidence fails. Instrumented observation timings do not replace the
unprofiled performance measurements.
