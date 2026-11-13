# QFN selected-attention workspace admission

## Finding and correction

QFN's main `.attn_q.weight` sources were sufficient for the workspace estimator
to charge a generic dense FA2 workspace. The main graph actually selects QSA:
`QwenGraphBase::buildFAAttentionGraphForLayer` returns the model's selected
consumer before constructing generic dense attention. Its QSA buffers already
belong to the activation arena. The separate learned predictor uses dense
attention and must retain its independent workspace.

`exactAttentionWorkspaceBytes` in `WorkspaceMemoryEstimator.cpp` now follows that
existing typed main-consumer choice. It excludes only the dense workspace for
the selected main consumer. The dense predictor union, full-context KV
conversion, RoPE, sparse activation arena and other memory owners remain
unchanged. This corrects a nonexistent consumer; it introduces no capacity
discount, new accounting ledger or reduced context.

## Exact local admission result

The same public configuration was admitted before and after the correction:
two MI50s running TP continuation and apportioned GPU experts, plus two CPU
sockets tensor-sharding the capacity-tier experts. Each socket has one MPI rank
and 28 physical workers. The model is the original four-shard
`Qwen3.8-Flash-Next-UD-Q4_K_XL`, with the shared Q8_0 MTP sidecar.

Context is 262144; resident prefill bucket is 512; MTP remains dynamic with its
retained capacity of 16 target rows. Expert balancing is enabled. Ngrams use
RAM backing and ConsumerLocal GPU caches, with the same configured row/byte
ceilings. Prefix-cache budgets remain 16 GiB RAM and 32 GiB disk.

| Per-GPU quantity | Before | Corrected |
|---|---:|---:|
| Main estimator attention bytes | 7,553,843,712 | 1,479,541,248 |
| Main estimator total workspace bytes | 10,780,353,792 | 4,706,051,328 |
| Actual admitted ExecutionWorkspace budget | 10,801,151,040 | 4,726,848,576 |
| Actual primary workspace allocation | 3,141,583,620 | 3,141,583,620 |
| Live expert weight bytes | 9,840,435,200 | 15,911,731,200 |
| Experts per layer | 50–51 | 82–83 |
| Expert slots across all 49 main/predictor entries | 2,497 | 4,037 |

The reservation correction is **6,074,302,464 bytes = 5.657 GiB/card**.
Actual GPU expert weight admission grows by 6,071,296,000 bytes/card; the
integer quota and final low-priority ngram grants consume the remaining
difference. GPU expert slots increase **61.67%**. The two-card domain now holds
164/166 of 512 experts per layer; the CPU domain holds 348/346. The prior
two-card quotas were 102 for entries 0–46 and 100 for entries 47–48.

The larger allocation completed graph preparation and the complete measured
generation workload. Driver-reported free memory was stable at 1659.0 MiB and
1630.5 MiB on the two MI50s at every post-warmup measurement boundary. The
benchmark's historical `MB` label divides by 1024², so those values are MiB.
The observations are driver free-memory readings, not sums of shared DRM BOs.

Per-process RSS grew by 38.32/37.64 MiB across the three measured requests;
the observed GPU memory series stayed flat. This short observation does not
certify leak freedom. Native processes retired, and the driver observer
reported zero findings. Both owned disk prefix-cache directories were removed
after retirement; the measurement records remain available.

## Regression and runtime qualification

`V2_Integration_QFNSelectedAttentionWorkspaceAdmission` is explicitly registered
in ProductionTestPreflight. It runs actual metadata admission without a GPU or
model payload and covers:

- CPU/CUDA/ROCm, TP1/2/4/8, contexts 4096 and 262144, and retained query rows
  0/2/4/16 for off, fixed and dynamic-capacity cases.
- Main source-marker ownership, with the independently required canonical
  RoPE contribution preserved, and large-to-small row geometry.
- Predictor-only PP, repeated metadata queries, conventional dense controls,
  and independent full-head versus TP attention/KV descriptor deltas.

The new main-ownership regression genuinely fails against the original binary
and passes with the correction. The complete WorkspaceMemoryEstimator Unit
suite and retained-MTP, FP16 geometry and compact-expert workspace regressions
pass. Both AVX2/AVX512 ordered-kernel, nonfinite/tail and fixture-identity
regressions pass after retiring the unprofitable experimental wide-row kernel.
The isolated feature checkpoint's workspace Unit and focused admission tests
also pass. Full production preflight was not rerun in this focused iteration.

The real-model Release workload uses one warmup and three measured requests,
each with an exact 512-token prompt and 256 generated tokens, seed 42 and
ordinary production sampling. Decode throughput counts 255 after-prefill
tokens. All requests complete 192 accepted draft tokens, zero rejected drafts
and 64 verifier transactions. Prefix misses are retained; no cache bypass or
deterministic test-only inference policy is introduced.

The same-source control changes only the admission correction; the CPU pair
kernel and other implementation source remain identical. The measured public
configurations differ only in derived expert placements and per-tier quotas,
besides output paths. All 768 generated token identities and all MTP work
counts match. Both runs retain every sample and use the same phase-boundary
memory probes outside the measured timers.

| Aggregate tok/s | Original admission | Corrected admission | Improvement |
|---|---:|---:|---:|
| Prefill | 161.929 | 184.840 | 14.15% |
| Decode after prefill | 29.380 | 30.809 | 4.86% |

These rates divide total measured tokens by total phase time. They are one
local matched pair, each with three measured requests, in corrected-then-control
order. No randomized or long-session repeatability claim is made. Earlier CPU
checkpoint rates used medians and a different pair-kernel implementation; they
are not used to establish this admission-only gain. After the control, the
corrected Release executable, core, decoder and MPI-session helper were
restored byte-identically to the measured fixed versions.

## Local evidence and remaining work

- Original admission and ownership proof:
  `/tmp/qfn-gpu-expert-capacity-r1028/`.
- Reviewed source/test packet:
  `/tmp/qfn-qsa-workspace-capacity-r1029/`.
- Corrected native admission, model, focused tests and resource audit:
  `/tmp/qfn-gpu-expert-capacity-fix-r1030/`.

The corrected workspace budget still exceeds the physical primary allocation;
that difference is not automatically free expert capacity. This slice retains
the predictor's existing compact-family envelope and does not claim to have
minimized every reservation. The 500 prefill / 40 decode tok/s target, complete
preflight, exact checkpoint model qualification and OpenCode session remain
open. This is a local, bounded workload qualification, not an image or general
accuracy certification.
