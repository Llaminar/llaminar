# QFN hybrid root timing and next performance target

The active target remains **500 prefill / 40 decode tok/s** on two MI50s and
two CPU sockets. This slice fixes phase attribution and measures the original
rooted CPU service. It introduces no kernel, placement or arithmetic policy.

## Matched model result

Unprofiled local Release, original four-shard Qwen3.8-Flash-Next UD-Q4_K_XL
plus shared-Q8_0 MTP sidecar; ROCm1/2 tensor-parallel continuation and apportioned
GPU experts; two MPI ranks/sockets with 28 physical CPU workers each and
complementary tensor-sharded CPU experts. Context262144, M512, dynamic MTP
maximum15, dynamic expert balancing, RAM ngrams, existing local GPU row caches,
and16/32GiB RAM/disk prefix budgets are unchanged. GPU admission retains
82–83 experts/card/layer after the selected-attention correction.

The workload is512 prompt tokens,256 generated tokens, seed42, one warmup and
three measured requests. Rates use total tokens divided by total phase time;
decode-after-prefill uses255 timed tokens per request.

| Local run | Prefill tok/s | Decode-after-prefill tok/s |
|---|---:|---:|
| Qualified admission correction, r1030 |184.8403|30.8094|
| Added phase/root observations, r1033 |184.3104|30.8388|

All768 generated IDs and actual per-request MTP counts match:192 accepted,
zero rejected,64 verifier runs and256 verifier rows per request. Prompt and
public/runtime configuration comparisons have no differences except output
paths. Dynamic expert movement and CPU routed work differ; they remain recorded
rather than being normalized away. This is a short instrumentation comparison,
not a new speed win, general accuracy gate or long-running leak certificate.
Both original native lifetimes and driver observers retire cleanly.

## What the measurements establish

The original retained CPU ticket worker waits for GPU-produced dispatch tickets
between layers. Its outer duration therefore includes device progress and peer
readiness, as well as useful CPU work. It is not a CPU-kernel timer.

| Root scope | Per512-token prefill | Per4-row verifier |
|---|---:|---:|
| Wait for GPU-produced dispatch tickets |886.439ms|41.837ms|
| Rooted gather, including peer readiness |490.648ms|4.176ms|
| Validated fixed-order shard combine |73.506ms|1.534ms|
| Original weighting and result publication |36.495ms|0.804ms|

These are host scopes within the original service, not a GPU critical-path
breakdown. Gather time is not isolated network/copy bandwidth. Related outer
service and follower timings overlap or nest and must not be added to this
table. Combine plus weighting is only110ms per prefill and2.34ms per verifier;
fusing those operations alone cannot explain or eliminate the larger gap.
The next priorities are a retained ROCm kernel/API/copy timeline and an audit of
the original two-socket gather implementation and live traffic.

Every root component has matching completed-stage counters:288 prefill,
9216 verifier and576 draft calls. Transaction counters own normalization,
including empty CPU endpoints whose compute timers are absent. This run omits
one empty prefill endpoint and125 empty verifier endpoints from packet-service
timers; those omissions are not missing root operations.

The apparent doubled prefill layer count is expected: each512-token request
uses disjoint448- and64-token chunks. Three requests ×48 main layers ×2 chunks
gives288 root FFNs in the preceding run. GPU participant counters describe both
TP graphs, not duplicate complete CPU experts. CPU sockets own F intervals
[0,320) and[320,640), with corresponding gate/up output and down input slices.
Follower raw statistics include a longer process-lifetime cohort; summing them
with the root's measured epoch would manufacture duplicated work.

## Implementation and regression contract

`MoECPUExpertShardRouteStage` exposes a nonmoving stack observation of its
original root-only gather, validated combine and weighting/publication. It owns
no buffers, routes, tickets or execution decisions. Each call resets the sample;
failure, omission and contributor execution publish no successful root sample.
Zero durations remain zero. The retained worker accumulates bounded scalar
totals by the original typed execution phase and publishes only after its
existing successful terminal join. Mapped joint-root work uses its existing
deferred layer sample. No new synchronization, allocation, transfer or graph
recapture is introduced by these observations.

The existing dispatch publication-wait timer previously inferred purpose from
row capacity, labeling grouped MTP as prefill or decode. It now uses the
immutable `ExpertHistogramSource`, including a one-row MTP predictor; absent
phase metadata is explicitly `unknown`. Older unrelated sparse-dispatch
two-phase counters are not silently reinterpreted by this fix.

Focused qualification:

- Ten Integration tests pass: root observation lifecycle, complete shard
  reducer Unit owner, original two-rank MPI stage/economy, and captured
  CUDA/ROCm prefill/decode/verifier publication. Lifecycle repeats20 times.
- The publication-phase regression and complete dispatch Unit owner pass.
  Both new phase test cases fail against the retained original Integration
  core for the intended phase mismatches, with its loader path authenticated.
- Four focused tests pass from the exact feature-worktree CPU-only build.
  Both new regressions are explicitly selected by ProductionTestPreflight.
- The matched native model run is from the broader development worktree with
  these same scoped changes. It does not certify every unrelated change or
  establish an exact feature-HEAD full-model result. Full preflight and the
  resumed OpenCode application gate remain open.

## Existing wider-row component result

The unchanged generic WideRows candidate was compared with current Auto,
observed as Pairwise, in a retained28-core AVX512 production fixture using
GPUAlignedExpert arithmetic and aligned output roles. All40 invocations pass
serial-byte, dirty-buffer, worker-placement and counter checks. First/reverse
order and enabled/disabled instrumentation samples are retained separately.

| Q4_K gate/up case | WideRows speedup, stats on/off |
|---|---:|
| Decode M3 |1.590 /1.508|
| Decode M4 |1.220 /1.346|
| Prefill M64 |1.060 /1.065|
| Prefill M192 |0.975 /0.959|

Q5_1 down M64 instead regresses to0.799/0.861 of Auto throughput. No blanket
WideRows default is promoted. Uniform70/35-projection component bundles are
not replays of the model's mixed-row expert batches. The existing shared shape
inventory already contains both CPU slice geometries; a future dispatch change
must retain the common producer-compatible certification workflow.

## Local evidence

- `/tmp/qfn-cpu-root-timing-qualification-r1033/`: model/admission and native
  closure receipts, phase/root comparison, driver/resource/placement records,
  all focused tests and original-phase negative control.
- `/tmp/qfn-cpu-current-component-r1032/`: paired component samples, immutable
  native closure, complete counter/placement evidence and conclusion.
- `/tmp/qfn-cpu-ordered-locality-r1031/DUPLICATION_AUDIT.md`: actual chunk,
  shard ownership and rank-epoch accounting.
- `/tmp/qfn-gpu-expert-capacity-fix-r1030/timing-audit/`: preceding timing scopes
  and explicit limits on hypothetical CPU-only speedup estimates.

Large local result files and model/cache payloads remain outside the source
repository. Completed owned disk-cache directories are removed after native
retirement; retained receipts never require scanning or hashing their payloads.
