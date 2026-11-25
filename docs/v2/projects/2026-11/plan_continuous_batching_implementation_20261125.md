# Continuous batching implementation plan and decisions (2026-11-25)

Companion to `design_continuous_batching_20261105.md` (revision 3). This file
records operator decisions and the implementation's current shape so reviews
do not reopen settled questions. Detailed evidence and progress live in the
reviewer dispositions and the project's `IMPLEMENTATION_PLAN.md`.

## Operator decisions

| Decision | Scope |
|---|---|
| G1 §5 + §8 + §9.4 adopted (2026-11-19) | Typed admission/lifetimes, cancellation, one-window grants, immutable publication receipts; DecodeWindow operation catalog; typed S/R/Q serving capacity with exact-or-reject resolution |
| Retire `GroupedServingEngine` (2026-11-24) | The engine, its `SharedExecutionProof` suite, its satellite models and the PMA aggregate-admission handle are removed; production serving state lives in the runner-owned `ServingSession` |
| Keep the `ServingWindowDecode` / DecodeWindow ABI (2026-11-24) | Reserved seam for GPU retained continuous execution; labelled unimplemented until a runner overrides `decodeServingWindowImpl` |
| NUMA-less-kernel fix stays on `feat/continuous-batching` (2026-11-24) | Hardened in place (agreeing sysfs / ENOSYS / hwloc observation, cached, injectable for tests) rather than split into its own PR |
| Implementation-first workflow (2026-11-24) | Focused tests per slice, Conventional Commits with `--no-verify` while the CPU-only lane carries the known GPU-family Unit failure, push to the draft PR, asynchronous milestone reviews |

## Current shape

- `OrchestrationRunner` is the sole lifecycle authority. It owns one
  `ServingSession` (`execution/serving/ServingSession.*`) per open serving batch
  and drives slot-addressed rows of the inference runner through
  `IServingRowExecutor`.
- One step: cancellations, admission into free slots (per-slot prefix restore),
  decisions from retained per-slot logits (forced, unpaced or granted), one
  decode forward, one prefill transaction with the oldest prefilling request's
  chunk, retirement. A failed shared forward seals the session.
- HTTP: `--serving-resident-slots S > 1` runs every chat request through
  `ServingBatchDriver` (driver thread owns the runner) and per-request
  `BatchedGenerationLane`s with one-window grants. Admission takes a
  `ServingResponseLeasePool` lease (R responses, S + Q executing) over the
  PMA `ServingResponsePool` line, which also prices one decision-logits row per
  slot. S = 1 keeps `SerializedInferenceExecutor`.
- A sealed session reports 503 on `/health` and inference (documented fatal
  policy; the supervisor restarts the server).

## Supported now / not yet

| Supported (CPU, single rank) | Fails closed at startup with the precise reason |
|---|---|
| Dense full-attention and hybrid recurrent (GDN, prefix cache off) models, greedy and seeded stochastic sampling, stop tokens, paced grants, forced tokens, cancellation, slow consumers, prefix cache (KV-only per slot, dense) | MTP draft state, hybrid + prefix cache, routed-expert (MoE) models, model-runtime prefix state, GPU retained execution, multi-rank |

Serving rows serve every resolved S, including the automatic S = 1 pool, so
default CPU serving reports `inference_queue.mode = continuous_batching`; the
topologies in the right column keep the serialized executor (logged and
recorded as PerfStats `server/serving_path` at startup).

## Next

1. Migrate S = 1 onto serving rows and retire the scalar serving path once rows
   cover every serving topology (G7).
2. MTP and hybrid rows on CPU.
3. GPU retained continuous execution through the DecodeWindow seam (requires
   CUDA/ROCm hardware).
4. Distributed batching and the Release benchmark matrix (requires hardware).
