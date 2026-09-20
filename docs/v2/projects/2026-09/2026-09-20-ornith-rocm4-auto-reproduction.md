# Ornith Q8 four-MI50 production and auto-selection reproduction

## Scope and provenance

Release source: `fc1a7569983cfceb9d0ffac79e55dd15828af532`, on `develop`.
No inference or planner implementation was changed for these measurements.
This is the short matched performance workload, not a new accuracy certificate.

- Four 32 GiB gfx906 MI50 devices, ordinals 0–3; driver reports no GPU P2P.
- ROCm 7.2.4; canonical RCCL at
  `/workspaces/llaminar/external/rccl/build/librccl.so`.
- `Ornith-1.5-35B-Q8_0.gguf`, 37,802,149,280 bytes, staged through the
  existing persistent tmpfs cache. No format conversion or precision tuning.
- Exact 512-token raw prompt from the comparison skill; SHA-256
  `5cbdeeafc357d7f0036f47ce25ba826809ca3f95f0fc6175f687a13a4ddf77b1`.
- 8,192-token capacity, 256 generated tokens, temperature zero, seed 42.
- FP32 model activations, FP16 KV, schema allreduce precision, MTP off.
- Profiling disabled for reported rates; captured inference and prefix misses
  verified in the receipts. Decode uses `decode_after_prefill` (255 timed steps).

Raw logs, individual iteration records, saved apply documents and profiling
evidence are in `/tmp/mx-llama-comparison.qrMGqk/results/`, outside source control.

## Reproduction results

| Run | Selected GPUs | Measured repetitions | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Saved previous-sprint Static result | 4 | 5 | 1,147.20 | 58.26 |
| Explicit four-card topology, production dispatch | 4 | 5 | 560.61 | 57.27 |
| Auto, Static maintenance override and 512/256 planning hint | 4 | 5 | 1,110.53 | 57.76 |
| Saved auto apply-document replay | 4 | 5 | 1,071.56 | 56.64 |
| Bare auto, production defaults | 4 | 3 | 1,092.37 | 55.15 |
| Bare auto plus only maintenance off (A/B diagnostic) | 4 | 3 | 1,090.13 | 57.68 |
| Identical bare-auto command, fresh startup | 3 | 3 | 1,315.97 | 55.86 |

Each run has one warmup. All generated IDs in every bare-auto and Static-auto
iteration exactly match the saved previous-sprint result. Prefix caching is
enabled, with no hits in these timed requests. The bare-auto graph includes
default dynamic maintenance; its receipt reports no completed movements on this
short workload. Do not claim this measures the long-horizon benefit of migration.

The four-device bare-auto run is approximately 5% below the saved Static
throughput, not an exact performance match. The one-setting maintenance A/B
recovers 57.68 tok/s: enabling default maintenance costs about 0.80 ms per timed
decode step on this short workload, even without completed movements. Prefill
does not materially change in that A/B.

Auto does not consistently select exactly four devices. An identical fresh
invocation selected ordinals 1–3, improving measured prefill and slightly
improving decode relative to the four-device default run. Its overall measured
request time was about 4.95 s versus 5.09 s for the four-device default. The
selector uses freshly measured primitive-service estimates, not whole-model
benchmarks; topology selection can vary between startups. The first command's
four-card choice is not a guarantee of stable four-card selection.

The explicit run's additional prefill cost has not been isolated; do not assume
it is a source regression or attribute it to NUMA placement without a controlled
measurement. Direct auto and saved-plan replay both recover the earlier prefill
performance range.

## Why the initial attempt was misleading

1. The reproduction command used `--deterministic`, not merely greedy sampling.
   That option publishes `LLAMINAR_DETERMINISTIC=1`. The current ROCm policy
   disables wave64 top-k dispatch and concurrent projections under that mode.
   In the slow trace, 40 serial routing kernels cost about 13.97 ms/token; the
   saved production trace's wave64 kernels cost about 0.91 ms/token. The slow
   run delivered about 33 tok/s. Using ordinary production dispatch with the
   same temperature-zero sampler restores approximately 58 tok/s without a
   code change or any changed generated tokens on this workload.
2. The earlier auto invocation imposed `--only-strategies expert-overlay`.
   `visitAutomaticOrchestrationCandidates()` currently labels a homogeneous
   multi-device pool `TensorParallel`, even when MoE execution is compiled into
   its sole ExpertOverlay authority. That filter therefore excluded the desired
   one-domain candidate. Unfiltered ROCm auto evaluated 61 admitted candidates
   and chose all four MI50s in one rank-local domain. It uses apportioned routed
   experts and `PrefillTensorParallelDecodeReplicated` dense execution.

The filtered two-domain selection also hit a separate graph-lowering error for
a same-rank secondary GPU expert domain. That defect is not fixed by removing
the filter and is not certified by these successful one-domain runs.

## Minimal reproduction command

No environment tuning, strategy filter, workload-cost hint, maintenance
override, explicit topology, precision override, or bucket override is needed:

```bash
./build_v2_release/llaminar2 benchmark \
  -m /mnt/llaminar-production-parity/cache/Ornith-1.5-35B-Q8_0.gguf \
  --only-backends rocm -c 8192 \
  --prompt-file /tmp/qwen38-exact512-prompt-nonl.txt \
  -n 256 --temperature 0 --seed 42 \
  --benchmark-json-output /tmp/ornith-auto-defaults.json
```

Auto is implicit when no placement is declared. The backend restriction scopes
this comparison to MI50s; the host also has CUDA devices. The other arguments
specify the measured workload, sampling, and output artifact, not tuning policy.
The benchmark defaults to one warmup and three measured repetitions.

## mx-llama.cpp comparison

The fork was built in Release for gfx906 at
`5542318e748c154b634211def405ae95da3dfaa9`, with HIP graphs and RCCL. It used the
same model and exact prompt, tensor split across four devices, FP16 KV, MTP off,
8,192-token capacity, temperature zero, seed 42, one warmup and five measured
requests. Every request reported 512 prompt tokens, no cached prompt tokens and
256 generated tokens. All returned the same 256 repetitions of ` test`, also
the content represented by the reproduced Llaminar token stream.

The successful fork run required Llaminar's canonical source-built RCCL on
`LD_LIBRARY_PATH`: the distribution RCCL crashed during the fork's initial
collective. This dependency substitution is part of the comparison provenance,
not evidence that the unmodified installed dependency works.

| Engine / configuration | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| mx-llama.cpp, four-card tensor split | 1,003.05 | 30.30 |
| Llaminar, bare ROCm auto defaults | 1,092.37 | 55.15 |

The initial bare-auto run is 8.9% faster in prefill and 82.0% faster in decode
on this exact short workload. These are completed-token rates from unprofiled
timing, not profiler kernel sums or a general result for every prompt/context.

Receipts: `mx-response-1.json` through `mx-response-5.json`,
`llaminar-auto-defaults.json`, `llaminar-auto-defaults-repeat.json`,
`llaminar-auto-defaults-maintenance-off.json`, `llaminar-auto-production-dispatch.json`,
`llaminar-auto-unrestricted-plan.json`, and `llaminar-auto-plan-replay.json`.
The comparison skill examples were corrected to distinguish temperature-zero
production sampling from diagnostic kernel determinism.
