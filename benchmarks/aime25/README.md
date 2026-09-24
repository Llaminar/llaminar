# AIME 2025 HTTP Benchmark

This workflow evaluates the production `Release` HTTP server against all 30
problems in the pinned `math-ai/aime25` test set. It deliberately uses two chat
requests per problem: the second request carries the complete system, user, and
assistant history and asks the model to independently review its answer. This
makes the score an end-to-end check of model math, request reset, chat-template
rendering, generation, response parsing, and multi-turn history execution.

The workflow is backend-neutral. Pass an explicit `cpu:N`, `cuda:N`, or
`rocm:N` device and choose a distinct result directory for that backend. For
example, run Qwen 3.6 35B-A3B MoE on the first ROCm device with its current
residency policy pinned explicitly:

```bash
python3 scripts/benchmarks/aime25_http_benchmark.py \
  --binary build_v2_release/llaminar2 \
  --model /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --device rocm:0 \
  --mtp-depth 3 \
  --prefix-cache \
  --server-extra-args='--moe-residency-maintenance off' \
  --output-dir benchmark_results/aime25/qwen36-35b-a3b-iq3s-rocm0
```

The same runner, scoring, history canary, checkpoint, and provenance contracts
apply with `--device cuda:0` and `--device cpu:0`. Server arguments are empty by
default because model- and backend-specific policy belongs in the invocation,
where it is captured by the immutable manifest.

`--mtp-depth 3` is a typed benchmark policy, not shorthand text hidden inside
the free-form server arguments. It enables fixed-depth speculative-sampling
MTP and requires the completed PerfStats ledger to report requested, captured,
and terminal depth 3 with nonzero device-resident verifier work.
`--prefix-cache` similarly selects the benchmark's canonical RAM-backed cache
policy: a 1024 MiB RAM tier, terminal-state storage in `auto` mode, and MoE
placement fingerprints. Before scoring, the runner issues one deterministic
chat request twice with byte-identical HTTP payloads and requires identical
model output. Certification then requires cache population, a real partial or
full hit, a restore, and, when MTP is enabled, restoration of an MTP-bearing
payload. The exact replay is intentionally separate from the history canary:
serializing an assistant response into a later chat request can change tokens
at the old generation boundary, so semantic message history alone is not proof
of a token-prefix cache hit. Raw server arguments that contradict either typed
policy are rejected before the server starts.

The runner downloads only the 15 KiB JSONL at pinned Hugging Face revision
`563bb8404243c5f09de6ec262f2db674fe5bce9b`, verifies SHA-256
`b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331`,
and caches it under `~/.cache/llaminar/benchmarks/aime25/`.

Each first turn is atomically checkpointed before its review begins, and each
completed result is fsynced immediately. Re-running the same command resumes
the exact missing turn after proving that the manifest still names the same
runner source, executable and resolved `libllaminar2_core.so`, model, dataset,
prompts, sampling settings, server arguments, and inherited Llaminar/backend
runtime knobs. Use a new output directory when any of those change.

After an owned server exits, the runner authenticates requested MTP and prefix
cache activity from `perfstats.json` and writes the evidence digest and counts
to `execution-contract.json`. The contract also binds the exact-replay canary
artifact by SHA-256. A successful CUDA or ROCm run additionally proves
both prefill and decode capture/replay, rejects segmented or manual graph
execution, and consumes the same fail-closed host-transfer policy as the
canonical server E2E gate. That policy permits compact final-response
materialization and explicit RAM/disk prefix-cache tier movement. ROCm may also
publish its exactly 48-byte, immutable, non-state scheduler ticket because HIP
graphs lack conditional nodes; CUDA may not, and any unknown or malformed host
read is fatal. Successful prefill and decode publications are also reconciled
against graph-lifecycle counters, so a partially eager run cannot pass on the
strength of one captured request. CPU runs use the same HTTP and accuracy
workflow without applying GPU-only graph or transfer invariants.

`summary.json` reports strict first-turn and reviewed accuracy. A response is
correct only when its assistant `content` contains an explicit `Answer: NNN` or
`\boxed{NNN}` matching the official integer; unrelated numbers are never used
as an answer. The default is deterministic greedy generation with thinking
enabled, a 4096-token first turn, a 2048-token review, and an 8192-token server
context. These are Llaminar's reproducible two-turn settings, not a claim of
drop-in comparability with evaluations using 32K generation budgets or repeated
stochastic samples.
