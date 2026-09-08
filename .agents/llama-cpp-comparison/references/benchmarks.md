# Matched Release benchmarks

## Inputs and provenance

Record executable paths/build types, source revisions and dirty changes,
compiler/ISA, driver/runtime, device IDs, clocks/power settings, CPU affinity,
thread counts, model path/size, and prepared-weight/workspace bytes. Use the
same GGUF for both engines. A persistent tmpfs copy avoids repeated model-load
I/O; use the repository staging authority, not a private second cache scheme.
Initialization, capture and reference generation are reported separately from
steady inference. They are not hidden inside a “decode kernel” number.

Inspect each binary's help before copying flags. llama.cpp flags and HTTP fields
evolve; a checked-out revision and its local source are the authority.
Build through `cmake --build ... --parallel` with the configured Ninja executable.
Never change the user's model format, activation policy, MTP depth or VRAM budget
to make either side faster without authorization.

## Exact 512-token prompt

`../assets/prompt-512-qwen38.json` preserves the tested raw prompt as a JSON
string, including its lack of a trailing newline. Extract the string with
`jq -jr .prompt` into an artifact-directory prompt file. Do not use shell
`echo`, add a chat template, normalize whitespace or add a newline. The hint
in the fixture is not proof that another model will tokenize it to 512 tokens.
Its repeated `test` suffix is intentional synthetic padding to reach that
bucket, not additional instructions or a representative natural-text corpus.
Keep it unchanged when reproducing that comparison. For a new workload, a
natural-language prompt is equally valid; authenticate its token count and
start fresh baselines for both engines instead of mixing prompt identities.

Authenticate the prompt through the running llama.cpp `/tokenize` endpoint
with its actual special-token settings, and independently through Llaminar's
benchmark prompt metadata. For the pinned Qwen workload raw tokenization uses
`add_special: false`; verify whether a different model needs BOS. The actual
completion's `timings.prompt_n` is the final admission check. Require 512,
`cache_n == 0`, the complete generation budget, and the expected retained
Llaminar M512 graph. A model's tokenizer may require a different prompt; adapt
and retokenize *before freezing both engines' shared input*.

“512-token prompt” does not imply every physical launch has M512. For example,
llama.cpp may execute 508 rows plus a 4-row tail. Attribute their combined work
and record that split; do not compare only the cheaper physical batch.

## Llaminar example: MTP off

Set `COMPARISON_MODEL`, `COMPARISON_PROMPT` and `COMPARISON_OUTPUT` to explicit
paths chosen for this run. Keep ordinary MPI bootstrap for canonical timing:

```bash
env LLAMINAR_BENCHMARK_ITERATIONS=5 LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  ./build_v2_release/llaminar2 benchmark \
  -m "$COMPARISON_MODEL" -d cuda:0 -c 4096 \
  --prompt-file "$COMPARISON_PROMPT" -n 256 --deterministic \
  --benchmark-json-output "$COMPARISON_OUTPUT"
```

Do not inherit profiling/debug overrides accidentally. Inspect `LLAMINAR_*`,
profiler injection variables, `CUDA_VISIBLE_DEVICES`/HIP device masks and CPU
affinity before measurement. Confirm `perf_stats.enabled == false`, MTP off,
successful prefill/decode and the exact requested token budget in the JSON.
Do not merely assume omitted MTP flags kept it off.

Use `throughput_tokens_per_sec.decode_after_prefill` for comparison with the
llama.cpp completion timing below. Llaminar's generic `decode` uses a different
denominator: a 256-output request usually has 255 after-prefill serial steps.
Inspect counts and timing definitions in `BenchmarkRunner` and the pinned
llama.cpp implementation instead of comparing similarly named fields blindly.
Retain individual iteration rates and all generated IDs, not just one mean.

## llama.cpp server example

For a CUDA-only, single-slot, fully GPU-resident MTP-off comparison:

```bash
env CUDA_VISIBLE_DEVICES=0 "$COMPARISON_LLAMA_SERVER" \
  -m "$COMPARISON_MODEL" -ngl 999 -c 4096 -b 512 -ub 512 -np 1 \
  --no-kv-unified -fa on -ctk f16 -ctv f16 --spec-type none \
  --host 127.0.0.1 --port 19082
```

The port is an example: ensure it is unused and record ownership of the process
you launch. Poll `/health` within a bounded readiness budget. Send one warmup
and five measured requests **sequentially**, writing complete response JSONs:

```bash
jq -n --rawfile prompt "$COMPARISON_PROMPT" \
  '{prompt:$prompt,n_predict:256,temperature:0,seed:42,ignore_eos:true,
    cache_prompt:false,return_tokens:true}' |
  curl --fail-with-body --max-time 600 \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:19082/completion
```

Use raw `/completion`, not `/v1/chat/completions`, unless a templated chat
workload is explicitly the comparison contract. Validate HTTP status and JSON
error fields, `prompt_n`, `cache_n`, `predicted_n`, truncation and token IDs on
every request. A prefix-cache hit or early EOS is not a faster matched run.
`ignore_eos` is only appropriate when both benchmarks deliberately exhaust the
same generation budget. Compare prompt/predicted timing definitions in source.

Terminate only the exact server/MPI process group launched for this experiment;
wait for shutdown and verify devices are released. Do not use broad `pkill`.

## MTP and fair interpretation

When MTP is requested, set the same explicit depth policy and verifier/sampling
contract on both engines; don't infer draft support from a model filename.
Record attempted/accepted drafts, transactions, rollback, emitted tokens and
depth trajectory. First measure MTP off when distinguishing base-kernel gaps
from acceptance or verifier overhead. Cross-engine token divergence can change
the later workload; distinguish that from a regression against Llaminar's own
serial-byte oracle. Never relax Llaminar's invariance to mimic llama.cpp.
