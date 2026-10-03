# Single-MI50 Qwen comparison: Llaminar dynamic MTP vs mx-llama.cpp

Measured 2026-09-26. This is a Release throughput comparison, not an accuracy
certificate or a kernel attribution study. No production source, defaults,
weight formats, or clock settings were changed.

## Result

Llaminar wins both phases on this workload against the fork's fastest tested
decode configuration, MTP depth 2. Values are tokens/second, using the median
of five measured requests after one warmup. The Llaminar headline is its second
independent run, after the fork's MTP sweep; its first run is retained below.

| Model / unchanged GGUF | Llaminar dynamic prefill | mx MTP-2 prefill | Llaminar dynamic decode | mx MTP-2 decode | Decode advantage |
|---|---:|---:|---:|---:|---:|
| Qwen 3.8 27B IQ4_XS | 178.21 | 159.29 | 43.44 | 31.65 | +37.3% |
| Qwen 3.6 35B-A3B UD-IQ3_S | 804.20 | 694.64 | 125.07 | 87.10 | +43.6% |

Prefill advantages against those same MTP-2 configurations are 11.9% and 15.8%.
The fork's MTP-off controls prefill faster than its MTP cases: 168.70 and
787.23 tok/s, respectively. Llaminar dynamic still exceeds those measurements,
but the MoE prefill margin against MTP-off is only 2.2%, not a large win.
The fork's MTP-off decode is slower than its MTP-2 decode on both models.

### All measured configurations

Each row is one warmup plus five measured requests, with no outlier removal.

| Model | Engine / MTP policy | Prefill median | Decode median | Driver interval |
|---|---|---:|---:|---|
| 27B | Llaminar learned dynamic, first run | 178.96 | 43.46 | Clean |
| 27B | mx off | 168.70 | 28.24 | Clean |
| 27B | mx depth 1 | 159.29 | 30.11 | **Failed: new driver warnings** |
| 27B | mx depth 2 | 159.29 | 31.65 | Clean |
| 27B | mx depth 3 | 159.10 | 29.50 | Clean |
| 27B | Llaminar learned dynamic, repeat | 178.21 | 43.44 | Clean |
| 35B MoE | Llaminar learned dynamic, first run | 803.21 | 124.88 | Clean |
| 35B MoE | mx off | 787.23 | 64.47 | Clean |
| 35B MoE | mx depth 1 | 695.79 | 81.02 | Clean |
| 35B MoE | mx depth 2 | 694.64 | 87.10 | Clean |
| 35B MoE | mx depth 3 | 685.35 | 84.91 | Clean |
| 35B MoE | Llaminar learned dynamic, repeat | 804.20 | 125.07 | Clean |

Llaminar's repeated decode medians changed by less than 0.2%. Its individual
repeat samples span 43.437–43.445 tok/s for dense and 125.056–125.087 for MoE.
The best fork MTP-2 samples span 31.498–31.903 and 86.724–87.116, respectively.
These are workload-specific results, not claims about every prompt, context,
quantization, or possible fork tuning configuration.

## Frozen workload and provenance

- One MI50 32 GB, `gfx906`, GPU 0, PCI `0000:8c:00.0`, NUMA node 1.
  `HIP_VISIBLE_DEVICES=0` on both engines. Other GPUs performed no inference.
  No concurrent CI GPU run; the existing CPU-only user service was left alone.
- Host Linux `6.14.0-37-generic`, AMD driver `7.1.9.31600000`, ROCm `7.2.4`.
- Llaminar: immutable published AVX512 Release image
  `ghcr.io/llaminar/llaminar:develop-557f0f97d15d447850b77b7ce4e91bb0663331b3`,
  image ID `sha256:872c2dd9c7bd2d3e1023bf297152d1a527ce9d4873842621f4f744ab3c2a7c78`.
- Fork: [mxxm-t/mx-llama.cpp at eefc4e7321c869496146697d63362f073941aed6](https://github.com/mxxm-t/mx-llama.cpp/tree/eefc4e7321c869496146697d63362f073941aed6),
  fetched from master and built without local patches. Release, HIP graphs,
  gfx906, MMQ MFMA and RCCL enabled; native CPU backend, profiling disabled.
  The fork's documented MTP optimization and default weight repacking were
  enabled; its log confirms Q5_K repacking.
- Both engines opened the same files in the persistent tmpfs, without copying
  or changing their quantization:
  `Qwen3.8-27B-IQ4_XS.gguf` (15,705,861,088 bytes) and
  `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (15,346,432,288 bytes).
  Their sizes and modification times were unchanged after measurement.
- Exact prompt: `.agents/llama-cpp-comparison/assets/prompt-512-qwen38.json`,
  extracted without a trailing newline. 2,893 bytes, prompt SHA256
  `5cbdeeafc357d7f0036f47ce25ba826809ca3f95f0fc6175f687a13a4ddf77b1`.
  Its repeated `test` suffix is synthetic token-count padding, not a realistic
  multi-prompt corpus. Both tokenizers and all completion receipts confirm 512
  prompt tokens.
- Context capacity 4,096; GPU KV FP16; temperature 0, seed 42; 256 emitted
  tokens per request. Engine-native activation arithmetic was left unchanged.
  No diagnostic `--deterministic`, profiler, or PerfStats instrumentation.
- No prefix-cache hits. Llaminar's production prefix-cache machinery stayed
  enabled; the benchmark resets it between requests. Its graph receipts show
  448 real tokens in the retained M512 graph and the final 64 in M256. Both
  pieces are included in prefill timing; this is not a single physical M512
  launch. The fork's reported prefill timing likewise covers its entire prompt.
- Decode rates compare Llaminar `decode_after_prefill` with fork
  `predicted_per_second`. Both exclude the first free token from the numerator:
  255 subsequent tokens divided by the full recorded decode duration.
  Startup, loading and initial preparation are excluded from phase throughput.

## Reproduction settings

The Llaminar command inside the image was:

```bash
env LLAMINAR_BENCHMARK_ITERATIONS=5 LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  HIP_VISIBLE_DEVICES=0 /usr/local/bin/llaminar2 benchmark \
  -m "$MODEL" --only-backends rocm --auto-device-counts rocm=1 \
  -c 4096 --prompt-file "$PROMPT" -n 256 --temperature 0 --seed 42 \
  --mtp --mtp-depth-policy dynamic --benchmark-json-output "$RESULT"
```

Normal MPI bootstrap and auto planning remained enabled. No strategy or learned
MTP override was applied. The runtime confirms `rocm-mi50` generated policy,
adaptive depth enabled, legal range 1–15, and no MTP bypass. Dense finished at
depth 2 with three policy updates per request; MoE held depth 3. Per measured
request, dense accepted 152 drafts out of 240 attempted, over 104 verifier runs;
MoE accepted 178 out of 234, over 78 verifier runs. These attempted-draft ratios
are distinct from Llaminar's checked-token acceptance rate.

The fork server command was:

```bash
env HIP_VISIBLE_DEVICES=0 HSA_FORCE_FINE_GRAIN_PCIE=1 \
  GPU_MAX_HW_QUEUES=8 LLAMA_ENABLE_MTP_OPT=1 \
  /tmp/mx-llama-comparison.qrMGqk/src/build/bin/llama-server \
  -m "$MODEL" -ngl 999 -c 4096 -b 512 -ub 512 -np 1 \
  --no-kv-unified -fa on -ctk f16 -ctv f16 -lm dio \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 127.0.0.1 --port "$PORT"
```

Depths 1 and 3 change only `--spec-draft-n-max`; MTP-off uses
`--spec-type none` without a draft-depth flag. Variables follow the
[fork's operating recommendations](https://github.com/mxxm-t/mx-llama.cpp/blob/eefc4e7321c869496146697d63362f073941aed6/FEATURES.md).
The full local ROCm environment also had `HSA_OVERRIDE_GFX_VERSION=9.0.6`.
All server launches used unused loopback ports and owned process groups.

One warmup and five sequential `/completion` requests used the frozen prompt,
`n_predict:256`, `temperature:0`, `seed:42`, `ignore_eos:true`,
`cache_prompt:false`, and `return_tokens:true`. Each response proved
`prompt_n:512`, `cache_n:0`, `predicted_n:256`, 256 returned IDs, no truncation,
and real draft activity when MTP was selected. MTP-2 accepted 143/224 drafts
on dense and 158/194 on MoE. Graph-reuse counters were present in the fork logs.

## Correctness and driver observations

Token IDs repeat exactly within every measured configuration, including across
the two Llaminar launches. They are **not equal between engines**: Llaminar
versus mx MTP-2 first differs at output index 1 on dense and 14 on MoE
(zero-based). Consequently this is a matched-input, free-generation speed
comparison, not a forced-identical-token kernel benchmark or an HF parity proof.
Different continuations also influence speculative acceptance. Neither engine's
accuracy was established by this timing exercise.

The canonical GPU-driver diagnostic brackets include startup and teardown.
All four Llaminar intervals and seven of eight fork intervals were clean.
The fork dense MTP-1 interval emitted two `ih2 ring buffer overflow` warnings,
at monotonic times `125669.357769` on `0000:8c:00.0` and `125692.893336` on
`0000:97:00.0`. The queue stopped on that failed check. Read-only health checks
found no new reset/page fault, no active KFD process, and released VRAM; then
the independent remaining cases continued with the same warning gate.
The warning-producing case is retained as failed, not silently certified by
its completed timings. This establishes the interval, not the overflow's root
cause or a conclusion that depth 1 itself caused it. No warning allowlist,
driver reset, library replacement, or fork patch was introduced.

After all measurements, every GPU returned to its starting 10,866,688 bytes
of idle VRAM and no KFD process remained. The existing user servers were not
restarted or modified.

## Evidence

Raw artifact root:
`/tmp/llaminar-mx-rocm1-20260926.aDPDKV/`.

It contains `run_comparison.py`, exact per-cell `command.json`, model metadata,
prompt files, all warmup/measured responses and token IDs, Llaminar graph/MTP
receipts, server logs, driver checkpoints/reports, image metadata, fork CMake
configuration, hardware snapshots, `results.csv`, `all-summaries.json`, and
`all-driver-reports.json`. Large local evidence is intentionally outside git.
No Unit/preflight rerun or kernel profiling was needed: this slice changed no
production code and used the already-published Release image.

## AVX2 copy-paste script verification

The subsequently supplied public Bash recipe was executed on the Docker host,
changing only its model-directory placeholder to the daemon-visible export of
the existing tmpfs. Because the interactive workspace is a devcontainer, a
scoped `nsenter` wrapper entered the host mount namespace before executing the
script; no inference arguments or benchmark assertions were changed.

The script successfully ran `docker pull`, downloaded the exact prompt from the
pinned public GitHub URL, ran both models sequentially, and passed its JSON
assertions. Its independently observed host exit status was **0**. The AVX2
image tag was
`ghcr.io/llaminar/llaminar:develop-557f0f97d15d447850b77b7ce4e91bb0663331b3-avx2`;
the pull resolved manifest digest
`sha256:e53daf9f4ca28deacafb921fd8b5f252f655ab8df48d762eee862866d51e312f`.

| Model / unchanged GGUF | AVX2 prefill median | AVX2 decode median |
|---|---:|---:|
| Qwen 3.8 27B IQ4_XS | 178.92 | 43.415 |
| Qwen 3.6 35B-A3B UD-IQ3_S | 804.91 | 125.123 |

Both cases retained the same 512/256-token workload, one warmup and five measured
requests, adaptive MTP enabled, graph replay, zero prefix-cache matches, and
PerfStats disabled. Token IDs are identical across their five repetitions and
match the corresponding AVX512 continuation. The entire script's GPU-driver
interval contains zero new records or warnings. All GPUs returned to their
starting idle VRAM usage, and no KFD process remained.

Local evidence is in
`parity-results/avx2-script-check.mXGUIr/`: `benchmark-avx2.sh` is the exact
executed recipe with the path substitution, `host-script.log` records the pull
and both summaries, `host-exit-status.txt` records exit 0, and
`llaminar-benchmark.qZCWqB/` contains both full JSON reports and engine logs.
The outer devcontainer socket's attach stream omitted stdout; logs and exit
status were therefore observed through the direct host Docker socket without
restarting either workload or changing the recipe. The host script itself
received and saved the normal engine output.
