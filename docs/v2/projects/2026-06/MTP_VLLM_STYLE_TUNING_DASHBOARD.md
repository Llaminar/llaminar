# vLLM-Style MTP Tuning Dashboard

Scope: Qwen3.6 dense/MoE MTP on CPU, CUDA, and ROCm. Keep implementation
detail in project handoffs.

RAG: **G** correct and economical, **A** correct but untuned/stale, **R**
failing or not yet proven. Token equality alone is not verifier parity proof.

## Current State

| Goal | Completion | Remaining proof |
|---|---:|---|
| SingleDevice fully device-resident MTP | 97% | refresh d1 economy and stochastic matrix |
| LocalTP fully device-resident MTP | 99% | remote participants and economy matrix |
| ExpertParallel fully device-resident MTP | 97% | mirrored-head batching across every EP mode |

- Homogeneous CUDA/ROCm execution is full-graph only. Device parameters select
  regimes inside one immutable capture; attention never requests recapture.
- GPU KV append, TurboQuant, verifier publication, and collectives use
  persistent workspace and explicit events. Hot paths have no allocation,
  blocking sync, segmentation, or intermediate host observation.
- Prefix restore preserves graph addresses through producer events. GPU
  FFN/MTP APIs reject null publication streams.

## Canonical CUDA2/ROCm2 LLEP Target

`LLEP` is the ordinary large-prefill routed-row assignment axis, not a
shorthand for an entire MoE execution mode. The active tuning target is this
explicit tuple:

| Axis | Required policy |
|---|---|
| Dense/shared trunk | tensor parallel |
| Routed expert storage/compute | apportioned whole experts |
| Routed phase | uniform |
| Grouped verifier/decode assignment | static owner |
| Ordinary large-prefill assignment | least-loaded resident (LLEP) |
| Small-prefill regime | static owner below the explicit routed-row threshold |
| MTP terminal norm/head | mirrored full vocabulary |
| Durable residency maintenance | off |
| Hot expert replica cache | off |
| Transport | one fully captured NCCL/RCCL graph |

The production large-prefill boundary is currently `M * top_k >= 8192` routed
rows. Setting it to zero is reserved for focused transfer-path proof. This is a
declared work-regime switch: grouped verifier/decode remains static-owner and
must never inherit least-loaded prefill assignment. Dynamic whole-expert
residency maintenance and hot replicas are independent experiments and are not
part of the canonical LLEP economy row.

Canonical tests must prove the tuple through PerfStats: static-owner grouped
verifier calls, least-loaded current-batch prefill when the threshold is met,
mirrored terminal-head execution, graph-captured NCCL/RCCL, no segmented
execution, and no durable maintenance or hot-cache activity. Performance
tuning starts with stochastic fixed depth 3 so graph/communication economics
are isolated; dynamic depth is tuned only after that baseline is sound.

## Production Matrix

| Mode | Device | Dense greedy/stoch | MoE greedy/stoch | Status |
|---|---|:---:|:---:|---|
| SingleDevice | CPU | R/R | A/A | refresh paused |
| SingleDevice | CUDA | A/G | A/R | dense d3 wins; d1/MoE need tuning |
| SingleDevice | ROCm | A/A | A/A | dense d3 wins; d1/MoE need tuning |
| LocalTP | CUDA2 | A/A | A/A | Dynamic-maintenance/current-batch-LLEP matrix green; perf active |
| LocalTP | ROCm2 | A/A | A/A | Dynamic-maintenance/current-batch-LLEP matrix green; perf pending |
| LocalTP | ROCm4 | A/R | R/R | full refresh pending |
| NodeLocalTP | CPU2 | A/A | R/R | dense E2E green; MoE/perf pending |
| ExpertParallel | GPU+CPU | A/R | G/R | explicit Dynamic/LLEP greedy+prefix green |

## Correctness Proof

- All-format grouped attention, TurboQuant, MoE routing, GDN, short-conv,
  stochastic target preparation, and draft publication sweeps are serial-row
  byte exact on their production backends and M ranges.
- CUDA MoE grouped verifier passed every native format at M=2..16 and M=31
  through the real router/expert path after the small-M grouping fusion.
- Release CUDA2/ROCm2 passed all eight Dynamic/LLEP cells and `166/166` checks
  through 2048 tokens with full capture and clean VRAM release.
- Release CUDA2 LLEP + RAM prefix + stochastic dynamic d4..15 passed `23/23`
  canonical server checks at context 4096 and 1024 output tokens.
  This includes deterministic prefix replay, forced movement, long recall,
  prefill replay, no segmented execution, clean shutdown, and PerfStats path
  assertions.
- The 2026-08-04 canonical grouped-verifier gate passed `89/89` explicit CPU,
  CUDA, CUDA2, ROCm, and ROCm2 cells. CUDA/ROCm KV publication is isolated by
  all 14 cache/source format pairs plus converted-read, logical-restore,
  adversarial, and TurboQuant lifecycle cells; each process uses the production
  graph-capture transaction and exact stream/event ownership.
- TurboQuant codebooks are uploaded once per cache/device on its construction
  stream and covered by the cache constructor's initialization fence. Launch
  wrappers cannot perform codebook upload, and no process-global ready flag can
  incorrectly alias initialization across devices. The complete Integration
  tree rebuilt cleanly and the device-free unit/source gate passed `585/585`.
- Expert-overlay policy is now represented as independent typed axes throughout
  config, graph lowering, fixtures, and E2E registration. The model-free
  canonical-tuple regression proves dense TP, apportioned/uniform routed work,
  static-owner grouped verification, least-loaded current-batch prefill,
  mirrored MTP head, and both durable maintenance and hot replicas Off. The
  complete Integration tree rebuilt all `1413/1413` edges and the device-free
  unit/source gate passed `585/585` on 2026-08-04.
- LocalTP routed-plus-shared publication now has one typed canonical transaction:
  every participant publishes router slots plus one rank-indexed shared bank,
  one rooted sum transports that payload, the root finalizes routes, shared
  banks, sigmoid gate, and residual in fixed serial order, and one broadcast
  publishes the terminal hidden rows. The previous routed reduce/broadcast plus
  independent shared allreduce/gate/combine transaction is absent from this
  policy. The production graph-lowering regression proves one reduce and one
  broadcast, exact producer dependencies, root-only finalization, and no old
  shared collective or epilogue nodes.
- The canonical publication kernels are byte exact against repeated production
  M=1 arithmetic for every `M=1..16,31`, full Qwen width 2048, and ragged/vector
  boundary widths `1,3,4,5,255,256,257,512,513`. Captured M=16 graphs replay at
  live M=11 without changing inactive rows. The complete CUDA and ROCm MoE
  grouped-verifier suites, including all native expert formats and real-path
  graph capture, passed in `189.83 s` and `109.47 s`, respectively. The full
  Integration tree then rebuilt all 971 affected targets and the complete
  device-free unit/source gate passed `585/585` in `143.32 s`. The strengthened
  source policy also forbids compute stages from inspecting raw tensor
  coherence after exact-stream publication; graph capture records that edge in
  its dependency ledger and replay owns the authority transition.
- The 2026-08-05 CurrentBatchLLEP long-context failure was a runtime-table
  identity defect, not a collective timeout. Ordinary prefill and durable
  static decode had aliased one placement table, allowing transient
  least-loaded assignment to leak into the next decode layer. Qwen3.5 MoE now
  declares typed `MainDecodeDurablePlacement`, `CurrentBatchLLEPPrefill`, and
  `MTPDepth` table roles; graph keys, prefix runtime state v4, bindings, and
  source/unit tests preserve those identities independently. CUDA2 and ROCm2
  CurrentBatchLLEP long-context cells passed after the change in `239.50 s` and
  `455.36 s`, respectively, with their canonical full-capture and PerfStats
  assertions enabled.
- A full CUDA FlashAttention fixture then exposed a second device-state defect:
  the cached-token parameter writer treated one shared FA2 parameter record as
  one logical query row, publishing `kv_len=1` for ordinary multirow prefill.
  CUDA and ROCm now distinguish shared prefill/M=1 records from row-local
  grouped-verifier records. Focused device-memory regressions cover initial and
  continuation prefill, active-row padded prefill, M=1 decode, and M=4 grouped
  verification. The complete CUDA and ROCm FlashAttention fixtures passed in
  `69.60 s` and `64.05 s`; CUDA captured cache growth/reset and ROCm all-format
  captured request-cache proofs remain byte exact. GPU cache views also require
  an explicit backend-qualified `DeviceId`, preventing a HIP allocation from
  being silently labeled as CUDA storage. The final Integration tree rebuilt
  all 784 affected targets and the device-free unit/source gate passed
  `585/585` in `135.90 s`.
- Long-context prefill now has an explicit `M=262145` totality gate on CPU,
  CUDA, and ROCm. The CPU cell uses Qwen2.5-0.5B geometry and one native Q8_1
  KV layer, processes exactly `64 * 4096 + 1` real rows, and proves native-byte
  first/final-row identity in `0.52 s`. Each GPU cell captures one 4096-row
  graph and replays it 64 times plus the one-row tail without per-chunk host
  slicing, transfer, allocation, or synchronization. A profiler-amplified
  final-tail miss exposed an unordered diagnostic D2H; the harness now records
  the terminal graph producer event and consumes it on the explicit observation
  stream before the sole terminal readback. CUDA and ROCm focused gates pass,
  including the exact `262145` device KV count. The CUDA materializer winner is
  32 threads at approximately `3.55 us`, 35 registers, and zero spills. The
  ROCm winner is 32 threads at `2.56 us` median and `2.734 us` pooled mean versus
  `2.72/2.890 us` for 256 threads; ISA evidence shows vectorized dwordx4
  loads/stores, 16 VGPRs, 52 logical SGPRs, and zero scratch, spills, LDS,
  barriers, or atomics.

## CUDA LLEP Economy

- Fixed-d3 control: Qwen3.6-35B-A3B IQ3_S, stochastic sampling, fixed prompt,
  425 prefill + 256 decode tokens, three iterations after one warmup.
- Stable M=4 grouping removed four launches/layer and moved verifier compute
  nodes `1536 -> 1376` per device. Direct FP32 route publication then removed
  one conversion kernel plus one memcpy/layer: compute nodes are now `1336`,
  captured prefill graphs are 80 total nodes/device smaller, and the native hot
  transaction is `1403` kernels including 67 NCCL/control sidecars.
- Runtime expert publication now uses one deterministic two-level warp/wave
  scan for counts, offsets, and active ranks instead of the former quadratic
  per-expert scan and second publication phase. Across M=2..31 the fused plan is
  `1.14x..1.26x` faster on CUDA and `1.16x..1.72x` faster on ROCm; at M=4 it
  moved `12.16 -> 9.76 us` and `30.28 -> 19.43 us`, respectively.
- The canonical routed-plus-shared transport transaction is `1.075x..1.151x`
  faster than the former three-collective transaction for M=2..16 in the
  isolated CUDA2 NCCL harness. Its fixed-order finalizer moved from `12.86` to
  `9.82 us` on CUDA with a 256-column tile and from `19.84` to `15.36 us` on
  ROCm with a 128-column tile. CUDA uses 30 registers/thread, 1.02 KiB shared
  memory, and zero spills; ROCm uses 24 VGPR, 48 SGPR, 512 bytes LDS, and zero
  scratch/spills. Narrower CUDA and wider ROCm alternatives were measured and
  rejected.
- The matched Release CUDA2 fixed-d3 benchmark validates the complete graph
  win: prefill improved `583.34 -> 654.43 tok/s` (`1.122x`) and decode improved
  `172.99 -> 180.43 tok/s` (`1.043x`). Generated token IDs, generated text
  bytes, and `79.25%` stochastic acceptance are identical to the baseline;
  verifier transaction validation failures remain zero. Both participants
  captured one complete prefill graph (`6387/6347` nodes), replayed it without
  recapture or segmentation, and all decode contexts report full-graph replay
  with deferred event completion. The mirrored LocalTP PerfStats records show
  participant-local outcomes with no outcome collective and no host shadow;
  the only D2H boundary is terminal response materialization.
- The M=4 CUDA fused kernel uses 40 registers/thread and 4.23 KiB shared memory,
  has zero spills, and retains 100% theoretical occupancy. The ROCm wave64
  kernel uses 45 VGPR, 89 SGPR, and 4,232 bytes LDS with zero private segment
  and zero spills. The post-change Integration tree rebuilt cleanly, the
  canonical grouped-verifier gate passed `89/89` in 834.31 seconds, and the
  device-free unit/source gate passed `585/585` in 134.99 seconds.
- Isolated M=4 grouping is `5.99 us`: 38 registers/thread, 4.10 KiB shared,
  zero spills, and 100% theoretical per-SM
  occupancy. Achieved whole-GPU occupancy is intentionally low for this single
  dependency block; splitting it would restore launch/dependency overhead.
- Matched decode improved `156.97 -> 164.93 -> 165.92 tok/s`; acceptance stayed
  `61.57%` with zero validation failures. Prefill is `179.51 tok/s`; decode is
  `3.43%` below the historical `171.81 tok/s` promotion target.
- Direct CUDA router publication is 40 registers, zero spills, 100% theoretical
  and 83% achieved occupancy. ROCm is 18 VGPR with no scratch or spills.
- Cooperative CUDA Top-K 40 cut target/draft distribution to `0.238/0.214 ms`;
  it is spill-free and byte exact through M=16.
- Qwen `N=512,K=2048` grouped projection is byte exact across all 21 CUDA
  formats at M=2..31 and gains `1.69x..2.21x`; its eight-CTA geometry remains a target.
- CUDA GDN is byte exact through M=31; at `d_k=d_v=128` it reaches `17.73 us`,
  541 GB/s, and zero spills.
- This RTX 3090 pair has no peer access; mirrored verification avoids its
  measured `941 us/layer` rooted M=5 NCCL collective.
- The post-fix long-context parity harness remains unsuitable as an economy
  result: CUDA2 CurrentBatchLLEP takes `239.50 s` and ROCm2 takes `455.36 s`,
  versus `31.72 s` for the earlier CUDA2 dynamic-maintenance control. The ROCm
  trace attributes roughly `194 s` to CPU-bound model preparation/loading and
  includes one `131.6 s` mapped-download wait plus per-step snapshot downloads.
  The next profile must separate test-only parity observation from the Release
  production graph before attributing those waits to LLEP itself; neither cost
  is accepted as part of the target device-resident inference transaction.

Matched llama.cpp master comparison, tok/s:

| Backend/model | d1 L/LC | d3 L/LC |
|---|---:|---:|
| CUDA dense 27B | `36.24/59.64` | `64.84/60.91` |
| CUDA MoE 35B | `93.01/153.92` | `159.61/171.81` |
| ROCm dense 27B | `20.08/25.15` | `39.28/22.66` |
| ROCm MoE 35B | `46.33/73.93` | `74.94/95.84` |

### Reproducible ROCm1 SingleDevice Reference

The active ROCm SingleDevice tuning control uses the Qwen3.6-35B-A3B MoE
model, a byte-stable 434-token Qwen assistant-generation prompt, stochastic
sampling, and MTP depth 3. Llaminar measures three steady-state graph replays
after one warmup; model loading, arena construction, graph capture, and the
warmup are outside the timing sample. MTP prefill is enabled, so each measured
prefill includes population of both the main and MTP KV caches.

`qwen36_mtp_fixed_chat.txt` is the production-valid benchmark prompt. Its
2,511 bytes have SHA-256
`63d628982074c2785953dfde6f327e4f0146162a5c51396a48b1f59a239a97ae` and
contain the exact Qwen user/assistant wrapper generated from the model's GGUF
chat template. The unwrapped `qwen36_mtp_fixed.txt` remains useful as a raw
tokenizer/continuation diagnostic, but the model closes that incomplete turn
with `<|im_end|>` as its first greedy token. Acceptance and throughput from
that malformed framing are not promotion evidence.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -d rocm:0 --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-rocm1-qwen36-35b-mtp-d3.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed_chat.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --moe-residency-maintenance off
```

The 2026-08-07 post-exact-GDN/grouped-verifier stochastic baseline is
`1301.15 tok/s` prefill and `120.39 tok/s` decode, with `72.27%` draft
acceptance and zero transaction-validation failures. Relative to the active
`1400/125 tok/s` targets, the remaining shortfall is `7.06%` for prefill and
`3.69%` for stochastic decode. On the same prompt at temperature zero,
Llaminar measured `1302.46 tok/s` prefill and `145.36 tok/s` decode with
`82.94%` acceptance.

The temperature-zero correctness matrix compares 256 generated token IDs from
serial and fixed-depth-3 MTP. Llaminar is identical for all 256 tokens on both
the production-valid prompt and the older raw diagnostic prompt. On the same
434-token production-valid input, llama.cpp MTP first diverges from its own
serial decode at generated token 147; it is therefore a useful performance
reference but not a byte-exact grouped-verifier oracle. The two engines' serial
lanes first differ at generated token 26, which is an ordinary cross-engine
kernel-math branch and is independent of Llaminar's internal MTP equivalence.

The focused Release ROCm1 fixed-depth-3 live-server gate passed `19/19`. It
proved repeated captured-prefill replay; beginning, middle, and end needle
recall; strict multi-needle JSON; a non-degenerate 1024-token completion; cache
reset; valid use of `3827/4096` context tokens; oversized-context rejection;
clean shutdown/VRAM release; and `8,405` PerfStats records. The direct grouped
verifier operation-equivalence matrix remains green on both CUDA and ROCm for
M=1..4, the first production transaction/publication, and the M=6
resident-sidecar device target.

The llama.cpp CLI command below intentionally receives the unwrapped source
file because `--conversation` applies the GGUF chat template itself. Its
effective 434-token prompt is byte-equivalent to Llaminar's checked-in
`qwen36_mtp_fixed_chat.txt`; passing the preformatted file here would wrap it
twice.

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  /workspaces/llama.cpp-reference/build-rocm/bin/llama-cli \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 \
  -f /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  -n 256 -c 4096 -ngl all --split-mode none --main-gpu 0 --fit off \
  --flash-attn on --seed 123 --temp 0.8 --top-k 40 --top-p 0.9 --min-p 0 \
  --repeat-penalty 1 --dry-multiplier 0 --perf \
  --conversation --single-turn --no-display-prompt
```

On current llama.cpp master `f9e832c10e94`, three runs measured prefill at
`408.7`, `405.3`, and `403.0 tok/s`, and generation at `78.4`, `77.7`, and
`78.4 tok/s`. The medians are therefore `405.3 tok/s` prefill and
`78.4 tok/s` generation.

### Reproducible Llaminar CUDA2 LLEP Reference

This is the canonical clean-throughput command for the explicit apportioned
LLEP policy tuple. It uses stochastic fixed depth 3 and the same prompt and
sampling controls as the llama.cpp reference below. The 425-token prompt is
below the production `M * top_k >= 8192` LLEP-prefill boundary, so this row
isolates grouped decode and two-device communication economy; use a longer
prompt to measure least-loaded current-batch expert movement itself.

```bash
env \
  LLAMINAR_BENCHMARK_ITERATIONS=3 \
  LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
  LLAMINAR_PREFILL_GRAPH_REQUIRED=1 \
  LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES=1 \
  LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=0 \
  /workspaces/llaminar/build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --context-length 4096 -n 256 \
  --benchmark-json-output /tmp/llaminar-cuda2-llep-d3-clean.json \
  --prompt-file /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  --seed 123 --temperature 0.8 --top-k 40 --top-p 0.9 \
  --mtp --mtp-draft-tokens 3 --mtp-depth-policy fixed \
  --mtp-verify-mode speculative-sampling \
  --mtp-terminal-head-policy mirrored-full-vocabulary \
  --moe-release-raw-expert-weights \
  --moe-residency-maintenance off --moe-hot-expert-cache off \
  --moe-routed-expert-placement tiered-overlay \
  --moe-routed-expert-continuation-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-base-model-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-shared-domain qwen36_moe_cuda_hot \
  --moe-routed-expert-residency static-by-id \
  --moe-continuation-dense-policy tensor-parallel \
  --moe-routed-expert-domain \
    'qwen36_moe_cuda_hot=cuda:0,cuda:1;scope=local;backend=nccl;routed_compute=apportioned;routed_phase=uniform;routed_decode_assignment=static-owner;routed_prefill_assignment=least-loaded-resident;owner=0' \
  --moe-routed-expert-tier \
    'hot@qwen36_moe_cuda_hot;priority=0;max-experts-per-layer=256;memory-mb=8192'
```

The 2026-08-05 post-correctness refresh measured `682.13 tok/s` prefill and
`124.07 tok/s` decode, with `52.89%` stochastic draft acceptance and zero
transaction-validation failures. Both devices captured and replayed one full
prefill graph (`5947/5907` nodes). This is the active economy baseline; the
older `180.43 tok/s` row had `79.25%` acceptance and must not be treated as
current until the acceptance regression is explained.

### Reproducible llama.cpp CUDA Reference

```bash
CUDA_VISIBLE_DEVICES=0 \
  /workspaces/llama.cpp-reference/build-cuda/bin/llama-cli \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 3 \
  -f /workspaces/llaminar/benchmarks/prompts/qwen36_mtp_fixed.txt \
  -n 256 -c 4096 -ngl all --split-mode none --main-gpu 0 --fit off \
  --flash-attn on --seed 123 --temp 0.8 --top-k 40 --top-p 0.9 --min-p 0 \
  --repeat-penalty 1 --dry-multiplier 0 --perf \
  --conversation --single-turn --no-display-prompt
```

On llama.cpp `5788b51`, three runs gave a `155.4 tok/s` decode median and
`1122.5 tok/s` prefill median. Promotion retains the stricter historical
`171.81 tok/s` decode target and requires at least `2245 tok/s` prefill.

## Next Gates

1. Profile the complete canonical CUDA2 d3 graph with `nsys`, then use `ncu` on
   its hottest non-collective kernels. Attribute NCCL primitives, verifier
   compute, prefill expert movement, and tiny control launches separately;
   retain full graph capture and exact output bytes throughout.
2. Close the prefill gap from `654.43 tok/s` to at least llama.cpp's current
   `1122.5 tok/s` median, then pursue the `2245 tok/s` two-times target with
   longer prompts that expose LLEP's theoretical crossover.
3. Keep native geometry for large kernels and fuse adjacent tiny/control work.
   `V2_Perf_CUDAPersistentVerifierGeometry` found a one-kernel CTA proxy `1.25x`
   faster, but full-lane/ALU8 proxies only `0.964x/0.952x`; ncu attributes
   `63-68%` of issue stalls to cooperative barriers. Zero spills.
4. Tune dynamic depth/hysteresis through depth 15 under deterministic prompt
   scenarios without regressing the fixed-d3 `180.43 tok/s` reference.
5. Repeat the economy pass for ROCm and the remaining SingleDevice/EP lanes.
6. After ROCm SingleDevice MTP is green, replace the current MTP KV-only
   prefill lowering with a dedicated captured K/V-only stage. Preserve the
   exact hidden/embedding norms, MTP projection, attention-input norm, K/V
   projection, K norm/RoPE, and cache-format append, while eliminating Q
   projection/gating, Q split/norm/RoPE, Q scratch, and unshiftable or padded
   row work. Promotion requires byte-identical MTP KV payloads across cache
   formats and prefix restore, plus CUDA/ROCm profiler evidence for occupancy,
   registers/VGPRs, zero spills, and end-to-end prefill gain.
