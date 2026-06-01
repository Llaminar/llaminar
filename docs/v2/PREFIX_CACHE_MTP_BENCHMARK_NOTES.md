# Prefix Cache And MTP Benchmark Notes

## 2026-06-01 Qwen3.6 Dense ROCm Slice

Hardware/topology:

- Model: `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf`
- Device: `rocm:0`
- Command shape: `./build_v2_release/llaminar2 benchmark -m <model> -d rocm:0 -n 16 ...`
- Benchmark mode: 1 warmup plus 3 measured iterations.
- Prompt length: 595 tokens.
- Decode length: 16 tokens.
- JSON artifacts:
  - `/tmp/llaminar_qwen36_dense_rocm_baseline.json`
  - `/tmp/llaminar_qwen36_dense_rocm_prefix_ram.json`
  - `/tmp/llaminar_qwen36_dense_rocm_mtp.json`
  - `/tmp/llaminar_qwen36_dense_rocm_prefix_ram_mtp.json`

| Scenario | Prefill ms | Decode ms | Total ms | Prefill tok/s | Decode tok/s | Overall tok/s | Key counters |
|----------|------------|-----------|----------|---------------|--------------|---------------|--------------|
| Baseline | 2619.80 | 862.65 | 3482.46 | 227.12 | 18.55 | 175.45 | prefix disabled, MTP disabled |
| RAM prefix cache | 84.93 | 865.65 | 950.59 | 7005.54 | 18.48 | 642.76 | 50 hits, 2975 matched tokens, 5 terminal hits, 340.20 MiB RAM |
| MTP greedy | 4275.88 | 8156.03 | 12431.92 | 139.15 | 1.96 | 49.15 | 32 draft steps, 64 accepted, 32 rejected, 32 rollbacks, 66.67% acceptance |
| RAM prefix cache plus MTP | 89.95 | 8198.35 | 8288.30 | 6614.57 | 1.95 | 73.72 | 50 hits, 2975 matched tokens, 5 terminal hits, 2.50 MiB MTP state, 66.67% MTP acceptance |

Observations:

- RAM prefix cache is performance-ready for this dense ROCm prompt class. Prefill improved from 2619.80 ms to 84.93 ms, a 30.8x prefill speedup, and total request time improved from 3482.46 ms to 950.59 ms, a 3.7x total speedup.
- Prefix counters explain the speedup: measured iterations restored 595-token full hits from RAM and restored terminal logits on the hit path.
- MTP greedy is correctness-ready but not performance-ready on this setup. Decode throughput regressed from 18.55 tok/s to about 1.96 tok/s even with 66.67% accepted draft tokens.
- Prefix plus MTP preserves the prefix-cache prefill speedup, but total request time is dominated by the current MTP decode overhead.

Follow-up blockers before MTP default enablement:

- Investigate why MTP sidecar execution is recorded under the prefill stage timeline and costs about 365 ms per small sidecar step on ROCm.
- Reduce MTP sidecar launch/graph overhead and verify the sidecar uses captured or fused graph paths where possible.
- Reduce rollback frequency or verifier replay cost. The trace shows 32 rollbacks for 32 draft steps, so accepted draft tokens are not translating into decode throughput.
- Keep MTP disabled by default until decode tok/s is faster than the baseline for the same model/backend/topology.

Telemetry fix landed with this benchmark slice:

- Benchmark JSON now computes top-level `mtp.acceptance_rate` as `accepted_tokens / (accepted_tokens + rejected_tokens)`, matching the per-request summary and keeping the value bounded between 0 and 1 when each draft step proposes multiple tokens.
