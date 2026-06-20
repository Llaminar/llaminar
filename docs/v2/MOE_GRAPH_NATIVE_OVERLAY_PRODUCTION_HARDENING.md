# Graph-Native MoE Overlay Production Hardening

This note captures the production checks added around graph-native Qwen3.5 MoE overlay placement. The production path is whole-expert expert parallelism: routed tiers own complete experts, sparse dispatch moves compact routed rows, sparse return/reduce scatters compact expert outputs, and there is no shadow LocalTP expert runtime.

## Placement Explanation

Use `--explain-placement --dry-run` to inspect a graph-native MoE overlay config without loading model weights:

```bash
./build_v2_release/llaminar2 oneshot \
  --config configs/moe_overlay/cuda_hot_rocm_warm_cpu_cold_static.yaml \
  --explain-placement --dry-run \
  -m /path/to/qwen35moe.gguf
```

The MoE overlay block reports:

- execution kind, residency policy, and the graph-native whole-expert ownership contract;
- continuation, base, and shared expert domains;
- dense and routed domain participants, backend, compute kind, owner, and rank hints;
- routed tiers with priority, capacity, memory budget, and fallback status;
- total non-fallback routed capacity and fallback domains;
- whether CPU fallback exists;
- coverage status when the routed expert count is known, or a note that coverage is validated during model-aware resolution.

Common production validation failures are intentionally explicit:

- routed `TensorParallelExperts` domains are rejected for graph-native routed tiers because no shadow LocalTP runtime exists;
- no-fallback plans must cover all routed experts when the model expert count is known;
- continuation, base, shared, and routed tier domains must reference declared execution domains;
- at most one fallback tier is allowed;
- overlay domain specs must include `scope=` and `compute=` hints, and multi-participant LocalTP domains need deterministic owner/rank hints before execution planning.

## Transfer Tracing For Decode

One-token decode should not transfer full hidden-state tensors between host and device as part of normal overlay routing. Expected MoE overlay movement is compact sparse payloads: row ids, entry offsets, expert ids, route weights, selected hidden rows, and compact return rows.

To catch accidental full dense device-to-host transfers, run a short decode with transfer tracing focused on large D2H movement:

```bash
LLAMINAR_TRACE_TRANSFERS=1 \
LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1 \
LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=1048576 \
./build_v2_release/llaminar2 oneshot \
  --config configs/moe_overlay/cuda_hot_rocm_warm_cpu_cold_static.yaml \
  -m /path/to/qwen35moe.gguf \
  -p "Sanity check" -n 4 -t 0
```

Use `LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=262144` for a stricter pass on smaller models. Leave the threshold high enough that compact sparse payloads do not dominate the log; for Qwen3.5 decode, unexpected dense hidden-state or logits transfers are the signal.

When `LLAMINAR_PROFILING=1` is enabled, MoE overlay profiler counters should agree with the tracing story:

- `compact_dispatch_bytes` and `compact_return_bytes` should scale with routed sparse rows, not full sequence hidden-state volume;
- `dense_bytes_avoided` should be positive for sparse tiers;
- `cpu_fallback_rows` should only appear for configured fallback tiers;
- `gpu_cached_rows` should rise as hot GPU tier capacity increases.

If transfer tracing shows large repeated D2H transfers during one-token decode, inspect recent stage logging, tensor validation, and debugging code for accidental `TensorBase::data()` calls on device-authoritative activations.