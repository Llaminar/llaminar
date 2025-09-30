# Quantized Golden Harness

This document describes the extended golden model loader quantized parity test (`test_model_loader_golden`, test case `CompareQuantizedTensorsWithLlama`). It validates that Llaminar's dequantization of GGUF quantized tensors exactly matches the embedded `llama.cpp` reference implementation.

## Purpose

1. Guard against silent corruption / mis-decoding of quantized tensors during model load.
2. Provide breadth‑first sampling across quant formats present in a model while keeping CI runtime bounded.
3. Offer actionable diagnostics (skip reasons + recommendations) when formats are unsampled due to size limits.
4. Allow optional partial (strided) sampling for extremely large single tensors (e.g. `output.weight` q8_0) without paying the full memory/time cost.

## Supported Quant Formats (raw block decode path)

- q2_K (Q2_K)
- q3_K (Q3_K)
- q4_0 (Q4_0)
- q4_1 (Q4_1)
- q5_0 (Q5_0)
- q5_1 (Q5_1)
- q5_K (Q5_K)
- q6_K (Q6_K)
- q8_0 (Q8_0)

(Other formats appear as `unsupported_format` in skip reasons.)

## Output Tags

| Tag | Meaning |
|-----|---------|
| `[QuantTensorCompare]` | Full tensor comparison; shows exact parity stats. |
| `[QuantTensorPartialCompare]` | Strided subset comparison for oversized tensor (partial sampling enabled). |
| `[QuantTensorSkip]` | A tensor not compared (reason enumerated). |
| `[QuantSummary]` | Aggregated summary of the run. |
| `[QuantRecommend]` | Recommendation to raise element cap for an unsampled type. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMINAR_GOLDEN_Q_ENABLE` | unset | Enable the quantized golden test. |
| `LLAMINAR_GOLDEN_Q_MAX_TENSORS` | 4 | Upper bound on number of (full or partial) tensor comparisons. |
| `LLAMINAR_GOLDEN_Q_PER_TYPE` | 1 | Breadth‑first target per quant type (once reached, further tensors of that type are deferred). |
| `LLAMINAR_GOLDEN_Q_MAX_ELEMS` | 500000 | Element cap beyond which a tensor is considered `too_large` (unless partial sampling enabled). |
| `LLAMINAR_GOLDEN_Q_RL2_TOL` | 1e-5 | Relative L2 tolerance (should usually be 0). |
| `LLAMINAR_GOLDEN_Q_DEBUG` | unset | If set (any value), prints debug info and recommendations. |
| `LLAMINAR_GOLDEN_Q_ALLOW_PARTIAL` | unset | If set, enables partial sampling path for oversized tensors. |
| `LLAMINAR_GOLDEN_Q_PARTIAL_ELEMS` | (inherits `MAX_ELEMS`) | Target number of sampled elements for partial comparison. |

## Skip Reasons

| Reason | Explanation |
|--------|-------------|
| `unsupported_format` | Quant type not implemented in raw block decode path or not supported by loader. |
| `too_large` | Tensor exceeded `MAX_ELEMS` and partial sampling disabled or not applicable. |
| `loader_fail` | Loader failed to produce a float buffer. |
| `no_ref` | Missing corresponding llama.cpp ggml tensor (naming mismatch). |
| `ref_unsupported` | Reference dequant path not implemented for that type. |
| `size_mismatch` / `ref_size_mismatch` | Inconsistent element counts between implementations. |
| `zero_elems` | Degenerate tensor with zero elements. |

## Partial Sampling Strategy

When `LLAMINAR_GOLDEN_Q_ALLOW_PARTIAL` is set and an oversized tensor (n > `MAX_ELEMS`) still needs coverage for its quant type quota:

1. Entire tensor is dequantized (both sides) once (opt-in due to memory/time cost).
2. A uniform stride is computed: `stride = max(1, n / sample_n)` where `sample_n = min(PARTIAL_ELEMS, n)`.
3. Strided slices form two sampled vectors compared with the usual parity metrics.
4. Output tag: `[QuantTensorPartialCompare]` shows both total and sampled cardinalities.

## Recommendations

If a quant type has only `too_large` skips and zero comparisons, a debug recommendation is emitted:
```
[QuantRecommend] type=q8_0 first_too_large_n=136134656 suggest_LLAMINAR_GOLDEN_Q_MAX_ELEMS>=136134656 to enable sampling
```
Raise `LLAMINAR_GOLDEN_Q_MAX_ELEMS` (or enable partial sampling) to capture it.

## Typical Workflows

Minimal parity smoke test:
```
LLAMINAR_GOLDEN_Q_ENABLE=1 ctest -R ModelLoaderGolden --output-on-failure
```

Force one example of each quant type (if small enough):
```
LLAMINAR_GOLDEN_Q_ENABLE=1 \
  LLAMINAR_GOLDEN_Q_MAX_TENSORS=16 \
  LLAMINAR_GOLDEN_Q_PER_TYPE=2 \
  ctest -R CompareQuantizedTensorsWithLlama --output-on-failure
```

Include partial sampling for giant head / embedding matrices:
```
LLAMINAR_GOLDEN_Q_ENABLE=1 \
  LLAMINAR_GOLDEN_Q_ALLOW_PARTIAL=1 \
  LLAMINAR_GOLDEN_Q_PARTIAL_ELEMS=2000000 \
  LLAMINAR_GOLDEN_Q_MAX_ELEMS=500000 \
  ctest -R CompareQuantizedTensorsWithLlama --output-on-failure
```

Show recommendations (debug mode):
```
LLAMINAR_GOLDEN_Q_ENABLE=1 LLAMINAR_GOLDEN_Q_DEBUG=1 ctest -R CompareQuantizedTensorsWithLlama
```

## Interpreting rel_l2

For identical dequant implementations, `rel_l2` and `max_abs` should be exactly 0. Any non‑zero discrepancy indicates a potential divergence in block decode logic or data corruption and should fail the test unless the tolerance was explicitly relaxed.

## Maintenance Notes

- When adding support for a new quant format, extend the `llama_dequant_tensor` switch and ensure `tensor_type_name()` returns a stable string.
- Keep the partial sampling path lightweight: avoid duplicating large allocations if future optimization introduces reusable buffers.
- Consider adding a second multi-pass phase if breadth-first quotas routinely leave some desired types unsampled (currently deferred to keep test simple).

## Future Enhancements (Backlog)

- Multi-pass backfill for unsatisfied per-type quotas.
- Hash-based sampling for deterministic representative subset across runs.
- Optional per-block validation mode (spot-checking raw block fields).
- Integration with RMSNorm / layer-wise forensic harness for unified reporting.

---
This harness prioritizes fast detection of quant load regressions while keeping CI costs controlled via explicit environment gating.
