# RMSNorm Forensics Instrumentation

This document describes the RMSNorm forensic instrumentation added to `MPITransformerPipeline::executeTransformerLayer` for layer-wise normalization diagnostics and divergence localization between implementations.

## Overview

When enabled, the instrumentation captures per-layer pre/post RMSNorm statistics and (optionally) a reference (recomputed) normalization for both attention pre-norm and feed-forward pre-norm phases (row-major path only—COSMA fused path pending). It reports:

- Pre / post min, max, mean, RMS
- Relative L2 vs locally recomputed reference (using current gamma & epsilon)
- Max absolute difference and worst element coordinates
- Optional row vector previews (first 16 elements) for selected rows

The goal is to quickly localize the earliest layer / substage where RMSNorm output diverges from expectation or shifts distribution in ways that propagate to large logit drift.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLAMINAR_RMS_FORENSICS` | Master enable switch. If unset, zero overhead fast path (just a static env parse once). | disabled |
| `LLAMINAR_RMS_FORENSICS_LAYERS` | Comma / range spec of layers to instrument (e.g. `0,1,5-7`). Empty => all layers. | all |
| `LLAMINAR_RMS_FORENSICS_ROWS` | Comma / range spec of row indices (sequence positions) to preview (prints first 16 cols pre/post[/ref]). | none |
| `LLAMINAR_RMS_FORENSICS_REL_L2_WARN` | Warn threshold for reference rel L2 (default 1e-5). | 1e-5 |
| `LLAMINAR_RMS_FORENSICS_TRACE` | If set, enable row vector previews (used with ROWS spec). | off |
| `LLAMINAR_RMS_FORENSICS_DIFF_ONLY` | Only emit rel L2 + max abs (skip full stats) for lean output. | off |

## Log Formats

### Summary (full mode)
```
[RMSSummary] layer=3 kind=attn rows=128 cols=896 pre_min=-6.125 pre_max=5.8125 pre_mean=0.012345 pre_rms=1.234567 post_min=-8.500000 post_max=7.750000 post_mean=0.000123 post_rms=1.000045 rel_l2_ref=2.3e-06 max_abs_ref=0.000122 worst_row=17 worst_col=512
```
`kind` is `attn` (attention pre-norm) or `ffn` (FFN pre-norm).

### Diff-only Mode
```
[RMSSummary] layer=3 kind=ffn rel_l2_ref=4.6e-06 max_abs_ref=0.000091
```

### Row Preview (when TRACE + ROWS)
```
[RMSForensics][Row] layer=3 tag=attn row=17 pre=0.12,-0.03,... post=0.09,-0.02,... ref=0.09,-0.02,...
```
(Shows first 16 columns; truncated with ellipses here.)

## Usage Examples

### Instrument First Two Layers Only
```
export LLAMINAR_RMS_FORENSICS=1
export LLAMINAR_RMS_FORENSICS_LAYERS=0-1
mpirun -np 2 ./build/test_prefill_attention_golden
```

### Deep Dive with Row Previews
```
export LLAMINAR_RMS_FORENSICS=1
export LLAMINAR_RMS_FORENSICS_LAYERS=0
export LLAMINAR_RMS_FORENSICS_ROWS=0,5,17
export LLAMINAR_RMS_FORENSICS_TRACE=1
mpirun -np 2 ./build/test_prefill_attention_golden
```

### Low-Noise Regression Gate
```
export LLAMINAR_RMS_FORENSICS=1
export LLAMINAR_RMS_FORENSICS_DIFF_ONLY=1
export LLAMINAR_RMS_FORENSICS_REL_L2_WARN=2e-6
mpirun -np 2 ./build/test_prefill_attention_golden
```

## Performance Characteristics

When disabled (default): only a single static environment parse guard executes on the first layer—negligible overhead.
When enabled: Added O(seq_len * hidden_size) passes for stats & reference build (two RMSNorms per layer). This is acceptable for short forensic runs (e.g. seq_len ≤ 512, few layers). Avoid enabling across all layers at large sequence lengths in routine CI.

## Current Limitations / Next Steps

- COSMA fused `fused_rmsnorm_qkv` path not yet instrumented (will require hooks inside `CosmaPrefillManager`).
- No per-row rel L2 breakout (aggregated only). Could add worst-row row-level RMS in future.
- Does not hash gamma / detect drift; can reuse existing gamma checksum env toggles if needed.
- Potential enhancement: capture distribution shift deltas (post - pre mean, scaling factor stats) compactly.

## Failure Triage Strategy
1. Enable for first layer only. Verify `rel_l2_ref` ≈ 0. If not, divergence likely upstream (embeddings / gamma mismatch). 
2. Incrementally extend `LLAMINAR_RMS_FORENSICS_LAYERS` until first layer exhibiting abnormal rel L2 located.
3. Use row previews to inspect anomalous rows (`worst_row`).
4. If only COSMA path diverges, add upcoming fused-path instrumentation (planned) or temporarily force replicated path and recompare.

## Exit Criteria
Instrumentation may be removed or reduced once persistent logits divergence root cause is resolved and a leaner steady-state health metric (e.g. periodic checksum) is adopted.

---
Author: Automated assistant (per project Doxygen guidelines use: David Sanftenberg for code doc blocks)
