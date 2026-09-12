# Production parity: Qwen2 CPU Q16 Top-5 boundary

## September 9 evidence

After the participant-local workspace correction, the unseen-only queue has
101/510 individual greens. Its next red is
`Qwen2_Q4_0_CPU0_ActFP32_KVQ16_1_MTPOff`. All eight canonical CSVs exist; all
24 layer rollups, incremental-decode checks and prefix-restore checks pass.
The sole assertion is prefill Top-5 overlap 80% versus the declared 95%.
Prefill cosine is 0.998758, KL 0.00187654 (limit 0.006), and Top-1 matches.

The model filename names Q4_0 overall, but its actual terminal `output.weight`
is Q8_0. Model activations remain FP32. NativeVNNI internally quantizes its
GEMM operand into transient Q8 blocks; this is independent of KV precision.
No weight, activation, cache precision, or threshold has been changed.

| Token | Original HF logit | FP32 equation on native hidden state | Native logit |
|---|---:|---:|---:|
| 345 | 15.899855 | 15.775319 | 15.757833 |
| 323 | 15.703446 | 15.758971 | 15.773873 |

Those two tokens exchange fifth and sixth position. The leading four tokens
remain identical. The HF fifth/sixth margin is 0.196408; after ordinary upstream
arithmetic drift the exact projection's margin is only 0.016348. Internal Q8
operand rounding then reverses their order.

## Independent arithmetic replay

Saved inputs/outputs come from the existing stage dumper, not a modified
inference path. Every prefill attention stage was replayed with FP64 QK,
causal softmax and PV using its exact saved operands. Worst relative L2 is
8.2743e-6 (layer zero); later layers are below 2.47e-6.

The exact GGUF output matrix is independently decoded through the Python
reference project. PyTorch FP32 projection on native hidden state differs from
production by relative L2 0.0111303, as expected when it omits the internal Q8
operand quantization. Applying the production quantization equation
independently before that projection reproduces native output with relative L2
2.2710e-7 and maximum absolute error 5.2452e-6, including the fifth/sixth swap.
This localizes the decisive swap to quantization error, not incorrect terminal
GEMM arithmetic. It is not a claim that every upstream kernel has been
individually replayed.

The 95% Top-5 requirement is effectively an exact five-of-five gate for one
terminal row. Permitting four-of-five would require an explicit threshold
decision; no such change is installed. The existing independent
route-conditioned MoE proof does not apply to vocabulary Top-5 validation.

Local evidence: `parity-results/ci-local-unseen-pass-02/report.json`,
`ci-qwen2-q16-stage-diagnostic-02/report.json`, `ci-qwen2-q16-stages/`, and
`diagnose-qwen2-q16.py` / `ci-qwen2-q16-independent-equations.log`. Early
diagnostic attempts used an unused snapshot switch and then an incorrect glob
filter; neither produced the final evidence. The installed stage-name filter
is substring-based. No previously green cell was rerun during diagnosis.
