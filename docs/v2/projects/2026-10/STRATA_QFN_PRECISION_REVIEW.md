# Strata Qwen Flash Next precision review

Source review on 2026-10-20, for the active QFN numerical investigation.
Repository: [Niko1221/Strata](https://github.com/Niko1221/Strata), pinned at
[`9ae93f7b1fe989d04a8e404d32cddaeeff10505f`](https://github.com/Niko1221/Strata/commit/9ae93f7b1fe989d04a8e404d32cddaeeff10505f).
This is source evidence, not a locally reproduced Strata benchmark or parity run.
The relevant local hardware is RTX 3090/SM86 and MI50/gfx906; newer-card results
do not qualify those devices.

## What arithmetic they actually use

The `--native` QFN preset enables the native dense, floating projection, GDN,
router and sparse-attention paths together. A default-false internal flag is
therefore not enough to determine the serving policy. The preset assignment is
in [generate.cpp](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/program/generate.cpp#L3161).

| Operation | Native decode | Batched prefill |
|---|---|---|
| Quantized GDN QKV, gate and output projections | Original GGUF blocks, Q8_1 activation blocks and integer dot products | GGUF weights decoded to FP16, FP16 inputs, FP32 GEMM accumulation/output on the ordinary CUDA path |
| Router, GDN alpha/beta, indexer and other BF16-weight projections | BF16 weights widened for FP32 input multiplication and FP32 accumulation | BF16 inputs/weights and FP32 accumulation, with an optional residual correction |
| GDN recurrent state | FP32 storage and FP32 recurrence/readout arithmetic | FP32 recurrent state; multiple recurrence implementations have different reduction layouts |

The quantized decode and floating projection dispatches are visible in
[layer.cpp](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/core/layer.cpp#L165).
The dense prefill conversion and FP32-accumulating BLAS calls are in
[gemm.cu](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/prefill/gemm.cu#L1231).
The fused GDN alpha/beta and recurrence kernels retain floating inputs and
state in [fused_gdn.cu](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/kernels/cuda/fused_gdn.cu#L19).
Consequently their prefill and decode are not one universal arithmetic policy.
HIP also has explicit alternative prompt arithmetic; its availability is not
evidence of qualification on gfx906.

For ordinary BF16 matvecs, Strata preserves ordered FP32 FMAs and a fixed
reduction tree adapted from llama.cpp. Its multi-row implementation retains an
independent accumulator per token with the single-row ordering. That is a
useful implementation example for grouped/serial equivalence, rather than a
reason to change Llaminar's own arithmetic contract.
[Source](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/kernels/cuda/native_bf16.cu#L44).

## Their targeted precision correction

`STRATA_PREFILL_BF16X2` implements:

```text
x_hi = BF16(x)
x_lo = BF16(x - FP32(x_hi))
y    = W * x_hi + W * x_lo
```

The low-part conversion is in
[prefill/kernels.cu](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/prefill/kernels.cu#L36).
Two GEMMs use the same weights; the second accumulates into the first result.
Mode 1 includes hyper-connections; mode 2 excludes those wider projections.
The default is zero. The optional HIP FP16-I/O prompt mode disables this BF16
correction. The source identifies activation rounding and router consistency
with decode as its motivation; the reviewed evidence does not establish its
end-to-end KL/performance benefit on our devices.
[Policy and implementation](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/prefill/prefill.cpp#L333).

This is the same residual-correction principle as Llaminar's existing
TwoTermQ8 diagnostic, with a different operand representation. It supports
testing selective correction, not replacing all Q8 GEMMs with floating-point
math or assuming that a smaller local error always lowers final model KL.

## Weight identity needs care

Strata's pack converter requires BF16 for several sensitive projections,
including hyper-connections, router, indexer and alpha/beta. An F32 tensor
already containing exact BF16 values can be narrowed losslessly. Other types
require the explicit `--compat-bf16` conversion, which can dequantize an
already-quantized tensor and round its values to BF16. It records that
conversion. This differs from evaluating the exact original GGUF values.
[Converter contract](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/tools/iq_pack.py#L63).

Llaminar retains its original-weight rule. Copying that conversion would
confound our same-GGUF comparison. A floating activation experiment, if needed,
must read the original quantized weight values without introducing a new
weight quantization. The current Q8 residual candidate already preserves them.

## Outlier handling addresses a different failure mode

Strata caps Q8_1's FP16 scale/sum metadata at the largest finite half and clamps
the corresponding integer codes. The purpose is to prevent large activations
from producing infinities and then NaNs in dot products. The helper preserves
NaNs for diagnosis; clipping is explicitly approximate, not recovery of the
original product.
[Q8_1 finite-range helper](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/include/strata/kernels/q8_1_finite.hpp).

It also saturates FP16 activation images, including large SwiGLU outputs, with
optional HIP range observations in
[prefill/kernels.cu](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/src/prefill/kernels.cu#L46).
This is not evidence of an outlier decomposition that restores discarded
information.

A read-only check of our archived layer-0 QKV inputs, all nine rows on each of
CPU/CUDA/ROCm, finds zero nonfinite values. Maximum absolute input is below
7.487, maximum absolute 32-value sum below 16.616, and maximum scale below
0.059, far inside FP16's 65504 limit. Therefore overflow clipping does not
explain the isolated QKV discrepancy. This observation covers those inputs,
not every layer or a long generation. Evidence:
`parity-results/qwen-flash-next/strata-precision-review45/qkv-range-check.json`.

## Their KL numbers are not our parity score

The QFN quant-quality tool compares a quantized Strata run with a Q8_0 Strata
run. Its KL uses the reference's top 64 tokens, renormalizes that subset and
substitutes the candidate's 64th log-probability for missing tokens. This is
an approximate restricted-support metric, not full-vocabulary KL against our
same-GGUF PyTorch reference. Its fixed controls disable expert adaptation,
tail skipping and prefix reuse, and select FP16 KV.
[Measurement implementation](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/tools/quant_quality.py#L1).

Other published measurements compare an optimization against Strata's own
previous/default configuration. For example, its Ampere table reports about
0.0018 KL for optional HC Q8 conversion against the existing engine. The
tail-skipping evaluation reports means around 0.003–0.006 and maxima around
0.3 for particular workloads. These are useful sensitivity measurements with
different baselines. Tail skipping changes which expert contributions are
computed and is not a numerical repair to adopt here.
[Published conditions and results](https://github.com/Niko1221/Strata/blob/9ae93f7b1fe989d04a8e404d32cddaeeff10505f/docs/DETAILS.md#cuda-opt-in-switches-measured-for-0145-rtx-3060-and-rtx-a4000-sm_86-all-off-unless-you-set-them).

## Consequences for the Llaminar investigation

Our same-input QKV evidence already isolates activation quantization: local
relative L2 falls from approximately 0.00221 to 0.0000082–0.0000087 with the
existing two-term Q8 operation. The residual native dot/reduction discrepancy
is approximately 1.2–1.8e-7. Merely changing the accumulator cannot recover
activation information discarded before the dot product.

The completed build45b matched controls also improve the worst of three
teacher-forced decode KL values, without passing the numerical gates:

| Backend | Single Q8 worst decode KL | QKV-only TwoTermQ8 worst decode KL |
|---|---:|---:|
| CPU | 1.07722 | 0.487234 |
| CUDA+CPU | 1.57389 | 0.361550 |
| ROCm+CPU | 0.461583 | 0.311559 |

All six cases complete their canonical artifacts with clean driver windows,
but all remain numerical red. Individual decode positions do not improve
monotonically. These are Integration correctness measurements, not Release
performance results. Complete prefill/decode metrics are in
`parity-results/qwen-flash-next/model45b-qkv-matched-six/{cpu,gpu}-summary.json`.

The next numerical comparison should combine QKV correction with the already
implemented HC/GDN-output correction, then attribute the first changed expert
selection and preceding residual drift through the canonical stage CSVs.
Keep unmatched expert-aligned diagnostic rows visible when selection differs;
do not pretend they form an identical paired inventory. Include original
alpha/beta as a separate control where its observed operand rounding matters.
Preserve ordinary Q8 expert GEMMs, original quantized weights, unchanged
full-vocabulary gates, fixed teacher-forced inputs and matched placement.

Qualification still requires multiple prompts/positions and contexts, all
three backends, original-weight same-input oracles, grouped/serial byte
equivalence, prefix restoration and MTP rollback. Measure component and full
model Release cost before promoting any correction. Strata's source provides
candidate mechanisms and useful controls; it does not establish that our
remaining KL spikes are acceptable or solved.
