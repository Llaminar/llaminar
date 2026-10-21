# vLLM Qwen Flash Next precision review

Source review on 2026-10-20, pinned to vLLM
[`cc8a41f328422e76697431e47993ee839489f572`](https://github.com/vllm-project/vllm/commit/cc8a41f328422e76697431e47993ee839489f572).
This is source and checkpoint-configuration evidence, not a local vLLM
accuracy or performance measurement. It supplements the
[Strata review](STRATA_QFN_PRECISION_REVIEW.md) and the
[QFN implementation plan](QWEN_FLASH_NEXT_IMPLEMENTATION.md).

## The material difference is where quantization is applied

The checked vLLM QFN evaluation selects
`Qwen/Qwen3.8-Flash-Next-FP8`. That checkpoint's configuration, pinned at
[`236dfdf285828023ca3bcd3f37366c58a3469b13`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/236dfdf285828023ca3bcd3f37366c58a3469b13/config.json),
uses BF16 model activations and dynamic FP8 quantization with 128×128 weight
blocks. Its exclusion list contains every main-layer GDN projection, QSA
projection/indexer, hyperconnection projection, router, and shared-expert
projection. A metadata-only audit found these complete exclusions:

| Main-model family | Excluded / expected projections |
|---|---:|
| GDN QKV, Z, alpha, beta and output | 180 / 180 |
| QSA Q, K, V, output and indexer QK | 60 / 60 |
| Routed-expert router | 48 / 48 |
| Shared-expert gate, up and down | 144 / 144 |
| Attention/MoE hyperconnection down, up and injection | 288 / 288 |

The recipe also excludes the terminal mixer, LM head, PLE key/value
projections, and predictor sensitive projections. It explicitly includes the
ngram embedding in conversion. This does not mean every operation in an
“FP8 model” uses eight-bit operands.

vLLM's [FP8 configuration implementation](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/model_executor/layers/quantization/fp8.py#L177)
selects an unquantized linear method for excluded projections. Its QFN
[CUDA hyperconnection](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/models/qwen4_exp/nvidia/hyperconnection.py#L95)
and [ROCm hyperconnection](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/models/qwen4_exp/amd/hyperconnection.py#L91)
also explicitly construct their projections without a quantization method.
Their model constructors select BF16 parameters.

Consequently this recipe is not a matched control for Llaminar's original
GGUF quantized projections with Q8 activation GEMMs. Its accuracy cannot
establish that our additional activation rounding is harmless, nor that our
native quantized kernels are wrong. Retain the original GGUF source bytes in
Llaminar; do not convert quantized weights into another export to imitate this
recipe.

## Router and expert selection

QFN inherits the [Qwen3Next MoE block](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/model_executor/models/qwen3_next.py#L206).
It constructs `GateLinear` without a quantization configuration, without
forcing FP32 computation, and without requesting an FP32 output. It therefore
does **not** establish a universal FP32-router policy. The shared-expert gate
is also unquantized. Shared-expert fusion can combine these gate matrices
through the [MoE runner](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L903).

[GateLinear](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/model_executor/layers/fused_moe/router/gate_linear.py)
has optional FP32-output, FP32-weight, and multi-term BF16 implementations.
Several require SM90/SM100 or gfx950, particular shapes, or an explicitly
requested output dtype. They must not be described as the ordinary QFN
router on RTX 3090 or MI50. Producing FP32 logits also cannot recover
information already lost in an incoming activation.

The standard [softmax/top-k implementation](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/csrc/moe/topk_softmax_kernels.cu#L355)
promotes loaded logits to FP32, computes probabilities and selected-weight
normalization in FP32, and breaks equal-score ties by index. It does not
recompute close expert rankings from a higher-precision upstream hidden
state. A NaN/Inf-to-zero repair in this kernel addresses padded-row safety,
not finite quantization error; it is not a policy to import for Llaminar's
live rows.

Expert arithmetic depends on the selected quantization/backend. The
[Triton expert kernel](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/model_executor/layers/fused_moe/fused_moe.py#L511)
uses FP32 accumulators, applies block scales, and rounds stored intermediates
to its compute dtype. The [FP8 backend policy](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/model_executor/layers/fused_moe/oracle/fp8.py#L738)
distinguishes activation-quantized implementations from Marlin/CPU W8A16.
“FP8” therefore does not prove identical expert operand precision across
hardware either.

## Numerical safeguards worth retaining

Hyperconnection norms, sigmoid gates, branch reductions and residual addition
use FP32 internal arithmetic, with BF16 storage boundaries. The fused
[HC down/SiLU implementation](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/models/qwen4_exp/nvidia/ops/cute_dsl/_hc_down_silu_mma.py#L518)
deliberately rounds the GEMM result to BF16 before SiLU to preserve the
unfused path's semantics. Fusion does not grant permission to silently remove
an arithmetic boundary.

The [GDN prefill preparation](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py#L126)
computes decay and beta in FP32. The
[recurrent implementation](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py#L123)
promotes Q/K/V and state for FP32 arithmetic, applies decay before the
delta correction, then computes the query output from the updated state.
Persistent state precision is configurable; FP32 arithmetic is not proof of
FP32 storage. Llaminar's existing GDN equation, additive-epsilon, state and
serial/grouped regressions remain the relevant checks.

In [CUDA QSA](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/vllm/models/qwen4_exp/nvidia/ops/qsa.py#L196),
score scaling occurs after the QK dot to avoid another low-precision query
rounding. Softmax and output accumulation use FP32; split results and
log-normalizers remain FP32 until merged. With FP8 cache input, the PV path
uses FP16 conversion for its extra mantissa precision. These are specific
precision boundaries, not a general all-FP32 or all-FP16 attention policy.

vLLM disables several shape-dependent optimized paths under
`VLLM_BATCH_INVARIANT`, including its specialized router dispatch and QFN
low-latency GEMM/HC fusion. That is a different concern from agreement with an
FP32 model oracle. Llaminar must continue proving grouped/serial byte
equivalence with its actual production arithmetic.

## What their tests accept, and what they do not prove

The [QSA selection tests](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/tests/models/qwen4_exp/test_qsa_reference.py#L803)
compare selected score multisets for FP8 cases because tied values can select
different indices. SM90 cases additionally permit a 1e-3 numerical tolerance;
the BF16 branch still checks selected indices. This is sparse-attention
selection evidence, not permission to substitute arbitrary MoE experts or
relax Llaminar's deterministic selector.

The checked [QFN evaluation](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/tests/evals/qwen4_exp/README.md)
uses GSM8K and AIME25 with the official FP8 checkpoint. Its configured score
targets are acceptance criteria, not results reproduced here. It does not
provide a same-GGUF, full-vocabulary PyTorch KL baseline. Current vLLM
[GGUF support](https://github.com/vllm-project/vllm/blob/cc8a41f328422e76697431e47993ee839489f572/docs/features/quantization/gguf.md)
lives in a separate plugin; compatibility with this exact multi-shard QFN
export has not been established by this review.

## Decision for Llaminar

The inspected sources contain no general correction that makes quantized
upstream GEMMs preserve FP32 expert choices. They instead protect sensitive
projections, retain precision at selected reductions, and qualify the
resulting model. This supports separating implementation correctness from
the quality cost of a deliberate quantization policy.

Our archived same-input router controls already support that distinction.
For the checked ROCm and CUDA cutoffs, native router probability relative L2
error was approximately 4e-7, while upstream input drift changed probabilities
by about 0.008 and 0.011 relative L2. Replaying the original F32 router on
the actual incoming hidden states reproduced each changed expert selection.
Evidence: `parity-results/qwen-flash-next/router-same-input39/summary.json`.
These two localized controls do not prove every layer or prompt is correct.

Qualification should therefore:

1. Keep independent FP32 same-GGUF checkpoints and report their actual KL.
   Additionally use exact native quantized operands in isolated operator
   oracles to separate quantization error from kernel error.
2. Record expert-set overlap, the top-10/top-11 score margin, local router
   error on identical inputs, and propagated input error. Compare expert
   contributions by expert identity, not slot, when order changes.
3. Treat small finite ranking changes as a possible Q8 quality trade-off only
   after the same-input explanation holds. A large-margin error, wrong
   normalization, stale expert data, or wrong source remains a defect.
4. Judge the final quality/performance menu across multiple prompts, prefill,
   teacher-forced decode and longer contexts. One near-tie swap can cause
   substantial downstream KL; “one expert changed” does not establish a small
   final quality loss.
5. Keep ordinary Q8 GEMMs as the baseline. Measure the existing selective
   QKV/HC/GDN-output residual correction as an explicit alternative, with
   unchanged source weights and measured Release cost. Do not require zero
   FP32-reference route drift as a substitute for a justified quality gate.

No precision default, canary threshold or MTP byte-equivalence requirement is
changed by this review. Source receipts and the metadata exclusion audit are
retained under `/tmp/vllm-qfn-precision-20261020/`; no weights were downloaded
or hashed and no vLLM inference was run.
