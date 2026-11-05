# Clover MoE: causal autoregressive source contract

`clover_moe` is a distinct registered graph and schema. It implements the causal
AR computation described by [llama.cpp PR #31306](https://github.com/ggml-org/llama.cpp/pull/31306),
pinned for this implementation at `ed26f2f7621306a8c393dd5193c18b3c3cea7940`.
Block diffusion and self-speculative diffusion proposals are not implemented.
An AR timing result must not be described as diffusion decoding performance.

## Ownership and arithmetic

`CloverMoEConfig` validates the original GGUF directory before configuration or
planning. Its immutable geometry supplies both `GraphConfig` and
`ModelMemoryProfile`; the latter contributes to `PhysicalMemoryAuthority`, not a
second capacity ledger. The graph and schema reuse the causal hybrid operators
from Qwen3.5 MoE without changing the source architecture or chat-template identity:

- Periodic causal GDN and full-attention blocks; global block indices survive PP.
- GDN uses causal width-four convolution, tiled value-head groups, normalized Q/K,
  delta-rule recurrence and gated output normalization.
- Full attention uses per-head Q/K normalization, partial NeoX RoPE and output gating.
- Each block uses normalized softmax top-k SwiGLU experts and an optional
  sigmoid-gated shared expert. Routed and shared intermediate widths are separate.
- The inherited participant-local graph retains native CPU/CUDA/ROCm kernel,
  arena, transfer, capture, prefix-state and expert-placement authorities. This
  module adds no kernels, collectives, device loops or alternate execution path.

GGUF norm gains, negative-exponentiated SSM A and tiled GDN value-head ordering
are already converted. Production consumes them directly: do not add another
`+1`, exponential or head permutation. Two-dimensional directory shapes are in
ModelLoader's `[N,K]` order; packed expert tensors retain GGUF's `[K,N,E]` order.
Absent `output.weight` declares a tied embedding/head rather than a missing
architecture implementation.

## Explicit exporter contracts

The official Preview inspected here is
[Diffuspace/Clover-35B-A3B-Preview-GGUF](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview-GGUF/tree/32e22a02b885e173177b7ce7661ad13448b4f89e).
Its directory omits `expert_shared_count` while declaring one shared expert's
width and complete weight family. The AR converter instead writes that count
explicitly. Both encodings are admitted; zero shared experts requires zero or
absent shared width and no shared tensors.

Preview diffusion operands are retained in the source directory, but excluded
from causal materialization and the consumed native-weight inventory:

- `ssm_conv1d.weight_right`
- `ssm_bd_history.bias`
- `ssm_bd_history.weight`
- `ssm_bd_logit_scale.weight`

If present, that family must be complete on every GDN block with the declared
shapes. Unknown tensors, incomplete causal tensors, inconsistent periodic masks,
nonzero learned-nextn depth and unsupported convolution/shared-expert geometry
fail admission. Recognizing Preview diffusion operands does not enable a diffusion
execution mode. Generic request sampling remains explicit; Qwen revision-specific
sampling recommendations and maintained chat templates are not imported by graph reuse.

## Validation boundaries

`V2_Unit_CloverMoEContract` exercises metadata rejection, the actual GGUF parser,
source-versus-consumed inventories, model-context configuration, PP indexing,
registration, weight binding and prepared-weight arithmetic policy. The same
contract is explicitly registered as `V2_Integration_CloverMoEAdmissionContract`
in `ProductionTestPreflight`. Its synthetic sparse source has no numerical model
observations and does not certify inference.

Real-weight prefill/decode parity, captured CUDA/ROCm inference, prefix/reset
behavior, distributed execution and Release benchmarks require separate native
qualification. Existing Qwen parity cells and reference dispatch are not Clover
certificates. Qualification evidence belongs with the PR and published test
results, not with the architecture registration itself.
