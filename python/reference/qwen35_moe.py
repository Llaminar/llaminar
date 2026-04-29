"""
Qwen 3.5 MoE (Mixture-of-Experts) Reference Implementation

PyTorch reference implementation for Qwen 3.5 MoE models using HuggingFace transformers.
Captures intermediate pipeline states for parity testing with Llaminar.

Architecture:
  - Same hybrid 3:1 GDN+FA attention layout as dense Qwen3.5
  - FFN replaced with SparseMoeBlock: router + 256 experts (top-8) + shared expert
  - MoE block: gate(router) → experts(routed) + shared_expert * sigmoid(shared_expert_gate)
  - Expert weights in GGUF are 3D packed [num_experts, intermediate, hidden]
  - GGUF stores gate_exps and up_exps separately; HF expects fused gate_up_proj

GGUF specifics (shared with dense Qwen3.5):
  - Norm weights use pre_rmsnorm_1p convention (GGUF stores w+1, subtract 1 on load)
  - in_proj_qkv, in_proj_z, in_proj_a, in_proj_b stored separately
  - ssm_a stores -exp(A_log), reversed via log(-x) on load
  - conv1d.weight squeezed to 2D in GGUF, unsqueezed back on load

MoE-specific GGUF tensors:
  - ffn_gate_inp.weight → router weights
  - ffn_gate_exps.weight, ffn_up_exps.weight → fused into gate_up_proj [num_experts, 2*intermediate, hidden]
  - ffn_down_exps.weight → down_proj [num_experts, hidden, intermediate]
  - ffn_gate_shexp.weight, ffn_up_shexp.weight, ffn_down_shexp.weight → shared expert
  - ffn_gate_inp_shexp.weight → shared expert gate

@author David Sanftenberg
"""

import re
from typing import Optional

import torch

from .base import HuggingFaceReferenceModel
from .pipeline_stages import PipelineStage
from .registry import ModelRegistry


class Qwen35MoEReferenceModel(HuggingFaceReferenceModel):
    """
    PyTorch reference implementation for Qwen 3.5 MoE models.

    Inherits shared GGUF loading and forward pass from HuggingFaceReferenceModel.
    Overrides:
      - _create_model_from_gguf_config: builds Qwen3_5MoeTextConfig + Qwen3_5MoeForCausalLM
      - _load_from_gguf: fuses GGUF's separate gate_exps+up_exps into HF's gate_up_proj
      - _register_hooks: adds MoE-specific hooks (router, experts, shared expert)
    """

    def _create_model_from_gguf_config(
        self,
        config_dict: dict,
        torch_dtype: Optional[torch.dtype],
    ) -> tuple:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextConfig,
            Qwen3_5MoeForCausalLM,
        )

        hf_config = self._build_hf_config(config_dict)

        if torch_dtype:
            hf_config.torch_dtype = torch_dtype

        model = Qwen3_5MoeForCausalLM(hf_config)
        return hf_config, model

    @staticmethod
    def _build_hf_config(config_dict: dict):
        """Convert GGUF config_dict to Qwen3_5MoeTextConfig parameters."""
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextConfig

        full_attn_interval = config_dict.get('full_attention_interval', 4)
        n_layers = config_dict.get('num_hidden_layers', 40)

        # Build layer_types list
        layer_types = []
        for i in range(n_layers):
            if (i + 1) % full_attn_interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")

        # Linear attention head counts
        linear_num_key_heads = config_dict.get('linear_num_key_heads', 16)
        linear_value_head_dim = config_dict.get('linear_value_head_dim', 128)
        ssm_inner_size = config_dict.get('ssm_inner_size', None)

        if ssm_inner_size is not None and linear_value_head_dim > 0:
            linear_num_value_heads = ssm_inner_size // linear_value_head_dim
        else:
            linear_num_value_heads = linear_num_key_heads

        linear_key_head_dim = config_dict.get('linear_key_head_dim', linear_value_head_dim)

        # MoE specific fields
        num_experts = config_dict.get('num_experts', 256)
        num_experts_per_tok = config_dict.get('num_experts_per_tok', 8)
        moe_intermediate_size = config_dict.get('moe_intermediate_size', 512)
        shared_expert_intermediate_size = config_dict.get('shared_expert_intermediate_size', 512)

        # RoPE parameters — Qwen3_5MoeTextConfig uses rope_parameters dict, not rope_theta
        rope_theta = config_dict.get('rope_theta', 10000000.0)
        rope_dimension_sections = config_dict.get('rope_dimension_sections', None)
        # mrope_section from rope_dimension_sections (e.g. [11, 11, 10] for head_dim=256)
        mrope_section = rope_dimension_sections if rope_dimension_sections else [11, 11, 10]
        rope_parameters = {
            'rope_type': 'default',
            'rope_theta': rope_theta,
            'mrope_interleaved': True,
            'mrope_section': mrope_section,
            'partial_rotary_factor': 0.25,
        }

        cfg = Qwen3_5MoeTextConfig(
            hidden_size=config_dict.get('hidden_size', 2048),
            num_hidden_layers=n_layers,
            num_attention_heads=config_dict.get('num_attention_heads', 16),
            num_key_value_heads=config_dict.get('num_key_value_heads', 2),
            head_dim=config_dict.get('head_dim', 256),
            max_position_embeddings=config_dict.get('max_position_embeddings', 262144),
            rms_norm_eps=config_dict.get('rms_norm_eps', 1e-6),
            rope_parameters=rope_parameters,
            vocab_size=config_dict.get('vocab_size', 248320),
            layer_types=layer_types,
            linear_key_head_dim=linear_key_head_dim or 128,
            linear_value_head_dim=config_dict.get('linear_value_head_dim', 128),
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_conv_kernel_dim=config_dict.get('linear_conv_kernel_dim', 4),
            # MoE fields
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )
        cfg._attn_implementation = "eager"
        return cfg

    def _tokenizer_fallbacks(self) -> list[str]:
        return ["Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B-Instruct"]

    # ------------------------------------------------------------------
    # GGUF loading with MoE expert tensor fusion
    # ------------------------------------------------------------------

    def _load_from_gguf(self, gguf_path: str, torch_dtype=None, **kwargs) -> None:
        """Load GGUF with MoE expert gate+up fusion into gate_up_proj."""
        import warnings
        from .loaders import GGUFLoader
        from transformers.initialization import no_init_weights

        print(f"Loading GGUF file: {gguf_path}")
        loader = GGUFLoader(gguf_path, verbose=self.verbose)
        config_dict, state_dict = loader.load(
            as_transformers_config=False,
            as_torch=True,
            show_progress=self.verbose,
        )

        # Fuse expert gate+up tensors: GGUF has separate gate_proj and up_proj,
        # HF expects a single gate_up_proj = cat(gate, up, dim=1)
        state_dict = self._fuse_expert_gate_up(state_dict)

        # Fix shared_expert_gate shape: GGUF stores as [hidden_size] (1D),
        # HF nn.Linear(hidden_size, 1) expects [1, hidden_size] (2D)
        for key in list(state_dict.keys()):
            if 'shared_expert_gate.weight' in key and state_dict[key].dim() == 1:
                state_dict[key] = state_dict[key].unsqueeze(0)

        with no_init_weights():
            self.hf_config, self.hf_model = self._create_model_from_gguf_config(
                config_dict, torch_dtype
            )

        print(f"Loading {len(state_dict)} tensors into model...")
        missing, unexpected = self.hf_model.load_state_dict(state_dict, strict=False)

        if "lm_head.weight" in missing or missing == ["lm_head.weight"]:
            print("Tying lm_head.weight to model.embed_tokens.weight (weight sharing)")
            self.hf_model.lm_head.weight = self.hf_model.model.embed_tokens.weight

        if missing and missing != ["lm_head.weight"]:
            warnings.warn(f"Missing keys when loading GGUF: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys when loading GGUF: {unexpected}")

        self.hf_model = self.hf_model.to(self.device)
        self.hf_model.eval()

        self._resolve_tokenizer(gguf_path)
        print("✓ GGUF MoE model loaded successfully")

    @staticmethod
    def _fuse_expert_gate_up(state_dict: dict) -> dict:
        """
        Fuse separate gate_proj and up_proj expert tensors into gate_up_proj.

        GGUF stores:
          model.layers.{i}.mlp.experts.gate_proj.weight  [num_experts, intermediate, hidden]
          model.layers.{i}.mlp.experts.up_proj.weight    [num_experts, intermediate, hidden]

        HF expects:
          model.layers.{i}.mlp.experts.gate_up_proj      [num_experts, 2*intermediate, hidden]
        """
        gate_pattern = re.compile(
            r'(model\.layers\.\d+\.mlp\.experts)\.gate_proj\.weight'
        )

        keys_to_remove = []
        keys_to_add = {}

        for key in list(state_dict.keys()):
            m = gate_pattern.match(key)
            if m:
                prefix = m.group(1)
                up_key = f"{prefix}.up_proj.weight"
                fused_key = f"{prefix}.gate_up_proj"

                if up_key in state_dict:
                    gate = state_dict[key]
                    up = state_dict[up_key]
                    # gate: [num_experts, intermediate, hidden]
                    # up:   [num_experts, intermediate, hidden]
                    # fused: [num_experts, 2*intermediate, hidden]
                    fused = torch.cat([gate, up], dim=1)
                    keys_to_add[fused_key] = fused
                    keys_to_remove.extend([key, up_key])

        for k in keys_to_remove:
            del state_dict[k]
        state_dict.update(keys_to_add)

        if keys_to_add:
            print(f"  Fused {len(keys_to_add)} expert gate+up tensors into gate_up_proj")

        return state_dict

    # ------------------------------------------------------------------
    # Hook registration for MoE layers
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """
        Register forward hooks for Qwen 3.5 MoE with heterogeneous layer types.

        Attention hooks are identical to dense Qwen3.5. FFN hooks are replaced
        with MoE-specific hooks on the SparseMoeBlock submodules:
          - mlp.gate (router) → MOE_ROUTER_OUTPUT
          - mlp.experts → MOE_EXPERT_OUTPUT
          - mlp.shared_expert → MOE_SHARED_EXPERT_OUTPUT
          - mlp.shared_expert_gate → MOE_SHARED_GATE_OUTPUT
          - mlp → MOE_COMBINED_OUTPUT (replaces FFN_DOWN)
        """
        if not self.hf_model:
            return

        model = self.hf_model.model

        # Embedding
        def _emb(module, inp, out):
            if self._should_capture(PipelineStage.EMBEDDING):
                self.capture_stage(PipelineStage.EMBEDDING, out, layer_idx=-1)
        self._hook_handles.append(model.embed_tokens.register_forward_hook(_emb))

        # Per-layer hooks
        for idx, layer in enumerate(model.layers):
            # Input layernorm (shared by both layer types)
            def _attn_norm(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.ATTENTION_NORM):
                    self.capture_stage(PipelineStage.ATTENTION_NORM, out, i)
            self._hook_handles.append(
                layer.input_layernorm.register_forward_hook(_attn_norm)
            )

            # --- Attention output + type-specific internals ---
            is_linear = hasattr(layer, 'linear_attn') and layer.linear_attn is not None
            attn_module = layer.linear_attn if is_linear else getattr(layer, 'self_attn', None)

            if attn_module is not None:
                def _attn_out(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.ATTENTION_OUTPUT):
                        h = out[0] if isinstance(out, tuple) else out
                        self.capture_stage(PipelineStage.ATTENTION_OUTPUT, h, i)
                self._hook_handles.append(
                    attn_module.register_forward_hook(_attn_out)
                )

            # FA-specific hooks for full attention layers
            if not is_linear and attn_module is not None:
                fa = attn_module

                def _q_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.Q_PROJECTION):
                        self.capture_stage(PipelineStage.Q_PROJECTION, out, i)
                self._hook_handles.append(
                    fa.q_proj.register_forward_hook(_q_proj)
                )

                def _k_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.K_PROJECTION):
                        self.capture_stage(PipelineStage.K_PROJECTION, out, i)
                self._hook_handles.append(
                    fa.k_proj.register_forward_hook(_k_proj)
                )

                def _v_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.V_PROJECTION):
                        self.capture_stage(PipelineStage.V_PROJECTION, out, i)
                self._hook_handles.append(
                    fa.v_proj.register_forward_hook(_v_proj)
                )

            # GDN-specific hooks for linear attention layers
            if is_linear:
                gdn = layer.linear_attn

                def _qkv_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.QKV_PROJECTION):
                        self.capture_stage(PipelineStage.QKV_PROJECTION, out, i)
                self._hook_handles.append(
                    gdn.in_proj_qkv.register_forward_hook(_qkv_proj)
                )

                def _conv1d_out(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_CONV1D_OUTPUT):
                        self.capture_stage(PipelineStage.GDN_CONV1D_OUTPUT, out, i)
                self._hook_handles.append(
                    gdn.conv1d.register_forward_hook(_conv1d_out)
                )

                def _z_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_Z_PROJECTION):
                        self.capture_stage(PipelineStage.GDN_Z_PROJECTION, out, i)
                self._hook_handles.append(
                    gdn.in_proj_z.register_forward_hook(_z_proj)
                )

                def _delta_rule_out(mod, inp, i=idx):
                    if self._should_capture(PipelineStage.GDN_DELTA_RULE_OUTPUT):
                        h = inp[0] if isinstance(inp, tuple) else inp
                        self.capture_stage(PipelineStage.GDN_DELTA_RULE_OUTPUT, h, i)
                self._hook_handles.append(
                    gdn.norm.register_forward_pre_hook(_delta_rule_out)
                )

                def _norm_gate_out(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_NORM_GATE_OUTPUT):
                        self.capture_stage(PipelineStage.GDN_NORM_GATE_OUTPUT, out, i)
                self._hook_handles.append(
                    gdn.norm.register_forward_hook(_norm_gate_out)
                )

            # --- Residual hooks (shared by both layer types) ---

            # Attention residual: pre-hook on post_attention_layernorm
            def _attn_residual(mod, inp, i=idx):
                if self._should_capture(PipelineStage.ATTENTION_RESIDUAL):
                    h = inp[0] if isinstance(inp, tuple) else inp
                    self.capture_stage(PipelineStage.ATTENTION_RESIDUAL, h, i)
            self._hook_handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(_attn_residual)
            )

            # Post-attention layernorm output
            def _ffn_norm(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.FFN_NORM):
                    self.capture_stage(PipelineStage.FFN_NORM, out, i)
            self._hook_handles.append(
                layer.post_attention_layernorm.register_forward_hook(_ffn_norm)
            )

            # --- MoE FFN hooks ---
            moe_block = layer.mlp

            # Router output (gate) + routing indices and weights
            def _router(mod, inp, out, i=idx):
                if isinstance(out, tuple) and len(out) >= 3:
                    # gate returns (router_logits, routing_weights, selected_experts)
                    if self._should_capture(PipelineStage.MOE_ROUTER_OUTPUT):
                        self.capture_stage(PipelineStage.MOE_ROUTER_OUTPUT, out[0], i)
                    if self._should_capture(PipelineStage.MOE_ROUTING_WEIGHTS):
                        self.capture_stage(PipelineStage.MOE_ROUTING_WEIGHTS, out[1], i)
                    if self._should_capture(PipelineStage.MOE_ROUTING_INDICES):
                        # selected_experts is int64 — store as float for snapshot compat
                        self.capture_stage(PipelineStage.MOE_ROUTING_INDICES, out[2].float(), i)
                else:
                    router_logits = out[0] if isinstance(out, tuple) else out
                    if self._should_capture(PipelineStage.MOE_ROUTER_OUTPUT):
                        self.capture_stage(PipelineStage.MOE_ROUTER_OUTPUT, router_logits, i)
            self._hook_handles.append(
                moe_block.gate.register_forward_hook(_router)
            )

            # Routed expert output
            def _experts(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.MOE_EXPERT_OUTPUT):
                    self.capture_stage(PipelineStage.MOE_EXPERT_OUTPUT, out, i)
            self._hook_handles.append(
                moe_block.experts.register_forward_hook(_experts)
            )

            # Shared expert output (before sigmoid gate)
            def _shared_expert(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.MOE_SHARED_EXPERT_OUTPUT):
                    self.capture_stage(PipelineStage.MOE_SHARED_EXPERT_OUTPUT, out, i)
            self._hook_handles.append(
                moe_block.shared_expert.register_forward_hook(_shared_expert)
            )

            # Shared expert gate (sigmoid scaling)
            # Capture the GATED result: sigmoid(gate_logit) * shared_expert_output
            # to match Llaminar's SharedExpertGateStage output [seq, d_model].
            # The shared_expert hook fires before this one, so MOE_SHARED_EXPERT_OUTPUT
            # is already captured in self.snapshots.
            def _shared_gate(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.MOE_SHARED_GATE_OUTPUT):
                    import torch.nn.functional as F
                    gate_weight = F.sigmoid(out)  # [seq, 1]
                    shared_key = (PipelineStage.MOE_SHARED_EXPERT_OUTPUT, i)
                    if shared_key in self.snapshots:
                        shared_np = self.snapshots[shared_key]
                        shared_t = torch.from_numpy(shared_np).to(gate_weight.device)
                        if shared_t.dim() == 2 and gate_weight.dim() == 2:
                            gated = shared_t * gate_weight  # [seq, d_model] * [seq, 1] broadcast
                            self.capture_stage(PipelineStage.MOE_SHARED_GATE_OUTPUT, gated, i)
                        else:
                            self.capture_stage(PipelineStage.MOE_SHARED_GATE_OUTPUT, out, i)
                    else:
                        self.capture_stage(PipelineStage.MOE_SHARED_GATE_OUTPUT, out, i)
            self._hook_handles.append(
                moe_block.shared_expert_gate.register_forward_hook(_shared_gate)
            )

            # Combined MoE output (routed + shared, reshaped)
            def _moe_combined(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.MOE_COMBINED_OUTPUT):
                    h = out[0] if isinstance(out, tuple) else out
                    self.capture_stage(PipelineStage.MOE_COMBINED_OUTPUT, h, i)
            self._hook_handles.append(
                moe_block.register_forward_hook(_moe_combined)
            )

            # FFN residual: post-hook on the full DecoderLayer
            def _ffn_residual(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.FFN_RESIDUAL):
                    h = out[0] if isinstance(out, tuple) else out
                    self.capture_stage(PipelineStage.FFN_RESIDUAL, h, i)
            self._hook_handles.append(layer.register_forward_hook(_ffn_residual))

        # Final norm
        def _fnorm(mod, inp, out):
            if self._should_capture(PipelineStage.FINAL_NORM):
                self.capture_stage(PipelineStage.FINAL_NORM, out, layer_idx=-1)
        self._hook_handles.append(model.norm.register_forward_hook(_fnorm))


# Register this implementation
ModelRegistry.register("qwen35_moe", Qwen35MoEReferenceModel)
