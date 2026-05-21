"""
Qwen 3.5 Gated Delta Net Reference Implementation

PyTorch reference implementation for Qwen 3.5 models using HuggingFace transformers.
Captures intermediate pipeline states for parity testing with Llaminar.

Architecture:
  - Hybrid 3:1 layout: 75% Gated DeltaNet (linear attention) + 25% full softmax GQA
  - Layer pattern: [linear, linear, linear, full_attn, ...] controlled by full_attention_interval
  - Linear layers have: in_proj_qkvz, in_proj_ba, conv1d, A_log, dt_bias, norm, out_proj
  - Full attention layers have: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm

GGUF specifics:
  - Norm weights use pre_rmsnorm_1p convention (GGUF stores w+1, subtract 1 on load)
  - in_proj_qkv, in_proj_z, in_proj_a, in_proj_b are stored separately in GGUF
    (attn_qkv, attn_gate, ssm_alpha, ssm_beta respectively — no fusion needed)
  - ssm_a stores -exp(A_log), reversed via log(-x) on load
  - conv1d.weight squeezed to 2D in GGUF, unsqueezed back on load

@author David Sanftenberg
"""

from typing import Optional
from abc import abstractmethod

import torch

from .base import HuggingFaceReferenceModel
from .pipeline_stages import PipelineStage
from .registry import ModelRegistry


class Qwen35ReferenceModel(HuggingFaceReferenceModel):
    """
    PyTorch reference implementation for Qwen 3.5 Gated Delta Net models.

    Inherits shared GGUF loading and forward pass from HuggingFaceReferenceModel.
    Overrides hook registration to handle heterogeneous layers (linear_attn vs self_attn).
    """

    def _create_model_from_gguf_config(
        self,
        config_dict: dict,
        torch_dtype: Optional[torch.dtype],
    ) -> tuple:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextConfig,
            Qwen3_5ForCausalLM,
        )

        # Build Qwen3_5TextConfig from GGUF-extracted config dict
        hf_config = self._build_hf_config(config_dict)

        if torch_dtype:
            hf_config.torch_dtype = torch_dtype

        model = Qwen3_5ForCausalLM(hf_config)
        return hf_config, model

    @staticmethod
    def _build_hf_config(config_dict: dict):
        """Convert GGUF config_dict to Qwen3_5TextConfig parameters."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig

        full_attn_interval = config_dict.get('full_attention_interval', 4)
        n_layers = config_dict.get('num_hidden_layers', 24)

        # Build layer_types list: "linear_attention" for GDN layers, "full_attention" for softmax
        layer_types = []
        for i in range(n_layers):
            if (i + 1) % full_attn_interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")

        # linear_num_value_heads: derive from ssm_inner_size / value_head_dim
        # (GGUF doesn't store this directly; for 0.8B n_k==n_v==16, for 4B n_k=16, n_v=32)
        linear_num_key_heads = config_dict.get('linear_num_key_heads', 16)
        linear_value_head_dim = config_dict.get('linear_value_head_dim', 128)
        ssm_inner_size = config_dict.get('ssm_inner_size', None)

        if ssm_inner_size is not None and linear_value_head_dim > 0:
            linear_num_value_heads = ssm_inner_size // linear_value_head_dim
        else:
            linear_num_value_heads = linear_num_key_heads

        # linear_key_head_dim: for Qwen3.5, d_k == d_v == state_size (128)
        # Don't derive from ssm_inner_size/n_k_heads — ssm_inner_size is value-total,
        # not key-total (they differ when n_k_heads != n_v_heads)
        linear_key_head_dim = config_dict.get('linear_key_head_dim', linear_value_head_dim)

        cfg = Qwen3_5TextConfig(
            hidden_size=config_dict.get('hidden_size', 1024),
            num_hidden_layers=n_layers,
            num_attention_heads=config_dict.get('num_attention_heads', 8),
            num_key_value_heads=config_dict.get('num_key_value_heads', 2),
            head_dim=config_dict.get('head_dim', 256),
            intermediate_size=config_dict.get('intermediate_size', 3584),
            max_position_embeddings=config_dict.get('max_position_embeddings', 262144),
            rms_norm_eps=config_dict.get('rms_norm_eps', 1e-6),
            rope_theta=config_dict.get('rope_theta', 10000000.0),
            vocab_size=config_dict.get('vocab_size', 248320),
            layer_types=layer_types,
            full_attention_interval=full_attn_interval,
            linear_key_head_dim=linear_key_head_dim or 128,
            linear_value_head_dim=config_dict.get('linear_value_head_dim', 128),
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_conv_kernel_dim=config_dict.get('linear_conv_kernel_dim', 4),
        )
        cfg._attn_implementation = "eager"
        return cfg

    def _tokenizer_fallbacks(self) -> list[str]:
        return ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B-Instruct"]

    # ------------------------------------------------------------------
    # Hook registration for heterogeneous layers
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """
        Register forward hooks for Qwen 3.5 with heterogeneous layer types.

        Linear attention layers have `linear_attn` module with GDN-specific
        sub-modules (conv1d, norm/gate, delta rule kernel).
        Full attention layers have `self_attn` module with standard Q/K/V/O.
        Both share `input_layernorm`, `post_attention_layernorm`, `mlp`.

        Hook strategy for residual connections:
          - ATTENTION_RESIDUAL: pre-hook on post_attention_layernorm captures its
            input, which is (residual + attn_output) per DecoderLayer.forward().
          - FFN_RESIDUAL: post-hook on the full DecoderLayer captures the final
            output, which is (residual + mlp_output).
        """
        if not self.hf_model:
            return

        model = self.hf_model.model
        self._last_fa_gate_by_layer = {}
        self._active_fa_rope_layer = None

        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_mod

        original_apply_rotary = qwen35_mod.apply_rotary_pos_emb

        def _flatten_head_tensor(t):
            if t.dim() == 4:
                return t.transpose(1, 2).contiguous().reshape(t.shape[0], t.shape[2], -1)
            return t

        def _flatten_seq_head_tensor(t):
            if t.dim() == 4:
                return t.contiguous().reshape(t.shape[0], t.shape[1], -1)
            return t

        def _conv1d_prefill_boundary(inp, out):
            seq_len = inp[0].shape[-1] if isinstance(inp, tuple) and inp else out.shape[-1]
            return torch.nn.functional.silu(out[:, :, :seq_len]).transpose(1, 2).contiguous()

        def _capture_apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
            q_rope, k_rope = original_apply_rotary(q, k, cos, sin, *args, **kwargs)
            layer_idx = self._active_fa_rope_layer
            if layer_idx is not None:
                if self._should_capture(PipelineStage.Q_ROPE):
                    self.capture_stage(PipelineStage.Q_ROPE, _flatten_head_tensor(q_rope), layer_idx)
                if self._should_capture(PipelineStage.K_ROPE):
                    self.capture_stage(PipelineStage.K_ROPE, _flatten_head_tensor(k_rope), layer_idx)
            return q_rope, k_rope

        qwen35_mod.apply_rotary_pos_emb = _capture_apply_rotary_pos_emb

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

                def _fa_pre(mod, inp, i=idx):
                    self._active_fa_rope_layer = i
                self._hook_handles.append(
                    fa.register_forward_pre_hook(_fa_pre)
                )

                def _fa_post(mod, inp, out, i=idx):
                    if self._active_fa_rope_layer == i:
                        self._active_fa_rope_layer = None
                self._hook_handles.append(
                    fa.register_forward_hook(_fa_post)
                )

                def _q_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.Q_PROJECTION):
                        self.capture_stage(PipelineStage.Q_PROJECTION, out, i)
                    if self._should_capture(PipelineStage.FA_GATE) or self._should_capture(PipelineStage.ATTENTION_CONTEXT):
                        head_dim = getattr(fa, 'head_dim', self.hf_config.head_dim)
                        q_gate = out.view(*out.shape[:-1], -1, head_dim * 2)
                        _, gate = torch.chunk(q_gate, 2, dim=-1)
                        gate = gate.reshape(*out.shape[:-1], -1)
                        self._last_fa_gate_by_layer[i] = gate.detach()
                        if self._should_capture(PipelineStage.FA_GATE):
                            self.capture_stage(PipelineStage.FA_GATE, gate, i)
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

                def _q_norm(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.Q_NORM):
                        self.capture_stage(PipelineStage.Q_NORM, _flatten_seq_head_tensor(out), i)
                self._hook_handles.append(
                    fa.q_norm.register_forward_hook(_q_norm)
                )

                def _k_norm(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.K_NORM):
                        self.capture_stage(PipelineStage.K_NORM, _flatten_seq_head_tensor(out), i)
                self._hook_handles.append(
                    fa.k_norm.register_forward_hook(_k_norm)
                )

                def _attn_context(mod, inp, i=idx):
                    h = inp[0] if isinstance(inp, tuple) else inp
                    if self._should_capture(PipelineStage.ATTENTION_CONTEXT):
                        gate_by_layer = getattr(self, '_last_fa_gate_by_layer', {})
                        gate = gate_by_layer.get(i)
                        if gate is not None:
                            gate = gate.to(device=h.device, dtype=h.dtype)
                            raw_context = h / torch.sigmoid(gate).clamp_min(1e-12)
                            self.capture_stage(PipelineStage.ATTENTION_CONTEXT, raw_context, i)
                    if self._should_capture(PipelineStage.ATTENTION_CONTEXT_GATED):
                        self.capture_stage(PipelineStage.ATTENTION_CONTEXT_GATED, h, i)
                self._hook_handles.append(
                    fa.o_proj.register_forward_pre_hook(_attn_context)
                )

            # GDN-specific hooks for linear attention layers
            if is_linear:
                gdn = layer.linear_attn

                # QKV projection (before conv1d)
                def _qkv_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.QKV_PROJECTION):
                        self.capture_stage(PipelineStage.QKV_PROJECTION, out, i)
                self._hook_handles.append(
                    gdn.in_proj_qkv.register_forward_hook(_qkv_proj)
                )

                # Conv1d output (after SiLU activation)
                def _conv1d_out(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_CONV1D_OUTPUT):
                        self.capture_stage(PipelineStage.GDN_CONV1D_OUTPUT, _conv1d_prefill_boundary(inp, out), i)
                self._hook_handles.append(
                    gdn.conv1d.register_forward_hook(_conv1d_out)
                )

                # Cached decode bypasses the Conv1d module and calls the
                # causal-conv update function directly, so a module hook never
                # sees the one-token convolution output. Wrap the update
                # function to capture that boundary without changing math.
                if not getattr(gdn, "_llaminar_conv_update_wrapped", False):
                    original_update = gdn.causal_conv1d_update

                    def _capture_conv1d_update(x, conv_state, weight, bias, activation,
                                               *args, _orig=original_update, i=idx, **kwargs):
                        out = _orig(x, conv_state, weight, bias, activation, *args, **kwargs)
                        if self._should_capture(PipelineStage.GDN_CONV1D_OUTPUT):
                            self.capture_stage(PipelineStage.GDN_CONV1D_OUTPUT, out, i)
                        return out

                    gdn.causal_conv1d_update = _capture_conv1d_update
                    gdn._llaminar_conv_update_wrapped = True

                # Z gate projection
                def _z_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_Z_PROJECTION):
                        self.capture_stage(PipelineStage.GDN_Z_PROJECTION, out, i)
                self._hook_handles.append(
                    gdn.in_proj_z.register_forward_hook(_z_proj)
                )

                # RMSNormGated input = delta rule output (before norm+gate)
                def _delta_rule_out(mod, inp, i=idx):
                    if self._should_capture(PipelineStage.GDN_DELTA_RULE_OUTPUT):
                        # norm forward takes (hidden_states, residual) — first arg
                        h = inp[0] if isinstance(inp, tuple) else inp
                        self.capture_stage(PipelineStage.GDN_DELTA_RULE_OUTPUT, h, i)
                self._hook_handles.append(
                    gdn.norm.register_forward_pre_hook(_delta_rule_out)
                )

                # RMSNormGated output (after norm + SiLU gate with z)
                def _norm_gate_out(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_NORM_GATE_OUTPUT):
                        self.capture_stage(PipelineStage.GDN_NORM_GATE_OUTPUT, out, i)
                self._hook_handles.append(
                    gdn.norm.register_forward_hook(_norm_gate_out)
                )

            # --- Residual and FFN hooks (shared by both layer types) ---

            # Attention residual: pre-hook on post_attention_layernorm
            # Its input is (residual + attn_output) from DecoderLayer.forward()
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

            # MLP output (FFN down projection)
            def _ffn_out(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.FFN_DOWN):
                    self.capture_stage(PipelineStage.FFN_DOWN, out, i)
            self._hook_handles.append(layer.mlp.register_forward_hook(_ffn_out))

            # FFN residual: post-hook on the full DecoderLayer
            # Output is (residual + mlp_output) — the final hidden state for this layer
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
ModelRegistry.register("qwen35", Qwen35ReferenceModel)
