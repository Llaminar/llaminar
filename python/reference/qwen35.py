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

import copy
import re
from pathlib import Path
from typing import Optional
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

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
    # Qwen3.6 next-token-prediction sidecar reference
    # ------------------------------------------------------------------

    def _prepare_gguf_state_dict(self, config_dict: dict, state_dict: dict) -> dict:
        """Retain the trailing Qwen3.6 ``nextn`` block outside the main graph.

        The GGUF parser deliberately excludes a real next-token-prediction
        block from ``num_hidden_layers`` so Hugging Face does not execute it as
        an ordinary decoder layer.  Its tensors remain in the state inventory,
        however, and form an independent real-weight oracle for production MTP
        checkpoint tests.
        """
        del config_dict
        self._capture_mtp_sidecar_state(state_dict)
        return state_dict

    def _capture_mtp_sidecar_state(self, state_dict: dict) -> None:
        """Retain only tensors owned by the first trailing ``nextn`` block."""
        nextn_layers = []
        for key in state_dict:
            match = re.match(r"blk\.(\d+)\.nextn\.eh_proj\.weight$", key)
            if match:
                nextn_layers.append(int(match.group(1)))

        nextn_layers = sorted(set(nextn_layers))
        self._mtp_sidecar_source_layer = nextn_layers[0] if nextn_layers else None
        self._mtp_sidecar_state = {}
        if self._mtp_sidecar_source_layer is None:
            return

        source = self._mtp_sidecar_source_layer
        raw_prefix = f"blk.{source}.nextn."
        layer_prefix = f"model.layers.{source}."
        for key, tensor in state_dict.items():
            if key.startswith(raw_prefix) or key.startswith(layer_prefix):
                # Retaining the immutable loader tensor avoids a second copy of
                # the sizeable dense sidecar FFN. ``load_state_dict`` reads but
                # does not mutate these source tensors.
                self._mtp_sidecar_state[key] = tensor.detach()

    @staticmethod
    def _mtp_rms_norm(
        value: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        *,
        pre_rmsnorm_1p: bool = False,
    ) -> torch.Tensor:
        """Apply the exact GGUF/Hugging Face RMSNorm weight convention."""
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + eps).to(dtype=value.dtype)
        effective_gamma = gamma.to(device=value.device, dtype=value.dtype)
        if pre_rmsnorm_1p:
            effective_gamma = effective_gamma + 1.0
        return normalized * effective_gamma

    @staticmethod
    def _mtp_snapshot_array(tensor: torch.Tensor) -> np.ndarray:
        """Detach one sidecar boundary in the canonical row-major FP32 form."""
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor.detach().cpu().float().numpy()

    def _mtp_tensor(self, suffix: str) -> torch.Tensor:
        """Resolve one nextn projection tensor from the retained GGUF state."""
        source = self._mtp_sidecar_source_layer
        key = f"blk.{source}.nextn.{suffix}"
        if key not in self._mtp_sidecar_state:
            raise RuntimeError(f"Missing dense MTP sidecar tensor {key}")
        return self._mtp_sidecar_state[key].to(self.device)

    def _make_mtp_sidecar_layer(self):
        """Build the single full-attention dense predictor layer from GGUF."""
        if not getattr(self, "_mtp_sidecar_state", None):
            return None

        from transformers.initialization import no_init_weights
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

        config = copy.deepcopy(self.hf_config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        with no_init_weights():
            layer = Qwen3_5DecoderLayer(config, 0)

        source = self._mtp_sidecar_source_layer
        layer_prefix = f"model.layers.{source}."
        layer_state = {
            key[len(layer_prefix):]: tensor
            for key, tensor in self._mtp_sidecar_state.items()
            if key.startswith(layer_prefix)
        }
        missing, unexpected = layer.load_state_dict(layer_state, strict=False)
        if missing:
            raise RuntimeError(
                f"Dense MTP sidecar reference layer missing weights: {missing}"
            )
        if unexpected:
            raise RuntimeError(
                f"Dense MTP sidecar reference layer had unexpected weights: {unexpected}"
            )
        layer._llaminar_mtp_config = config
        return layer.to(self.device).eval()

    def generate_mtp_sidecar_decode_snapshots(
        self,
        prompt: str,
        decode_steps: int,
        output_dir,
        max_draft_depth: int = 3,
        verbose: bool = False,
    ) -> int:
        """Generate recursive MTP0..MTP2 checkpoints from real dense weights.

        The reference follows production's shifted-cache contract.  Depth zero
        consumes the sampled main-model condition token; each deeper predictor
        consumes the previous predictor's shared-head-normalized hidden state
        and greedy token.  Only depth zero is committed to the cache before the
        next main-model row, while deeper speculative rows are cropped.
        """
        if not getattr(self, "_mtp_sidecar_state", None):
            raise RuntimeError(
                "The dense GGUF has no retained Qwen3.6 nextn/MTP sidecar"
            )
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        if max_draft_depth < 1 or max_draft_depth > 3:
            raise ValueError("max_draft_depth must be in [1, 3]")
        if decode_steps <= 0:
            return 0

        from transformers.cache_utils import DynamicCache

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        last_main_layer = self.hf_config.num_hidden_layers - 1
        sidecar_layer = self._make_mtp_sidecar_layer()
        if sidecar_layer is None:
            raise RuntimeError("Dense MTP sidecar layer could not be constructed")
        sidecar_cache = DynamicCache(config=sidecar_layer._llaminar_mtp_config)

        hnorm = self._mtp_tensor("hnorm.weight")
        enorm = self._mtp_tensor("enorm.weight")
        eh_proj = self._mtp_tensor("eh_proj.weight")
        shared_head_norm = self._mtp_tensor("shared_head_norm.weight")

        encoding = self.tokenizer(prompt, return_tensors="pt")
        token_ids = encoding["input_ids"][0].tolist()
        if len(token_ids) < 2:
            raise RuntimeError("Dense MTP reference requires at least two prompt tokens")

        result = self.forward(
            token_ids,
            clear_snapshots=True,
            use_cache=True,
            capture_stages=[PipelineStage.FFN_RESIDUAL],
        )
        main_cache = result["past_key_values"]
        last_hidden = torch.from_numpy(
            self.snapshots[(PipelineStage.FFN_RESIDUAL, last_main_layer)]
        ).to(self.device)
        if last_hidden.dim() == 2:
            last_hidden = last_hidden.unsqueeze(0)

        def project_sidecar_hidden(
            terminal_hidden: torch.Tensor,
            token_id: int,
            depth_index: int,
        ) -> dict[str, torch.Tensor]:
            prefix = f"MTP{depth_index}_"
            token = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
            embedding = self.hf_model.model.embed_tokens(token)
            norm_hidden = self._mtp_rms_norm(
                terminal_hidden,
                hnorm,
                self.hf_config.rms_norm_eps,
                pre_rmsnorm_1p=True,
            )
            norm_embedding = self._mtp_rms_norm(
                embedding,
                enorm,
                self.hf_config.rms_norm_eps,
                pre_rmsnorm_1p=True,
            )
            concat = torch.cat([norm_embedding, norm_hidden], dim=-1)
            projected = F.linear(concat, eh_proj)
            return {
                f"{prefix}TERMINAL_HIDDEN_ROW_SELECT": terminal_hidden,
                f"{prefix}EMBEDDING": embedding,
                f"{prefix}NORM_HIDDEN": norm_hidden,
                f"{prefix}NORM_EMBEDDING": norm_embedding,
                f"{prefix}CONCAT": concat,
                f"{prefix}FC": projected,
            }

        def capture(captures: dict, key: str, tensor: torch.Tensor) -> None:
            captures[key] = self._mtp_snapshot_array(tensor)

        def flatten_seq_heads(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 4:
                return tensor.contiguous().reshape(
                    tensor.shape[0], tensor.shape[1], -1
                )
            return tensor

        def install_capture_hooks(captures: dict, depth_index: int) -> list:
            handles = []
            runtime = {}
            prefix = f"MTP{depth_index}_"
            key = lambda suffix: f"{prefix}{suffix}"

            handles.append(
                sidecar_layer.input_layernorm.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("ATTENTION_NORM"), output
                    )
                )
            )

            attention = sidecar_layer.self_attn

            def q_projection(_module, _inputs, output):
                capture(captures, key("Q_PROJECTION"), output)
                head_dim = getattr(attention, "head_dim", self.hf_config.head_dim)
                q_gate = output.view(*output.shape[:-1], -1, head_dim * 2)
                _, gate = torch.chunk(q_gate, 2, dim=-1)
                runtime["fa_gate"] = gate.reshape(*output.shape[:-1], -1).detach()
                capture(captures, key("FA_GATE"), runtime["fa_gate"])

            handles.append(attention.q_proj.register_forward_hook(q_projection))
            handles.append(
                attention.k_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("K_PROJECTION"), output
                    )
                )
            )
            handles.append(
                attention.v_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("V_PROJECTION"), output
                    )
                )
            )
            handles.append(
                attention.q_norm.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("Q_NORM"), flatten_seq_heads(output)
                    )
                )
            )
            handles.append(
                attention.k_norm.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("K_NORM"), flatten_seq_heads(output)
                    )
                )
            )

            def attention_context(_module, inputs):
                gated = inputs[0] if isinstance(inputs, tuple) else inputs
                capture(captures, key("ATTENTION_CONTEXT_GATED"), gated)
                gate = runtime.get("fa_gate")
                if gate is not None:
                    raw = gated / torch.sigmoid(
                        gate.to(device=gated.device, dtype=gated.dtype)
                    ).clamp_min(1e-12)
                    capture(captures, key("ATTENTION_CONTEXT"), raw)

            handles.append(attention.o_proj.register_forward_pre_hook(attention_context))
            handles.append(
                attention.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures,
                        key("ATTENTION_OUTPUT"),
                        output[0] if isinstance(output, tuple) else output,
                    )
                )
            )
            handles.append(
                sidecar_layer.post_attention_layernorm.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("FFN_NORM"), output
                    )
                )
            )

            mlp = sidecar_layer.mlp
            handles.append(
                mlp.gate_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("FFN_GATE"), output
                    )
                )
            )
            handles.append(
                mlp.up_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("FFN_UP"), output
                    )
                )
            )
            handles.append(
                mlp.down_proj.register_forward_pre_hook(
                    lambda _module, inputs: capture(
                        captures,
                        key("FFN_SWIGLU"),
                        inputs[0] if isinstance(inputs, tuple) else inputs,
                    )
                )
            )
            handles.append(
                mlp.down_proj.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures, key("FFN_DOWN"), output
                    )
                )
            )
            handles.append(
                sidecar_layer.register_forward_hook(
                    lambda _module, _inputs, output: capture(
                        captures,
                        key("FFN_RESIDUAL"),
                        output[0] if isinstance(output, tuple) else output,
                    )
                )
            )
            return handles

        def sidecar_forward(
            projected: torch.Tensor,
            position: int,
            cache,
            depth_index: int = 0,
            captures: Optional[dict] = None,
        ) -> tuple[torch.Tensor, object]:
            text_positions = torch.tensor(
                [[position]], device=self.device, dtype=torch.long
            )
            rope_positions = text_positions[None, ...].expand(3, 1, 1)
            position_embeddings = self.hf_model.model.rotary_emb(
                projected, rope_positions
            )
            handles = (
                install_capture_hooks(captures, depth_index)
                if captures is not None
                else []
            )
            try:
                hidden = sidecar_layer(
                    projected,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=text_positions,
                    past_key_values=cache,
                    use_cache=True,
                )
            finally:
                for handle in handles:
                    handle.remove()
            return hidden, cache

        with torch.no_grad():
            # Shift the prompt into the sidecar KV cache. Row N consumes the
            # next prompt token and terminal main hidden row N-1.
            for row in range(len(token_ids) - 1):
                pieces = project_sidecar_hidden(
                    last_hidden[:, row:row + 1, :], token_ids[row + 1], 0
                )
                _, sidecar_cache = sidecar_forward(
                    pieces["MTP0_FC"], row + 1, sidecar_cache
                )

            next_token = int(result["logits"][0, -1, :].argmax())
            total = 0
            for step in range(decode_steps):
                committed_cache_length = sidecar_cache.get_seq_length()
                draft_hidden = last_hidden[:, -1:, :]
                draft_condition_token = next_token
                for depth_index in range(max_draft_depth):
                    prefix = f"MTP{depth_index}_"
                    pieces = project_sidecar_hidden(
                        draft_hidden, draft_condition_token, depth_index
                    )
                    sidecar_captures = {}
                    hidden, sidecar_cache = sidecar_forward(
                        pieces[f"{prefix}FC"],
                        len(token_ids) + step + depth_index,
                        sidecar_cache,
                        depth_index,
                        sidecar_captures,
                    )
                    final_hidden = self._mtp_rms_norm(
                        hidden,
                        shared_head_norm,
                        self.hf_config.rms_norm_eps,
                        pre_rmsnorm_1p=True,
                    )
                    logits = self.hf_model.lm_head(final_hidden)

                    snapshots = {
                        key: self._mtp_snapshot_array(value)
                        for key, value in pieces.items()
                    }
                    snapshots.update(sidecar_captures)
                    snapshots[f"{prefix}FFN_RESIDUAL"] = self._mtp_snapshot_array(
                        hidden
                    )
                    snapshots[f"{prefix}FINAL_NORM"] = self._mtp_snapshot_array(
                        final_hidden
                    )
                    snapshots[f"{prefix}LM_HEAD"] = self._mtp_snapshot_array(logits)
                    if depth_index == 0:
                        snapshots["MTP_TERMINAL_HIDDEN_ROW_SELECT"] = snapshots[
                            "MTP0_TERMINAL_HIDDEN_ROW_SELECT"
                        ]

                    for snapshot_key, payload in snapshots.items():
                        np.save(
                            output_dir / f"decode_step{step}_{snapshot_key}.npy",
                            payload,
                        )
                        total += 1
                        if verbose:
                            print(
                                f"  Saved decode_step{step}_{snapshot_key}: "
                                f"shape={list(payload.shape)}"
                            )

                    draft_condition_token = int(logits[0, -1, :].argmax())
                    draft_hidden = final_hidden

                # Depth zero is the sole committed shifted-MTP row. Deeper
                # recursive predictions are discarded with their cache state.
                sidecar_cache.crop(committed_cache_length + 1)

                result = self.forward(
                    [next_token],
                    clear_snapshots=True,
                    past_key_values=main_cache,
                    use_cache=True,
                    capture_stages=[PipelineStage.FFN_RESIDUAL],
                )
                main_cache = result["past_key_values"]
                last_hidden = torch.from_numpy(
                    self.snapshots[(PipelineStage.FFN_RESIDUAL, last_main_layer)]
                ).to(self.device)
                if last_hidden.dim() == 2:
                    last_hidden = last_hidden.unsqueeze(0)
                next_token = int(result["logits"][0, -1, :].argmax())

        return total

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

                def _alpha_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_ALPHA):
                        self.capture_stage(PipelineStage.GDN_ALPHA, out, i)
                self._hook_handles.append(
                    gdn.in_proj_a.register_forward_hook(_alpha_proj)
                )

                def _beta_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_BETA):
                        self.capture_stage(PipelineStage.GDN_BETA, out, i)
                self._hook_handles.append(
                    gdn.in_proj_b.register_forward_hook(_beta_proj)
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
