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

import copy
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import HuggingFaceReferenceModel
from .mtp_sidecar_reference import (
    mtp_sidecar_replay_depth,
    save_mtp_snapshot_atomic,
)
from .pipeline_stages import PipelineStage
from .registry import ModelRegistry


def production_router_distribution(router_output) -> torch.Tensor:
    """Return the full post-softmax distribution published by production.

    Hugging Face names the first tuple member ``router_logits`` even though the
    module has already applied softmax. Keeping this validation in one helper
    prevents main-model and recursive-MTP hooks from drifting back to a
    reconstructed raw projection that no captured backend publishes.
    """

    if not isinstance(router_output, tuple) or len(router_output) < 3:
        raise RuntimeError(
            "Qwen3.5 MoE router did not return probabilities, weights, and indices"
        )
    return router_output[0]


def materialize_route_contributions(
    experts,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Reproduce Hugging Face's weighted output for every selected route.

    Hugging Face's public experts result is the sum over the top-k routes. A
    summed row cannot isolate one moved expert when quantized and FP32 routers
    exchange an unrelated, low-weight boundary expert. This reference-only
    recorder evaluates the same expert equations and preserves each route in
    ``[token, slot, hidden]`` order. It never participates in inference; its
    sole purpose is to provide an unambiguous numerical parity oracle.
    """

    if hidden_states.ndim != 2:
        raise RuntimeError("Qwen3.5 MoE expert input must be a rank-2 tensor")
    if (
        selected_experts.ndim != 2
        or routing_weights.shape != selected_experts.shape
    ):
        raise RuntimeError(
            "Qwen3.5 MoE route IDs and weights must share rank-2 geometry"
        )
    if selected_experts.shape[0] != hidden_states.shape[0]:
        raise RuntimeError(
            "Qwen3.5 MoE route rows must match the flattened expert input"
        )

    route_contributions = hidden_states.new_zeros(
        (
            hidden_states.shape[0],
            selected_experts.shape[1],
            hidden_states.shape[1],
        )
    )
    for encoded_expert in torch.unique(selected_experts):
        expert_idx = int(encoded_expert.item())
        if expert_idx < 0 or expert_idx >= int(experts.num_experts):
            raise RuntimeError(
                f"Qwen3.5 MoE route names invalid expert {expert_idx}"
            )
        positions = torch.nonzero(
            selected_experts == expert_idx, as_tuple=False
        )
        token_indices = positions[:, 0]
        route_slots = positions[:, 1]
        current_state = hidden_states[token_indices]
        gate, up = F.linear(
            current_state, experts.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        expert_output = F.linear(
            experts.act_fn(gate) * up,
            experts.down_proj[expert_idx],
        )
        weighted_output = expert_output * routing_weights[
            token_indices, route_slots, None
        ]
        route_contributions[token_indices, route_slots] = weighted_output.to(
            route_contributions.dtype
        )
    return route_contributions


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
        """Stream GGUF tensors directly into the final Hugging Face model.

        A 122B checkpoint expands to roughly 488 GB in FP32. Materializing a
        second state dictionary beside the model exceeds host memory and makes
        split-model parity impossible. The bounded loader publishes ordinary
        parameters and each half of the fused expert tensor directly into its
        final allocation; the trailing MTP decoder is loaded into its own
        layer by the same transaction.
        """
        from .loaders import GGUFLoader
        from .loaders.gguf_parser import GGUFParser
        from transformers.initialization import no_init_weights
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeForCausalLM,
        )

        print(f"Loading GGUF file: {gguf_path}")
        loader = GGUFLoader(gguf_path, verbose=self.verbose)
        parser = GGUFParser(gguf_path)
        parser.parse()
        try:
            config_dict = loader.load_config(
                parser=parser, as_transformers_config=False
            )
            sidecar_reference_pack = kwargs.get("mtp_sidecar_reference_pack")
            self._mtp_sidecar_reference_pack = (
                Path(sidecar_reference_pack)
                if sidecar_reference_pack is not None
                else None
            )
            with no_init_weights():
                self.hf_config = self._build_hf_config(config_dict)
                if torch_dtype:
                    self.hf_config.torch_dtype = torch_dtype
                if self._mtp_sidecar_reference_pack is None:
                    self.hf_model = Qwen3_5MoeForCausalLM(self.hf_config)
                else:
                    # An additive branch oracle consumes the authenticated
                    # main-model trajectory already stored in the canonical
                    # Hugging Face pack. It owns only graph-external MTP state:
                    # embeddings, LM head, rotary state, and one sidecar layer.
                    # Retaining 48 unused FP32 main layers would add roughly
                    # 450 GiB and defeat campaign-level context amortization.
                    shell_config = copy.deepcopy(self.hf_config)
                    shell_config.num_hidden_layers = 0
                    shell_config.layer_types = []
                    self.hf_model = Qwen3_5MoeForCausalLM(shell_config)
                    # The top-level Transformers model resolves its concrete
                    # expert implementation during construction. Propagate
                    # that typed choice to the original full-depth config used
                    # by the standalone sidecar layer so both canonical and
                    # additive contexts execute the same HF expert kernel.
                    self.hf_config._experts_implementation = (
                        shell_config._experts_implementation
                    )

            nextn_sources = sorted(
                {
                    int(match.group(1))
                    for tensor in parser.tensors
                    if (
                        match := re.fullmatch(
                            r"blk\.(\d+)\.nextn\.eh_proj\.weight",
                            tensor.name,
                        )
                    )
                }
            )
            if len(nextn_sources) > 1:
                raise RuntimeError(
                    f"Multiple MTP source layers are unsupported: {nextn_sources}"
                )
            self._mtp_sidecar_source_layer = (
                nextn_sources[0] if nextn_sources else None
            )
            self._mtp_sidecar_state = {}
            self._mtp_sidecar_layer = None

            sidecar_parameters = {}
            sidecar_loaded = set()
            sidecar_expert_halves = {}
            if self._mtp_sidecar_source_layer is not None:
                sidecar_config = copy.deepcopy(self.hf_config)
                sidecar_config.num_hidden_layers = 1
                sidecar_config.layer_types = ["full_attention"]
                with no_init_weights():
                    self._mtp_sidecar_layer = Qwen3_5MoeDecoderLayer(
                        sidecar_config, 0
                    )
                self._mtp_sidecar_layer._llaminar_mtp_config = sidecar_config
                sidecar_parameters = dict(
                    self._mtp_sidecar_layer.named_parameters()
                )

            main_parameters = dict(self.hf_model.named_parameters())
            main_loaded = set()
            main_expert_halves = {}
            unexpected = []
            source_layer_prefix = (
                f"model.layers.{self._mtp_sidecar_source_layer}."
                if self._mtp_sidecar_source_layer is not None
                else None
            )
            raw_nextn_prefix = (
                f"blk.{self._mtp_sidecar_source_layer}.nextn."
                if self._mtp_sidecar_source_layer is not None
                else None
            )

            def include_sidecar_tensor(mapped_name: str) -> bool:
                """Select tensors owned by the additive sidecar context."""

                if self._mtp_sidecar_reference_pack is None:
                    return True
                return (
                    mapped_name in main_parameters
                    or (
                        raw_nextn_prefix is not None
                        and mapped_name.startswith(raw_nextn_prefix)
                    )
                    or (
                        source_layer_prefix is not None
                        and mapped_name.startswith(source_layer_prefix)
                    )
                )

            for mapped_name, tensor in loader.iter_state_dict(
                parser=parser,
                as_torch=True,
                show_progress=self.verbose,
                max_in_flight=8,
                include_mapped_name=include_sidecar_tensor,
            ):
                if raw_nextn_prefix and mapped_name.startswith(raw_nextn_prefix):
                    self._mtp_sidecar_state[mapped_name] = tensor.detach()
                    continue
                if source_layer_prefix and mapped_name.startswith(
                    source_layer_prefix
                ):
                    sidecar_name = mapped_name[len(source_layer_prefix):]
                    if not self._copy_streamed_parameter(
                        sidecar_name,
                        tensor,
                        sidecar_parameters,
                        sidecar_loaded,
                        sidecar_expert_halves,
                    ):
                        unexpected.append(mapped_name)
                    continue
                if not self._copy_streamed_parameter(
                    mapped_name,
                    tensor,
                    main_parameters,
                    main_loaded,
                    main_expert_halves,
                ):
                    if mapped_name not in {
                        "rope_freqs.weight",
                        "rope.freqs",
                        "pos_embd.weight",
                    }:
                        unexpected.append(mapped_name)

            main_missing = set(main_parameters) - main_loaded
            if main_missing == {"lm_head.weight"} and (
                "model.embed_tokens.weight" in main_loaded
            ):
                print("Tying lm_head.weight to model.embed_tokens.weight")
                self.hf_model.lm_head.weight = (
                    self.hf_model.model.embed_tokens.weight
                )
                main_missing.clear()
            if main_missing:
                raise RuntimeError(
                    "Streamed GGUF model is missing parameters: "
                    f"{sorted(main_missing)}"
                )

            sidecar_missing = set(sidecar_parameters) - sidecar_loaded
            if sidecar_missing:
                raise RuntimeError(
                    "Streamed GGUF MTP layer is missing parameters: "
                    f"{sorted(sidecar_missing)}"
                )
            if unexpected:
                raise RuntimeError(
                    "Streamed GGUF contains unexpected tensors: "
                    f"{sorted(unexpected)}"
                )
            if self._mtp_sidecar_source_layer is not None:
                for suffix in (
                    "eh_proj.weight",
                    "enorm.weight",
                    "hnorm.weight",
                    "shared_head_norm.weight",
                ):
                    self._mtp_tensor(suffix)

            self.hf_model = self.hf_model.to(self.device).eval()
            if self._mtp_sidecar_layer is not None:
                self._mtp_sidecar_layer = (
                    self._mtp_sidecar_layer.to(self.device).eval()
                )
        finally:
            parser.close()

        self._resolve_tokenizer(gguf_path)
        print("✓ GGUF MoE model loaded successfully")

    @staticmethod
    def _copy_streamed_parameter(
        name: str,
        tensor: torch.Tensor,
        parameters: dict[str, torch.nn.Parameter],
        loaded: set[str],
        expert_halves: dict[str, set[str]],
    ) -> bool:
        """Copy one mapped tensor into its final parameter allocation.

        GGUF stores expert gate and up projections independently while
        Transformers owns one ``gate_up_proj`` parameter. Copying each source
        into its exact slice avoids constructing a second fused 6+ GB tensor.
        """
        expert_match = re.fullmatch(
            r"((?:.*\.)?mlp\.experts)\.(gate_proj|up_proj)\.weight",
            name,
        )
        target_name = name
        target_view = None
        half_name = None
        if expert_match:
            target_name = f"{expert_match.group(1)}.gate_up_proj"
            if target_name not in parameters or tensor.ndim != 3:
                return False
            target = parameters[target_name]
            if target.shape[0] != tensor.shape[0] or target.shape[2] != tensor.shape[2]:
                raise RuntimeError(
                    f"Expert tensor shape mismatch for {name}: "
                    f"source={tuple(tensor.shape)}, target={tuple(target.shape)}"
                )
            if target.shape[1] != tensor.shape[1] * 2:
                raise RuntimeError(
                    f"Expert fused width mismatch for {name}: "
                    f"source={tuple(tensor.shape)}, target={tuple(target.shape)}"
                )
            half_name = expert_match.group(2)
            offset = 0 if half_name == "gate_proj" else tensor.shape[1]
            target_view = target[:, offset:offset + tensor.shape[1], :]
        elif target_name not in parameters:
            return False
        else:
            target_view = parameters[target_name]
            if name.endswith("mlp.shared_expert_gate.weight") and tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)

        if tuple(target_view.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"Parameter shape mismatch for {name}: "
                f"source={tuple(tensor.shape)}, target={tuple(target_view.shape)}"
            )
        with torch.no_grad():
            target_view.copy_(
                tensor.to(device=target_view.device, dtype=target_view.dtype)
            )

        if half_name is None:
            loaded.add(target_name)
        else:
            halves = expert_halves.setdefault(target_name, set())
            halves.add(half_name)
            if halves == {"gate_proj", "up_proj"}:
                loaded.add(target_name)
        return True

    def _capture_mtp_sidecar_state(self, state_dict: dict) -> None:
        """Keep only the trailing nextn tensors needed for MTP sidecar parity."""
        nextn_layers = []
        for key in state_dict:
            m = re.match(r"blk\.(\d+)\.nextn\.eh_proj\.weight$", key)
            if m:
                nextn_layers.append(int(m.group(1)))
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
                captured = tensor.detach().clone()
                if key.endswith("mlp.shared_expert_gate.weight") and captured.dim() == 1:
                    captured = captured.unsqueeze(0)
                self._mtp_sidecar_state[key] = captured

    @staticmethod
    def _rms_norm(
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        *,
        pre_rmsnorm_1p: bool = False,
    ) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
        effective_gamma = gamma.to(device=x.device, dtype=x.dtype)
        if pre_rmsnorm_1p:
            effective_gamma = effective_gamma + 1.0
        return y * effective_gamma

    @staticmethod
    def _flatten_for_snapshot(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor.detach().cpu().float().numpy()

    def _make_mtp_sidecar_layer(self):
        if getattr(self, "_mtp_sidecar_layer", None) is not None:
            return self._mtp_sidecar_layer
        if not getattr(self, "_mtp_sidecar_state", None):
            return None

        from transformers.initialization import no_init_weights
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

        cfg = copy.deepcopy(self.hf_config)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ["full_attention"]
        with no_init_weights():
            layer = Qwen3_5MoeDecoderLayer(cfg, 0)

        source = self._mtp_sidecar_source_layer
        layer_prefix = f"model.layers.{source}."
        layer_state = {}
        for key, tensor in self._mtp_sidecar_state.items():
            if key.startswith(layer_prefix):
                layer_state[key[len(layer_prefix):]] = tensor
        missing, unexpected = layer.load_state_dict(layer_state, strict=False)
        if missing:
            raise RuntimeError(f"MTP sidecar reference layer missing weights: {missing}")
        if unexpected:
            raise RuntimeError(f"MTP sidecar reference layer had unexpected weights: {unexpected}")
        layer._llaminar_mtp_config = cfg
        return layer.to(self.device).eval()

    def _mtp_tensor(self, suffix: str) -> torch.Tensor:
        source = self._mtp_sidecar_source_layer
        key = f"blk.{source}.nextn.{suffix}"
        if key not in self._mtp_sidecar_state:
            raise RuntimeError(f"Missing MTP sidecar tensor {key}")
        return self._mtp_sidecar_state[key].to(self.device)

    def _load_mtp_reference_pack_trajectory(
        self,
        prompt: str,
        decode_steps: int,
    ) -> tuple[list[int], list[int], torch.Tensor]:
        """Load the authenticated main-model trajectory for a branch oracle.

        Additive MTP branches do not alter committed main-model execution. The
        canonical Hugging Face pack therefore is the exact authority for the
        prompt's final-layer hidden rows and committed greedy tokens. Reusing
        those immutable FP32 artifacts lets one small sidecar context evaluate
        every production-observed branch without loading the 48-layer main
        model again.

        Returns:
            Prompt token IDs, committed decode tokens, and the prefill final-
            layer residual tensor.
        """

        pack = getattr(self, "_mtp_sidecar_reference_pack", None)
        if pack is None:
            raise RuntimeError("No MTP sidecar reference pack was configured")
        metadata_path = pack / "metadata.txt"
        if not metadata_path.is_file():
            raise RuntimeError(
                f"MTP sidecar reference pack has no metadata: {metadata_path}"
            )

        metadata = {}
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

        def require(key: str) -> str:
            value = metadata.get(key)
            if value is None or not value:
                raise RuntimeError(
                    f"MTP sidecar reference metadata is missing {key}"
                )
            return value

        if require("reference_engine") != "pytorch":
            raise RuntimeError("MTP sidecar branches require a PyTorch reference pack")
        if require("reference_dtype") != "float32":
            raise RuntimeError("MTP sidecar branches require an FP32 reference pack")
        if int(require("n_layers")) != self.hf_config.num_hidden_layers:
            raise RuntimeError(
                "MTP sidecar reference layer count does not match the GGUF model"
            )
        if int(require("decode_steps")) != decode_steps:
            raise RuntimeError(
                "MTP sidecar reference decode length does not match the request"
            )

        token_ids = [int(token) for token in require("token_ids").split(",")]
        encoded_ids = self.tokenizer(prompt, return_tensors="pt")[
            "input_ids"
        ][0].tolist()
        if encoded_ids != token_ids:
            raise RuntimeError(
                "MTP sidecar reference prompt tokens do not match the request"
            )

        decode_tokens = [
            int(token) for token in require("decode_tokens").split(",")
        ]
        if len(decode_tokens) != decode_steps + 1:
            raise RuntimeError(
                "MTP sidecar reference pack must contain the prefill token and "
                "one successor token for every decode step"
            )

        last_main_layer = self.hf_config.num_hidden_layers - 1
        hidden_path = pack / f"layer{last_main_layer}_FFN_RESIDUAL.npy"
        last_hidden = self._load_mtp_reference_hidden(hidden_path)
        if last_hidden.shape[:2] != (1, len(token_ids)):
            raise RuntimeError(
                f"Unexpected MTP prefill trajectory shape {tuple(last_hidden.shape)}"
            )
        return token_ids, decode_tokens, last_hidden

    def _load_mtp_reference_hidden(self, path: Path) -> torch.Tensor:
        """Load and validate one immutable FP32 hidden-state checkpoint."""

        if not path.is_file():
            raise RuntimeError(f"Missing MTP main-model trajectory tensor: {path}")
        payload = np.load(path, allow_pickle=False)
        if payload.dtype != np.float32 or payload.ndim != 3:
            raise RuntimeError(
                f"Invalid MTP trajectory tensor {path}: "
                f"dtype={payload.dtype}, shape={payload.shape}"
            )
        if payload.shape[-1] != self.hf_config.hidden_size:
            raise RuntimeError(
                f"MTP trajectory hidden width does not match the model: {path}"
            )
        return torch.from_numpy(payload).to(self.device)

    def generate_mtp_sidecar_decode_snapshots(
        self,
        prompt: str,
        decode_steps: int,
        output_dir,
        max_draft_depth: int = 3,
        verbose: bool = False,
        draft_token_overrides: Optional[dict[int, list[int]]] = None,
        reuse_canonical_main_trajectory: bool = False,
    ) -> int:
        """Generate recursive MTP0..MTPN checkpoints for the Qwen3.6 sidecar.

        Each repeated predictor call consumes the previous call's shared-head-
        normalized hidden state.  That tensor is both the LM-head input and the
        hidden-state result published by the reference Qwen3.5 MTP module; the
        pre-normalized decoder residual is an intermediate checkpoint only.

        ``draft_token_overrides`` names production-selected recursive condition
        tokens by decode step.  Entry zero is consumed by MTP1, entry one by
        MTP2, and so on; MTP0 always consumes the main-model condition token.
        Alternate snapshots include the consumed branch in their filename so
        they can coexist with the canonical Hugging Face greedy branch.

        ``reuse_canonical_main_trajectory`` explicitly authorizes canonical
        MTP0..MTPN generation from an existing authenticated main-model pack.
        The sidecar-only model context then consumes the immutable FP32 hidden
        rows and committed token history in that pack.  Keeping this authority
        explicit prevents an arbitrary directory from silently replacing a
        complete-model reference run while allowing deeper predictor capacity
        to be added without loading the complete 35B/122B model again.
        """
        if not getattr(self, "_mtp_sidecar_state", None):
            return 0
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        if max_draft_depth < 1 or max_draft_depth > 15:
            raise ValueError("max_draft_depth must be in [1, 15]")
        draft_token_overrides = draft_token_overrides or {}
        for step, tokens in draft_token_overrides.items():
            if step < 0 or step >= decode_steps:
                raise ValueError(
                    f"MTP draft override step {step} is outside decode range"
                )
            if len(tokens) > max_draft_depth - 1 or any(token < 0 for token in tokens):
                raise ValueError(
                    f"MTP draft override step {step} has invalid recursive tokens"
                )

        from pathlib import Path
        from transformers.cache_utils import DynamicCache

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = self._mtp_sidecar_source_layer
        last_main_layer = self.hf_config.num_hidden_layers - 1
        sidecar_layer = self._make_mtp_sidecar_layer()
        sidecar_cache = DynamicCache(config=sidecar_layer._llaminar_mtp_config)
        reference_pack = getattr(self, "_mtp_sidecar_reference_pack", None)
        if (
            reference_pack is not None
            and not draft_token_overrides
            and not reuse_canonical_main_trajectory
        ):
            raise RuntimeError(
                "A sidecar-only context may generate additive branch snapshots "
                "only unless reuse_canonical_main_trajectory explicitly "
                "authorizes an authenticated canonical trajectory"
            )

        hnorm = self._mtp_tensor("hnorm.weight")
        enorm = self._mtp_tensor("enorm.weight")
        eh_proj = self._mtp_tensor("eh_proj.weight")
        shared_head_norm = self._mtp_tensor("shared_head_norm.weight")

        if reference_pack is None:
            encoding = self.tokenizer(prompt, return_tensors="pt")
            token_ids = encoding["input_ids"][0].tolist()
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
            committed_decode_tokens = None
        else:
            token_ids, committed_decode_tokens, last_hidden = (
                self._load_mtp_reference_pack_trajectory(prompt, decode_steps)
            )
            main_cache = None

        def project_sidecar_hidden(
            terminal_hidden: torch.Tensor,
            token_id: int,
            depth_index: int,
        ) -> dict:
            prefix = f"MTP{depth_index}_"
            token = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
            embedding = self.hf_model.model.embed_tokens(token)
            norm_hidden = self._rms_norm(
                terminal_hidden, hnorm, self.hf_config.rms_norm_eps,
                pre_rmsnorm_1p=True)
            norm_embedding = self._rms_norm(
                embedding, enorm, self.hf_config.rms_norm_eps,
                pre_rmsnorm_1p=True)
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

        def _capture_mtp(captures: dict, key: str, tensor: torch.Tensor) -> None:
            captures[key] = self._flatten_for_snapshot(tensor)

        def _flatten_head_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 4:
                return tensor.transpose(1, 2).contiguous().reshape(tensor.shape[0], tensor.shape[2], -1)
            return tensor

        def _flatten_seq_head_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 4:
                return tensor.contiguous().reshape(tensor.shape[0], tensor.shape[1], -1)
            return tensor

        def _install_sidecar_capture_hooks(
            captures: dict,
            depth_index: int,
        ) -> list:
            handles = []
            runtime = {}
            prefix = f"MTP{depth_index}_"
            key = lambda suffix: f"{prefix}{suffix}"

            handles.append(sidecar_layer.input_layernorm.register_forward_hook(
                lambda _mod, _inp, out: _capture_mtp(captures, key("ATTENTION_NORM"), out)))

            fa = sidecar_layer.self_attn

            def _q_proj(_mod, _inp, out):
                _capture_mtp(captures, key("Q_PROJECTION"), out)
                head_dim = getattr(fa, "head_dim", self.hf_config.head_dim)
                q_gate = out.view(*out.shape[:-1], -1, head_dim * 2)
                _, gate = torch.chunk(q_gate, 2, dim=-1)
                runtime["fa_gate"] = gate.reshape(*out.shape[:-1], -1).detach()
                _capture_mtp(captures, key("FA_GATE"), runtime["fa_gate"])

            handles.append(fa.q_proj.register_forward_hook(_q_proj))
            handles.append(fa.k_proj.register_forward_hook(
                lambda _mod, _inp, out: _capture_mtp(captures, key("K_PROJECTION"), out)))
            handles.append(fa.v_proj.register_forward_hook(
                lambda _mod, _inp, out: _capture_mtp(captures, key("V_PROJECTION"), out)))
            handles.append(fa.q_norm.register_forward_hook(
                lambda _mod, _inp, out: _capture_mtp(captures, key("Q_NORM"), _flatten_seq_head_tensor(out))))
            handles.append(fa.k_norm.register_forward_hook(
                lambda _mod, _inp, out: _capture_mtp(captures, key("K_NORM"), _flatten_seq_head_tensor(out))))

            def _attention_context(_mod, inp):
                h = inp[0] if isinstance(inp, tuple) else inp
                _capture_mtp(captures, key("ATTENTION_CONTEXT_GATED"), h)
                gate = runtime.get("fa_gate")
                if gate is not None:
                    raw_context = h / torch.sigmoid(gate.to(device=h.device, dtype=h.dtype)).clamp_min(1e-12)
                    _capture_mtp(captures, key("ATTENTION_CONTEXT"), raw_context)

            handles.append(fa.o_proj.register_forward_pre_hook(_attention_context))

            def _attention_output(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                _capture_mtp(captures, key("ATTENTION_OUTPUT"), h)

            handles.append(fa.register_forward_hook(_attention_output))

            handles.append(sidecar_layer.post_attention_layernorm.register_forward_hook(
                lambda _mod, _inp, out: _capture_mtp(captures, key("FFN_NORM"), out)))

            moe_block = sidecar_layer.mlp

            def _router(_mod, _inp, out):
                # Qwen3_5MoeTopKRouter calls its first result `router_logits`,
                # but the value is the full post-softmax expert distribution.
                # Llaminar intentionally publishes that same live routing
                # workspace under the historical MOE_ROUTER_OUTPUT key.  Keep
                # the reference at the production boundary instead of
                # reconstructing a pre-softmax tensor that the captured graph
                # does not expose after routing has completed.
                _capture_mtp(
                    captures,
                    key("MOE_ROUTER_OUTPUT"),
                    production_router_distribution(out),
                )
                _capture_mtp(captures, key("MOE_ROUTING_WEIGHTS"), out[1])
                _capture_mtp(captures, key("MOE_ROUTING_INDICES"), out[2].float())

            handles.append(moe_block.gate.register_forward_hook(_router))
            def _mtp_experts(mod, inp, out):
                _capture_mtp(captures, key("MOE_EXPERT_OUTPUT"), out)
                if not isinstance(inp, tuple) or len(inp) != 3:
                    raise RuntimeError(
                        "Qwen3.5 MoE MTP experts did not receive complete route inputs"
                    )
                _capture_mtp(
                    captures,
                    key("MOE_ROUTE_CONTRIBUTIONS"),
                    materialize_route_contributions(
                        mod,
                        inp[0],
                        inp[1],
                        inp[2],
                    ),
                )

            handles.append(
                moe_block.experts.register_forward_hook(_mtp_experts)
            )

            def _shared_expert(_mod, _inp, out):
                runtime["shared_expert_output"] = out.detach()
                _capture_mtp(captures, key("MOE_SHARED_EXPERT_OUTPUT"), out)

            handles.append(moe_block.shared_expert.register_forward_hook(_shared_expert))

            def _shared_gate(_mod, _inp, out):
                shared = runtime.get("shared_expert_output")
                if shared is not None:
                    gated = shared.to(device=out.device, dtype=out.dtype) * torch.sigmoid(out)
                    _capture_mtp(captures, key("MOE_SHARED_GATE_OUTPUT"), gated)
                else:
                    _capture_mtp(captures, key("MOE_SHARED_GATE_OUTPUT"), out)

            handles.append(moe_block.shared_expert_gate.register_forward_hook(_shared_gate))

            def _moe_combined(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                _capture_mtp(captures, key("MOE_COMBINED_OUTPUT"), h)

            handles.append(moe_block.register_forward_hook(_moe_combined))

            def _ffn_residual(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                _capture_mtp(captures, key("FFN_RESIDUAL"), h)

            handles.append(sidecar_layer.register_forward_hook(_ffn_residual))
            return handles

        def sidecar_forward(
            projected: torch.Tensor,
            position: int,
            cache,
            depth_index: int = 0,
            captures: Optional[dict] = None,
        ) -> tuple[torch.Tensor, object]:
            text_position_ids = torch.tensor([[position]], device=self.device, dtype=torch.long)
            rope_position_ids = text_position_ids[None, ...].expand(3, 1, 1)
            position_embeddings = self.hf_model.model.rotary_emb(projected, rope_position_ids)
            handles = (
                _install_sidecar_capture_hooks(captures, depth_index)
                if captures is not None
                else []
            )
            try:
                hidden = sidecar_layer(
                    projected,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=text_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            finally:
                for handle in handles:
                    handle.remove()
            return hidden, cache

        with torch.no_grad():
            for row in range(len(token_ids) - 1):
                pieces = project_sidecar_hidden(
                    last_hidden[:, row:row + 1, :],
                    token_ids[row + 1],
                    0)
                _, sidecar_cache = sidecar_forward(
                    pieces["MTP0_FC"],
                    row + 1,
                    sidecar_cache)

            next_token = (
                int(result["logits"][0, -1, :].argmax())
                if committed_decode_tokens is None
                else committed_decode_tokens[0]
            )
            total = 0
            for step in range(decode_steps):
                committed_cache_length = sidecar_cache.get_seq_length()
                draft_hidden = last_hidden[:, -1:, :]
                draft_condition_token = next_token
                step_overrides = draft_token_overrides.get(step)
                consumed_recursive_tokens = []
                replay_depth = mtp_sidecar_replay_depth(
                    step, max_draft_depth, draft_token_overrides)
                for depth_index in range(replay_depth):
                    if (
                        depth_index > 0
                        and step_overrides is not None
                        and depth_index - 1 < len(step_overrides)
                    ):
                        draft_condition_token = step_overrides[depth_index - 1]
                    if depth_index > 0:
                        consumed_recursive_tokens.append(draft_condition_token)
                    prefix = f"MTP{depth_index}_"
                    pieces = project_sidecar_hidden(
                        draft_hidden,
                        draft_condition_token,
                        depth_index)
                    sidecar_captures = {}
                    hidden, sidecar_cache = sidecar_forward(
                        pieces[f"{prefix}FC"],
                        len(token_ids) + step + depth_index,
                        sidecar_cache,
                        depth_index,
                        sidecar_captures)
                    final_hidden = self._rms_norm(
                        hidden,
                        shared_head_norm,
                        self.hf_config.rms_norm_eps,
                        pre_rmsnorm_1p=True)
                    logits = self.hf_model.lm_head(final_hidden)

                    snapshots = {
                        f"{prefix}TERMINAL_HIDDEN_ROW_SELECT": self._flatten_for_snapshot(
                            pieces[f"{prefix}TERMINAL_HIDDEN_ROW_SELECT"]),
                        f"{prefix}EMBEDDING": self._flatten_for_snapshot(
                            pieces[f"{prefix}EMBEDDING"]),
                        f"{prefix}NORM_HIDDEN": self._flatten_for_snapshot(
                            pieces[f"{prefix}NORM_HIDDEN"]),
                        f"{prefix}NORM_EMBEDDING": self._flatten_for_snapshot(
                            pieces[f"{prefix}NORM_EMBEDDING"]),
                        f"{prefix}CONCAT": self._flatten_for_snapshot(
                            pieces[f"{prefix}CONCAT"]),
                        f"{prefix}FC": self._flatten_for_snapshot(
                            pieces[f"{prefix}FC"]),
                        f"{prefix}FFN_RESIDUAL": self._flatten_for_snapshot(hidden),
                        f"{prefix}FINAL_NORM": self._flatten_for_snapshot(final_hidden),
                        f"{prefix}LM_HEAD": self._flatten_for_snapshot(logits),
                    }
                    if depth_index == 0:
                        snapshots["MTP_TERMINAL_HIDDEN_ROW_SELECT"] = snapshots[
                            "MTP0_TERMINAL_HIDDEN_ROW_SELECT"
                        ]
                    snapshots.update(sidecar_captures)
                    branch_qualifier = ""
                    if step_overrides is not None and depth_index > 0:
                        branch_qualifier = "_BRANCH_" + "_".join(
                            str(token) for token in consumed_recursive_tokens
                        )
                    persist_snapshot = (
                        not draft_token_overrides
                        or (step_overrides is not None and depth_index > 0)
                    )
                    for snapshot_key, payload in snapshots.items():
                        if not persist_snapshot:
                            continue
                        snapshot_path = (
                            output_dir
                            / f"decode_step{step}{branch_qualifier}_{snapshot_key}.npy"
                        )
                        if draft_token_overrides:
                            save_mtp_snapshot_atomic(snapshot_path, payload)
                        else:
                            np.save(snapshot_path, payload)
                        total += 1
                        if verbose:
                            print(
                                f"  Saved decode_step{step}_{snapshot_key}: "
                                f"shape={list(payload.shape)}"
                            )

                    draft_condition_token = int(logits[0, -1, :].argmax())
                    draft_hidden = final_hidden

                # Only the first sidecar row belongs to the next committed main
                # position.  Deeper rows are speculative and must not leak into
                # the reference cache used by the following main decode step.
                sidecar_cache.crop(committed_cache_length + 1)

                if committed_decode_tokens is None:
                    result = self.forward(
                        [next_token],
                        clear_snapshots=True,
                        past_key_values=main_cache,
                        use_cache=True,
                        capture_stages=[PipelineStage.FFN_RESIDUAL],
                    )
                    main_cache = result["past_key_values"]
                    last_hidden = torch.from_numpy(
                        self.snapshots[
                            (PipelineStage.FFN_RESIDUAL, last_main_layer)
                        ]
                    ).to(self.device)
                    if last_hidden.dim() == 2:
                        last_hidden = last_hidden.unsqueeze(0)
                    next_token = int(result["logits"][0, -1, :].argmax())
                else:
                    hidden_path = (
                        reference_pack
                        / f"decode_step{step}_layer{last_main_layer}_"
                        "FFN_RESIDUAL.npy"
                    )
                    last_hidden = self._load_mtp_reference_hidden(hidden_path)
                    if last_hidden.shape[:2] != (1, 1):
                        raise RuntimeError(
                            "MTP decode trajectory must contain exactly one row: "
                            f"{hidden_path}"
                        )
                    next_token = committed_decode_tokens[step + 1]

        return total

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
        self._last_fa_gate_by_layer = {}
        self._active_fa_rope_layer = None

        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen35_moe_mod

        original_apply_rotary = qwen35_moe_mod.apply_rotary_pos_emb

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

        qwen35_moe_mod.apply_rotary_pos_emb = _capture_apply_rotary_pos_emb

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

                def _qkv_proj(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.QKV_PROJECTION):
                        self.capture_stage(PipelineStage.QKV_PROJECTION, out, i)
                self._hook_handles.append(
                    gdn.in_proj_qkv.register_forward_hook(_qkv_proj)
                )

                def _conv1d_out(mod, inp, out, i=idx):
                    if self._should_capture(PipelineStage.GDN_CONV1D_OUTPUT):
                        self.capture_stage(PipelineStage.GDN_CONV1D_OUTPUT, _conv1d_prefill_boundary(inp, out), i)
                self._hook_handles.append(
                    gdn.conv1d.register_forward_hook(_conv1d_out)
                )

                # Cached decode uses causal_conv1d_update directly instead of
                # invoking the Conv1d module, so the forward hook above only
                # covers prefill/fallback paths. Wrap the update function to
                # expose the same post-conv boundary for one-token decode.
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
                # The first result is the complete post-softmax distribution.
                # CUDA and ROCm retain that same full routing workspace through
                # captured replay, so this is the only live cross-backend
                # checkpoint boundary. Top-k weights and indices remain the
                # second and third results respectively.
                if self._should_capture(PipelineStage.MOE_ROUTER_OUTPUT):
                    self.capture_stage(
                        PipelineStage.MOE_ROUTER_OUTPUT,
                        production_router_distribution(out),
                        i,
                    )
                if self._should_capture(PipelineStage.MOE_ROUTING_WEIGHTS):
                    self.capture_stage(PipelineStage.MOE_ROUTING_WEIGHTS, out[1], i)
                if self._should_capture(PipelineStage.MOE_ROUTING_INDICES):
                    # selected_experts is int64 — store as float for snapshot compat
                    self.capture_stage(PipelineStage.MOE_ROUTING_INDICES, out[2].float(), i)
            self._hook_handles.append(
                moe_block.gate.register_forward_hook(_router)
            )

            # Routed expert output
            def _experts(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.MOE_EXPERT_OUTPUT):
                    self.capture_stage(PipelineStage.MOE_EXPERT_OUTPUT, out, i)
                if self._should_capture(PipelineStage.MOE_ROUTE_CONTRIBUTIONS):
                    if not isinstance(inp, tuple) or len(inp) != 3:
                        raise RuntimeError(
                            "Qwen3.5 MoE experts did not receive hidden rows, route IDs, and route weights"
                        )
                    self.capture_stage(
                        PipelineStage.MOE_ROUTE_CONTRIBUTIONS,
                        materialize_route_contributions(
                            mod,
                            inp[0],
                            inp[1],
                            inp[2],
                        ),
                        i,
                    )
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
