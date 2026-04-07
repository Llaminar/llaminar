"""
HuggingFace GGUF Compatibility Shim for Qwen3.5

Patches transformers' GGUF loading infrastructure to support the ``qwen35``
architecture string emitted by llama.cpp GGUF files for Qwen 3.5 models.

As of transformers 5.5.0 and gguf 0.18.0, the ``gguf`` Python package
recognises ``qwen35`` but transformers does not wire up the config mapping,
architecture normalisation, or model-type overrides needed to load these
checkpoints via ``AutoModelForCausalLM.from_pretrained(..., gguf_file=...)``.

Usage::

    # Import before any GGUF loading through HF's AutoModel path
    from python.reference.loaders.gguf_compat import patch_hf_gguf_qwen35
    patch_hf_gguf_qwen35()

The patch is idempotent and safe to call multiple times.
"""

from __future__ import annotations

import functools
import re
from typing import Any

import numpy as np

_PATCHED = False


# ── Config mapping: GGUF metadata key → HF config parameter ─────────
QWEN35_CONFIG_MAPPING: dict[str, str | None] = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
    # Qwen 3.5 GDN-specific fields
    "full_attention_interval": "full_attention_interval",
    "ssm.conv_kernel": "linear_conv_kernel_dim",
    "ssm.group_count": "linear_num_key_heads",
    "ssm.inner_size": "ssm_inner_size",
    "ssm.state_size": "linear_value_head_dim",
    "attention.key_length": "head_dim",
    "rope.dimension_sections": "rope_dimension_sections",
}


# ── Tensor processor: weight transforms for Qwen 3.5 ────────────────

class Qwen35TensorProcessor:
    """
    Transform GGUF tensors to match HuggingFace Qwen3_5ForCausalLM expectations.

    llama.cpp's ``convert_hf_to_gguf.py`` applies several transforms when
    converting Qwen 3.5 checkpoints from HF to GGUF.  We reverse them here:

      - **norm weights** (except ``linear_attn.norm``): GGUF stores ``w + 1``
        (pre_rmsnorm_1p convention) → subtract 1
      - **A_log** (``ssm_a``): GGUF stores ``-exp(A_log)`` → apply ``log(-x)``
      - **conv1d**: GGUF squeezes ``[C, 1, K]`` → ``[C, K]`` → unsqueeze back

    Name mapping fixups:

      - **dt_bias**: HF model has ``linear_attn.dt_bias`` (bare parameter,
        not a submodule), but the gguf package maps ``dt_proj`` →
        ``blk.N.ssm_dt``.  We add the mapping via ``perform_fallback_tensor_mapping``.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def preprocess_name(self, hf_name: str) -> str:
        return hf_name

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map: dict[str, str], suffix: str, qual_name: str, hf_name: str,
    ):
        # HF param:   model.layers.N.linear_attn.dt_bias
        # GGUF tensor: blk.N.ssm_dt.bias
        # The gguf name_map doesn't match dt_bias because it's not dt_proj
        m = re.match(r"model\.layers\.(\d+)\.linear_attn\.dt_bias$", hf_name)
        if m:
            bid = m.group(1)
            gguf_to_hf_name_map[f"blk.{bid}.ssm_dt.bias"] = qual_name + hf_name

    def process(self, weights: np.ndarray, name: str, **kwargs):
        from transformers.modeling_gguf_pytorch_utils import GGUFTensor

        metadata: dict[str, Any] = {}

        # conv1d: unsqueeze [C, K] → [C, 1, K]
        # GGUF name: blk.N.ssm_conv1d.weight
        if "ssm_conv1d" in name and weights.ndim == 2:
            weights = np.expand_dims(weights, axis=1)

        # A_log: GGUF stores -exp(A_log) → reverse to A_log
        # GGUF name: blk.N.ssm_a
        if ".ssm_a" in name:
            weights = np.log(np.maximum(-weights.astype(np.float32), 1e-30))

        # Norm weights: reverse pre_rmsnorm_1p (+1 convention)
        # GGUF names: blk.N.attn_norm.weight, blk.N.ffn_norm.weight, output_norm.weight
        # Exclude ssm_norm (linear_attn.norm) which is NOT pre_rmsnorm_1p
        if name.endswith("_norm.weight") and "ssm_norm" not in name:
            weights = weights.astype(np.float32) - 1.0

        return GGUFTensor(weights, name, metadata)


def _build_layer_types(num_layers: int, full_attention_interval: int) -> list[str]:
    """Build the ``layer_types`` list expected by ``Qwen3_5TextConfig``."""
    types: list[str] = []
    for i in range(num_layers):
        if (i + 1) % full_attention_interval == 0:
            types.append("full_attention")
        else:
            types.append("linear_attention")
    return types


def _postprocess_qwen35_config(config: dict[str, Any]) -> None:
    """Derive parameters that GGUF metadata doesn't store directly."""
    # model_type must match HF's CONFIG_MAPPING key for the text backbone
    config["model_type"] = "qwen3_5_text"
    config["architectures"] = ["Qwen3_5ForCausalLM"]

    # Build layer_types from full_attention_interval
    n_layers = config.get("num_hidden_layers", 24)
    interval = config.get("full_attention_interval", 4)
    config["layer_types"] = _build_layer_types(n_layers, interval)

    # Derive linear_num_value_heads from ssm_inner_size / linear_value_head_dim
    ssm_inner = config.get("ssm_inner_size")
    v_dim = config.get("linear_value_head_dim")
    if ssm_inner and v_dim and v_dim > 0:
        config["linear_num_value_heads"] = ssm_inner // v_dim

    # linear_key_head_dim defaults to linear_value_head_dim
    if "linear_key_head_dim" not in config:
        config["linear_key_head_dim"] = config.get("linear_value_head_dim", 128)


def patch_hf_gguf_qwen35() -> None:
    """
    Monkey-patch transformers to recognise the ``qwen35`` GGUF architecture.

    Idempotent — safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return

    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

    # 1. Register config mapping
    if "qwen35" not in GGUF_CONFIG_MAPPING:
        GGUF_CONFIG_MAPPING["qwen35"] = QWEN35_CONFIG_MAPPING
        # Keep top-level reference in sync
        gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"] = GGUF_CONFIG_MAPPING

    # 2. Update supported architectures list (snapshot computed at import time)
    if "qwen35" not in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    # 2b. Register custom tensor processor for weight transforms
    if "qwen35" not in gguf_utils.TENSOR_PROCESSORS:
        gguf_utils.TENSOR_PROCESSORS["qwen35"] = Qwen35TensorProcessor

    # 3. Wrap load_gguf_checkpoint to add architecture normalisation and
    #    post-processing that the upstream code doesn't have for qwen35.
    _original_load = gguf_utils.load_gguf_checkpoint.__wrapped__ if hasattr(
        gguf_utils.load_gguf_checkpoint, "__wrapped__"
    ) else gguf_utils.load_gguf_checkpoint

    @functools.wraps(_original_load)
    def _patched_load(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
        result = _original_load(gguf_checkpoint_path, return_tensors, model_to_load)
        cfg = result.get("config", {})
        # Detect qwen35 by checking if our GDN-specific keys are present
        if cfg.get("model_type") == "qwen35" or cfg.get("full_attention_interval") is not None:
            _postprocess_qwen35_config(cfg)
        return result

    _patched_load.__wrapped__ = _original_load
    gguf_utils.load_gguf_checkpoint = _patched_load

    # Also patch the cached reference in configuration_utils (imported at
    # module level via ``from .modeling_gguf_pytorch_utils import …``).
    try:
        from transformers import configuration_utils as _cfg_utils
        _cfg_utils.load_gguf_checkpoint = _patched_load
    except (ImportError, AttributeError):
        pass

    # 4. Wrap get_gguf_hf_weights_map to map qwen3_5_text → qwen35 for
    #    the gguf package's MODEL_ARCH_NAMES lookup.
    _original_weights_map = gguf_utils.get_gguf_hf_weights_map.__wrapped__ if hasattr(
        gguf_utils.get_gguf_hf_weights_map, "__wrapped__"
    ) else gguf_utils.get_gguf_hf_weights_map

    @functools.wraps(_original_weights_map)
    def _patched_weights_map(hf_model, processor, model_type=None, num_layers=None, qual_name=""):
        if model_type is None and hf_model is not None:
            mt = getattr(hf_model.config, "model_type", None)
            if mt in ("qwen3_5_text", "qwen3_5"):
                model_type = "qwen35"
        elif model_type in ("qwen3_5_text", "qwen3_5"):
            model_type = "qwen35"
        return _original_weights_map(hf_model, processor, model_type, num_layers, qual_name)

    _patched_weights_map.__wrapped__ = _original_weights_map
    gguf_utils.get_gguf_hf_weights_map = _patched_weights_map

    _PATCHED = True


def is_patched() -> bool:
    """Return whether the HF GGUF compatibility patch has been applied."""
    return _PATCHED


def unpatch_hf_gguf_qwen35() -> None:
    """
    Remove the monkey-patch (for testing teardown).

    Restores the original ``load_gguf_checkpoint`` and
    ``get_gguf_hf_weights_map`` functions, removes the ``qwen35`` config
    mapping and supported architecture entry.
    """
    global _PATCHED
    if not _PATCHED:
        return

    import transformers.modeling_gguf_pytorch_utils as gguf_utils
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING

    # Restore original functions
    if hasattr(gguf_utils.load_gguf_checkpoint, "__wrapped__"):
        original_load = gguf_utils.load_gguf_checkpoint.__wrapped__
        gguf_utils.load_gguf_checkpoint = original_load
        # Restore configuration_utils reference too
        try:
            from transformers import configuration_utils as _cfg_utils
            _cfg_utils.load_gguf_checkpoint = original_load
        except (ImportError, AttributeError):
            pass
    if hasattr(gguf_utils.get_gguf_hf_weights_map, "__wrapped__"):
        gguf_utils.get_gguf_hf_weights_map = gguf_utils.get_gguf_hf_weights_map.__wrapped__

    # Remove config mapping
    GGUF_CONFIG_MAPPING.pop("qwen35", None)
    gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"] = GGUF_CONFIG_MAPPING

    # Remove tensor processor
    gguf_utils.TENSOR_PROCESSORS.pop("qwen35", None)

    # Remove from supported architectures
    try:
        gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.remove("qwen35")
    except ValueError:
        pass

    _PATCHED = False
