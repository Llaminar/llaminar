"""
Tests for HuggingFace GGUF Compatibility Shim (gguf_compat.py)

Unit tests (no model files required):
  - Patch/unpatch lifecycle and idempotency
  - Config mapping completeness
  - _build_layer_types logic
  - _postprocess_qwen35_config derivations
  - Qwen35TensorProcessor transforms (conv1d, A_log, norm)
  - Qwen35TensorProcessor dt_bias fallback mapping

Integration tests (require models/Qwen3.5-4B-Q8_0.gguf):
  - Full AutoModelForCausalLM.from_pretrained loading
  - All 426 tensors mapped and loaded
  - Config values match expected model dimensions
"""

import pytest
import numpy as np
from pathlib import Path

from python.reference.loaders.gguf_compat import (
    QWEN35_CONFIG_MAPPING,
    Qwen35TensorProcessor,
    _build_layer_types,
    _postprocess_qwen35_config,
    is_patched,
    patch_hf_gguf_qwen35,
    unpatch_hf_gguf_qwen35,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QWEN35_4B_GGUF = Path("models/Qwen3.5-4B-Q8_0.gguf")


@pytest.fixture(autouse=True)
def _ensure_unpatched():
    """Ensure each test starts and ends with a clean (unpatched) state."""
    unpatch_hf_gguf_qwen35()
    yield
    unpatch_hf_gguf_qwen35()


# ---------------------------------------------------------------------------
# Unit Tests — No model files required
# ---------------------------------------------------------------------------


class TestPatchLifecycle:
    """Test patch/unpatch idempotency and state management."""

    def test_initially_unpatched(self):
        assert not is_patched()

    def test_patch_sets_flag(self):
        patch_hf_gguf_qwen35()
        assert is_patched()

    def test_unpatch_clears_flag(self):
        patch_hf_gguf_qwen35()
        unpatch_hf_gguf_qwen35()
        assert not is_patched()

    def test_patch_idempotent(self):
        """Calling patch twice should not raise or double-register."""
        patch_hf_gguf_qwen35()
        patch_hf_gguf_qwen35()
        assert is_patched()
        unpatch_hf_gguf_qwen35()
        assert not is_patched()

    def test_unpatch_idempotent(self):
        """Calling unpatch when not patched should not raise."""
        unpatch_hf_gguf_qwen35()
        assert not is_patched()

    def test_patch_registers_architecture(self):
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        patch_hf_gguf_qwen35()
        assert "qwen35" in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES

    def test_unpatch_removes_architecture(self):
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        patch_hf_gguf_qwen35()
        unpatch_hf_gguf_qwen35()
        assert "qwen35" not in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES

    def test_patch_registers_config_mapping(self):
        from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
        patch_hf_gguf_qwen35()
        assert "qwen35" in GGUF_CONFIG_MAPPING
        assert GGUF_CONFIG_MAPPING["qwen35"] is QWEN35_CONFIG_MAPPING

    def test_unpatch_removes_config_mapping(self):
        from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
        patch_hf_gguf_qwen35()
        unpatch_hf_gguf_qwen35()
        assert "qwen35" not in GGUF_CONFIG_MAPPING

    def test_patch_registers_tensor_processor(self):
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        patch_hf_gguf_qwen35()
        assert "qwen35" in gguf_utils.TENSOR_PROCESSORS
        assert gguf_utils.TENSOR_PROCESSORS["qwen35"] is Qwen35TensorProcessor

    def test_unpatch_removes_tensor_processor(self):
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        patch_hf_gguf_qwen35()
        unpatch_hf_gguf_qwen35()
        assert "qwen35" not in gguf_utils.TENSOR_PROCESSORS


class TestConfigMapping:
    """Test QWEN35_CONFIG_MAPPING completeness."""

    def test_required_keys_present(self):
        required = [
            "context_length",
            "block_count",
            "feed_forward_length",
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "vocab_size",
        ]
        for key in required:
            assert key in QWEN35_CONFIG_MAPPING, f"Missing mapping for '{key}'"

    def test_gdn_specific_keys(self):
        gdn_keys = [
            "full_attention_interval",
            "ssm.conv_kernel",
            "ssm.group_count",
            "ssm.inner_size",
            "ssm.state_size",
        ]
        for key in gdn_keys:
            assert key in QWEN35_CONFIG_MAPPING, f"Missing GDN mapping for '{key}'"

    def test_rope_keys(self):
        assert "rope.freq_base" in QWEN35_CONFIG_MAPPING
        assert QWEN35_CONFIG_MAPPING["rope.freq_base"] == "rope_theta"

    def test_null_mapping_for_derived_keys(self):
        """Keys mapped to None are intentionally excluded from config."""
        assert QWEN35_CONFIG_MAPPING["rope.dimension_count"] is None


class TestBuildLayerTypes:
    """Test _build_layer_types() layer pattern generation."""

    def test_interval_4_layers_8(self):
        types = _build_layer_types(8, 4)
        assert len(types) == 8
        # Layer indices 0-based: full attention at (i+1) % 4 == 0 → i=3,7
        assert types == [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ]

    def test_interval_4_layers_32(self):
        """Qwen3.5-4B: 32 layers with interval 4."""
        types = _build_layer_types(32, 4)
        assert len(types) == 32
        full_count = sum(1 for t in types if t == "full_attention")
        linear_count = sum(1 for t in types if t == "linear_attention")
        assert full_count == 8
        assert linear_count == 24

    def test_interval_4_layers_24(self):
        """Qwen3.5-0.8B: 24 layers."""
        types = _build_layer_types(24, 4)
        assert len(types) == 24
        full_indices = [i for i, t in enumerate(types) if t == "full_attention"]
        assert full_indices == [3, 7, 11, 15, 19, 23]

    def test_interval_1_all_full(self):
        """Every layer is full attention if interval is 1."""
        types = _build_layer_types(4, 1)
        assert all(t == "full_attention" for t in types)

    def test_single_layer(self):
        types = _build_layer_types(1, 4)
        assert types == ["linear_attention"]


class TestPostprocessConfig:
    """Test _postprocess_qwen35_config() derived value computation."""

    def test_model_type(self):
        cfg = {"num_hidden_layers": 32, "full_attention_interval": 4}
        _postprocess_qwen35_config(cfg)
        assert cfg["model_type"] == "qwen3_5_text"

    def test_architectures(self):
        cfg = {"num_hidden_layers": 32, "full_attention_interval": 4}
        _postprocess_qwen35_config(cfg)
        assert cfg["architectures"] == ["Qwen3_5ForCausalLM"]

    def test_layer_types_generated(self):
        cfg = {"num_hidden_layers": 8, "full_attention_interval": 4}
        _postprocess_qwen35_config(cfg)
        assert len(cfg["layer_types"]) == 8
        assert cfg["layer_types"][3] == "full_attention"
        assert cfg["layer_types"][0] == "linear_attention"

    def test_linear_num_value_heads(self):
        """linear_num_value_heads = ssm_inner_size / linear_value_head_dim."""
        cfg = {
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
            "ssm_inner_size": 4096,
            "linear_value_head_dim": 128,
        }
        _postprocess_qwen35_config(cfg)
        assert cfg["linear_num_value_heads"] == 32

    def test_linear_key_head_dim_defaults(self):
        """linear_key_head_dim defaults to linear_value_head_dim."""
        cfg = {
            "num_hidden_layers": 32,
            "full_attention_interval": 4,
            "linear_value_head_dim": 128,
        }
        _postprocess_qwen35_config(cfg)
        assert cfg["linear_key_head_dim"] == 128


class TestTensorProcessorTransforms:
    """Test Qwen35TensorProcessor weight transforms."""

    def setup_method(self):
        self.processor = Qwen35TensorProcessor()

    def test_preprocess_name_passthrough(self):
        """preprocess_name should return names unchanged."""
        assert self.processor.preprocess_name("model.layers.0.linear_attn.dt_bias") == \
            "model.layers.0.linear_attn.dt_bias"
        assert self.processor.preprocess_name("model.embed_tokens.weight") == \
            "model.embed_tokens.weight"

    def test_conv1d_unsqueeze(self):
        """conv1d weights: [C, K] → [C, 1, K]."""
        weights_2d = np.random.randn(8192, 4).astype(np.float32)
        result = self.processor.process(weights_2d, "blk.0.ssm_conv1d.weight")
        assert result.weights.shape == (8192, 1, 4)
        np.testing.assert_array_equal(result.weights[:, 0, :], weights_2d)

    def test_conv1d_3d_passthrough(self):
        """If conv1d weights are already 3D, don't unsqueeze again."""
        weights_3d = np.random.randn(8192, 1, 4).astype(np.float32)
        result = self.processor.process(weights_3d, "blk.0.ssm_conv1d.weight")
        assert result.weights.shape == (8192, 1, 4)

    def test_a_log_transform(self):
        """ssm_a: GGUF stores -exp(A_log) → recover A_log via log(-x)."""
        a_log_original = np.array([-2.0, -1.5, -1.0, -0.5], dtype=np.float32)
        gguf_stored = -np.exp(a_log_original)  # What the converter stored
        result = self.processor.process(gguf_stored, "blk.0.ssm_a")
        np.testing.assert_allclose(result.weights, a_log_original, atol=1e-6)

    def test_norm_weight_minus_one(self):
        """Norm weights: reverse pre_rmsnorm_1p (GGUF stores w+1)."""
        original = np.array([0.5, 1.0, -0.3], dtype=np.float32)
        gguf_stored = original + 1.0
        result = self.processor.process(gguf_stored, "blk.0.attn_norm.weight")
        np.testing.assert_allclose(result.weights, original, atol=1e-6)

    def test_norm_weight_ffn_norm(self):
        """ffn_norm should also get -1 treatment."""
        original = np.ones(10, dtype=np.float32) * 0.7
        result = self.processor.process(original + 1.0, "blk.5.ffn_norm.weight")
        np.testing.assert_allclose(result.weights, original, atol=1e-6)

    def test_norm_weight_output_norm(self):
        """output_norm should also get -1 treatment."""
        original = np.ones(10, dtype=np.float32) * 0.9
        result = self.processor.process(original + 1.0, "output_norm.weight")
        np.testing.assert_allclose(result.weights, original, atol=1e-6)

    def test_ssm_norm_excluded(self):
        """ssm_norm (linear_attn.norm) should NOT get -1 treatment."""
        weights = np.ones(10, dtype=np.float32) * 1.5
        result = self.processor.process(weights.copy(), "blk.0.ssm_norm.weight")
        np.testing.assert_array_equal(result.weights, weights)

    def test_regular_weight_passthrough(self):
        """Weights without special name patterns pass through unchanged."""
        weights = np.random.randn(100, 200).astype(np.float32)
        result = self.processor.process(weights.copy(), "blk.0.attn_q.weight")
        np.testing.assert_array_equal(result.weights, weights)


class TestTensorProcessorDtBiasFallback:
    """Test dt_bias → ssm_dt.bias fallback mapping."""

    def setup_method(self):
        self.processor = Qwen35TensorProcessor()

    def test_dt_bias_mapping_layer_0(self):
        mapping = {}
        self.processor.perform_fallback_tensor_mapping(
            mapping, "", "", "model.layers.0.linear_attn.dt_bias")
        assert "blk.0.ssm_dt.bias" in mapping
        assert mapping["blk.0.ssm_dt.bias"] == "model.layers.0.linear_attn.dt_bias"

    def test_dt_bias_mapping_layer_23(self):
        mapping = {}
        self.processor.perform_fallback_tensor_mapping(
            mapping, "", "", "model.layers.23.linear_attn.dt_bias")
        assert "blk.23.ssm_dt.bias" in mapping
        assert mapping["blk.23.ssm_dt.bias"] == "model.layers.23.linear_attn.dt_bias"

    def test_dt_bias_with_qual_name(self):
        """qual_name prefix is prepended to the HF name in the map."""
        mapping = {}
        self.processor.perform_fallback_tensor_mapping(
            mapping, "", "model.", "model.layers.5.linear_attn.dt_bias")
        assert mapping["blk.5.ssm_dt.bias"] == "model.model.layers.5.linear_attn.dt_bias"

    def test_non_dt_bias_no_mapping(self):
        """Non-dt_bias names should not add any fallback entries."""
        mapping = {}
        self.processor.perform_fallback_tensor_mapping(
            mapping, "", "", "model.layers.0.linear_attn.conv1d.weight")
        assert len(mapping) == 0


# ---------------------------------------------------------------------------
# Integration Tests — Require model files
# ---------------------------------------------------------------------------


class TestGGUFCompatLoading:
    """Integration: full AutoModel loading via patched HF path."""

    @pytest.fixture(autouse=True)
    def _check_model(self):
        if not QWEN35_4B_GGUF.exists():
            pytest.skip(f"Model not found: {QWEN35_4B_GGUF}")

    def test_automodel_loads_successfully(self):
        """AutoModelForCausalLM.from_pretrained with gguf_file works."""
        import torch
        patch_hf_gguf_qwen35()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            '.', gguf_file=str(QWEN35_4B_GGUF), dtype='float32',
        )
        assert type(model).__name__ == "Qwen3_5ForCausalLM"

    def test_all_tensors_loaded(self):
        """All 426 GGUF tensors map to HF parameters (no missing keys)."""
        import torch
        patch_hf_gguf_qwen35()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            '.', gguf_file=str(QWEN35_4B_GGUF), dtype='float32',
        )
        # No missing keys in state dict
        state = model.state_dict()
        for name, param in model.named_parameters():
            assert name in state, f"Missing param: {name}"
            assert param.shape == state[name].shape

    def test_config_values(self):
        """Config values match Qwen3.5-4B architecture."""
        import torch
        patch_hf_gguf_qwen35()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            '.', gguf_file=str(QWEN35_4B_GGUF), dtype='float32',
        )
        cfg = model.config
        assert cfg.model_type == "qwen3_5_text"
        assert cfg.num_hidden_layers == 32
        assert cfg.hidden_size == 2560
        assert len(cfg.layer_types) == 32
        # 8 full attention layers (every 4th)
        assert sum(1 for t in cfg.layer_types if t == "full_attention") == 8

    def test_conv1d_weights_3d(self):
        """conv1d weights should be 3D [C, 1, K] after transform."""
        import torch
        patch_hf_gguf_qwen35()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            '.', gguf_file=str(QWEN35_4B_GGUF), dtype='float32',
        )
        conv1d = model.state_dict()["model.layers.0.linear_attn.conv1d.weight"]
        assert conv1d.ndim == 3
        assert conv1d.shape[1] == 1  # group dimension

    def test_dt_bias_loaded(self):
        """dt_bias parameters should be present and non-zero."""
        import torch
        patch_hf_gguf_qwen35()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            '.', gguf_file=str(QWEN35_4B_GGUF), dtype='float32',
        )
        dt_bias = model.state_dict()["model.layers.0.linear_attn.dt_bias"]
        assert dt_bias is not None
        assert dt_bias.numel() > 0
        # Should not be all zeros (that would mean it wasn't loaded)
        assert not torch.all(dt_bias == 0).item()
