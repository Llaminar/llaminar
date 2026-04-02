"""
Transformers API Compatibility Tests

Catches breaking changes in HuggingFace Transformers APIs that the snapshot
generation and reference model code depends on. These are the fragile internal
API touchpoints that have broken across transformers version upgrades (e.g.,
the 4.x → 5.x `apply_rotary_pos_emb` signature change).

All tests in this module are fast unit tests with NO model loading.

@author David Sanftenberg
@date 2026-04-02
"""

import inspect
import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# 1. apply_rotary_pos_emb signature & behavior
# ---------------------------------------------------------------------------


class TestApplyRotaryPosEmb:
    """Verify apply_rotary_pos_emb can be imported and called correctly."""

    def _get_apply_rotary_pos_emb(self, model_module: str):
        """Import apply_rotary_pos_emb from a specific model module."""
        mod = __import__(
            f"transformers.models.{model_module}.modeling_{model_module}",
            fromlist=["apply_rotary_pos_emb"],
        )
        return mod.apply_rotary_pos_emb

    @pytest.mark.parametrize("model_module", ["qwen2", "qwen3"])
    def test_import_succeeds(self, model_module):
        """apply_rotary_pos_emb is importable from Qwen2 and Qwen3."""
        fn = self._get_apply_rotary_pos_emb(model_module)
        assert callable(fn)

    @pytest.mark.parametrize("model_module", ["qwen2", "qwen3"])
    def test_signature_has_unsqueeze_dim(self, model_module):
        """5.x signature must accept (q, k, cos, sin, unsqueeze_dim=1)."""
        fn = self._get_apply_rotary_pos_emb(model_module)
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params[:4] == ["q", "k", "cos", "sin"], (
            f"Expected (q, k, cos, sin, ...) but got {params}"
        )
        assert "unsqueeze_dim" in sig.parameters, (
            "Missing 'unsqueeze_dim' parameter — API may have changed"
        )

    @pytest.mark.parametrize("model_module", ["qwen2", "qwen3"])
    def test_no_position_ids_param(self, model_module):
        """position_ids must NOT be a parameter (removed in 5.x)."""
        fn = self._get_apply_rotary_pos_emb(model_module)
        sig = inspect.signature(fn)
        assert "position_ids" not in sig.parameters, (
            "Found 'position_ids' parameter — this is the old 4.x signature. "
            "Snapshot generation scripts pass position_ids as a positional arg, "
            "which breaks when 5.x expects unsqueeze_dim (an int)."
        )

    @pytest.mark.parametrize("model_module", ["qwen2", "qwen3"])
    def test_callable_with_four_tensor_args(self, model_module):
        """Calling with (q, k, cos, sin) must succeed and return two tensors."""
        fn = self._get_apply_rotary_pos_emb(model_module)
        batch, heads, seq, dim = 1, 2, 4, 8
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        cos = torch.randn(1, seq, dim)
        sin = torch.randn(1, seq, dim)
        q_out, k_out = fn(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    @pytest.mark.parametrize("model_module", ["qwen2", "qwen3"])
    def test_rejects_tensor_as_fifth_positional_arg(self, model_module):
        """Passing a tensor where unsqueeze_dim (int) is expected must fail.

        This is the exact bug that broke generate_qwen_pipeline_snapshots.py
        when transformers was upgraded from 4.x to 5.x.
        """
        fn = self._get_apply_rotary_pos_emb(model_module)
        batch, heads, seq, dim = 1, 2, 4, 8
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        cos = torch.randn(1, seq, dim)
        sin = torch.randn(1, seq, dim)
        position_ids = torch.arange(seq).unsqueeze(0)  # old-style arg

        with pytest.raises(TypeError):
            fn(q, k, cos, sin, position_ids)


# ---------------------------------------------------------------------------
# 2. repeat_kv import and signature
# ---------------------------------------------------------------------------


class TestRepeatKV:
    """Verify repeat_kv API stability."""

    def test_import_from_qwen2(self):
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
        assert callable(repeat_kv)

    def test_signature(self):
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
        sig = inspect.signature(repeat_kv)
        params = list(sig.parameters.keys())
        assert "hidden_states" in params
        assert "n_rep" in params

    def test_expansion_shape(self):
        """repeat_kv should expand KV heads correctly."""
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
        batch, kv_heads, seq, dim = 1, 2, 4, 8
        x = torch.randn(batch, kv_heads, seq, dim)
        out = repeat_kv(x, n_rep=7)
        assert out.shape == (batch, 14, seq, dim)

    def test_identity_when_n_rep_1(self):
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
        x = torch.randn(1, 4, 8, 16)
        out = repeat_kv(x, n_rep=1)
        assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# 3. Model class imports (Qwen2, Qwen3, Qwen3.5, Llama)
# ---------------------------------------------------------------------------


class TestModelClassImports:
    """Verify core model classes are importable from transformers."""

    def test_qwen2_classes(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        assert Qwen2Config is not None
        assert Qwen2ForCausalLM is not None

    def test_qwen3_classes(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        assert Qwen3Config is not None
        assert Qwen3ForCausalLM is not None

    def test_qwen35_classes(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextConfig,
            Qwen3_5ForCausalLM,
        )
        assert Qwen3_5TextConfig is not None
        assert Qwen3_5ForCausalLM is not None

    def test_llama_classes(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        assert LlamaConfig is not None
        assert LlamaForCausalLM is not None

    def test_auto_classes(self):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        assert AutoConfig is not None
        assert AutoModelForCausalLM is not None
        assert AutoTokenizer is not None


# ---------------------------------------------------------------------------
# 4. Snapshot generation helpers — pure functions that don't need a model
# ---------------------------------------------------------------------------


class TestInferSnapshotOutputDir:
    """Test infer_snapshot_output_dir path sanitization."""

    def test_basic_gguf(self):
        from python.reference.generate_qwen_pipeline_snapshots import (
            infer_snapshot_output_dir,
        )
        result = infer_snapshot_output_dir("models/qwen2.5-0.5b-instruct-q4_0.gguf")
        assert str(result) == "pytorch_qwen2_5_0_5b_instruct_q4_0_snapshots"

    def test_empty_stem(self):
        from python.reference.generate_qwen_pipeline_snapshots import (
            infer_snapshot_output_dir,
        )
        # ".gguf" has stem ".gguf" which sanitizes to "gguf"
        result = infer_snapshot_output_dir(".gguf")
        assert "pytorch_" in str(result)
        assert "snapshots" in str(result)

    def test_special_characters(self):
        from python.reference.generate_qwen_pipeline_snapshots import (
            infer_snapshot_output_dir,
        )
        result = infer_snapshot_output_dir("path/to/my model (v2).gguf")
        # Should sanitize spaces and parens
        name = str(result)
        assert " " not in name
        assert "(" not in name
        assert ")" not in name


class TestApplyRmsNorm:
    """Test the manual RMSNorm implementation in QwenPipelineCapture."""

    def test_rmsnorm_basic(self):
        """RMSNorm output should be normalized and scaled by gamma."""
        from python.reference.generate_qwen_pipeline_snapshots import (
            QwenPipelineCapture,
        )
        cap = QwenPipelineCapture.__new__(QwenPipelineCapture)

        hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        gamma = torch.ones(4)
        out = cap._apply_rmsnorm(hidden, gamma)
        assert out.shape == hidden.shape
        # With gamma=1, RMSNorm normalizes to unit RMS
        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_rmsnorm_with_gamma(self):
        """Gamma should scale the output element-wise."""
        from python.reference.generate_qwen_pipeline_snapshots import (
            QwenPipelineCapture,
        )
        cap = QwenPipelineCapture.__new__(QwenPipelineCapture)

        hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        gamma_ones = torch.ones(4)
        gamma_twos = 2.0 * torch.ones(4)
        out_1 = cap._apply_rmsnorm(hidden, gamma_ones)
        out_2 = cap._apply_rmsnorm(hidden, gamma_twos)
        assert torch.allclose(out_2, 2.0 * out_1, atol=1e-6)

    def test_rmsnorm_batch_seqlen(self):
        """Should work with arbitrary batch and sequence dimensions."""
        from python.reference.generate_qwen_pipeline_snapshots import (
            QwenPipelineCapture,
        )
        cap = QwenPipelineCapture.__new__(QwenPipelineCapture)

        hidden = torch.randn(2, 5, 64)
        gamma = torch.ones(64)
        out = cap._apply_rmsnorm(hidden, gamma)
        assert out.shape == (2, 5, 64)
        assert not torch.isnan(out).any()


class TestExpandKVHeadsForGQA:
    """Test the GQA head expansion helper."""

    def test_no_expansion_mha(self):
        """When n_heads == n_kv_heads, no expansion needed."""
        from python.reference.generate_qwen_pipeline_snapshots import (
            QwenPipelineCapture,
        )
        cap = QwenPipelineCapture.__new__(QwenPipelineCapture)

        k = torch.randn(1, 14, 8, 64)
        v = torch.randn(1, 14, 8, 64)
        k_out, v_out = cap._expand_kv_heads_for_gqa(k, v, n_heads=14, n_kv_heads=14)
        assert torch.equal(k, k_out)
        assert torch.equal(v, v_out)

    def test_gqa_expansion_shape(self):
        """Qwen2.5-0.5B: 14 heads, 2 KV heads → 7× expansion."""
        from python.reference.generate_qwen_pipeline_snapshots import (
            QwenPipelineCapture,
        )
        cap = QwenPipelineCapture.__new__(QwenPipelineCapture)

        k = torch.randn(1, 2, 8, 64)
        v = torch.randn(1, 2, 8, 64)
        k_out, v_out = cap._expand_kv_heads_for_gqa(k, v, n_heads=14, n_kv_heads=2)
        assert k_out.shape == (1, 14, 8, 64)
        assert v_out.shape == (1, 14, 8, 64)

    def test_gqa_expansion_values(self):
        """Expanded heads must be copies of the original KV heads."""
        from python.reference.generate_qwen_pipeline_snapshots import (
            QwenPipelineCapture,
        )
        cap = QwenPipelineCapture.__new__(QwenPipelineCapture)

        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        k_out, v_out = cap._expand_kv_heads_for_gqa(k, v, n_heads=6, n_kv_heads=2)
        # Head 0,1,2 should all be copies of KV head 0
        assert torch.equal(k_out[:, 0], k_out[:, 1])
        assert torch.equal(k_out[:, 0], k_out[:, 2])
        # Head 3,4,5 should all be copies of KV head 1
        assert torch.equal(k_out[:, 3], k_out[:, 4])
        assert torch.equal(k_out[:, 3], k_out[:, 5])


# ---------------------------------------------------------------------------
# 5. Snapshot script import smoke tests — verify no stale API calls
# ---------------------------------------------------------------------------


class TestSnapshotScriptImports:
    """Verify that snapshot generation scripts can be imported without errors.

    These catch top-level import-time calls to transformers APIs that may have
    changed (e.g., the `from transformers.models.qwen2.modeling_qwen2 import
    apply_rotary_pos_emb, repeat_kv` at the top of generate_test_snapshots.py).
    """

    def test_generate_qwen_pipeline_snapshots_importable(self):
        import python.reference.generate_qwen_pipeline_snapshots as mod
        assert hasattr(mod, "QwenPipelineCapture")
        assert hasattr(mod, "infer_snapshot_output_dir")

    def test_generate_test_snapshots_importable(self):
        import python.reference.generate_test_snapshots as mod
        assert hasattr(mod, "PipelineStageCapture")

    def test_generate_attention_snapshots_importable(self):
        import python.reference.generate_attention_snapshots as mod
        assert hasattr(mod, "AttentionSnapshotCapture")


# ---------------------------------------------------------------------------
# 6. Model reference module import smoke tests
# ---------------------------------------------------------------------------


class TestReferenceModuleImports:
    """Verify that all model reference modules can be imported."""

    def test_qwen_module(self):
        from python.reference.qwen import QwenReferenceModel
        assert QwenReferenceModel is not None

    def test_qwen35_module(self):
        from python.reference.qwen35 import Qwen35ReferenceModel
        assert Qwen35ReferenceModel is not None

    def test_llama_module(self):
        from python.reference.llama import LlamaReferenceModel
        assert LlamaReferenceModel is not None

    def test_base_module(self):
        from python.reference.base import (
            AbstractReferenceModel,
            HuggingFaceReferenceModel,
        )
        assert AbstractReferenceModel is not None
        assert HuggingFaceReferenceModel is not None

    def test_registry_has_all_models(self):
        from python.reference.registry import ModelRegistry
        models = ModelRegistry.list_models()
        assert "qwen" in models
        assert "llama" in models
        assert "qwen35" in models


# ---------------------------------------------------------------------------
# 7. Qwen3.5 model-specific API surface
# ---------------------------------------------------------------------------


class TestQwen35TransformersAPI:
    """Verify Qwen 3.5 specific classes have expected attributes."""

    def test_text_config_accepts_gdn_params(self):
        """Qwen3_5TextConfig must accept GDN-specific kwargs."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextConfig
        # These are the GDN-specific config fields our code depends on
        cfg = Qwen3_5TextConfig(
            vocab_size=151936,
            hidden_size=896,
            num_hidden_layers=4,
            num_attention_heads=14,
            num_key_value_heads=2,
            intermediate_size=4864,
            attention_head_dim=64,
            short_conv_width=4,
            gated_linear_unit_activation="silu",
        )
        assert cfg.hidden_size == 896
        assert cfg.num_hidden_layers == 4

    def test_model_has_linear_attn_layers(self):
        """Qwen3_5ForCausalLM must create linear_attn on GDN layers."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextConfig,
            Qwen3_5ForCausalLM,
        )
        cfg = Qwen3_5TextConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            attention_head_dim=32,
            short_conv_width=4,
        )
        model = Qwen3_5ForCausalLM(cfg)
        # Qwen3.5 3:1 pattern: layers 0,1,2 are GDN, layer 3 is full attention
        for i in range(3):
            layer = model.model.layers[i]
            assert hasattr(layer, "linear_attn"), (
                f"Layer {i} should have linear_attn (GDN layer)"
            )
        # Layer 3 should be full attention (self_attn, no linear_attn)
        layer3 = model.model.layers[3]
        assert hasattr(layer3, "self_attn")
