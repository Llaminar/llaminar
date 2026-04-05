"""
Tests for Qwen 3.5-4B Reference Model (Q8_0)

Unit tests (no model files required):
  - 4B-specific config extraction
  - Asymmetric KV heads (n_kv_heads=4, n_v_heads=32)
  - Layer type pattern for 32 layers

Integration tests (require models/Qwen3.5-4B-Q8_0.gguf):
  - GGUF loading and state dict completeness
  - Forward pass produces valid logits
  - Heterogeneous layer snapshot capture
  - GDN vs FA layer dispatch verification

@author David Sanftenberg
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from python.reference.loaders.tensor_name_mapper import TensorNameMapper
from python.reference.loaders.gguf_loader import GGUFLoader
from python.reference.pipeline_stages import PipelineStage


# ---------------------------------------------------------------------------
# Unit Tests — No model files required
# ---------------------------------------------------------------------------


class TestQwen35_4B_Config:
    """Test 4B-specific configuration dimensions."""

    def test_config_from_4b_metadata(self):
        """Build config dict from simulated Qwen 3.5-4B GGUF metadata."""
        from python.reference.loaders.gguf_parser import GGUFParser

        parser = GGUFParser.__new__(GGUFParser)
        parser.metadata = {
            'general.architecture': 'qwen35',
            'qwen35.embedding_length': 2560,
            'qwen35.attention.head_count': 32,
            'qwen35.block_count': 32,
            'qwen35.feed_forward_length': 9216,
            'qwen35.context_length': 262144,
            'qwen35.attention.head_count_kv': 4,
            'qwen35.attention.key_length': 160,
            'qwen35.attention.layer_norm_rms_epsilon': 1e-6,
            'qwen35.rope.freq_base': 10000000.0,
            'qwen35.full_attention_interval': 4,
            'qwen35.ssm.conv_kernel': 4,
            'qwen35.ssm.group_count': 16,
            'qwen35.ssm.inner_size': 2560,
            'qwen35.ssm.state_size': 64,
            'qwen35.rope.dimension_sections': [14, 14, 12, 0],
            'tokenizer.ggml.tokens': ['a'] * 248320,
        }

        config = parser.get_config_dict()

        assert config['hidden_size'] == 2560
        assert config['num_attention_heads'] == 32
        assert config['num_hidden_layers'] == 32
        assert config['intermediate_size'] == 9216
        assert config['num_key_value_heads'] == 4
        assert config['head_dim'] == 160
        assert config['full_attention_interval'] == 4
        assert config['rope_dimension_sections'] == [14, 14, 12, 0]

    def test_4b_layer_type_pattern(self):
        """32-layer model should have FA at layers 3,7,11,15,19,23,27,31."""
        from python.reference.qwen35 import Qwen35ReferenceModel

        config_dict = {
            'num_hidden_layers': 32,
            'full_attention_interval': 4,
            'hidden_size': 2560,
            'num_attention_heads': 32,
            'num_key_value_heads': 4,
            'head_dim': 160,
            'intermediate_size': 9216,
            'vocab_size': 248320,
            'linear_num_key_heads': 16,
            'linear_value_head_dim': 64,
            'linear_conv_kernel_dim': 4,
            'ssm_inner_size': 2560,
        }
        cfg = Qwen35ReferenceModel._build_hf_config(config_dict)

        fa_layers = [i for i, t in enumerate(cfg.layer_types) if t == 'full_attention']
        gdn_layers = [i for i, t in enumerate(cfg.layer_types) if t == 'linear_attention']

        assert fa_layers == [3, 7, 11, 15, 19, 23, 27, 31]
        assert len(gdn_layers) == 24  # 32 - 8 FA layers

    def test_4b_tensor_name_mapping(self):
        """4B model uses same tensor name patterns as 0.8B."""
        mapper = TensorNameMapper('qwen35', full_attention_interval=4)

        # GDN layer (layer 0)
        assert (
            mapper.map_name('blk.0.attn_qkv.weight')
            == 'model.layers.0.linear_attn.in_proj_qkv.weight'
        )

        # FA layer (layer 3)
        assert (
            mapper.map_name('blk.3.attn_q.weight')
            == 'model.layers.3.self_attn.q_proj.weight'
        )

        # Deep layers (layer 31 = FA)
        assert (
            mapper.map_name('blk.31.attn_output.weight')
            == 'model.layers.31.self_attn.o_proj.weight'
        )

    def test_4b_asymmetric_kv_heads(self):
        """4B model: n_kv_heads=4 for K, n_v_heads=32 for V (repeat_interleave)."""
        from python.reference.qwen35 import Qwen35ReferenceModel

        config_dict = {
            'num_hidden_layers': 4,
            'full_attention_interval': 4,
            'hidden_size': 2560,
            'num_attention_heads': 32,
            'num_key_value_heads': 4,
            'head_dim': 160,
            'intermediate_size': 9216,
            'vocab_size': 100,
            'linear_num_key_heads': 16,
            'linear_value_head_dim': 64,
            'linear_conv_kernel_dim': 4,
            'ssm_inner_size': 2560,
        }
        cfg = Qwen35ReferenceModel._build_hf_config(config_dict)

        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 4
        # K needs repeat_interleave by 32/4 = 8


# ---------------------------------------------------------------------------
# Integration Tests — Require models/Qwen3.5-4B-Q8_0.gguf
# ---------------------------------------------------------------------------


class TestQwen35_4B_GGUFLoading:
    """Integration tests for 4B GGUF loading."""

    @pytest.fixture(scope='class')
    def state_dict(self, qwen35_4b_q8_model):
        """Load 4B GGUF once and share across all tests."""
        loader = GGUFLoader(str(qwen35_4b_q8_model))
        _, sd = loader.load(as_transformers_config=False)
        return sd

    def test_state_dict_has_gdn_projections(self, state_dict):
        """Layer 0 (GDN) should have separate projection tensors."""
        assert 'model.layers.0.linear_attn.in_proj_qkv.weight' in state_dict
        assert 'model.layers.0.linear_attn.in_proj_z.weight' in state_dict
        assert 'model.layers.0.linear_attn.in_proj_a.weight' in state_dict
        assert 'model.layers.0.linear_attn.in_proj_b.weight' in state_dict

    def test_state_dict_has_fa_tensors(self, state_dict):
        """Layer 3 (FA) should have standard Q/K/V/O projections."""
        assert 'model.layers.3.self_attn.q_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.k_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.v_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.o_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.q_norm.weight' in state_dict
        assert 'model.layers.3.self_attn.k_norm.weight' in state_dict

    def test_qkv_projection_shape(self, state_dict):
        """in_proj_qkv should have correct shape for 4B dimensions."""
        qkv = state_dict['model.layers.0.linear_attn.in_proj_qkv.weight']
        assert qkv.shape[1] == 2560, f"K-dim should be hidden_size=2560, got {qkv.shape[1]}"

    def test_fa_projection_shapes(self, state_dict):
        """FA layer projections should match 4B dimensions."""
        q = state_dict['model.layers.3.self_attn.q_proj.weight']
        k = state_dict['model.layers.3.self_attn.k_proj.weight']
        v = state_dict['model.layers.3.self_attn.v_proj.weight']

        # Q: [n_heads * head_dim, hidden] — K-dim should be 2560
        assert q.shape[1] == 2560, f"Q K-dim: {q.shape[1]}"
        # K and V: [n_kv_heads * head_dim, hidden] — K-dim should be 2560
        assert k.shape[1] == 2560, f"K K-dim: {k.shape[1]}"
        assert v.shape[1] == 2560, f"V K-dim: {v.shape[1]}"
        # K is smaller than Q due to GQA (n_kv_heads=4 < n_heads=32)
        assert k.shape[0] < q.shape[0], "K should be smaller than Q (GQA)"

    def test_ffn_shapes(self, state_dict):
        """FFN weight shapes should match 4B dimensions."""
        gate = state_dict['model.layers.0.mlp.gate_proj.weight']
        up = state_dict['model.layers.0.mlp.up_proj.weight']
        down = state_dict['model.layers.0.mlp.down_proj.weight']

        assert gate.shape == (9216, 2560), f"gate_proj shape: {gate.shape}"
        assert up.shape == (9216, 2560), f"up_proj shape: {up.shape}"
        assert down.shape == (2560, 9216), f"down_proj shape: {down.shape}"

    def test_conv1d_shape(self, state_dict):
        """conv1d.weight should be 3D."""
        conv = state_dict['model.layers.0.linear_attn.conv1d.weight']
        assert conv.dim() == 3
        # conv1d operates on the concatenated QKV+Z projection output
        assert conv.shape[1] == 1
        assert conv.shape[2] == 4  # kernel_dim=4

    def test_all_32_layers_present(self, state_dict):
        """All 32 layers should have at least norm and FFN weights."""
        for i in range(32):
            assert f'model.layers.{i}.input_layernorm.weight' in state_dict, \
                f"Missing input_layernorm for layer {i}"
            assert f'model.layers.{i}.mlp.gate_proj.weight' in state_dict, \
                f"Missing gate_proj for layer {i}"


class TestQwen35_4B_Inference:
    """Integration tests for 4B model inference."""

    @pytest.fixture(scope='class')
    def loaded_model(self, qwen35_4b_q8_model):
        """Load 4B model once and share across tests."""
        from python.reference import ModelRegistry
        model = ModelRegistry.create(
            'qwen35', checkpoint_path=str(qwen35_4b_q8_model), auto_load=False
        )
        model.load_model()
        return model

    def test_forward_produces_logits(self, loaded_model):
        """Forward pass should produce logits with correct vocab dim."""
        result = loaded_model.forward([1, 2, 3, 4, 5])
        logits = result['logits']
        assert logits.shape[0] == 1  # batch
        assert logits.shape[1] == 5  # seq_len
        assert logits.shape[2] == loaded_model.hf_config.vocab_size
        assert np.isfinite(logits).all(), "Logits contain NaN/Inf"

    def test_forward_no_nan(self, loaded_model):
        """All outputs should be finite."""
        result = loaded_model.forward([10, 20, 30])
        assert np.isfinite(result['logits']).all(), "NaN in logits"
        assert np.isfinite(result['hidden_states']).all(), "NaN in hidden states"

    def test_heterogeneous_snapshots(self, loaded_model):
        """GDN-specific stages should fire on GDN layers only."""
        result = loaded_model.forward(
            [1, 2, 3],
            capture_stages=[
                PipelineStage.GDN_CONV1D_OUTPUT,
                PipelineStage.GDN_DELTA_RULE_OUTPUT,
                PipelineStage.ATTENTION_OUTPUT,
            ],
        )
        snapshots = result['snapshots']

        # ATTENTION_OUTPUT fires on all 32 layers
        attn_layers = sorted([k[1] for k in snapshots if k[0] == PipelineStage.ATTENTION_OUTPUT])
        assert len(attn_layers) == 32

        # GDN stages fire on 24 GDN layers (not FA layers 3,7,11,15,19,23,27,31)
        gdn_layers = sorted([k[1] for k in snapshots if k[0] == PipelineStage.GDN_CONV1D_OUTPUT])
        assert len(gdn_layers) == 24
        assert 3 not in gdn_layers  # FA layer
        assert 7 not in gdn_layers
        assert 31 not in gdn_layers

    def test_embedding_shape(self, loaded_model):
        """Embedding output should be (1, seq_len, 2560) for 4B model."""
        result = loaded_model.forward(
            [1, 2, 3],
            capture_stages=[PipelineStage.EMBEDDING],
        )
        emb = result['snapshots'][(PipelineStage.EMBEDDING, -1)]
        assert emb.shape == (1, 3, 2560), f"Expected (1, 3, 2560), got {emb.shape}"
