"""
Tests for Qwen 3.5 Gated Delta Net Reference Implementation

Unit tests (no model files required):
  - Tensor name mapping
  - Config extraction from GGUF metadata
  - Fused tensor reconstruction (QKVZ, BA)
  - Transform functions (norm -1, A_log, conv1d unsqueeze)
  - Model registry

Integration tests (require models/Qwen3.5-0.8B-Q4_0.gguf):
  - GGUF loading with fused tensors
  - State dict completeness (no missing keys)
  - Forward pass produces valid logits
  - Snapshot capture for heterogeneous layers

@author David Sanftenberg
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from python.reference.loaders.tensor_name_mapper import (
    TensorNameMapper,
    detect_model_type_from_metadata,
)
from python.reference.loaders.gguf_loader import GGUFLoader
from python.reference.pipeline_stages import PipelineStage, stage_to_string


# ---------------------------------------------------------------------------
# Unit Tests — No model files required
# ---------------------------------------------------------------------------


class TestQwen35TensorNameMapping:
    """Test GGUF → HuggingFace tensor name mapping for Qwen 3.5."""

    def setup_method(self):
        self.mapper = TensorNameMapper('qwen35', full_attention_interval=4)

    def test_model_type_detection(self):
        """Detect qwen35 from GGUF metadata."""
        meta = {'general.architecture': 'qwen35', 'qwen35.block_count': 24}
        assert detect_model_type_from_metadata(meta) == 'qwen35'

    def test_global_tensors(self):
        """Global (non-layer) tensor mappings."""
        assert self.mapper.map_name('token_embd.weight') == 'model.embed_tokens.weight'
        assert self.mapper.map_name('output_norm.weight') == 'model.norm.weight'
        assert self.mapper.map_name('output.weight') == 'lm_head.weight'

    def test_shared_layer_tensors(self):
        """Tensors shared by both linear and full attention layers."""
        assert (
            self.mapper.map_name('blk.0.attn_norm.weight')
            == 'model.layers.0.input_layernorm.weight'
        )
        assert (
            self.mapper.map_name('blk.5.post_attention_norm.weight')
            == 'model.layers.5.post_attention_layernorm.weight'
        )
        assert (
            self.mapper.map_name('blk.2.ffn_gate.weight')
            == 'model.layers.2.mlp.gate_proj.weight'
        )

    def test_full_attention_layer_tensors(self):
        """Full (softmax) attention layer tensors."""
        assert (
            self.mapper.map_name('blk.3.attn_q.weight')
            == 'model.layers.3.self_attn.q_proj.weight'
        )
        assert (
            self.mapper.map_name('blk.3.attn_k.weight')
            == 'model.layers.3.self_attn.k_proj.weight'
        )
        assert (
            self.mapper.map_name('blk.3.attn_v.weight')
            == 'model.layers.3.self_attn.v_proj.weight'
        )
        assert (
            self.mapper.map_name('blk.3.attn_output.weight')
            == 'model.layers.3.self_attn.o_proj.weight'
        )
        assert (
            self.mapper.map_name('blk.3.attn_q_norm.weight')
            == 'model.layers.3.self_attn.q_norm.weight'
        )
        assert (
            self.mapper.map_name('blk.3.attn_k_norm.weight')
            == 'model.layers.3.self_attn.k_norm.weight'
        )

    def test_linear_attention_direct_tensors(self):
        """Linear (GDN) attention layer tensors with direct mappings."""
        assert (
            self.mapper.map_name('blk.0.ssm_conv1d.weight')
            == 'model.layers.0.linear_attn.conv1d.weight'
        )
        assert (
            self.mapper.map_name('blk.0.ssm_dt.bias')
            == 'model.layers.0.linear_attn.dt_bias'
        )
        assert (
            self.mapper.map_name('blk.0.ssm_a')
            == 'model.layers.0.linear_attn.A_log'
        )
        assert (
            self.mapper.map_name('blk.0.ssm_norm.weight')
            == 'model.layers.0.linear_attn.norm.weight'
        )
        assert (
            self.mapper.map_name('blk.0.ssm_out.weight')
            == 'model.layers.0.linear_attn.out_proj.weight'
        )

    def test_linear_attention_projection_tensors(self):
        """Linear attention projection tensors should map directly (no fusion)."""
        assert (
            self.mapper.map_name('blk.0.attn_qkv.weight')
            == 'model.layers.0.linear_attn.in_proj_qkv.weight'
        )
        assert (
            self.mapper.map_name('blk.0.attn_gate.weight')
            == 'model.layers.0.linear_attn.in_proj_z.weight'
        )
        assert (
            self.mapper.map_name('blk.0.ssm_alpha.weight')
            == 'model.layers.0.linear_attn.in_proj_a.weight'
        )
        assert (
            self.mapper.map_name('blk.0.ssm_beta.weight')
            == 'model.layers.0.linear_attn.in_proj_b.weight'
        )


class TestQwen35Transforms:
    """Test the data transforms applied during loading."""

    def test_norm_subtract_one(self):
        """Norm weights (except linear_attn.norm) should have -1 applied."""
        loader = GGUFLoader.__new__(GGUFLoader)
        t = torch.tensor([2.0, 1.5, 1.0])

        # input_layernorm: should subtract 1
        result = loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.attn_norm.weight',
            'model.layers.0.input_layernorm.weight'
        )
        torch.testing.assert_close(result, torch.tensor([1.0, 0.5, 0.0]))

        # post_attention_layernorm: should subtract 1
        result = loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.post_attention_norm.weight',
            'model.layers.0.post_attention_layernorm.weight'
        )
        torch.testing.assert_close(result, torch.tensor([1.0, 0.5, 0.0]))

        # output_norm: should subtract 1
        result = loader._apply_qwen35_transforms(
            t.clone(), 'output_norm.weight', 'model.norm.weight'
        )
        torch.testing.assert_close(result, torch.tensor([1.0, 0.5, 0.0]))

    def test_linear_attn_norm_no_subtract(self):
        """linear_attn.norm.weight should NOT have -1 applied."""
        loader = GGUFLoader.__new__(GGUFLoader)
        t = torch.tensor([2.0, 1.5, 1.0])

        result = loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.ssm_norm.weight',
            'model.layers.0.linear_attn.norm.weight'
        )
        torch.testing.assert_close(result, torch.tensor([2.0, 1.5, 1.0]))

    def test_a_log_transform(self):
        """A_log: GGUF stores -exp(A_log), should reverse to log(-x)."""
        loader = GGUFLoader.__new__(GGUFLoader)
        original_a_log = torch.tensor([1.0, 2.0, 0.5])
        gguf_value = -torch.exp(original_a_log)

        result = loader._apply_qwen35_transforms(
            gguf_value.clone(), 'blk.0.ssm_a',
            'model.layers.0.linear_attn.A_log'
        )
        torch.testing.assert_close(result, original_a_log, atol=1e-6, rtol=1e-5)

    def test_conv1d_unsqueeze(self):
        """conv1d: GGUF squeezes dim-1, should unsqueeze back to (out, 1, kernel)."""
        loader = GGUFLoader.__new__(GGUFLoader)
        t = torch.randn(6144, 4)

        result = loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.ssm_conv1d.weight',
            'model.layers.0.linear_attn.conv1d.weight'
        )
        assert result.shape == (6144, 1, 4)

    def test_non_transform_passthrough(self):
        """Tensors without special transforms should pass through unchanged."""
        loader = GGUFLoader.__new__(GGUFLoader)
        t = torch.randn(3584, 1024)

        result = loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.ffn_gate.weight',
            'model.layers.0.mlp.gate_proj.weight'
        )
        torch.testing.assert_close(result, t)


class TestQwen35ConfigExtraction:
    """Test GGUF config dict extraction for Qwen 3.5."""

    def test_qwen35_config_from_metadata(self):
        """Build config dict from simulated Qwen 3.5 GGUF metadata."""
        from python.reference.loaders.gguf_parser import GGUFParser

        # Create a mock parser with qwen35 metadata
        parser = GGUFParser.__new__(GGUFParser)
        parser.metadata = {
            'general.architecture': 'qwen35',
            'qwen35.embedding_length': 1024,
            'qwen35.attention.head_count': 8,
            'qwen35.block_count': 24,
            'qwen35.feed_forward_length': 3584,
            'qwen35.context_length': 262144,
            'qwen35.attention.head_count_kv': 2,
            'qwen35.attention.key_length': 256,
            'qwen35.attention.layer_norm_rms_epsilon': 1e-6,
            'qwen35.rope.freq_base': 10000000.0,
            'qwen35.full_attention_interval': 4,
            'qwen35.ssm.conv_kernel': 4,
            'qwen35.ssm.group_count': 16,
            'qwen35.ssm.inner_size': 2048,
            'qwen35.ssm.state_size': 128,
            'qwen35.rope.dimension_sections': [11, 11, 10, 0],
            'tokenizer.ggml.tokens': ['a'] * 248320,
        }

        config = parser.get_config_dict()

        assert config['hidden_size'] == 1024
        assert config['num_attention_heads'] == 8
        assert config['num_hidden_layers'] == 24
        assert config['intermediate_size'] == 3584
        assert config['num_key_value_heads'] == 2
        assert config['head_dim'] == 256
        assert config['full_attention_interval'] == 4
        assert config['linear_num_key_heads'] == 16
        assert config['linear_value_head_dim'] == 128
        assert config['linear_conv_kernel_dim'] == 4
        assert config['rope_dimension_sections'] == [11, 11, 10, 0]

    def test_qwen36_nextn_sidecar_excluded_from_main_layers(self):
        """Qwen3.6 nextn/MTP sidecar blocks are not main decoder layers."""
        from python.reference.loaders.gguf_parser import GGUFParser

        parser = GGUFParser.__new__(GGUFParser)
        parser.metadata = {
            'general.architecture': 'qwen35',
            'qwen35.embedding_length': 5120,
            'qwen35.attention.head_count': 40,
            'qwen35.block_count': 65,
            'qwen35.feed_forward_length': 27648,
            'qwen35.context_length': 262144,
            'qwen35.nextn_predict_layers': 1,
            'qwen35.full_attention_interval': 4,
            'tokenizer.ggml.tokens': ['a'] * 248320,
        }
        parser.tensors = [
            'blk.63.attn_norm.weight',
            'blk.64.attn_norm.weight',
            'blk.64.nextn.eh_proj.weight',
        ]

        config = parser.get_config_dict()

        assert config['num_hidden_layers'] == 64


class TestQwen35ModelRegistry:
    """Test model registry for Qwen 3.5."""

    def test_registry_has_qwen35(self):
        from python.reference import ModelRegistry
        assert ModelRegistry.is_registered('qwen35')

    def test_create_without_loading(self):
        from python.reference import ModelRegistry
        model = ModelRegistry.create('qwen35', checkpoint_path='dummy.gguf', auto_load=False)
        assert model.model_name == 'qwen35'


class TestQwen35SnapshotGeneration:
    """Unit tests for snapshot/metadata generation helpers."""

    def test_metadata_only_prefill_and_decode_suppresses_snapshot_capture(self, tmp_path):
        from python.reference.generate_qwen35_pipeline_snapshots import run_prefill_and_decode

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                assert prompt == "hello"
                assert return_tensors == "pt"
                return {"input_ids": torch.tensor([[10, 11]])}

        class FakeCache:
            def __init__(self):
                self.length = 2

            def get_seq_length(self):
                return self.length

        class FakeModel:
            def __init__(self):
                self.tokenizer = FakeTokenizer()
                self.capture_args = []
                self.clear_count = 0

            def forward(self, token_ids, **kwargs):
                self.capture_args.append(kwargs.get("capture_stages"))
                vocab = 32
                logits = np.zeros((1, len(token_ids), vocab), dtype=np.float32)
                logits[0, -1, 20 + len(self.capture_args)] = 1.0
                return {"logits": logits, "past_key_values": FakeCache()}

            def get_snapshots(self):
                raise AssertionError("metadata-only generation should not request snapshots")

            def clear_snapshots(self):
                self.clear_count += 1

        model = FakeModel()
        total, token_ids, decode_tokens = run_prefill_and_decode(
            model,
            "hello",
            decode_steps=2,
            output_dir=tmp_path,
            save_snapshots=False,
        )

        assert total == 0
        assert token_ids == [10, 11]
        assert decode_tokens == [21, 22, 23]
        assert model.capture_args == [[], [], []]
        assert model.clear_count == 3
        assert not list(tmp_path.glob("*.npy"))

    def test_decode_only_snapshots_skip_prefill_payloads(self, tmp_path):
        from python.reference.generate_qwen35_pipeline_snapshots import run_prefill_and_decode

        class FakeTokenizer:
            def __call__(self, prompt, return_tensors):
                assert prompt == "hello"
                assert return_tensors == "pt"
                return {"input_ids": torch.tensor([[10, 11]])}

        class FakeModel:
            def __init__(self):
                self.tokenizer = FakeTokenizer()
                self.capture_args = []
                self.clear_count = 0

            def forward(self, token_ids, **kwargs):
                self.capture_args.append(kwargs.get("capture_stages"))
                vocab = 32
                logits = np.zeros((1, len(token_ids), vocab), dtype=np.float32)
                logits[0, -1, 20 + len(self.capture_args)] = 1.0
                return {"logits": logits, "past_key_values": object()}

            def get_snapshots(self):
                return {
                    (PipelineStage.LM_HEAD, -1):
                        np.array([[float(len(self.capture_args))]], dtype=np.float32)
                }

            def clear_snapshots(self):
                self.clear_count += 1

        model = FakeModel()
        total, token_ids, decode_tokens = run_prefill_and_decode(
            model,
            "hello",
            decode_steps=2,
            output_dir=tmp_path,
            save_snapshots=True,
            save_prefill_snapshots=False,
            save_decode_snapshots=True,
        )

        assert total == 2
        assert token_ids == [10, 11]
        assert decode_tokens == [21, 22, 23]
        assert model.capture_args == [[], None, None]
        assert model.clear_count == 1
        assert sorted(p.name for p in tmp_path.glob("*.npy")) == [
            "decode_step0_LM_HEAD.npy",
            "decode_step1_LM_HEAD.npy",
        ]

    def test_write_metadata_creates_output_directory(self, tmp_path):
        from python.reference.generate_qwen35_pipeline_snapshots import write_metadata

        class FakeConfig:
            architectures = ["Qwen3_5TextConfig"]
            num_hidden_layers = 64
            num_attention_heads = 24
            num_key_value_heads = 4
            hidden_size = 5120
            head_dim = 256
            intermediate_size = 17408
            vocab_size = 248320

        class FakeHFModel:
            config = FakeConfig()

        class FakeModel:
            hf_model = FakeHFModel()

        output_dir = tmp_path / "missing" / "qwen36"
        write_metadata(
            output_dir,
            "/models/qwen36.gguf",
            FakeModel(),
            "hello",
            [1, 2],
            1,
            [3, 4],
        )

        metadata = output_dir / "metadata.txt"
        assert metadata.exists()
        text = metadata.read_text()
        assert "token_ids: 1,2" in text
        assert "decode_tokens: 3,4" in text


class TestQwen35PipelineStages:
    """Unit tests for GDN-specific pipeline stages (no model required)."""

    def test_gdn_stages_exist_in_enum(self):
        """GDN pipeline stages should be defined in PipelineStage enum."""
        assert hasattr(PipelineStage, 'GDN_CONV1D_OUTPUT')
        assert hasattr(PipelineStage, 'GDN_DELTA_RULE_OUTPUT')
        assert hasattr(PipelineStage, 'GDN_NORM_GATE_OUTPUT')

    def test_gdn_stages_have_string_mapping(self):
        """GDN stages should have string representations."""
        assert stage_to_string(PipelineStage.GDN_CONV1D_OUTPUT) == 'GDN_CONV1D_OUTPUT'
        assert stage_to_string(PipelineStage.GDN_DELTA_RULE_OUTPUT) == 'GDN_DELTA_RULE_OUTPUT'
        assert stage_to_string(PipelineStage.GDN_NORM_GATE_OUTPUT) == 'GDN_NORM_GATE_OUTPUT'

    def test_residual_stages_exist(self):
        """Residual stages used by hooks should exist."""
        assert hasattr(PipelineStage, 'ATTENTION_RESIDUAL')
        assert hasattr(PipelineStage, 'FFN_RESIDUAL')

    def test_all_stages_have_string_mapping(self):
        """Every PipelineStage enum value should have a string mapping."""
        for stage in PipelineStage:
            name = stage_to_string(stage)
            assert isinstance(name, str) and len(name) > 0


class TestQwen35HFConfig:
    """Unit tests for _build_hf_config (no model loading required)."""

    def test_build_hf_config_layer_types(self):
        """Config should produce correct 3:1 linear/full attention pattern."""
        from python.reference.qwen35 import Qwen35ReferenceModel

        config_dict = {
            'num_hidden_layers': 8,
            'full_attention_interval': 4,
            'hidden_size': 1024,
            'num_attention_heads': 8,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'intermediate_size': 3584,
            'vocab_size': 1000,
            'linear_num_key_heads': 16,
            'linear_value_head_dim': 128,
            'linear_conv_kernel_dim': 4,
            'ssm_inner_size': 2048,
        }
        cfg = Qwen35ReferenceModel._build_hf_config(config_dict)

        assert cfg.num_hidden_layers == 8
        # Pattern: [linear, linear, linear, full, linear, linear, linear, full]
        expected = ['linear_attention'] * 3 + ['full_attention'] + ['linear_attention'] * 3 + ['full_attention']
        assert cfg.layer_types == expected

    def test_build_hf_config_linear_key_head_dim_from_ssm(self):
        """linear_key_head_dim should be derived from ssm_inner_size / linear_num_key_heads."""
        from python.reference.qwen35 import Qwen35ReferenceModel

        config_dict = {
            'num_hidden_layers': 4,
            'full_attention_interval': 4,
            'hidden_size': 1024,
            'num_attention_heads': 8,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'intermediate_size': 3584,
            'vocab_size': 1000,
            'linear_num_key_heads': 16,
            'linear_value_head_dim': 128,
            'linear_conv_kernel_dim': 4,
            'ssm_inner_size': 2048,
        }
        cfg = Qwen35ReferenceModel._build_hf_config(config_dict)
        assert cfg.linear_key_head_dim == 2048 // 16  # 128

    def test_build_hf_config_eager_attention(self):
        """Config should force eager attention implementation."""
        from python.reference.qwen35 import Qwen35ReferenceModel

        config_dict = {
            'num_hidden_layers': 4,
            'full_attention_interval': 4,
            'hidden_size': 1024,
            'num_attention_heads': 8,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'intermediate_size': 3584,
            'vocab_size': 1000,
            'linear_num_key_heads': 16,
            'linear_value_head_dim': 128,
            'linear_conv_kernel_dim': 4,
            'ssm_inner_size': 2048,
        }
        cfg = Qwen35ReferenceModel._build_hf_config(config_dict)
        assert cfg._attn_implementation == 'eager'


class TestQwen35HookRegistration:
    """Unit tests for hook registration on a tiny model (no GGUF required)."""

    @pytest.fixture(scope='class')
    def tiny_model(self):
        """Create a tiny Qwen 3.5 model for hook testing."""
        from python.reference.qwen35 import Qwen35ReferenceModel

        config_dict = {
            'num_hidden_layers': 4,
            'full_attention_interval': 4,
            'hidden_size': 64,
            'num_attention_heads': 2,
            'num_key_value_heads': 1,
            'head_dim': 32,
            'intermediate_size': 128,
            'vocab_size': 100,
            'linear_num_key_heads': 2,
            'linear_num_value_heads': 2,
            'linear_value_head_dim': 32,
            'linear_key_head_dim': 32,
            'linear_conv_kernel_dim': 4,
            'ssm_inner_size': 64,
        }
        model = Qwen35ReferenceModel('qwen35', checkpoint_path='dummy.gguf', auto_load=False)
        hf_config = Qwen35ReferenceModel._build_hf_config(config_dict)

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        model.hf_config = hf_config
        model.hf_model = Qwen3_5ForCausalLM(hf_config)
        model.hf_model.eval()
        model._register_hooks()
        return model

    def test_hook_handles_registered(self, tiny_model):
        """Hooks should be registered on the model."""
        # The exact count depends on the number of pipeline stages captured.
        # This is a regression guard — if hooks change, update this count
        # after verifying correctness.
        assert len(tiny_model._hook_handles) == 49

    def test_gdn_hooks_capture_on_linear_layers(self, tiny_model):
        """GDN-specific stages should fire only on linear attention layers."""
        tokens = torch.randint(0, 100, (1, 5))
        result = tiny_model.forward(
            tokens,
            capture_stages=[
                PipelineStage.QKV_PROJECTION,
                PipelineStage.GDN_CONV1D_OUTPUT,
                PipelineStage.GDN_DELTA_RULE_OUTPUT,
                PipelineStage.GDN_NORM_GATE_OUTPUT,
            ],
        )
        snapshots = result['snapshots']

        # Layers 0,1,2 are linear; layer 3 is full attention
        for stage in [PipelineStage.QKV_PROJECTION,
                      PipelineStage.GDN_CONV1D_OUTPUT,
                      PipelineStage.GDN_DELTA_RULE_OUTPUT,
                      PipelineStage.GDN_NORM_GATE_OUTPUT]:
            captured_layers = sorted([k[1] for k in snapshots if k[0] == stage])
            assert captured_layers == [0, 1, 2], (
                f"{stage.name}: expected [0,1,2], got {captured_layers}"
            )

    def test_residual_hooks_capture_all_layers(self, tiny_model):
        """Residual hooks should fire on all layers (both linear and full attention)."""
        tokens = torch.randint(0, 100, (1, 5))
        result = tiny_model.forward(
            tokens,
            capture_stages=[
                PipelineStage.ATTENTION_RESIDUAL,
                PipelineStage.FFN_RESIDUAL,
            ],
        )
        snapshots = result['snapshots']

        for stage in [PipelineStage.ATTENTION_RESIDUAL, PipelineStage.FFN_RESIDUAL]:
            captured_layers = sorted([k[1] for k in snapshots if k[0] == stage])
            assert captured_layers == [0, 1, 2, 3], (
                f"{stage.name}: expected [0,1,2,3], got {captured_layers}"
            )

    def test_attention_residual_is_sum_of_input_and_attn(self, tiny_model):
        """ATTENTION_RESIDUAL should equal the layer input + attention output."""
        tokens = torch.randint(0, 100, (1, 5))
        result = tiny_model.forward(
            tokens,
            capture_stages=[
                PipelineStage.EMBEDDING,
                PipelineStage.ATTENTION_OUTPUT,
                PipelineStage.ATTENTION_RESIDUAL,
                PipelineStage.FFN_RESIDUAL,
            ],
        )
        snapshots = result['snapshots']

        # For layer 0: residual = embedding + attention_output
        emb = snapshots[(PipelineStage.EMBEDDING, -1)]
        attn_out = snapshots[(PipelineStage.ATTENTION_OUTPUT, 0)]
        attn_resid = snapshots[(PipelineStage.ATTENTION_RESIDUAL, 0)]
        np.testing.assert_allclose(attn_resid, emb + attn_out, rtol=1e-5, atol=1e-6)

        # For layer 1: residual = layer0_ffn_residual + layer1_attention_output
        ffn_resid_0 = snapshots[(PipelineStage.FFN_RESIDUAL, 0)]
        attn_out_1 = snapshots[(PipelineStage.ATTENTION_OUTPUT, 1)]
        attn_resid_1 = snapshots[(PipelineStage.ATTENTION_RESIDUAL, 1)]
        np.testing.assert_allclose(attn_resid_1, ffn_resid_0 + attn_out_1, rtol=1e-5, atol=1e-6)

    def test_snapshot_shapes_consistent(self, tiny_model):
        """All per-layer snapshots should have shape (1, seq_len, hidden_size)."""
        tokens = torch.randint(0, 100, (1, 7))
        result = tiny_model.forward(
            tokens,
            capture_stages=[
                PipelineStage.ATTENTION_NORM,
                PipelineStage.ATTENTION_OUTPUT,
                PipelineStage.ATTENTION_RESIDUAL,
                PipelineStage.FFN_NORM,
                PipelineStage.FFN_DOWN,
                PipelineStage.FFN_RESIDUAL,
            ],
        )
        snapshots = result['snapshots']
        hidden_size = tiny_model.hf_config.hidden_size  # 64

        for key, tensor in snapshots.items():
            stage, layer_idx = key
            if layer_idx >= 0:
                assert tensor.shape == (1, 7, hidden_size), (
                    f"{stage.name} layer {layer_idx}: "
                    f"expected (1, 7, {hidden_size}), got {tensor.shape}"
                )

    def test_selective_capture_filters_correctly(self, tiny_model):
        """Only requested stages should appear in snapshots."""
        tokens = torch.randint(0, 100, (1, 3))
        result = tiny_model.forward(
            tokens,
            capture_stages=[PipelineStage.EMBEDDING, PipelineStage.LM_HEAD],
        )
        snapshots = result['snapshots']
        stages_captured = {k[0] for k in snapshots}
        assert stages_captured == {PipelineStage.EMBEDDING, PipelineStage.LM_HEAD}

    def test_all_snapshots_finite(self, tiny_model):
        """All captured snapshots should contain finite values."""
        tokens = torch.randint(0, 100, (1, 5))
        result = tiny_model.forward(tokens)  # capture_stages=None → all
        for key, tensor in result['snapshots'].items():
            assert np.isfinite(tensor).all(), f"{key[0].name} layer {key[1]} has non-finite values"


# ---------------------------------------------------------------------------
# Integration Tests — Require models/Qwen3.5-0.8B-Q4_0.gguf
# ---------------------------------------------------------------------------


class TestQwen35GGUFLoading:
    """Integration tests for GGUF loading with fused tensor reconstruction."""

    @pytest.fixture(scope='class')
    def shared_state_dict(self, qwen35_q4_model):
        """Load GGUF once and share across all tests in this class."""
        loader = GGUFLoader(str(qwen35_q4_model))
        _, state_dict = loader.load(as_transformers_config=False)
        return state_dict

    def test_state_dict_has_fused_tensors(self, shared_state_dict):
        """Separate in_proj_qkv, in_proj_z, in_proj_a, in_proj_b should be present."""
        state_dict = shared_state_dict

        # Layer 0 is linear attention — expect separate projection tensors
        assert 'model.layers.0.linear_attn.in_proj_qkv.weight' in state_dict
        assert 'model.layers.0.linear_attn.in_proj_z.weight' in state_dict
        assert 'model.layers.0.linear_attn.in_proj_a.weight' in state_dict
        assert 'model.layers.0.linear_attn.in_proj_b.weight' in state_dict

        qkv = state_dict['model.layers.0.linear_attn.in_proj_qkv.weight']
        z = state_dict['model.layers.0.linear_attn.in_proj_z.weight']

        # in_proj_qkv: (q_dim + k_dim + v_dim) x hidden_size
        assert qkv.shape[1] == 1024  # hidden_size
        # in_proj_z: hidden_size x hidden_size (gating projection)
        assert z.shape[1] == 1024

    def test_state_dict_has_full_attention_tensors(self, shared_state_dict):
        """Full attention layers (3, 7, 11, ...) should have standard Q/K/V/O."""
        state_dict = shared_state_dict

        assert 'model.layers.3.self_attn.q_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.k_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.v_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.o_proj.weight' in state_dict
        assert 'model.layers.3.self_attn.q_norm.weight' in state_dict
        assert 'model.layers.3.self_attn.k_norm.weight' in state_dict

    def test_conv1d_shape(self, shared_state_dict):
        """conv1d.weight should be 3D (unsqueezed from GGUF's 2D)."""
        conv = shared_state_dict['model.layers.0.linear_attn.conv1d.weight']
        assert conv.dim() == 3
        assert tuple(conv.shape) == (6144, 1, 4)

    def test_a_log_values(self, shared_state_dict):
        """A_log values should be finite (not NaN/Inf from log transform)."""
        a_log = shared_state_dict['model.layers.0.linear_attn.A_log']
        assert torch.isfinite(a_log).all(), f"A_log has non-finite values: {a_log}"

    def test_no_fused_components_in_state_dict(self, shared_state_dict):
        """Raw GGUF fused component names should NOT appear in state dict."""
        for key in shared_state_dict:
            assert 'attn_qkv' not in key, f"Raw attn_qkv found: {key}"
            assert 'attn_gate' not in key, f"Raw attn_gate found: {key}"
            assert 'ssm_alpha' not in key, f"Raw ssm_alpha found: {key}"
            assert 'ssm_beta' not in key, f"Raw ssm_beta found: {key}"


class TestQwen35Inference:
    """Integration tests for model inference."""

    @pytest.fixture(scope='class')
    def loaded_model(self, qwen35_q4_model):
        """Load model once and share across all tests in this class."""
        from python.reference import ModelRegistry
        model = ModelRegistry.create(
            'qwen35', checkpoint_path=str(qwen35_q4_model), auto_load=False
        )
        model.load_model()
        return model

    @pytest.fixture(autouse=True)
    def setup_model(self, loaded_model):
        self.model = loaded_model

    def test_forward_produces_logits(self):
        """Forward pass should produce logits with correct vocab size."""
        result = self.model.forward([1, 2, 3, 4, 5])
        logits = result['logits']
        assert logits.shape == (1, 5, 248320)
        assert np.isfinite(logits).all()

    def test_forward_captures_snapshots(self):
        """Forward pass should capture snapshots from heterogeneous layers."""
        from python.reference import PipelineStage

        result = self.model.forward(
            [1, 2, 3],
            capture_stages=[
                PipelineStage.EMBEDDING,
                PipelineStage.ATTENTION_NORM,
                PipelineStage.QKV_PROJECTION,
                PipelineStage.ATTENTION_OUTPUT,
                PipelineStage.ATTENTION_RESIDUAL,
                PipelineStage.FFN_NORM,
                PipelineStage.FFN_DOWN,
                PipelineStage.FFN_RESIDUAL,
                PipelineStage.FINAL_NORM,
                PipelineStage.LM_HEAD,
                # GDN-specific
                PipelineStage.GDN_CONV1D_OUTPUT,
                PipelineStage.GDN_DELTA_RULE_OUTPUT,
                PipelineStage.GDN_NORM_GATE_OUTPUT,
            ],
        )
        snapshots = result['snapshots']

        # Non-layer stages
        assert (PipelineStage.EMBEDDING, -1) in snapshots
        assert (PipelineStage.FINAL_NORM, -1) in snapshots
        assert (PipelineStage.LM_HEAD, -1) in snapshots

        # Per-layer stages present for all 24 layers
        n_layers = self.model.hf_config.num_hidden_layers
        for stage in [PipelineStage.ATTENTION_NORM, PipelineStage.ATTENTION_OUTPUT,
                      PipelineStage.ATTENTION_RESIDUAL, PipelineStage.FFN_NORM,
                      PipelineStage.FFN_DOWN, PipelineStage.FFN_RESIDUAL]:
            captured = [k for k in snapshots if k[0] == stage]
            assert len(captured) == n_layers, (
                f"{stage.name}: expected {n_layers} layers, got {len(captured)}"
            )

        # GDN-specific stages only on linear attention layers (layers 0,1,2, 4,5,6, ...)
        linear_layers = [i for i in range(n_layers) if (i + 1) % 4 != 0]
        for stage in [PipelineStage.QKV_PROJECTION,
                      PipelineStage.GDN_CONV1D_OUTPUT,
                      PipelineStage.GDN_DELTA_RULE_OUTPUT,
                      PipelineStage.GDN_NORM_GATE_OUTPUT]:
            captured = sorted([k[1] for k in snapshots if k[0] == stage])
            assert captured == linear_layers, (
                f"{stage.name}: expected layers {linear_layers}, got {captured}"
            )

    def test_forward_no_nan(self):
        """Output logits and hidden states should contain no NaN values."""
        result = self.model.forward([10, 20, 30, 40, 50])
        assert np.isfinite(result['logits']).all(), "NaN in logits"
        assert np.isfinite(result['hidden_states']).all(), "NaN in hidden states"

    def test_state_dict_load_no_missing_keys(self):
        """
        Loading state dict into model should have no missing keys
        (except possibly lm_head.weight which is tied).
        """
        hf_model = self.model.hf_model
        state_dict = {k: v for k, v in hf_model.state_dict().items()}

        # Create a fresh model on meta device (no weight init) to check key names
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        with torch.device('meta'):
            fresh = Qwen3_5ForCausalLM(self.model.hf_config)
        expected_keys = set(fresh.state_dict().keys())
        actual_keys = set(state_dict.keys())

        # lm_head.weight may be missing (tied with embed_tokens)
        missing = expected_keys - actual_keys - {'lm_head.weight'}
        unexpected = actual_keys - expected_keys
        assert len(missing) == 0, f"Missing keys: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
