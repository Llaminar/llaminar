"""
Tests for Qwen 3.5 MoE GGUF loading — V-head reorder reversal and MoE mappings.

Unit tests (no model files required):
  - V-head reorder roundtrip (_reorder_v_heads is its own inverse with swapped args)
  - _reverse_v_head_reorder for each GDN tensor type
  - Conv1d unsqueeze interplay with V-head reorder
  - MoE tensor name mapping (router, experts, shared expert)
  - Config extraction from qwen35moe GGUF metadata
  - Backward compatibility: _apply_qwen35_transforms with metadata=None

@author David Sanftenberg
"""

import pytest
import torch
import numpy as np
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from python.reference.loaders.gguf_loader import GGUFLoader
from python.reference.loaders import gguf_loader as gguf_loader_module
from python.reference.loaders.tensor_name_mapper import (
    TensorNameMapper,
    detect_model_type_from_metadata,
)
from python.reference.qwen35_moe import (
    Qwen35MoEReferenceModel,
    mtp_sidecar_replay_depth,
    production_router_distribution,
)
from python.reference.generate_qwen35_moe_pipeline_snapshots import (
    normalize_mtp_branch_override_batches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Qwen3.5-35B-A3B MoE GDN config (from GGUF metadata)
MOE_METADATA = {
    'qwen35moe.ssm.group_count': 16,      # num_k_heads
    'qwen35moe.ssm.time_step_rank': 32,   # num_v_heads
    'qwen35moe.ssm.state_size': 128,      # head_k_dim = head_v_dim
    'qwen35moe.ssm.conv_kernel': 4,
    'qwen35moe.ssm.inner_size': 4096,
}

NUM_K = 16
NUM_V = 32
NUM_V_PER_K = NUM_V // NUM_K  # 2
HEAD_DIM = 128
HIDDEN = 2048
KEY_DIM = NUM_K * HEAD_DIM   # 2048
VALUE_DIM = NUM_V * HEAD_DIM  # 4096
CONV_KERNEL = 4


def _make_loader():
    """Create a bare GGUFLoader without opening a file."""
    return GGUFLoader.__new__(GGUFLoader)


def _forward_reorder(tensor, dim, num_k, num_v_per_k, head_dim):
    """Simulate the converter's forward V-head reorder (grouped→tiled)."""
    return GGUFLoader._reorder_v_heads(tensor, dim, num_k, num_v_per_k, head_dim)


def test_router_snapshot_uses_live_post_softmax_distribution():
    """Main and MTP parity must not reconstruct retired raw router logits."""

    probabilities = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32)
    weights = torch.tensor([[0.7, 0.2]], dtype=torch.float32)
    indices = torch.tensor([[2, 1]], dtype=torch.int64)
    observed = production_router_distribution(
        (probabilities, weights, indices)
    )

    assert observed is probabilities
    assert torch.allclose(observed.sum(dim=-1), torch.ones(1))
    with pytest.raises(RuntimeError, match="probabilities, weights, and indices"):
        production_router_distribution((probabilities,))


def test_moe_snapshot_generator_help_exposes_diagnostic_snapshot_modes():
    """MoE parity can refresh metadata or decode-only diagnostic snapshots."""
    script = Path(__file__).resolve().parents[1] / "generate_qwen35_moe_pipeline_snapshots.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--metadata-only" in result.stdout
    assert "--decode-snapshots-only" in result.stdout
    assert "--mtp-max-draft-depth" in result.stdout


def test_additive_mtp_branch_replays_only_committed_prefix_and_requested_depth():
    """Branch generation does not recompute the combinatorial canonical pack."""

    overrides = {2: [101, 202, 303, 404]}
    assert mtp_sidecar_replay_depth(0, 15, overrides) == 1
    assert mtp_sidecar_replay_depth(1, 15, overrides) == 1
    assert mtp_sidecar_replay_depth(2, 15, overrides) == 5
    assert mtp_sidecar_replay_depth(3, 15, overrides) == 1
    assert mtp_sidecar_replay_depth(2, 15, {}) == 15


def test_mtp_branch_override_campaign_batches_preserve_independent_trajectories():
    """Many observed branches share one HF model without merging trajectories."""

    assert normalize_mtp_branch_override_batches({"3": [11, 22]}) == [
        {3: [11, 22]}
    ]
    assert normalize_mtp_branch_override_batches(
        [{"1": [7]}, {"1": [8, 9]}, {"3": [10, 11, 12]}]
    ) == [{1: [7]}, {1: [8, 9]}, {3: [10, 11, 12]}]

    with pytest.raises(ValueError, match="flat token array"):
        normalize_mtp_branch_override_batches({"2": [[1, 2], [3, 4]]})


def test_streaming_loader_filters_before_read_and_dequantize(monkeypatch):
    """Sidecar-only loading must not touch excluded main-model tensor bytes."""

    class FakeMapper:
        @staticmethod
        def map_name(name):
            return f"mapped.{name}"

    class FakeParser:
        metadata = {}
        tensors = [
            SimpleNamespace(name="skip", type=SimpleNamespace(name="F32"), shape=(1,)),
            SimpleNamespace(name="keep", type=SimpleNamespace(name="F32"), shape=(1,)),
        ]

        def __init__(self):
            self.read_names = []

        @staticmethod
        def get_model_type():
            return "test"

        def read_tensor_data(self, tensor_info):
            self.read_names.append(tensor_info.name)
            if tensor_info.name == "skip":
                raise AssertionError("excluded tensor bytes were read")
            return b"payload"

    monkeypatch.setattr(
        gguf_loader_module,
        "create_mapper_from_metadata",
        lambda _metadata: FakeMapper(),
    )
    monkeypatch.setattr(
        gguf_loader_module.dequantize,
        "dequantize",
        lambda _raw, _type, _shape: np.array([7.0], dtype=np.float32),
    )
    loader = GGUFLoader.__new__(GGUFLoader)
    loader.file_path = Path("unused.gguf")
    loader.verbose = False
    parser = FakeParser()

    tensors = list(
        loader.iter_state_dict(
            parser=parser,
            as_torch=False,
            max_in_flight=1,
            include_mapped_name=lambda name: name == "mapped.keep",
        )
    )

    assert parser.read_names == ["keep"]
    assert tensors[0][0] == "mapped.keep"
    np.testing.assert_array_equal(tensors[0][1], np.array([7.0], dtype=np.float32))


def test_mtp_sidecar_reference_pack_reuses_exact_main_trajectory(tmp_path):
    """Additive branches consume canonical FP32 hidden rows and token history."""

    (tmp_path / "metadata.txt").write_text(
        "\n".join(
            [
                "reference_engine: pytorch",
                "reference_dtype: float32",
                "n_layers: 2",
                "decode_steps: 2",
                "token_ids: 10,20,30",
                "decode_tokens: 40,50,60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    expected = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    np.save(tmp_path / "layer1_FFN_RESIDUAL.npy", expected)

    model = Qwen35MoEReferenceModel.__new__(Qwen35MoEReferenceModel)
    model._hook_handles = []
    model._mtp_sidecar_reference_pack = tmp_path
    model.hf_config = SimpleNamespace(num_hidden_layers=2, hidden_size=4)
    model.device = torch.device("cpu")
    model.tokenizer = lambda _prompt, return_tensors: {
        "input_ids": torch.tensor([[10, 20, 30]])
    }

    token_ids, decode_tokens, hidden = model._load_mtp_reference_pack_trajectory(
        "authenticated prompt", 2
    )

    assert token_ids == [10, 20, 30]
    assert decode_tokens == [40, 50, 60]
    torch.testing.assert_close(hidden, torch.from_numpy(expected))


def test_bf16_gguf_tensor_type_dequantizes_to_fp32():
    """Qwen3.6 MoE GGUFs may contain BF16 tensors in an otherwise IQ file."""
    from python.reference.loaders.gguf_parser import GGUFTensorType
    from python.reference.loaders import dequantize

    raw_bf16 = np.array([0x3F80, 0xBF80, 0x4000], dtype="<u2").tobytes()
    out = dequantize.dequantize(raw_bf16, GGUFTensorType(30), (3,))

    assert GGUFTensorType(30) == GGUFTensorType.BF16
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.array([1.0, -1.0, 2.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# _reorder_v_heads: roundtrip & basic properties
# ---------------------------------------------------------------------------


class TestReorderVHeads:
    """Test the low-level _reorder_v_heads helper."""

    def test_roundtrip_2d(self):
        """forward(reverse(x)) == x for a 2D tensor along dim 0."""
        original = torch.arange(NUM_V * HEAD_DIM).reshape(NUM_V * HEAD_DIM, 1).float()
        tiled = _forward_reorder(original, 0, NUM_K, NUM_V_PER_K, HEAD_DIM)
        restored = _forward_reorder(tiled, 0, NUM_V_PER_K, NUM_K, HEAD_DIM)
        torch.testing.assert_close(restored, original)

    def test_roundtrip_1d(self):
        """Roundtrip for 1D tensor (e.g. A_log, dt_bias)."""
        original = torch.arange(NUM_V).float()
        unsq = original.unsqueeze(-1)
        tiled = _forward_reorder(unsq, 0, NUM_K, NUM_V_PER_K, 1)
        restored = _forward_reorder(tiled, 0, NUM_V_PER_K, NUM_K, 1)
        torch.testing.assert_close(restored.squeeze(-1), original)

    def test_roundtrip_columns(self):
        """Roundtrip along dim 1 (e.g. out_proj columns)."""
        original = torch.randn(HIDDEN, VALUE_DIM)
        tiled = _forward_reorder(original, 1, NUM_K, NUM_V_PER_K, HEAD_DIM)
        restored = _forward_reorder(tiled, 1, NUM_V_PER_K, NUM_K, HEAD_DIM)
        torch.testing.assert_close(restored, original)

    def test_noop_when_k_equals_v(self):
        """When num_k == num_v (num_v_per_k=1), reorder is a no-op."""
        original = torch.randn(8 * HEAD_DIM, HIDDEN)
        reordered = _forward_reorder(original, 0, 8, 1, HEAD_DIM)
        torch.testing.assert_close(reordered, original)

    def test_shape_preserved(self):
        """Reorder should not change tensor shape."""
        t = torch.randn(VALUE_DIM, HIDDEN)
        result = _forward_reorder(t, 0, NUM_K, NUM_V_PER_K, HEAD_DIM)
        assert result.shape == t.shape

    def test_actual_permutation(self):
        """Verify the reorder actually moves data.

        With num_k=2, num_v_per_k=2, head_dim=1:
          Grouped:  [K0_V0, K0_V1, K1_V0, K1_V1]
          Tiled:    [K0_V0, K1_V0, K0_V1, K1_V1]
        """
        # 4 elements: grouped order [0, 1, 2, 3]
        grouped = torch.tensor([0.0, 1.0, 2.0, 3.0]).unsqueeze(-1)  # (4, 1)
        tiled = _forward_reorder(grouped, 0, 2, 2, 1)
        expected_tiled = torch.tensor([0.0, 2.0, 1.0, 3.0]).unsqueeze(-1)
        torch.testing.assert_close(tiled, expected_tiled)


# ---------------------------------------------------------------------------
# _reverse_v_head_reorder: per-tensor-type tests
# ---------------------------------------------------------------------------


class TestReverseVHeadReorder:
    """Test reversal of V-head tiled→grouped reordering for each GDN tensor type."""

    def setup_method(self):
        self.loader = _make_loader()

    # -- in_proj_qkv --

    def test_in_proj_qkv_q_k_unchanged(self):
        """Q and K portions of in_proj_qkv should be unchanged by reorder reversal."""
        q_orig = torch.randn(KEY_DIM, HIDDEN)
        k_orig = torch.randn(KEY_DIM, HIDDEN)
        v_orig = torch.randn(VALUE_DIM, HIDDEN)

        # Forward reorder (only V portion)
        v_tiled = _forward_reorder(v_orig, 0, NUM_K, NUM_V_PER_K, HEAD_DIM)
        tiled_qkv = torch.cat([q_orig, k_orig, v_tiled], dim=0)

        result, handled = self.loader._reverse_v_head_reorder(
            tiled_qkv,
            'model.layers.0.linear_attn.in_proj_qkv.weight',
            MOE_METADATA,
        )

        # Q and K should be unchanged
        torch.testing.assert_close(result[:KEY_DIM], q_orig)
        torch.testing.assert_close(result[KEY_DIM:2 * KEY_DIM], k_orig)

    def test_in_proj_qkv_v_restored(self):
        """V portion of in_proj_qkv should be restored to grouped order."""
        q_orig = torch.randn(KEY_DIM, HIDDEN)
        k_orig = torch.randn(KEY_DIM, HIDDEN)
        v_orig = torch.randn(VALUE_DIM, HIDDEN)

        v_tiled = _forward_reorder(v_orig, 0, NUM_K, NUM_V_PER_K, HEAD_DIM)
        tiled_qkv = torch.cat([q_orig, k_orig, v_tiled], dim=0)

        result, _ = self.loader._reverse_v_head_reorder(
            tiled_qkv,
            'model.layers.0.linear_attn.in_proj_qkv.weight',
            MOE_METADATA,
        )

        torch.testing.assert_close(result[2 * KEY_DIM:], v_orig)

    def test_in_proj_qkv_shape(self):
        """Shape should be preserved: (q_dim + k_dim + v_dim, hidden)."""
        total_dim = KEY_DIM + KEY_DIM + VALUE_DIM  # 8192
        t = torch.randn(total_dim, HIDDEN)
        result, _ = self.loader._reverse_v_head_reorder(
            t, 'model.layers.0.linear_attn.in_proj_qkv.weight', MOE_METADATA)
        assert result.shape == (total_dim, HIDDEN)

    # -- in_proj_z --

    def test_in_proj_z_roundtrip(self):
        """in_proj_z rows should roundtrip through forward+reverse."""
        original = torch.randn(VALUE_DIM, HIDDEN)
        tiled = _forward_reorder(original, 0, NUM_K, NUM_V_PER_K, HEAD_DIM)

        result, _ = self.loader._reverse_v_head_reorder(
            tiled, 'model.layers.0.linear_attn.in_proj_z.weight', MOE_METADATA)
        torch.testing.assert_close(result, original)

    # -- in_proj_a / in_proj_b --

    def test_in_proj_a_roundtrip(self):
        """in_proj_a (num_v_heads × hidden) rows should roundtrip."""
        original = torch.randn(NUM_V, HIDDEN)
        tiled = _forward_reorder(original, 0, NUM_K, NUM_V_PER_K, 1)

        result, _ = self.loader._reverse_v_head_reorder(
            tiled, 'model.layers.0.linear_attn.in_proj_a.weight', MOE_METADATA)
        torch.testing.assert_close(result, original)

    def test_in_proj_b_roundtrip(self):
        """in_proj_b (num_v_heads × hidden) rows should roundtrip."""
        original = torch.randn(NUM_V, HIDDEN)
        tiled = _forward_reorder(original, 0, NUM_K, NUM_V_PER_K, 1)

        result, _ = self.loader._reverse_v_head_reorder(
            tiled, 'model.layers.5.linear_attn.in_proj_b.weight', MOE_METADATA)
        torch.testing.assert_close(result, original)

    # -- A_log (1D) --

    def test_a_log_1d_roundtrip(self):
        """A_log is 1D (num_v_heads,) — should roundtrip."""
        original = torch.randn(NUM_V)
        tiled = _forward_reorder(original.unsqueeze(-1), 0, NUM_K, NUM_V_PER_K, 1).squeeze(-1)

        result, _ = self.loader._reverse_v_head_reorder(
            tiled, 'model.layers.0.linear_attn.A_log', MOE_METADATA)
        torch.testing.assert_close(result, original)

    # -- dt_bias (1D) --

    def test_dt_bias_1d_roundtrip(self):
        """dt_bias is 1D (num_v_heads,) — should roundtrip."""
        original = torch.randn(NUM_V)
        tiled = _forward_reorder(original.unsqueeze(-1), 0, NUM_K, NUM_V_PER_K, 1).squeeze(-1)

        result, _ = self.loader._reverse_v_head_reorder(
            tiled, 'model.layers.0.linear_attn.dt_bias', MOE_METADATA)
        torch.testing.assert_close(result, original)

    # -- conv1d --

    def test_conv1d_v_channels_roundtrip(self):
        """conv1d: V channel portion should roundtrip; QK channels unchanged."""
        qk_channels = KEY_DIM * 2  # 4096
        qk_orig = torch.randn(qk_channels, CONV_KERNEL)
        v_orig = torch.randn(VALUE_DIM, CONV_KERNEL)

        v_tiled = _forward_reorder(v_orig, 0, NUM_K, NUM_V_PER_K, HEAD_DIM)
        tiled_conv = torch.cat([qk_orig, v_tiled], dim=0)  # (8192, 4)

        result, handled = self.loader._reverse_v_head_reorder(
            tiled_conv, 'model.layers.0.linear_attn.conv1d.weight', MOE_METADATA)

        # Conv1d reversal also unsqueezes to (channels, 1, kernel)
        assert result.shape == (qk_channels + VALUE_DIM, 1, CONV_KERNEL)
        assert handled is True

        # QK channels unchanged
        torch.testing.assert_close(result[:qk_channels].squeeze(1), qk_orig)
        # V channels restored
        torch.testing.assert_close(result[qk_channels:].squeeze(1), v_orig)

    # -- out_proj --

    def test_out_proj_columns_roundtrip(self):
        """out_proj: columns (input dim = value_dim) should roundtrip."""
        original = torch.randn(HIDDEN, VALUE_DIM)
        tiled = _forward_reorder(original, 1, NUM_K, NUM_V_PER_K, HEAD_DIM)

        result, _ = self.loader._reverse_v_head_reorder(
            tiled, 'model.layers.0.linear_attn.out_proj.weight', MOE_METADATA)
        torch.testing.assert_close(result, original)

    # -- Edge cases --

    def test_no_reorder_when_k_equals_v(self):
        """No reorder when num_k_heads == num_v_heads."""
        meta = {**MOE_METADATA, 'qwen35moe.ssm.time_step_rank': 16}
        t = torch.randn(VALUE_DIM, HIDDEN)
        result, handled = self.loader._reverse_v_head_reorder(
            t, 'model.layers.0.linear_attn.in_proj_z.weight', meta)
        torch.testing.assert_close(result, t)
        assert handled is False

    def test_missing_metadata_returns_unchanged(self):
        """Missing metadata keys should return tensor unchanged."""
        t = torch.randn(VALUE_DIM, HIDDEN)
        result, handled = self.loader._reverse_v_head_reorder(
            t, 'model.layers.0.linear_attn.in_proj_z.weight', {})
        torch.testing.assert_close(result, t)
        assert handled is False

    def test_non_linear_attn_tensor_unchanged(self):
        """Non-linear-attn tensors should not trigger conv1d_handled flag."""
        t = torch.randn(100, HIDDEN)
        result, handled = self.loader._reverse_v_head_reorder(
            t, 'model.layers.0.mlp.gate_proj.weight', MOE_METADATA)
        torch.testing.assert_close(result, t)
        assert handled is False


# ---------------------------------------------------------------------------
# _apply_qwen35_transforms: integration with V-head reorder
# ---------------------------------------------------------------------------


class TestApplyTransformsWithMetadata:
    """Test _apply_qwen35_transforms with metadata for V-head reversal."""

    def setup_method(self):
        self.loader = _make_loader()

    def test_backward_compat_no_metadata(self):
        """Without metadata, transforms still work (conv1d unsqueezes, etc)."""
        t = torch.randn(8192, 4)
        result = self.loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.ssm_conv1d.weight',
            'model.layers.0.linear_attn.conv1d.weight',
            metadata=None,
        )
        assert result.shape == (8192, 1, 4)

    def test_conv1d_with_metadata_unsqueezes(self):
        """Conv1d with metadata should still produce (channels, 1, kernel) shape."""
        total_channels = KEY_DIM * 2 + VALUE_DIM  # 8192
        t = torch.randn(total_channels, CONV_KERNEL)
        result = self.loader._apply_qwen35_transforms(
            t, 'blk.0.ssm_conv1d.weight',
            'model.layers.0.linear_attn.conv1d.weight',
            metadata=MOE_METADATA,
        )
        assert result.shape == (total_channels, 1, CONV_KERNEL)

    def test_norm_subtract_still_works_with_metadata(self):
        """Norm subtraction should work whether or not metadata is passed."""
        t = torch.tensor([2.0, 1.5, 1.0])

        result = self.loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.attn_norm.weight',
            'model.layers.0.input_layernorm.weight',
            metadata=MOE_METADATA,
        )
        torch.testing.assert_close(result, torch.tensor([1.0, 0.5, 0.0]))

    def test_linear_attn_norm_excluded_with_metadata(self):
        """linear_attn.norm.weight should NOT subtract 1 even with metadata."""
        t = torch.tensor([2.0, 1.5, 1.0])
        result = self.loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.ssm_norm.weight',
            'model.layers.0.linear_attn.norm.weight',
            metadata=MOE_METADATA,
        )
        torch.testing.assert_close(result, torch.tensor([2.0, 1.5, 1.0]))

    def test_nextn_manual_rms_norm_uses_pre_rmsnorm_1p_effective_gamma(self):
        """Manual nextn/MTP RMSNorm applies gamma_effective = 1 + stored."""
        x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        stored_gamma = torch.tensor([-0.5, 0.25], dtype=torch.float32)
        result = Qwen35MoEReferenceModel._rms_norm(
            x,
            stored_gamma,
            0.0,
            pre_rmsnorm_1p=True,
        )
        base = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))
        expected = base * torch.tensor([0.5, 1.25], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_streamed_expert_halves_fill_one_final_parameter(self):
        """Bounded loading must not allocate a second fused expert tensor."""
        target = torch.nn.Parameter(torch.zeros(2, 6, 4))
        parameters = {"layer.mlp.experts.gate_up_proj": target}
        loaded = set()
        halves = {}
        gate = torch.full((2, 3, 4), 2.0)
        up = torch.full((2, 3, 4), 7.0)

        assert Qwen35MoEReferenceModel._copy_streamed_parameter(
            "layer.mlp.experts.gate_proj.weight",
            gate,
            parameters,
            loaded,
            halves,
        )
        assert loaded == set()
        assert Qwen35MoEReferenceModel._copy_streamed_parameter(
            "layer.mlp.experts.up_proj.weight",
            up,
            parameters,
            loaded,
            halves,
        )

        assert loaded == {"layer.mlp.experts.gate_up_proj"}
        torch.testing.assert_close(target[:, :3, :], gate)
        torch.testing.assert_close(target[:, 3:, :], up)

    def test_streamed_sidecar_expert_name_has_no_model_prefix(self):
        """A stripped MTP-layer name must still recognize fused experts."""
        target = torch.nn.Parameter(torch.zeros(1, 4, 2))
        parameters = {"mlp.experts.gate_up_proj": target}
        loaded = set()
        halves = {}
        for projection, value in (("gate_proj", 3.0), ("up_proj", 5.0)):
            assert Qwen35MoEReferenceModel._copy_streamed_parameter(
                f"mlp.experts.{projection}.weight",
                torch.full((1, 2, 2), value),
                parameters,
                loaded,
                halves,
            )
        assert loaded == {"mlp.experts.gate_up_proj"}
        torch.testing.assert_close(target[:, :2, :], torch.full((1, 2, 2), 3.0))
        torch.testing.assert_close(target[:, 2:, :], torch.full((1, 2, 2), 5.0))

    def test_streamed_shared_gate_restores_linear_weight_shape(self):
        """The GGUF vector form must publish into HF's [1, hidden] tensor."""
        target = torch.nn.Parameter(torch.zeros(1, 4))
        loaded = set()
        assert Qwen35MoEReferenceModel._copy_streamed_parameter(
            "layer.mlp.shared_expert_gate.weight",
            torch.arange(4, dtype=torch.float32),
            {"layer.mlp.shared_expert_gate.weight": target},
            loaded,
            {},
        )
        assert loaded == {"layer.mlp.shared_expert_gate.weight"}
        torch.testing.assert_close(
            target, torch.arange(4, dtype=torch.float32).unsqueeze(0)
        )

    def test_a_log_transform_with_metadata(self):
        """A_log transform + V-head reversal should both apply."""
        original_a_log = torch.randn(NUM_V)
        # Converter: A_log → -exp(A_log) → reorder
        gguf_val = -torch.exp(original_a_log)
        gguf_val_reordered = _forward_reorder(
            gguf_val.unsqueeze(-1), 0, NUM_K, NUM_V_PER_K, 1).squeeze(-1)

        result = self.loader._apply_qwen35_transforms(
            gguf_val_reordered, 'blk.0.ssm_a',
            'model.layers.0.linear_attn.A_log',
            metadata=MOE_METADATA,
        )
        # After log(-x) and reverse reorder, should get original back
        torch.testing.assert_close(result, original_a_log, atol=1e-5, rtol=1e-5)

    def test_in_proj_z_full_roundtrip(self):
        """in_proj_z: reorder in converter → reverse in loader should be identity."""
        original = torch.randn(VALUE_DIM, HIDDEN)
        # Converter reorders rows
        tiled = _forward_reorder(original, 0, NUM_K, NUM_V_PER_K, HEAD_DIM)

        result = self.loader._apply_qwen35_transforms(
            tiled, 'blk.0.attn_gate.weight',
            'model.layers.0.linear_attn.in_proj_z.weight',
            metadata=MOE_METADATA,
        )
        torch.testing.assert_close(result, original)

    def test_non_gdn_tensor_passthrough(self):
        """Non-GDN tensors should pass through unchanged even with metadata."""
        t = torch.randn(1024, HIDDEN)
        result = self.loader._apply_qwen35_transforms(
            t.clone(), 'blk.0.ffn_gate.weight',
            'model.layers.0.mlp.gate_proj.weight',
            metadata=MOE_METADATA,
        )
        torch.testing.assert_close(result, t)


# ---------------------------------------------------------------------------
# MoE tensor name mapping
# ---------------------------------------------------------------------------


class TestQwen35MoeTensorNameMapping:
    """Test GGUF → HuggingFace tensor name mapping for Qwen 3.5 MoE."""

    def setup_method(self):
        self.mapper = TensorNameMapper('qwen35moe', full_attention_interval=4)

    def test_model_type_detection(self):
        meta = {'general.architecture': 'qwen35moe', 'qwen35moe.block_count': 40}
        assert detect_model_type_from_metadata(meta) == 'qwen35moe'

    # -- Global tensors --

    def test_global_tensors(self):
        assert self.mapper.map_name('token_embd.weight') == 'model.embed_tokens.weight'
        assert self.mapper.map_name('output_norm.weight') == 'model.norm.weight'
        assert self.mapper.map_name('output.weight') == 'lm_head.weight'

    # -- MoE FFN tensors --

    def test_router(self):
        assert (
            self.mapper.map_name('blk.0.ffn_gate_inp.weight')
            == 'model.layers.0.mlp.gate.weight'
        )

    def test_expert_gate(self):
        assert (
            self.mapper.map_name('blk.0.ffn_gate_exps.weight')
            == 'model.layers.0.mlp.experts.gate_proj.weight'
        )

    def test_expert_up(self):
        assert (
            self.mapper.map_name('blk.0.ffn_up_exps.weight')
            == 'model.layers.0.mlp.experts.up_proj.weight'
        )

    def test_expert_down(self):
        """down_proj maps to bare nn.Parameter (no .weight suffix)."""
        assert (
            self.mapper.map_name('blk.0.ffn_down_exps.weight')
            == 'model.layers.0.mlp.experts.down_proj'
        )

    def test_shared_expert(self):
        assert (
            self.mapper.map_name('blk.0.ffn_gate_shexp.weight')
            == 'model.layers.0.mlp.shared_expert.gate_proj.weight'
        )
        assert (
            self.mapper.map_name('blk.0.ffn_up_shexp.weight')
            == 'model.layers.0.mlp.shared_expert.up_proj.weight'
        )
        assert (
            self.mapper.map_name('blk.0.ffn_down_shexp.weight')
            == 'model.layers.0.mlp.shared_expert.down_proj.weight'
        )

    def test_shared_expert_gate(self):
        assert (
            self.mapper.map_name('blk.0.ffn_gate_inp_shexp.weight')
            == 'model.layers.0.mlp.shared_expert_gate.weight'
        )

    # -- Per-layer norms (should NOT have dense FFN mappings) --

    def test_layer_norms(self):
        assert (
            self.mapper.map_name('blk.5.attn_norm.weight')
            == 'model.layers.5.input_layernorm.weight'
        )
        assert (
            self.mapper.map_name('blk.5.post_attention_norm.weight')
            == 'model.layers.5.post_attention_layernorm.weight'
        )

    # -- GDN tensors still available --

    def test_gdn_qkv(self):
        assert (
            self.mapper.map_name('blk.0.attn_qkv.weight')
            == 'model.layers.0.linear_attn.in_proj_qkv.weight'
        )

    def test_gdn_gate(self):
        assert (
            self.mapper.map_name('blk.0.attn_gate.weight')
            == 'model.layers.0.linear_attn.in_proj_z.weight'
        )

    # -- Full attention tensors --

    def test_full_attention_q(self):
        assert (
            self.mapper.map_name('blk.3.attn_q.weight')
            == 'model.layers.3.self_attn.q_proj.weight'
        )

    def test_full_attention_k(self):
        assert (
            self.mapper.map_name('blk.3.attn_k.weight')
            == 'model.layers.3.self_attn.k_proj.weight'
        )

    # -- Layer index (higher layers) --

    def test_higher_layer_moe(self):
        assert (
            self.mapper.map_name('blk.39.ffn_gate_inp.weight')
            == 'model.layers.39.mlp.gate.weight'
        )


# ---------------------------------------------------------------------------
# MoE config extraction
# ---------------------------------------------------------------------------


class TestQwen35MoeConfigExtraction:
    """Test GGUF config dict extraction for Qwen 3.5 MoE."""

    def test_qwen35moe_config_from_metadata(self):
        from python.reference.loaders.gguf_parser import GGUFParser

        parser = GGUFParser.__new__(GGUFParser)
        parser.metadata = {
            'general.architecture': 'qwen35moe',
            'qwen35moe.embedding_length': 2048,
            'qwen35moe.attention.head_count': 16,
            'qwen35moe.block_count': 40,
            'qwen35moe.context_length': 262144,
            'qwen35moe.attention.head_count_kv': 2,
            'qwen35moe.attention.key_length': 256,
            'qwen35moe.attention.value_length': 256,
            'qwen35moe.attention.layer_norm_rms_epsilon': 1e-6,
            'qwen35moe.rope.freq_base': 10000000.0,
            'qwen35moe.full_attention_interval': 4,
            'qwen35moe.ssm.conv_kernel': 4,
            'qwen35moe.ssm.group_count': 16,
            'qwen35moe.ssm.inner_size': 4096,
            'qwen35moe.ssm.state_size': 128,
            'qwen35moe.ssm.time_step_rank': 32,
            'qwen35moe.expert_count': 256,
            'qwen35moe.expert_used_count': 8,
            'qwen35moe.expert_feed_forward_length': 512,
            'qwen35moe.expert_shared_feed_forward_length': 512,
            'qwen35moe.rope.dimension_sections': [11, 11, 10, 0],
            'tokenizer.ggml.tokens': ['a'] * 248320,
        }

        config = parser.get_config_dict()

        assert config['hidden_size'] == 2048
        assert config['num_attention_heads'] == 16
        assert config['num_hidden_layers'] == 40
        assert config['num_key_value_heads'] == 2
        assert config['head_dim'] == 256
        assert config['full_attention_interval'] == 4
        assert config['linear_num_key_heads'] == 16
        assert config['linear_value_head_dim'] == 128
        assert config['ssm_inner_size'] == 4096
        assert config['ssm_time_step_rank'] == 32
        assert config.get('num_experts') == 256
        assert config.get('num_experts_per_tok') == 8
        assert config.get('moe_intermediate_size') == 512
        assert config.get('shared_expert_intermediate_size') == 512
