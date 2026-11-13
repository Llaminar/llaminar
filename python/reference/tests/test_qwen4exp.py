"""Device-free Qwen4Exp exact-integer, native-codec and bounded-row regressions.

These tests verify reference ownership and independently specified block/hash
arithmetic. They do not load the real model or certify backend execution.
"""
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from python.reference.loaders.dequantize import dequantize, get_quantization_info
from python.reference.loaders.gguf_parser import GGUFTensorType as T
from python.reference.pipeline_stages import PipelineStage as S, stage_to_string, string_to_stage
from python.reference.qwen4exp import (
    NativeSliceWeights, Qwen4ExpReferenceModel, ple_hash_indices, rms_gain,
)


@pytest.fixture(scope='module', autouse=True)
def bounded_math_fixture_workers():
    """Keep tiny device-free arithmetic proofs independent of host-wide Torch teams."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_iq4nl_decodes_all_codes_in_half_block_order():
    codebook = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
    packed = bytes(i | ((15 - i) << 4) for i in range(16))
    raw = struct.pack('<e', 0.5) + packed
    expected = np.concatenate((codebook, codebook[::-1])) * 0.5
    np.testing.assert_array_equal(dequantize(raw, T.IQ4_NL, (32,)), expected)
    assert get_quantization_info(T.IQ4_NL)['block_bytes'] == 18
    with pytest.raises(ValueError, match='complete, exact'):
        dequantize(raw[:-1], T.IQ4_NL, (32,))


def test_q5_1_affine_codes_include_every_high_bit_and_both_nibble_halves():
    codes = np.arange(32, dtype=np.uint8)
    high = sum(int(code >> 4) << i for i, code in enumerate(codes))
    low = bytes(int(codes[i] & 15) | int(codes[i + 16] & 15) << 4 for i in range(16))
    raw = struct.pack('<eeI', 0.25, -2, high) + low
    np.testing.assert_array_equal(dequantize(raw, T.Q5_1, (32,)), codes * 0.25 - 2)
    assert get_quantization_info(T.Q5_1)['block_bytes'] == 24


def test_hashes_retain_45bit_multipliers_and_reset_only_after_primary_eos():
    eos = 248044
    multipliers = [23703573157769, 20109073645365, 8052911324071]
    offsets = [i * 20000003 for i in range(16)]
    sizes = [20000003 + i * 2 for i in range(16)]
    tokens = torch.tensor([[17, eos, 31, 248046, 47]])
    previous = torch.tensor([[eos, eos]])
    result = ple_hash_indices(tokens, previous, multipliers, offsets, sizes, eos)
    expected = []
    segment = []
    for token in tokens[0].tolist():
        row = []
        suffix = [token] + list(reversed(segment[-2:]))
        suffix += [eos] * (3 - len(suffix))
        for head in range(16):
            order = 2 + head // 8
            mixed = 0
            for index in range(order):
                mixed ^= suffix[index] * multipliers[index]
            row.append(mixed % sizes[head] + offsets[head])
        expected.append(row)
        segment.append(token)
        if token == eos:
            segment.clear()
    np.testing.assert_array_equal(result.numpy(), np.array([expected], dtype=np.int64))
    assert result.dtype == torch.int64
    assert result.max() > 1 << 24
    # Chunked hashing retains exactly the two-token suffix, including EOS cuts.
    left = ple_hash_indices(tokens[:, :2], previous, multipliers, offsets, sizes, eos)
    right = ple_hash_indices(tokens[:, 2:], tokens[:, :2], multipliers, offsets, sizes, eos)
    assert torch.equal(result, torch.cat((left, right), 1))


def test_integer_stage_capture_does_not_round_large_row_ids():
    model = Qwen4ExpReferenceModel('qwen4exp', 'unused')
    model.capture_stages = None
    ids = torch.tensor([16777217, 319999999], dtype=torch.int64)
    model.capture_stage(S.PLE_HASH_INDICES, ids, 1)
    np.testing.assert_array_equal(model.get_snapshots()[(S.PLE_HASH_INDICES, 1)], ids.numpy())
    assert model.get_snapshots()[(S.PLE_HASH_INDICES, 1)].dtype == np.int64
    for stage in [S.GR_NORM, S.GR_READ, S.GR_WRITE, S.QSA_INDEXER_QUERY,
                  S.QSA_INDEXER_POOLED_KEYS, S.QSA_SELECTED_INDICES,
                  S.PLE_HASH_INDICES, S.PLE_GATHER, S.PLE_GATE, S.PLE_CONVOLUTION]:
        assert string_to_stage(stage_to_string(stage)) == stage


def test_grouped_rms_uses_final_gain_and_independent_branch_width():
    value = torch.tensor([[3., 4., 0., 2.]])
    gain = torch.tensor([2., 1., 3., 4.])
    expected = torch.tensor([[6 / np.sqrt(12.5 + 1e-6), 4 / np.sqrt(12.5 + 1e-6),
                              0, 8 / np.sqrt(2 + 1e-6)]], dtype=torch.float32)
    torch.testing.assert_close(rms_gain(value, gain, 1e-6, 2), expected)


def test_row_store_decodes_only_selected_extents_and_enforces_cache_budget():
    # An immutable fake mapping describes a huge table but supplies only three
    # native rows. The accessor must never ask dequantize() for table geometry.
    payload = b''.join(struct.pack('<e', float(i + 1)) + bytes([0x88]) * 16 for i in range(3))
    weight = NativeSliceWeights.__new__(NativeSliceWeights)
    from collections import OrderedDict
    weight.cache = OrderedDict()
    weight.decoded_bytes = 0
    weight.cache_bytes = 128
    weight.directory = {'per_layer_token_embd.weight': SimpleNamespace(shape=(320001536, 32), type=T.IQ4_NL)}
    weight.parser = SimpleNamespace(read_tensor_data=lambda _: memoryview(payload))
    indices = torch.tensor([[2, 0, 2]])
    rows = weight.rows('per_layer_token_embd.weight', indices)
    assert rows.shape == (1, 3, 32)
    assert torch.equal(rows[0, 0], torch.full((32,), 3.))
    assert torch.equal(rows[0, 1], torch.ones(32))
    assert weight.decoded_bytes <= 128
    with pytest.raises(ValueError, match='selected-row'):
        weight.tensor('per_layer_token_embd.weight')


def test_gdn_tiled_head_order_and_sigmoid_norm_match_shared_hf_recurrence():
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    torch.manual_seed(19)
    model = Qwen4ExpReferenceModel('qwen4exp', 'unused')
    model.epsilon = 1e-6
    model.capture_stages = None
    batch, length = 1, 3
    value = torch.randn(batch, length, 8)
    projected = {
        'attn_qkv': torch.randn(batch, length, 10240) * .1,
        'attn_gate': torch.randn(batch, length, 6144),
        'ssm_alpha': torch.randn(batch, length, 48),
        'ssm_beta': torch.randn(batch, length, 48),
    }
    tensors = {
        'ssm_conv1d.weight': torch.randn(10240, 4) * .1,
        'ssm_a': -torch.rand(48),
        'ssm_dt.bias': torch.randn(48),
        'ssm_norm.weight': torch.rand(128),
    }
    out_weight = torch.randn(8, 6144) * .01
    model.weights = SimpleNamespace(tensor=lambda name: tensors[name])
    model._linear = lambda x, prefix, stem: F.linear(x, out_weight) if stem == 'ssm_out' else projected[stem]
    from torch.nn import functional as F
    cache = {}
    result = model._gdn(value, '', cache, 0)
    conv = F.silu(F.conv1d(F.pad(projected['attn_qkv'].transpose(1, 2), (3, 0)),
                          tensors['ssm_conv1d.weight'][:, None], groups=10240).transpose(1, 2))
    q, k, v = torch.split(conv, [2048, 2048, 6144], -1)
    q = q.reshape(batch, length, 16, 128).repeat(1, 1, 3, 1)
    k = k.reshape(batch, length, 16, 128).repeat(1, 1, 3, 1)
    v = v.reshape(batch, length, 48, 128)
    decay = tensors['ssm_a'] * F.softplus(projected['ssm_alpha'] + tensors['ssm_dt.bias'])
    core, state = torch_recurrent_gated_delta_rule(
        q, k, v, decay, projected['ssm_beta'].sigmoid(), None, True, use_qk_l2norm_in_kernel=True)
    normed = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + 1e-6)
    normed = normed * tensors['ssm_norm.weight'] * projected['attn_gate'].reshape_as(core).sigmoid()
    torch.testing.assert_close(result, F.linear(normed.flatten(-2), out_weight), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(cache['gdn_state'], state, rtol=1e-5, atol=1e-7)


def test_qsa_prefill_decode_selection_and_context_match_independent_dense_membership():
    # Seven tokens exercise a complete4-token block plus a3-token tail.
    # Nonzero cache continuation proves block boundaries do not reset on decode.
    torch.manual_seed(23)
    model = Qwen4ExpReferenceModel('qwen4exp', 'unused')
    model.epsilon, model.rope_base = 1e-6, 1e7
    model.capture_stages = None
    length = 7
    values = torch.randn(1, length, 5)
    sizes = {'indexer.q_proj': 512, 'indexer.k_proj': 128,
             'attn_q': 12288, 'attn_k': 512, 'attn_v': 512}
    matrices = {name: torch.randn(size, 5) * .1 for name, size in sizes.items()}
    out_weight = torch.randn(5, 6144) * .01
    def linear(x, prefix, name):
        return F.linear(x, out_weight if name == 'attn_output' else matrices[name])
    from torch.nn import functional as F
    model._linear = linear
    model._norm = lambda x, name, group=None: rms_gain(x, torch.ones(x.shape[-1]), 1e-6)
    whole = model._qsa(values, '', {}, 3, 0)
    selected = model.snapshots[(S.QSA_SELECTED_INDICES, 3)]
    for row in range(length):
        np.testing.assert_array_equal(selected[0, row, :row + 1], np.arange(row + 1))
        assert np.all(selected[0, row, row + 1:] == -1)
    # At this length every visible token is selected. A separate dense causal
    # attention equation must therefore match the selected-gather path.
    positions = torch.arange(length)[None]
    q, gate = linear(values, '', 'attn_q').reshape(1, length, 24, 512).split(256, -1)
    q = model._rope(model._norm(q, ''), positions)
    k = model._rope(model._norm(linear(values, '', 'attn_k').reshape(1, length, 2, 256), ''), positions)
    v = linear(values, '', 'attn_v').reshape(1, length, 2, 256)
    k, v = k.repeat_interleave(12, 2), v.repeat_interleave(12, 2)
    logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / 16
    mask = torch.ones(length, length, dtype=torch.bool).tril()
    logits = logits.masked_fill(~mask, -torch.inf)
    context = torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), v)
    expected = F.linear((context * gate.sigmoid()).flatten(-2), out_weight)
    torch.testing.assert_close(whole, expected, rtol=1e-5, atol=1e-6)
    cache = {}
    first = model._qsa(values[:, :3], '', cache, 3, 0)
    second = model._qsa(values[:, 3:], '', cache, 3, 3)
    torch.testing.assert_close(torch.cat((first, second), 1), whole, rtol=1e-5, atol=1e-6)


def test_ple_gate_and_dilated_convolution_match_explicit_equation_and_chunk_history():
    torch.manual_seed(29)
    from torch.nn import functional as F
    model = Qwen4ExpReferenceModel('qwen4exp', 'unused')
    model.epsilon = 1e-6
    model.capture_stages = None
    tokens = torch.tensor([[17, 19, 23, 248044, 29, 31, 37, 41, 248046, 43]])
    trunk = torch.randn(1, 10, 10240)
    model.meta = {'qwen4exp.ple.' + key: value for key, value in {
        'layer_multipliers': [23703573157769, 20109073645365, 8052911324071],
        'head_offsets': [i * 20000003 for i in range(16)],
        'head_vocab_sizes': [20000003 + 2 * i for i in range(16)], 'eos_token_id': 248044,
    }.items()}
    gains = {name: torch.linspace(.5, 1.5, 10240) for name in
             ['ple_norm_key.weight', 'ple_norm_query.weight', 'ple_norm_conv.weight']}
    conv_weight = torch.randn(10240, 4) * .03
    def rows(name, indices):
        return ((indices[..., None] % 97).float() + torch.arange(160)) / 128
    model.weights = SimpleNamespace(rows=rows, tensor=lambda name:
        conv_weight if name.endswith('ple_conv1d.weight') else gains[name.removeprefix('blk.1.')])
    model._linear = lambda x, prefix, name: x.repeat(1, 1, 4) if name == 'ple_key' else x * .25
    previous = torch.full((1, 2), 248044)
    result = model._ple(trunk, tokens, previous, {}, 1)
    embedding = model.snapshots[(S.PLE_GATHER, 1)]
    embedding = torch.from_numpy(embedding)
    key = rms_gain(embedding.repeat(1, 1, 4), gains['ple_norm_key.weight'], 1e-6, 2560)
    query = rms_gain(trunk, gains['ple_norm_query.weight'], 1e-6, 2560)
    score = (key.reshape(1, 10, 4, 2560) * query.reshape(1, 10, 4, 2560)).sum(-1) / np.sqrt(2560)
    gate = (score.abs().clamp_min(1e-6).sqrt() * score.sign()).sigmoid()
    gated = (gate[..., None] * (embedding * .25)[..., None, :]).flatten(-2)
    normalized = rms_gain(gated, gains['ple_norm_conv.weight'], 1e-6, 2560)
    explicit_conv = torch.zeros_like(gated)
    for row in range(10):
        for tap in range(4):
            predecessor = row - 9 + tap * 3
            if predecessor >= 0:
                explicit_conv[:, row] += normalized[:, predecessor] * conv_weight[:, tap]
    expected = trunk + gated + F.silu(explicit_conv)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.snapshots[(S.PLE_GATE, 1)], gated.numpy(), rtol=1e-6)
    np.testing.assert_allclose(model.snapshots[(S.PLE_CONVOLUTION, 1)], expected.numpy(), rtol=1e-5, atol=1e-6)
    cache = {}
    first = model._ple(trunk[:, :4], tokens[:, :4], previous, cache, 1)
    second = model._ple(trunk[:, 4:], tokens[:, 4:], tokens[:, 2:4], cache, 1)
    torch.testing.assert_close(torch.cat((first, second), 1), result, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('tensor_type,block_bytes,maximum', [(T.Q4_K, 144, 15), (T.Q5_K, 176, 31)])
def test_kquant_selected_expert511_uses_original_native_superblocks(tensor_type, block_bytes, maximum):
    """A native K-quant parent shape selects expert511 without full decode or requantization."""
    from collections import OrderedDict
    # Exact Q4_K block: d=1, dmin=0, eight 6-bit subscales=1,
    # eight minima=0, all packed nibbles equal to this expert's code.
    scales = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    blocks = []
    for expert in range(512):
        code = expert % 16
        high = bytes([255 if expert & 16 else 0]) * 32 if tensor_type == T.Q5_K else b''
        block = struct.pack('<ee', 1., 0.) + scales + high + bytes([code | code << 4]) * 128
        blocks.append(block * 2)  # Independently encoded [2,256] matrix per expert.
    native = b''.join(blocks)
    weight = NativeSliceWeights.__new__(NativeSliceWeights)
    weight.cache = OrderedDict()
    weight.decoded_bytes = 0
    weight.cache_bytes = 4096
    info = SimpleNamespace(shape=(512, 2, 256), type=tensor_type)
    weight.directory = {'blk.0.ffn_gate_exps.weight': info}
    weight.parser = SimpleNamespace(read_tensor_data=lambda _: memoryview(native))
    assert get_quantization_info(tensor_type)['block_size'] == 256
    assert get_quantization_info(tensor_type)['block_bytes'] == block_bytes
    result = weight.slice('blk.0.ffn_gate_exps.weight', 511)
    assert result.shape == (2, 256)
    assert torch.equal(result, torch.full((2, 256), float(maximum)))
    assert weight.decoded_bytes == 2 * 256 * 4
    assert list(weight.cache) == [('blk.0.ffn_gate_exps.weight', 511)]
    with pytest.raises(ValueError, match='selected-expert'):
        weight.tensor('blk.0.ffn_gate_exps.weight')
    with pytest.raises(IndexError):
        weight.slice('blk.0.ffn_gate_exps.weight', 512)
