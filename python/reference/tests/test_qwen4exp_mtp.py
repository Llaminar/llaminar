"""Small independent Dense0 equations; no model payloads or accelerator access.

NumPy/scalar expected values deliberately do not call the oracle's norm, HC,
attention or terminal helpers. Unequal branch amplitudes, asymmetric projection
halves, distinct KV heads and future values make the source conventions visible.
These are arithmetic/contract proofs, not whole-model or native-kernel parity.
"""
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from python.reference.qwen4exp_mtp import Dense0Geometry, Dense0KV, Qwen4ExpDense0Reference


@pytest.fixture(scope='module', autouse=True)
def one_cpu_thread():
    """Keep tiny reference workloads cheap and independent of model gate workers."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TensorReader:
    """Existing NativeSliceWeights surface over explicitly small source tensors."""

    def __init__(self, tensors):
        self.data = tensors
        self.directory = {k: SimpleNamespace(shape=tuple(v.shape)) for k, v in tensors.items()}
        self.calls = []

    def tensor(self, name):
        self.calls.append(('tensor', name))
        return self.data[name]

    def rows(self, name, ids):
        self.calls.append(('rows', name))
        return self.data[name][ids]

    def linear(self, x, name, expert=None):
        self.calls.append(('linear', name, expert))
        weight = self.data[name] if expert is None else self.data[name][expert]
        return F.linear(x, weight)

    def row_range(self, name, start, end):
        self.calls.append(('row_range', name, start, end))
        return self.data[name][start:end]


def fixture(branches=4, batch=2, length=3, query_heads=4):
    """Build asymmetric inputs and all consumed exact names, excluding dormant roles."""
    g = Dense0Geometry(width=6, branches=branches, low_rank=3, query_heads=query_heads,
                       kv_heads=2, head_dim=6, rotary_dim=4, experts=3,
                       experts_used=2, expert_width=5, shared_width=5)
    generator = torch.Generator().manual_seed(173)
    rand = lambda *shape: torch.randn(*shape, generator=generator) * .35
    p = 'blk.48.'
    tensors = {}
    def put(role, value):
        tensors[p + role + '.weight'] = value
    for role in ('hc_attn', 'hc_ffn', 'nextn.hc_head'):
        put(role + '_norm', torch.linspace(.4, 1.7, g.trunk_width))
        put(role + '_down', rand(g.low_rank, g.trunk_width))
        put(role + '_up', rand(g.trunk_width, g.low_rank))
        if role != 'nextn.hc_head':
            put(role + '_inject', rand(branches, g.trunk_width))
    put('nextn.enorm', torch.linspace(.7, 1.2, g.width))
    put('nextn.hnorm', torch.linspace(.3, 1.5, g.trunk_width))
    put('nextn.eh_proj', rand(g.width, 2 * g.width))
    put('attn_q', rand(2 * g.query_heads * g.head_dim, g.width))
    put('attn_k', rand(g.kv_heads * g.head_dim, g.width))
    put('attn_v', rand(g.kv_heads * g.head_dim, g.width))
    put('attn_output', rand(g.width, g.query_heads * g.head_dim))
    put('attn_q_norm', torch.tensor([.6, 1.2, 1.5, .8, 1.1, .9]))
    put('attn_k_norm', torch.tensor([1.4, .7, .9, 1.1, .6, 1.3]))
    put('ffn_gate_inp', rand(g.experts, g.width))
    for role in ('ffn_gate_exps', 'ffn_up_exps'):
        put(role, rand(g.experts, g.expert_width, g.width))
    put('ffn_down_exps', rand(g.experts, g.width, g.expert_width))
    for role in ('ffn_gate_shexp', 'ffn_up_shexp'):
        put(role, rand(g.shared_width, g.width))
    put('ffn_down_shexp', rand(g.width, g.shared_width))
    put('ffn_gate_inp_shexp', rand(g.width))
    target = TensorReader({'token_embd.weight': rand(13, g.width), 'output.weight': rand(11, g.width)})
    reader = TensorReader(tensors)
    model = Qwen4ExpDense0Reference(reader, target, g, rope_base=10000)
    hidden = rand(batch, length, branches, g.width)
    hidden *= torch.logspace(-1, 2, branches).reshape(1, 1, branches, 1)
    hidden = hidden.flatten(-2)
    tokens = torch.arange(batch * length).reshape(batch, length) % 13
    positions = torch.arange(17, 17 + length).expand(batch, -1).clone()
    return model, hidden, tokens, positions


def npvalue(value):
    """Use higher precision in the independent scalar oracle, never torch helpers."""
    return value.detach().numpy().astype(np.float64)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(x, gain, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * gain


def matrix(model, role):
    return npvalue(model.weights.data[model.prefix + role + '.weight'])


def np_mix(model, hidden, tokens, *, flattened=False, reverse=False, pooled=False):
    """Explicit W_e@e + W_h@h per branch with adversarial wrong-convention controls."""
    g = model.g
    h = npvalue(hidden).reshape(*hidden.shape[:2], g.branches, g.width)
    gain = matrix(model, 'nextn.hnorm').reshape(g.branches, g.width)
    if flattened:
        h = normalize(h.reshape(*h.shape[:2], -1), gain.reshape(-1), model.epsilon).reshape(h.shape)
    else:
        h = normalize(h, gain, model.epsilon)
    if pooled:
        h = np.broadcast_to(h.mean(-2, keepdims=True), h.shape)
    e = npvalue(model.target_weights.data['token_embd.weight'])[tokens.numpy()]
    e = normalize(e, matrix(model, 'nextn.enorm'), model.epsilon)
    w = matrix(model, 'nextn.eh_proj')
    result = np.zeros_like(h)
    for b in range(h.shape[0]):
        for t in range(h.shape[1]):
            for branch in range(g.branches):
                if reverse:
                    result[b, t, branch] = w[:, :g.width] @ h[b, t, branch] + w[:, g.width:] @ e[b, t]
                else:
                    result[b, t, branch] = w[:, :g.width] @ e[b, t] + w[:, g.width:] @ h[b, t, branch]
    return result.reshape(hidden.shape)


def np_hc(model, trunk, role):
    """Independent branch normalization, low-rank nonlinear read and injection."""
    g = model.g
    x = np.asarray(trunk).reshape(*trunk.shape[:2], g.branches, g.width)
    xn = normalize(x, matrix(model, role + '_norm').reshape(g.branches, g.width), model.epsilon)
    flat = xn.reshape(trunk.shape)
    lo = flat @ matrix(model, role + '_down').T / g.branches
    lo = lo * sigmoid(lo)
    gate = sigmoid(lo @ matrix(model, role + '_up').T)
    read = (flat * gate).reshape(x.shape).mean(-2)
    if role == 'nextn.hc_head':
        return read, flat, gate
    inject = 2 * sigmoid(flat @ matrix(model, role + '_inject').T / g.branches)
    return read, flat, inject


def np_attention(model, read, positions, cache=None, *, wrong_layout=False, wrong_gqa=False, noncausal=False):
    """Scalar dense causal oracle, with independent head layout and rotation loops."""
    g = model.g
    b, t = read.shape[:2]
    qraw = read @ matrix(model, 'attn_q').T
    if wrong_layout:
        q, gate = np.split(qraw, 2, -1)
        q = q.reshape(b, t, g.query_heads, g.head_dim)
        gate = gate.reshape(q.shape)
    else:
        q, gate = np.split(qraw.reshape(b, t, g.query_heads, 2 * g.head_dim), 2, -1)
    q = normalize(q, matrix(model, 'attn_q_norm'), model.epsilon)
    k = (read @ matrix(model, 'attn_k').T).reshape(b, t, g.kv_heads, g.head_dim)
    k = normalize(k, matrix(model, 'attn_k_norm'), model.epsilon)
    v = (read @ matrix(model, 'attn_v').T).reshape(k.shape)
    for values in (q, k):
        for batch in range(b):
            for row in range(t):
                for head in range(values.shape[2]):
                    for i in range(g.rotary_dim // 2):
                        angle = float(positions[batch, row]) * model.rope_base ** (-2 * i / g.rotary_dim)
                        x, y = values[batch, row, head, i], values[batch, row, head, i + g.rotary_dim // 2]
                        values[batch, row, head, i] = x * math.cos(angle) - y * math.sin(angle)
                        values[batch, row, head, i + g.rotary_dim // 2] = y * math.cos(angle) + x * math.sin(angle)
    keys, vals, kp = k, v, np.asarray(positions)
    if cache is not None:
        keys = np.concatenate((npvalue(cache.keys), k), axis=1)
        vals = np.concatenate((npvalue(cache.values), v), axis=1)
        kp = np.concatenate((cache.positions.numpy(), kp), axis=1)
    context = np.zeros(q.shape)
    probability = np.zeros((b, g.query_heads, t, keys.shape[1]))
    for batch in range(b):
        for row in range(t):
            visible = np.arange(keys.shape[1]) if noncausal else np.where(kp[batch] <= positions[batch, row])[0]
            for head in range(g.query_heads):
                kv = head % g.kv_heads if wrong_gqa else head // (g.query_heads // g.kv_heads)
                logits = np.array([q[batch, row, head] @ keys[batch, n, kv] / math.sqrt(g.head_dim) for n in visible])
                p = np.exp(logits - logits.max())
                p /= p.sum()
                for j, n in enumerate(visible):
                    probability[batch, head, row, n] = p[j]
                    context[batch, row, head] += p[j] * vals[batch, n, kv]
    out = (context * sigmoid(gate)).reshape(b, t, -1) @ matrix(model, 'attn_output').T
    return out, probability, (k, v)


def np_moe(model, read):
    """Independent per-token selected experts plus the single shared expert gate."""
    g = model.g
    flat = read.reshape(-1, g.width)
    output = np.zeros_like(flat)
    for row, x in enumerate(flat):
        logits = matrix(model, 'ffn_gate_inp') @ x
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        ids = np.argsort(-prob, kind='stable')[:g.experts_used]
        weights = prob[ids] / prob[ids].sum()
        for expert in sorted(ids):
            gate = matrix(model, 'ffn_gate_exps')[expert] @ x
            up = matrix(model, 'ffn_up_exps')[expert] @ x
            y = matrix(model, 'ffn_down_exps')[expert] @ (gate * sigmoid(gate) * up)
            output[row] += y * weights[np.where(ids == expert)[0][0]]
        gate = matrix(model, 'ffn_gate_shexp') @ x
        up = matrix(model, 'ffn_up_shexp') @ x
        shared = matrix(model, 'ffn_down_shexp') @ (gate * sigmoid(gate) * up)
        output[row] += shared * sigmoid(matrix(model, 'ffn_gate_inp_shexp') @ x)
    return output.reshape(read.shape)


@pytest.mark.parametrize('branches,batch,length', [(2, 1, 1), (4, 1, 3), (4, 2, 3)])
def test_embedding_first_mix_preserves_independently_normalized_hidden_branches(branches, batch, length):
    m, h, tokens, _ = fixture(branches, batch, length)
    actual, snapshots = m.mix_inputs(tokens, h)
    expected = np_mix(m, h, tokens)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-7)
    assert snapshots['mtp_concat'].shape == (batch, length, branches, 2 * m.g.width)
    for wrong in ({'flattened': True}, {'reverse': True}, {'pooled': True}):
        assert np.max(np.abs(expected - np_mix(m, h, tokens, **wrong))) > .05


@pytest.mark.parametrize('with_prefix', [False, True])
@pytest.mark.parametrize('query_heads', [4, 24])
def test_dense_causal_gqa_matches_scalar_oracle_and_prefix(with_prefix, query_heads):
    m, h, _, positions = fixture(query_heads=query_heads)
    read = h[..., :m.g.width] * 7
    prefix = None
    if with_prefix:
        g = m.g
        keys = torch.linspace(-1, 1, 2 * 2 * g.kv_heads * g.head_dim).reshape(2, 2, g.kv_heads, g.head_dim)
        values = keys.flip(-1) * 2 + .3
        prefix = Dense0KV(keys, values, torch.tensor([[3, 11], [2, 8]]))
    out, cache, snapshots = m.dense_attention(read, positions, prefix)
    expected, probs, kv = np_attention(m, npvalue(read), positions.numpy(), prefix)
    np.testing.assert_allclose(out, expected, rtol=4e-6, atol=8e-7)
    np.testing.assert_allclose(snapshots['mtp_attn_probability'], probs, rtol=4e-6, atol=5e-7)
    np.testing.assert_allclose(cache.keys[:, -3:], kv[0], rtol=3e-6, atol=6e-7)
    if prefix is not None:
        assert torch.equal(cache.keys[:, :2], prefix.keys)
        assert torch.equal(cache.values[:, :2], prefix.values)
    for wrong in ({'wrong_layout': True}, {'wrong_gqa': True}, {'noncausal': True}):
        wrong_out, _, _ = np_attention(m, npvalue(read), positions.numpy(), prefix, **wrong)
        assert np.max(np.abs(expected - wrong_out)) > .01


def test_dense_prefix_chunking_and_input_cache_remain_independent():
    m, h, _, positions = fixture()
    read = h[..., :m.g.width] * 7
    whole, _, _ = m.dense_attention(read, positions)
    first, cache, _ = m.dense_attention(read[:, :1], positions[:, :1])
    original = cache.keys.clone()
    rest, continued, _ = m.dense_attention(read[:, 1:], positions[:, 1:], cache)
    torch.testing.assert_close(torch.cat((first, rest), 1), whole, rtol=3e-6, atol=5e-7)
    assert torch.equal(cache.keys, original) and cache.keys.shape[1] == 1
    assert continued.keys.shape[1] == 3


def test_published_rotary64_matches_scalar_text_phases_and_preserves_tail():
    g = Dense0Geometry()
    m = Qwen4ExpDense0Reference(TensorReader({}), TensorReader({}), g)
    generator = torch.Generator().manual_seed(29)
    value = torch.randn(1, 2, 3, 256, generator=generator)
    positions = torch.tensor([[0, 19]])
    actual = m._rope(value, positions)
    expected = npvalue(value)
    original = expected.copy()
    for row, position in enumerate([0, 19]):
        for head in range(3):
            for lane in range(32):
                phase = position * 1e7 ** (-2 * lane / 64)
                x, y = original[0, row, head, lane], original[0, row, head, lane + 32]
                expected[0, row, head, lane] = x * math.cos(phase) - y * math.sin(phase)
                expected[0, row, head, lane + 32] = y * math.cos(phase) + x * math.sin(phase)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=8e-7)
    assert torch.equal(actual[..., 64:], value[..., 64:])


@pytest.mark.parametrize('tile_rows', [1, 4, 32])
def test_predictor_hc_terminal_uses_own_final_gains_and_borrowed_lm_head(tile_rows):
    m, h, _, _ = fixture()
    logits, head, _ = m.terminal(h, tile_rows=tile_rows)
    expected, normed, gate = np_hc(m, npvalue(h), 'nextn.hc_head')
    expected_logits = expected @ npvalue(m.target_weights.data['output.weight']).T
    np.testing.assert_allclose(head, expected, rtol=3e-6, atol=5e-7)
    np.testing.assert_allclose(logits, expected_logits, rtol=3e-6, atol=6e-7)
    # No extra final normalization; gains are already final GGUF (no added 1).
    extra = normalize(expected, np.ones(m.g.width), m.epsilon)
    assert np.max(np.abs(extra - expected)) > .1
    assert np.max(np.abs(normed * gate - normed * .5)) > .01
    tiles = [call for call in m.target_weights.calls if call[0] == 'row_range']
    assert sum(end - start for _, _, start, end in tiles) == 11
    assert max(end - start for _, _, start, end in tiles) <= tile_rows
    assert all('output_hc' not in call[1] and 'shared_head_norm' not in call[1] for call in m.weights.calls)


def test_complete_small_predictor_matches_independent_equations_without_dormant_roles():
    m, h, tokens, pos = fixture()
    actual = m.forward(tokens, h, pos)
    trunk = np_mix(m, h, tokens)
    read, _, inject = np_hc(m, trunk, 'hc_attn')
    attn, _, _ = np_attention(m, read, pos.numpy())
    trunk = trunk + (inject[..., None] * attn[..., None, :]).reshape(trunk.shape)
    read, _, inject = np_hc(m, trunk, 'hc_ffn')
    ffn = np_moe(m, read)
    trunk = trunk + (inject[..., None] * ffn[..., None, :]).reshape(trunk.shape)
    head, _, _ = np_hc(m, trunk, 'nextn.hc_head')
    logits = head @ npvalue(m.target_weights.data['output.weight']).T
    np.testing.assert_allclose(actual['hidden_trunk'], trunk, rtol=5e-6, atol=1e-6)
    np.testing.assert_allclose(actual['head'], head, rtol=5e-6, atol=1e-6)
    np.testing.assert_allclose(actual['logits'], logits, rtol=5e-6, atol=1e-6)
    assert actual['snapshots']['h_nextn'] is actual['hidden_trunk']
    assert actual['hidden_trunk'].shape[-1] == m.g.trunk_width
    assert actual['head'].shape[-1] == m.g.width
    assert {call[1] for call in m.target_weights.calls} == {'token_embd.weight', 'output.weight'}
    assert not any('indexer' in c[1] or 'ple' in c[1] or 'shared_head_norm' in c[1] for c in m.weights.calls)


@pytest.mark.parametrize('defect', ['collapsed_hidden', 'float_tokens', 'nonmonotone_positions',
                                  'overlap_prefix', 'wrong_gain', 'wrong_matrix'])
def test_invalid_input_contracts_fail_explicitly(defect):
    m, h, tokens, pos = fixture()
    if defect == 'collapsed_hidden':
        with pytest.raises(ValueError, match='requires shape'):
            m.mix_inputs(tokens, h[..., :m.g.width])
    elif defect == 'float_tokens':
        with pytest.raises(ValueError, match='CPU int64'):
            m.mix_inputs(tokens.float(), h)
    elif defect == 'nonmonotone_positions':
        pos[:, 1] = pos[:, 0]
        with pytest.raises(ValueError, match='strictly increasing'):
            m.dense_attention(h[..., :m.g.width], pos)
    elif defect == 'overlap_prefix':
        _, cache, _ = m.dense_attention(h[:, :1, :m.g.width], pos[:, :1])
        saved = cache.keys.clone()
        with pytest.raises(ValueError, match='strictly earlier'):
            m.dense_attention(h[..., :m.g.width], pos, cache)
        assert torch.equal(cache.keys, saved)
    elif defect == 'wrong_gain':
        m.weights.data[m.prefix + 'nextn.hnorm.weight'] = torch.ones(m.g.width)
        with pytest.raises(ValueError, match='requires shape'):
            m.mix_inputs(tokens, h)
    else:
        m.weights.directory[m.prefix + 'nextn.eh_proj.weight'].shape = (m.g.width, m.g.width)
        with pytest.raises(ValueError, match='requires native shape'):
            m.mix_inputs(tokens, h)


@pytest.mark.parametrize('bad', [{'query_heads': 3}, {'rotary_dim': 3}, {'rotary_dim': 6},
                                 {'experts_used': 4}, {'width': 0}])
def test_invalid_geometry_is_unrepresentable(bad):
    fields = dict(width=6, branches=4, low_rank=3, query_heads=4, kv_heads=2,
                  head_dim=4, rotary_dim=2, experts=3, experts_used=2,
                  expert_width=5, shared_width=5)
    fields.update(bad)
    with pytest.raises(ValueError):
        Dense0Geometry(**fields)
