"""Exact-GGUF CPU/FP32 Qwen4Exp oracle in the canonical reference framework.

Equations are pinned to Transformers 28034545c35acac05793695bd5c878f66e6b544e
and GGUF transforms to llama.cpp 1b70350c92563449576bcbbf552994b1d1ed847d.
Native directory slices are the sole weight source. PLE and expert tables are
never materialized wholesale; a bounded immutable decode cache holds only
requested source slices. This diagnostic oracle is independent of production
kernels and does not certify production graph execution or backend support.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
import math

import numpy as np
import torch
import torch.nn.functional as F

from .base import AbstractReferenceModel
from .registry import ModelRegistry
from .pipeline_stages import PipelineStage as S
from .loaders.gguf_parser import GGUFParser
from .loaders.dequantize import dequantize, get_quantization_info

CHECKPOINT_REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
TRANSFORMERS_REVISION = '28034545c35acac05793695bd5c878f66e6b544e'
CONVERTER_REVISION = '1b70350c92563449576bcbbf552994b1d1ed847d'


def validate_qwen4exp_main_metadata(metadata):
    """Validate the unchanged canonical main header without reading tensor payloads.

    Both normal main construction and sidecar-only replay must authenticate the
    same original mixer, PLE and activation geometry. The returned layer counts
    preserve the existing inline-predictor allowance; external-pair binding then
    explicitly requires zero inline predictors.
    """
    if metadata.get('general.architecture') != 'qwen4exp':
        raise ValueError('exact qwen4exp GGUF required')
    expected = {
        'embedding_length': 2560, 'hyper_connection.count': 4,
        'hyper_connection.low_rank': 320, 'attention.head_count': 24,
        'attention.head_count_kv': 2, 'attention.key_length': 256,
        'rope.dimension_count': 64, 'ssm.group_count': 16,
        'ssm.time_step_rank': 48, 'ssm.state_size': 128,
        'ssm.conv_kernel': 4, 'expert_count': 512, 'expert_used_count': 10,
        'attention.indexer.head_count': 4, 'attention.indexer.key_length': 128,
        'attention.indexer.top_k': 2048, 'ple.ngram_size': 3,
        'ple.heads_per_ngram': 8, 'ple.conv_kernel': 4,
        'embedding_length_per_layer_input': 160, 'ple.eos_token_id': 248044,
    }
    for suffix, value in expected.items():
        if metadata.get('qwen4exp.' + suffix) != value:
            raise ValueError(f'qwen4exp.{suffix}: requires pinned geometry {value}')
    n_layers = int(metadata['qwen4exp.block_count'])
    predictor_layers = int(metadata.get('qwen4exp.nextn_predict_layers', 0))
    if n_layers - predictor_layers != 48 or predictor_layers not in (0, 1):
        raise ValueError('requires 48 main blocks and zero or one embedded predictor')
    ratios = metadata['qwen4exp.attention.compress_ratios']
    if ratios != [4 if (i + 1) % 4 == 0 or i >= 48 else 0 for i in range(n_layers)]:
        raise ValueError('qwen4exp attention compression/mixer identity differs')
    if metadata['qwen4exp.ple.layers'] != [1]:
        raise ValueError('pinned PLE is decoder layer1')
    return n_layers, predictor_layers


class NativeSliceWeights:
    """Own split-file mappings and decode only complete native row/expert slices."""

    def __init__(self, path, cache_bytes=1 << 30):
        """Parse immutable directories without reading the model payload."""
        self.parser = GGUFParser(path)
        self.parser.parse()
        self.directory = {t.name: t for t in self.parser.tensors}
        self.cache_bytes = int(cache_bytes)
        if self.cache_bytes < 0:
            raise ValueError('weight cache budget must be nonnegative')
        self.cache = OrderedDict()
        self.decoded_bytes = 0

    def close(self):
        """Retire decoded tensors before the owning immutable mappings."""
        self.cache.clear()
        self.decoded_bytes = 0
        self.parser.close()

    def _decode(self, name, index=None):
        """Decode a whole ordinary tensor or one independently encoded outer slice."""
        key = (name, index)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        info = self.directory[name]
        if name == 'per_layer_token_embd.weight' and index is None:
            raise ValueError('PLE table requires selected-row access')
        if len(info.shape) == 3 and index is None:
            raise ValueError('expert tensor requires selected-expert access')
        shape = info.shape if index is None else info.shape[1:]
        if index is not None and not 0 <= index < info.shape[0]:
            raise IndexError(f'{name}: source slice {index} is out of range')
        fmt = get_quantization_info(info.type)
        count = math.prod(shape)
        if not fmt['block_size']:
            raise ValueError(f'{name}: source format {info.type.name} has no native block geometry')
        if count % fmt['block_size']:
            raise ValueError(f'{name}: source slice is not native-block aligned')
        size = count // fmt['block_size'] * fmt['block_bytes']
        start = 0 if index is None else index * size
        raw = self.parser.read_tensor_data(info)
        try:
            # Copy only the selected native extent; decoded tensors never retain
            # exported mmap views, so close() has no outstanding buffer leases.
            payload = bytes(raw[start:start + size])
        finally:
            raw.release()
        if len(payload) != size:
            raise ValueError(f'{name}: selected native extent is truncated')
        result = torch.from_numpy(dequantize(payload, info.type, shape).copy())
        size = result.numel() * result.element_size()
        if size <= self.cache_bytes:
            while self.decoded_bytes + size > self.cache_bytes and self.cache:
                _, old = self.cache.popitem(last=False)
                self.decoded_bytes -= old.numel() * old.element_size()
            self.cache[key] = result
            self.decoded_bytes += size
        return result

    def tensor(self, name):
        """Read an ordinary tensor; immutable row tables reject this operation."""
        return self._decode(name)

    def slice(self, name, index):
        """Read one native outer row or expert, preserving source format."""
        return self._decode(name, int(index))

    def rows(self, name, indices):
        """Gather only requested rows while deduplicating repeated hash IDs."""
        unique, inverse = torch.unique(indices.cpu().long(), sorted=True, return_inverse=True)
        decoded = torch.stack([self.slice(name, i) for i in unique.tolist()])
        return decoded[inverse].reshape(*indices.shape, -1)

    def linear(self, value, name, expert=None):
        """Apply one exact-GGUF matrix using the independently decoded FP32 gain."""
        weight = self.tensor(name) if expert is None else self.slice(name, expert)
        return F.linear(value, weight)

    def row_range(self, name, start, end):
        """Decode one contiguous bounded terminal-matrix tile without row allocations."""
        info = self.directory[name]
        if len(info.shape) != 2 or not 0 <= start < end <= info.shape[0]:
            raise ValueError(f'{name}: invalid row tile')
        if name == 'per_layer_token_embd.weight':
            raise ValueError('PLE table requires selected-row access')
        fmt = get_quantization_info(info.type)
        width = info.shape[1]
        if not fmt['block_size'] or width % fmt['block_size']:
            raise ValueError(f'{name}: row tile is not native-block aligned')
        stride = width // fmt['block_size'] * fmt['block_bytes']
        raw = self.parser.read_tensor_data(info)
        try:
            payload = bytes(raw[start * stride:end * stride])
        finally:
            raw.release()
        if len(payload) != (end - start) * stride:
            raise ValueError(f'{name}: terminal matrix tile is truncated')
        return torch.from_numpy(dequantize(payload, info.type, (end - start, width)).copy())


@dataclass
class Qwen4ExpCache:
    """Reference-owned serial cache; never a mirror of production execution state."""
    tokens: list = field(default_factory=list)
    layers: dict = field(default_factory=dict)
    position: int = 0


def rms_gain(value, gain, epsilon, group_width=None):
    """Normalize before multiplying GGUF's final gain (no HF zero-center +1)."""
    shape = value.shape
    grouped = value if group_width is None else value.reshape(*shape[:-1], -1, group_width)
    normed = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + epsilon)
    return normed.reshape(shape) * gain


def ple_hash_indices(token_ids, previous, multipliers, offsets, vocab_sizes, eos):
    """Compute exact signed-int64 suffix XOR hashes with predecessor-EOS resets."""
    batch, length = token_ids.shape
    history = torch.cat((previous, token_ids.long()), dim=1)
    positions = torch.arange(history.shape[1]).expand(batch, -1)
    eos_positions = torch.where(history == eos, positions, -1)
    inclusive = torch.cummax(eos_positions, dim=1).values
    prior = F.pad(inclusive[:, :-1], (1, 0), value=-1)
    shifted = []
    for shift in range(len(multipliers)):
        source = positions - shift
        valid = (source >= 0) & (positions - prior - 1 >= shift)
        shifted.append(torch.where(valid, history.gather(1, source.clamp_min(0)), eos))
    mixed = shifted[0] * multipliers[0]
    blocks = []
    heads = len(offsets) // (len(multipliers) - 1)
    for order in range(2, len(multipliers) + 1):
        mixed = torch.bitwise_xor(mixed, shifted[order - 1] * multipliers[order - 1])
        start = (order - 2) * heads
        sizes = torch.tensor(vocab_sizes[start:start + heads], dtype=torch.int64)
        base = torch.tensor(offsets[start:start + heads], dtype=torch.int64)
        blocks.append(torch.remainder(mixed[..., None], sizes) + base)
    return torch.cat(blocks, -1)[:, -length:]


class Qwen4ExpReferenceModel(AbstractReferenceModel):
    """Pinned main-model oracle with GR, QSA, PLE, GDN and routed-MoE checkpoints."""

    def load_model(self, tokenizer=None, cache_bytes=1 << 30, **kwargs):
        """Require exact qwen4exp source geometry and CPU FP32 reference arithmetic."""
        if self.device.type != 'cpu' or self.dtype != torch.float32:
            raise ValueError('Qwen4Exp parity reference requires CPU/FP32')
        self.weights = NativeSliceWeights(self.checkpoint_path, cache_bytes)
        self.meta = self.weights.parser.metadata
        self.n_layers, self.predictor_layers = validate_qwen4exp_main_metadata(self.meta)
        self.ratios = self.meta['qwen4exp.attention.compress_ratios']
        self.epsilon = float(self.meta['qwen4exp.attention.layer_norm_rms_epsilon'])
        self.rope_base = float(self.meta['qwen4exp.rope.freq_base'])
        self.width, self.branches = 2560, 4
        self.tokenizer = tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                'Qwen/Qwen3.8-Flash-Next', revision=CHECKPOINT_REVISION)
        # Reuse canonical pack metadata writer without constructing a second HF model.
        config = SimpleNamespace(architectures=['Qwen4ExpForCausalLM'], num_hidden_layers=48,
            num_attention_heads=24, num_key_value_heads=2, hidden_size=2560,
            head_dim=256, moe_intermediate_size=640, vocab_size=248320)
        self.hf_model = SimpleNamespace(config=config)
        self.model = self

    def close(self):
        """Release all exact-source mappings after pack generation."""
        self.weights.close()

    def capture_stage(self, stage, tensor, layer_idx=-1):
        """Preserve integer checkpoints exactly; convert only floating values to FP32."""
        if self.capture_stages is not None and stage not in self.capture_stages:
            return
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.float()
        self.snapshots[(stage, layer_idx)] = value.numpy().copy()

    def _norm(self, value, name, group=None):
        """Apply the source's final gain in independently normalized groups."""
        return rms_gain(value, self.weights.tensor(name), self.epsilon, group)

    def _linear(self, value, prefix, stem):
        """Resolve one canonical tensor role without name remapping or aliases."""
        return self.weights.linear(value, prefix + stem + '.weight')

    def _rope(self, value, positions):
        """Apply text-position interleaved mRoPE, rotating only the first64 lanes."""
        inv = self.rope_base ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
        angles = positions.float()[..., None] * inv
        angles = torch.cat((angles, angles), -1)
        while angles.ndim < value.ndim:
            angles = angles.unsqueeze(-2)
        first = value[..., :64]
        half = torch.cat((-first[..., 32:], first[..., :32]), -1)
        return torch.cat((first * angles.cos() + half * angles.sin(), value[..., 64:]), -1)

    def _gr_read(self, trunk, prefix, role, layer):
        """Read four normalized residual branches through the low-rank gate."""
        normed = self._norm(trunk, prefix + role + '_norm.weight', self.width)
        low = F.silu(self._linear(normed, prefix, role + '_down') / self.branches)
        gates = torch.sigmoid(self._linear(low, prefix, role + '_up'))
        mixed = (gates.reshape(*trunk.shape[:-1], 4, 2560) *
                 normed.reshape(*trunk.shape[:-1], 4, 2560)).mean(-2)
        if role == 'hc_attn':
            self.capture_stage(S.GR_NORM, normed, layer)
            self.capture_stage(S.GR_READ, mixed, layer)
        return mixed, normed

    def _gr_write(self, trunk, normed, output, prefix, role):
        """Inject the compute-width result into every retained residual branch."""
        inject = 2 * torch.sigmoid(self._linear(normed, prefix, role + '_inject') / 4)
        return trunk + (inject[..., None] * output[..., None, :]).flatten(-2)

    def _conv(self, values, weight, cache, key, dilation=1):
        """Run one causal depthwise convolution with exact retained lexical history."""
        history = (weight.shape[-1] - 1) * dilation
        previous = cache.get(key)
        if previous is None:
            previous = values.new_zeros(values.shape[0], history, values.shape[-1])
        joined = torch.cat((previous, values), 1)
        cache[key] = joined[:, -history:].clone()
        result = F.conv1d(joined.transpose(1, 2), weight[:, None, :],
                          groups=values.shape[-1], dilation=dilation)
        return F.silu(result.transpose(1, 2))

    def _ple(self, trunk, tokens, previous, cache, layer):
        """Gather16 native rows, gate each branch and add the dilated short convolution."""
        prefix = f'blk.{layer}.'
        get = lambda name: self.meta['qwen4exp.ple.' + name]
        indices = ple_hash_indices(tokens, previous, get('layer_multipliers'),
                                   get('head_offsets'), get('head_vocab_sizes'), get('eos_token_id'))
        rows = self.weights.rows('per_layer_token_embd.weight', indices)
        embedding = rows.flatten(-2)
        key = self._norm(self._linear(embedding, prefix, 'ple_key'), prefix + 'ple_norm_key.weight', 2560)
        query = self._norm(trunk, prefix + 'ple_norm_query.weight', 2560)
        score = (key.reshape(*tokens.shape, 4, 2560) * query.reshape(*tokens.shape, 4, 2560)).sum(-1)
        score = score / math.sqrt(2560)
        gate = torch.sigmoid(score.abs().clamp_min(1e-6).sqrt() * score.sign())
        value = self._linear(embedding, prefix, 'ple_value')
        gated = (gate[..., None] * value[..., None, :]).flatten(-2)
        conv = self._conv(self._norm(gated, prefix + 'ple_norm_conv.weight', 2560),
                          self.weights.tensor(prefix + 'ple_conv1d.weight'), cache, 'ple_conv', 3)
        self.capture_stage(S.PLE_HASH_INDICES, indices, layer)
        self.capture_stage(S.PLE_GATHER, embedding, layer)
        self.capture_stage(S.PLE_GATE, gated, layer)
        self.capture_stage(S.PLE_CONVOLUTION, trunk + gated + conv, layer)
        return trunk + gated + conv

    def _gdn(self, value, prefix, cache, layer):
        """Use GGUF's tiled V-head order with a fixed serial recurrent FP32 oracle."""
        mixed = self._linear(value, prefix, 'attn_qkv')
        z = self._linear(value, prefix, 'attn_gate')
        alpha = self._linear(value, prefix, 'ssm_alpha')
        beta_projection = self._linear(value, prefix, 'ssm_beta')
        conv = self._conv(mixed, self.weights.tensor(prefix + 'ssm_conv1d.weight'), cache, 'gdn_conv')
        q, k, v = torch.split(conv, [2048, 2048, 6144], -1)
        shape = value.shape[:2]
        q, k = q.reshape(*shape, 16, 128), k.reshape(*shape, 16, 128)
        # Converter changes [keyhead, V-per-key] to [V-per-key, keyhead].
        q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).repeat(1, 1, 3, 1) / math.sqrt(128)
        k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).repeat(1, 1, 3, 1)
        v = v.reshape(*shape, 48, 128)
        decay = self.weights.tensor(prefix + 'ssm_a') * F.softplus(
            alpha + self.weights.tensor(prefix + 'ssm_dt.bias'))
        beta = torch.sigmoid(beta_projection)
        state = cache.get('gdn_state', value.new_zeros(shape[0], 48, 128, 128))
        outputs = []
        for row in range(shape[1]):
            state = state * decay[:, row, :, None, None].exp()
            memory = (state * k[:, row, :, :, None]).sum(-2)
            delta = (v[:, row] - memory) * beta[:, row, :, None]
            state = state + k[:, row, :, :, None] * delta[..., None, :]
            outputs.append((state * q[:, row, :, :, None]).sum(-2))
        cache['gdn_state'] = state
        core = torch.stack(outputs, 1)
        # Unlike Qwen3.5, pinned Qwen4Exp chooses sigmoid output gating.
        normed = self._norm(core, prefix + 'ssm_norm.weight') * torch.sigmoid(z.reshape(*shape, 48, 128))
        for stage, tensor in ((S.QKV_PROJECTION, mixed), (S.GDN_Z_PROJECTION, z),
                              (S.GDN_ALPHA, alpha), (S.GDN_BETA, beta_projection),
                              (S.GDN_CONV1D_OUTPUT, conv), (S.GDN_DELTA_RULE_OUTPUT, core),
                              (S.GDN_NORM_GATE_OUTPUT, normed.flatten(-2))):
            self.capture_stage(stage, tensor, layer)
        return self._linear(normed.flatten(-2), prefix, 'ssm_out')

    def _qsa(self, value, prefix, cache, layer, start):
        """Score pooled blocks, select live token IDs and run sparse grouped-query attention."""
        batch, length, _ = value.shape
        positions = torch.arange(start, start + length).expand(batch, -1)
        iq = self._linear(value, prefix, 'indexer.q_proj').reshape(batch, length, 4, 128)
        iq = self._rope(self._norm(iq, prefix + 'indexer.q_norm.weight'), positions)
        raw = self._linear(value, prefix, 'indexer.k_proj')
        raw = torch.cat((cache['index_keys'], raw), 1) if 'index_keys' in cache else raw
        cache['index_keys'] = raw
        full_blocks = raw.shape[1] // 4
        pooled = raw[:, :full_blocks * 4].reshape(batch, full_blocks, 4, 128).mean(2)
        pooled = self._norm(pooled, prefix + 'indexer.k_norm.weight')
        pooled = self._rope(pooled, (torch.arange(full_blocks) * 4).expand(batch, -1))
        qgate = self._linear(value, prefix, 'attn_q').reshape(batch, length, 24, 512)
        q, gate = qgate.split(256, -1)
        q = self._rope(self._norm(q, prefix + 'attn_q_norm.weight'), positions)
        k = self._linear(value, prefix, 'attn_k').reshape(batch, length, 2, 256)
        k = self._rope(self._norm(k, prefix + 'attn_k_norm.weight'), positions)
        v = self._linear(value, prefix, 'attn_v').reshape(batch, length, 2, 256)
        k = torch.cat((cache['keys'], k), 1) if 'keys' in cache else k
        v = torch.cat((cache['values'], v), 1) if 'values' in cache else v
        cache['keys'], cache['values'] = k, v
        selected = torch.full((batch, length, 2051), -1, dtype=torch.int64)
        contexts = []
        for row in range(length):
            visible = start + row + 1
            blocks = visible // 4
            scores = torch.einsum('bhd,bnd->bhn', iq[:, row], pooled[:, :blocks])
            scores = scores.relu().sum(1) / math.sqrt(128)
            # Stable lower-ID ties match the production typed selection contract.
            chosen = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :min(512, blocks)]
            ids = (chosen[..., None] * 4 + torch.arange(4)).flatten(1)
            tail = torch.arange(blocks * 4, visible).expand(batch, -1)
            ids = torch.cat((ids, tail), -1)
            # HF creates a membership mask: core attention visits selected tokens
            # in chronological order, independently of score-ranked block order.
            ordered = ids.sort(-1).values
            selected[:, row, :ordered.shape[-1]] = ordered
            keys = k.gather(1, ordered[..., None, None].expand(-1, -1, 2, 256)).repeat_interleave(12, 2)
            vals = v.gather(1, ordered[..., None, None].expand(-1, -1, 2, 256)).repeat_interleave(12, 2)
            logits = torch.einsum('bhd,bnhd->bhn', q[:, row], keys) / 16
            probability = logits.softmax(-1)
            contexts.append(torch.einsum('bhn,bnhd->bhd', probability, vals))
        context = torch.stack(contexts, 1)
        gated = context * gate.sigmoid()
        for stage, tensor in ((S.QSA_INDEXER_QUERY, iq), (S.QSA_INDEXER_POOLED_KEYS, pooled),
                              (S.QSA_SELECTED_INDICES, selected.float()), (S.Q_PROJECTION, qgate.flatten(-2)),
                              (S.Q_ROPE, q), (S.K_ROPE, k[:, -length:]),
                              (S.ATTENTION_CONTEXT, context.flatten(-2)),
                              (S.ATTENTION_CONTEXT_GATED, gated.flatten(-2))):
            self.capture_stage(stage, tensor, layer)
        return self._linear(gated.flatten(-2), prefix, 'attn_output')

    def _moe(self, value, prefix, layer):
        """Decode selected experts and reproduce pinned HF's ascending-expert accumulation."""
        shape = value.shape
        flat = value.reshape(-1, 2560)
        probabilities = self._linear(flat, prefix, 'ffn_gate_inp').softmax(-1)
        ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :10]
        weights = probabilities.gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True)
        contributions = flat.new_zeros(flat.shape[0], 10, 2560)
        routed = torch.zeros_like(flat)
        for expert in ids.unique(sorted=True).tolist():
            rows, routes = torch.where(ids == expert)
            x = flat[rows]
            gate = self.weights.linear(x, prefix + 'ffn_gate_exps.weight', expert)
            up = self.weights.linear(x, prefix + 'ffn_up_exps.weight', expert)
            out = self.weights.linear(F.silu(gate) * up, prefix + 'ffn_down_exps.weight', expert)
            contributions[rows, routes] = out * weights[rows, routes, None]
            routed.index_add_(0, rows, contributions[rows, routes])
        shared = self._linear(F.silu(self._linear(flat, prefix, 'ffn_gate_shexp')) *
                              self._linear(flat, prefix, 'ffn_up_shexp'), prefix, 'ffn_down_shexp')
        shared_gate = torch.sigmoid(self._linear(flat, prefix, 'ffn_gate_inp_shexp')).reshape(-1, 1)
        gated_shared = shared * shared_gate
        for stage, tensor in ((S.MOE_ROUTER_OUTPUT, probabilities), (S.MOE_ROUTING_INDICES, ids.float()),
                              (S.MOE_ROUTING_WEIGHTS, weights), (S.MOE_ROUTE_CONTRIBUTIONS, contributions),
                              (S.MOE_EXPERT_OUTPUT, routed), (S.MOE_SHARED_EXPERT_OUTPUT, shared),
                              (S.MOE_SHARED_GATE_OUTPUT, gated_shared),
                              (S.MOE_COMBINED_OUTPUT, routed + gated_shared)):
            self.capture_stage(stage, tensor, layer)
        return (routed + gated_shared).reshape(shape)

    @torch.no_grad()
    def forward(self, token_ids, capture_stages=None, clear_snapshots=True,
                past_key_values=None, use_cache=False, **kwargs):
        """Execute main-model prefill/decode and publish canonical checkpoint arrays."""
        if not hasattr(self, 'weights'):
            raise RuntimeError('load_model() is required')
        tokens = torch.as_tensor(token_ids, dtype=torch.int64)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError('token IDs require a nonempty [batch, sequence] tensor')
        cache = past_key_values if past_key_values is not None else Qwen4ExpCache()
        if not isinstance(cache, Qwen4ExpCache):
            raise TypeError('past_key_values must be Qwen4ExpCache')
        if clear_snapshots:
            self.clear_snapshots()
        self.capture_stages = capture_stages
        previous = torch.full((tokens.shape[0], 2), 248044, dtype=torch.int64)
        if cache.tokens:
            previous = torch.tensor(cache.tokens, dtype=torch.int64)
        embedding = self.weights.rows('token_embd.weight', tokens)
        trunk = embedding.repeat(1, 1, 4)
        self.capture_stage(S.EMBEDDING, embedding)
        for layer in range(48):
            prefix = f'blk.{layer}.'
            state = cache.layers.setdefault(layer, {})
            if layer == 1:
                trunk = self._ple(trunk, tokens, previous, state, layer)
            read, normed = self._gr_read(trunk, prefix, 'hc_attn', layer)
            self.capture_stage(S.ATTENTION_NORM, read, layer)
            out = self._qsa(read, prefix, state, layer, cache.position) if self.ratios[layer] else self._gdn(read, prefix, state, layer)
            self.capture_stage(S.ATTENTION_OUTPUT, out, layer)
            trunk = self._gr_write(trunk, normed, out, prefix, 'hc_attn')
            self.capture_stage(S.GR_WRITE, trunk, layer)
            self.capture_stage(S.ATTENTION_RESIDUAL, trunk, layer)
            read, normed = self._gr_read(trunk, prefix, 'hc_ffn', layer)
            self.capture_stage(S.FFN_NORM, read, layer)
            trunk = self._gr_write(trunk, normed, self._moe(read, prefix, layer), prefix, 'hc_ffn')
            self.capture_stage(S.FFN_RESIDUAL, trunk, layer)
        head, _ = self._gr_read(trunk, '', 'output_hc', -1)
        self.capture_stage(S.FINAL_NORM, head)
        # Stream the terminal matrix in native source rows as well: no full
        # 2.5-GiB FP32 LM-head materialization is needed for a diagnostic pack.
        chunks = []
        for start in range(0, 248320, 1024):
            tile = self.weights.row_range('output.weight', start, min(start + 1024, 248320))
            chunks.append(F.linear(head, tile))
        logits = torch.cat(chunks, -1)
        self.capture_stage(S.LM_HEAD, logits)
        cache.tokens = torch.cat((previous, tokens), 1)[:, -2:].tolist()
        cache.position += tokens.shape[1]
        result = {'logits': logits.numpy(), 'hidden_states': head.numpy(), 'snapshots': self.get_snapshots()}
        if use_cache:
            result['past_key_values'] = cache
        return result


ModelRegistry.register('qwen4exp', Qwen4ExpReferenceModel)
