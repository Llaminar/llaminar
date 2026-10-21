"""CPU/FP32 arithmetic oracle for the published Qwen4Exp Dense0 sidecar.

This companion consumes the existing exact-GGUF ``NativeSliceWeights`` API.
It neither registers another main model nor changes its checkpoint/gold output.
The equations and native names follow danielhanchen/llama.cpp commit
6fcaa16f4b360649933a54d1f91ad40ed35c0e11, src/models/qwen4exp.cpp graph_mtp.
This is explicitly the shared-target, Dense0, PerBranch predictor convention;
it is not the flattened-hidden-norm HF/vLLM convention or a sparse predictor.

Callers provide the already-paired target/predictor readers and the *wide*
post-FFN target residual (existing layer47_FFN_RESIDUAL), not hidden_states.
Token/hidden alignment, artifact admission, sampling and accepted-cache policy
belong to the parity transaction driver and are deliberately not inferred here.
The immutable geometry defaults describe the published artifact; smaller exact
equations are permitted for independent model-free tests. KV is reference-owned
FP32 state, never a mirror of production device/cache state.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .qwen4exp import rms_gain


PUBLISHER_REVISION = '6fcaa16f4b360649933a54d1f91ad40ed35c0e11'
PUBLISHER_MODEL_SHA256 = '9f6ccddded3066cc442c853eeb555070bbcabf5fdcf5fa0e581b89820c04db2d'
PUBLISHER_CONVERTER_SHA256 = '6dab74eb519ea3b12787a28cc3fd5039ff65135ff14ba44321c9855edf1f8289'


@dataclass(frozen=True)
class Dense0Geometry:
    """Equation geometry, independent of artifact admission and cache capacity."""
    width: int = 2560
    branches: int = 4
    low_rank: int = 320
    query_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    experts: int = 512
    experts_used: int = 10
    expert_width: int = 640
    shared_width: int = 640

    def __post_init__(self):
        """Reject geometries that cannot express grouped causal attention/MoE."""
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError('Dense0 dimensions must be positive integers')
        if self.query_heads % self.kv_heads or self.rotary_dim % 2:
            raise ValueError('Dense0 requires integral GQA groups and even rotary width')
        if self.rotary_dim > self.head_dim or self.experts_used > self.experts:
            raise ValueError('Dense0 rotary/expert selection exceeds its source geometry')

    @property
    def trunk_width(self):
        """Return branch-major retained hidden width, distinct from compute width."""
        return self.width * self.branches


@dataclass(frozen=True)
class Dense0KV:
    """Append-only oracle KV with explicit absolute positions and rotated keys."""
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor


def _fp32(value, shape, label):
    """Validate reference placement/shape without an implicit cast or device move."""
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32:
        raise ValueError(f'{label}: requires CPU FP32')
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f'{label}: requires shape {tuple(shape)}, got {tuple(value.shape)}')
    return value


class Qwen4ExpDense0Reference:
    """Explicit shared-sidecar predictor; native globals keep their target owner.

    ``weights`` is the sidecar reader, ``target_weights`` the already-admitted
    main reader. Their tensor/slice/rows/row_range/linear methods are the existing
    ``NativeSliceWeights`` contract. No inline-global fallback, role alias,
    dormant indexer access, PLE, source autodiscovery or main graph is used.
    """

    def __init__(self, weights, target_weights, geometry=Dense0Geometry(), *,
                 source_layer=48, epsilon=1e-6, rope_base=1e7):
        """Retain explicit immutable readers/policy; do not read weight payloads."""
        if not isinstance(geometry, Dense0Geometry):
            raise TypeError('geometry must be Dense0Geometry')
        if type(source_layer) is not int or source_layer < 0:
            raise ValueError('source_layer must be a nonnegative exact source identity')
        if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError('epsilon and rope_base must be finite and positive')
        self.weights, self.target_weights = weights, target_weights
        self.g, self.prefix = geometry, f'blk.{source_layer}.'
        self.epsilon, self.rope_base = float(epsilon), float(rope_base)

    def _tensor(self, role, shape):
        """Resolve one exact predictor role and reject malformed final gains."""
        name = self.prefix + role + '.weight'
        return _fp32(self.weights.tensor(name), shape, name)

    def _linear(self, value, role, out_width, *, expert=None):
        """Validate native directory geometry before decoding any matrix slice."""
        name = self.prefix + role + '.weight'
        shape = (out_width, value.shape[-1])
        directory_shape = (self.g.experts, *shape) if expert is not None else shape
        if tuple(self.weights.directory[name].shape) != directory_shape:
            raise ValueError(f'{name}: requires native shape {directory_shape}')
        return self.weights.linear(value, name, expert)

    def _trunk(self, trunk):
        """Require batch/sequence axes and the full retained branch-major state."""
        if not isinstance(trunk, torch.Tensor) or trunk.ndim != 3:
            raise ValueError('hidden_trunk requires [batch, sequence, branches*width]')
        if not trunk.shape[0] or not trunk.shape[1]:
            raise ValueError('Dense0 requires nonempty batch and sequence')
        return _fp32(trunk, (*trunk.shape[:2], self.g.trunk_width), 'hidden_trunk')

    @torch.no_grad()
    def mix_inputs(self, token_ids, hidden_trunk):
        """Normalize per branch, repeat embeddings, then apply [W_e | W_h]."""
        self._trunk(hidden_trunk)
        b, t = hidden_trunk.shape[:2]
        if (not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.int64 or
                token_ids.device.type != 'cpu' or tuple(token_ids.shape) != (b, t)):
            raise ValueError('token_ids requires CPU int64 [batch, sequence]')
        embedding = self.target_weights.rows('token_embd.weight', token_ids)
        _fp32(embedding, (b, t, self.g.width), 'target token embedding')
        enorm = rms_gain(embedding, self._tensor('nextn.enorm', (self.g.width,)), self.epsilon)
        hnorm = rms_gain(hidden_trunk, self._tensor('nextn.hnorm', (self.g.trunk_width,)),
                         self.epsilon, self.g.width)
        repeated = enorm.unsqueeze(-2).expand(b, t, self.g.branches, self.g.width)
        # Converter concatenates fc_embedding and fc_hidden along matrix input
        # width. Each branch has its own h_norm; pooling here loses wide state.
        joined = torch.cat((repeated, hnorm.reshape(b, t, self.g.branches, self.g.width)), -1)
        projected = self._linear(joined, 'nextn.eh_proj', self.g.width).flatten(-2)
        return projected, {'mtp_tok_embd': embedding, 'mtp_hnorm': hnorm,
                           'mtp_enorm': repeated, 'mtp_concat': joined,
                           'mtp_eh_proj': projected}

    @torch.no_grad()
    def hc_read(self, trunk, role):
        """Read per-branch RMS gains through SiLU(down/branches), up, sigmoid."""
        self._trunk(trunk)
        if role not in ('hc_attn', 'hc_ffn', 'nextn.hc_head'):
            raise ValueError('unknown Dense0 HC read role')
        normed = rms_gain(trunk, self._tensor(role + '_norm', (self.g.trunk_width,)),
                          self.epsilon, self.g.width)
        low = F.silu(self._linear(normed, role + '_down', self.g.low_rank) / self.g.branches)
        gate = self._linear(low, role + '_up', self.g.trunk_width).sigmoid()
        mixed = (normed * gate).reshape(*trunk.shape[:2], self.g.branches, self.g.width).mean(-2)
        return mixed, normed, gate

    def hc_write(self, trunk, normed, output, role):
        """Add the same block output with independent 2*sigmoid(inject/HC) weights."""
        self._trunk(trunk)
        _fp32(normed, trunk.shape, 'HC normalized trunk')
        _fp32(output, (*trunk.shape[:2], self.g.width), 'HC block output')
        if role not in ('hc_attn', 'hc_ffn'):
            raise ValueError('terminal HC read has no injection role')
        scale = 2 * torch.sigmoid(self._linear(normed, role + '_inject', self.g.branches) / self.g.branches)
        return trunk + (scale.unsqueeze(-1) * output.unsqueeze(-2)).flatten(-2)

    def _rope(self, value, positions):
        """Use text mRoPE: all section coordinates equal the supplied absolute position."""
        r = self.g.rotary_dim
        inv = self.rope_base ** (-torch.arange(0, r, 2, dtype=torch.float32) / r)
        phase = positions.float().unsqueeze(-1) * inv
        phase = torch.cat((phase, phase), -1).unsqueeze(-2)
        first = value[..., :r]
        half = torch.cat((-first[..., r // 2:], first[..., :r // 2]), -1)
        return torch.cat((first * phase.cos() + half * phase.sin(), value[..., r:]), -1)

    @torch.no_grad()
    def dense_attention(self, read, positions, cache=None):
        """Run dense causal GQA over explicit prefix KV; no QSA/indexer operands."""
        if not isinstance(read, torch.Tensor) or read.ndim != 3:
            raise ValueError('attention input requires [batch, sequence, width]')
        b, t = read.shape[:2]
        _fp32(read, (b, t, self.g.width), 'attention input')
        if not b or not t:
            raise ValueError('Dense0 attention requires nonempty batch and sequence')
        if (not isinstance(positions, torch.Tensor) or positions.device.type != 'cpu' or
                positions.dtype != torch.int64 or tuple(positions.shape) != (b, t) or
                torch.any(positions < 0) or torch.any(positions[:, 1:] <= positions[:, :-1])):
            raise ValueError('positions require strictly increasing CPU int64 [batch, sequence]')
        qh, kh, d = self.g.query_heads, self.g.kv_heads, self.g.head_dim
        qgate = self._linear(read, 'attn_q', 2 * qh * d).reshape(b, t, qh, 2 * d)
        q, gate = qgate.split(d, -1)  # Per-head [query | gate], not two whole banks.
        q = rms_gain(q, self._tensor('attn_q_norm', (d,)), self.epsilon)
        k = self._linear(read, 'attn_k', kh * d).reshape(b, t, kh, d)
        k = rms_gain(k, self._tensor('attn_k_norm', (d,)), self.epsilon)
        v = self._linear(read, 'attn_v', kh * d).reshape(b, t, kh, d)
        q, k = self._rope(q, positions), self._rope(k, positions)
        keys, values, key_positions = k, v, positions
        if cache is not None:
            if not isinstance(cache, Dense0KV):
                raise TypeError('cache must be Dense0KV')
            n = cache.keys.shape[1]
            _fp32(cache.keys, (b, n, kh, d), 'prefix keys')
            _fp32(cache.values, (b, n, kh, d), 'prefix values')
            p = cache.positions
            if (p.device.type != 'cpu' or p.dtype != torch.int64 or tuple(p.shape) != (b, n) or
                    torch.any(p < 0) or torch.any(p[:, 1:] <= p[:, :-1]) or
                    (n and torch.any(positions[:, 0] <= p[:, -1]))):
                raise ValueError('prefix positions must be strictly earlier and ordered')
            keys, values = torch.cat((cache.keys, k), 1), torch.cat((cache.values, v), 1)
            key_positions = torch.cat((p, positions), 1)
        repeated_keys = keys.repeat_interleave(qh // kh, 2)
        repeated_values = values.repeat_interleave(qh // kh, 2)
        scores = torch.einsum('bthd,bnhd->bhtn', q, repeated_keys) / math.sqrt(d)
        causal = key_positions[:, None, :] <= positions[:, :, None]
        scores = scores.masked_fill(~causal[:, None], float('-inf'))
        probability = scores.softmax(-1)
        context = torch.einsum('bhtn,bnhd->bthd', probability, repeated_values)
        gated = context * gate.sigmoid()
        out = self._linear(gated.flatten(-2), 'attn_output', self.g.width)
        snapshots = {'mtp_Qcur_full': qgate.flatten(-2), 'mtp_Qcur': q, 'mtp_Kcur': k,
                     'mtp_Vcur': v, 'mtp_gate': gate.flatten(-2),
                     'mtp_attn_probability': probability, 'mtp_attn_pregate': context.flatten(-2),
                     'mtp_attn_gated': gated.flatten(-2), 'mtp_attn_out': out}
        return out, Dense0KV(keys.clone(), values.clone(), key_positions.clone()), snapshots

    @torch.no_grad()
    def expert_equation(self, read, expert):
        """Evaluate the canonical original-source SwiGLU expert on explicit FP32 rows.

        This is shared by the ordinary sidecar equation and its independent
        HF-input bank; it accepts no production output or normalization fit.
        """
        if (not isinstance(read, torch.Tensor) or read.ndim != 2 or not read.shape[0] or
                read.device.type != 'cpu' or read.dtype != torch.float32 or
                read.shape[1] != self.g.width or not torch.isfinite(read).all() or
                type(expert) is not int or not 0 <= expert < self.g.experts):
            raise ValueError('expert equation requires finite CPU FP32 rows and an exact original expert')
        gate = self._linear(read, 'ffn_gate_exps', self.g.expert_width, expert=expert)
        up = self._linear(read, 'ffn_up_exps', self.g.expert_width, expert=expert)
        return self._linear(F.silu(gate) * up, 'ffn_down_exps', self.g.width, expert=expert)

    @torch.no_grad()
    def moe(self, read):
        """Apply the existing oracle's selected-expert equations with exact native roles.

        Stable router ties and ascending-expert accumulation retain the existing
        main-reference convention. This numerical oracle does not assert ggml
        kernel reduction bytes or certify grouped production arithmetic.
        """
        shape = read.shape
        _fp32(read, (*shape[:2], self.g.width), 'FFN input')
        flat = read.reshape(-1, self.g.width)
        probability = self._linear(flat, 'ffn_gate_inp', self.g.experts).softmax(-1)
        ids = probability.argsort(dim=-1, descending=True, stable=True)[:, :self.g.experts_used]
        route_weights = probability.gather(1, ids)
        route_weights = route_weights / route_weights.sum(-1, keepdim=True)
        routed = torch.zeros_like(flat)
        for expert in ids.unique(sorted=True).tolist():
            rows, slots = torch.where(ids == expert)
            selected = flat[rows]
            out = self.expert_equation(selected, expert)
            routed.index_add_(0, rows, out * route_weights[rows, slots, None])
        shared = self._linear(F.silu(self._linear(flat, 'ffn_gate_shexp', self.g.shared_width)) *
                              self._linear(flat, 'ffn_up_shexp', self.g.shared_width),
                              'ffn_down_shexp', self.g.width)
        # Native shared gate is a vector [H], not a synthesized [1,H] matrix.
        gain = self._tensor('ffn_gate_inp_shexp', (self.g.width,))
        shared_ungated = shared
        shared = shared * F.linear(flat, gain).sigmoid().unsqueeze(-1)
        return (routed + shared).reshape(shape), {
            'mtp_router': probability.reshape(*shape[:2], -1),
            'mtp_expert_ids': ids.reshape(*shape[:2], -1),
            'mtp_expert_weights': route_weights.reshape(*shape[:2], -1),
            'mtp_routed': routed.reshape(shape), 'mtp_shared': shared.reshape(shape),
            'mtp_shared_ungated': shared_ungated.reshape(shape)}

    @torch.no_grad()
    def terminal(self, trunk, *, tile_rows=1024):
        """Use predictor HC head then the target's native output rows, without extra norm."""
        if type(tile_rows) is not int or tile_rows <= 0:
            raise ValueError('terminal tile_rows must be positive')
        head, normed, gate = self.hc_read(trunk, 'nextn.hc_head')
        shape = tuple(self.target_weights.directory['output.weight'].shape)
        if len(shape) != 2 or shape[1] != self.g.width or not shape[0]:
            raise ValueError('target output.weight has incompatible native geometry')
        tiles = []
        for start in range(0, shape[0], tile_rows):
            end = min(start + tile_rows, shape[0])
            weight = self.target_weights.row_range('output.weight', start, end)
            _fp32(weight, (end - start, self.g.width), 'target output tile')
            tiles.append(F.linear(head, weight))
        logits = torch.cat(tiles, -1)
        return logits, head, {'mtp_hc_head_norm': normed, 'mtp_hc_head_gate': gate,
                              'mtp_hc_head': head, 'result_output': logits}

    @torch.no_grad()
    def forward(self, token_ids, hidden_trunk, positions, *, cache=None):
        """Compose one predictor block, returning its pre-collapse trunk for recursion.

        All current rows are retained (the publisher's ungathered output case).
        Drivers may select returned rows explicitly; this API does not invent
        accepted-prefix coordinates or silently shift tokens/hidden/positions.
        The input cache remains immutable; the returned cache includes new KV.
        """
        trunk, snapshots = self.mix_inputs(token_ids, hidden_trunk)
        read, normed, _ = self.hc_read(trunk, 'hc_attn')
        snapshots.update(mtp_hc_attn_norm=normed, mtp_hc_attn_pre=read)
        out, next_cache, attn = self.dense_attention(read, positions, cache)
        snapshots.update(attn)
        trunk = self.hc_write(trunk, normed, out, 'hc_attn')
        snapshots['mtp_hc_attn_post'] = trunk
        read, normed, _ = self.hc_read(trunk, 'hc_ffn')
        snapshots.update(mtp_hc_ffn_norm=normed, mtp_hc_ffn_pre=read)
        out, ffn = self.moe(read)
        snapshots.update(ffn)
        snapshots['mtp_ffn_out'] = out
        trunk = self.hc_write(trunk, normed, out, 'hc_ffn')
        snapshots.update(mtp_hc_ffn_post=trunk, h_nextn=trunk)
        logits, head, terminal = self.terminal(trunk)
        snapshots.update(terminal)
        return {'logits': logits, 'hidden_trunk': trunk, 'head': head,
                'past_key_values': next_cache, 'snapshots': snapshots}
