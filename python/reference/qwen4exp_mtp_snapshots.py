"""Bind the existing Dense0 equation to canonical QFN paired-source snapshots.

This is a reference adapter, not an inference runner. Only the existing original
GGUF readers and authenticated CPU/FP32 main trajectory supply values. QFN's
visible token-zero/zero-hidden seed and wide PerBranch recursion are explicit;
no production cache/device state or normalized-head substitute is consumed.
"""
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch

from .loaders.gguf_parser import GGUFTensorType
from .loaders.dequantize import get_quantization_info
from .mtp_sidecar_reference import (
    mtp_branch_qualifier, mtp_sidecar_replay_depth,
    normalize_mtp_branch_override_batches, save_mtp_snapshot_atomic,
)
from .qwen4exp import (CHECKPOINT_REVISION, TRANSFORMERS_REVISION, CONVERTER_REVISION,
                       validate_qwen4exp_main_metadata)
from .qwen4exp_mtp import Dense0Geometry, Dense0KV, Qwen4ExpDense0Reference
from .snapshot_metadata import build_reference_identity, write_metadata_atomically

SCHEMA = 1
ORIGIN = 'visible_zero_hidden_token_zero'
SOURCE_LAYER = 48


def read_metadata(path):
    """Read scalar metadata without accepting duplicate authority fields."""
    fields = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        if key in fields:
            raise ValueError(f'duplicate reference metadata field {key}')
        fields[key] = value.strip()
    return fields


def _typed_equal(left, right):
    """Preserve bool/integer and nested tokenizer policy distinctions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _native_role(reader, name, shape, fp32=False):
    """Validate original native descriptor/extents without reading payload bytes."""
    directory = reader.directory
    info = directory.get(name)
    if info is None or tuple(info.shape) != tuple(shape):
        raise ValueError(f'{name}: paired source shape differs')
    if sum(item.name == name for item in reader.parser.tensors) != 1:
        raise ValueError(f'{name}: duplicate source role')
    if fp32 and info.type != GGUFTensorType.F32:
        raise ValueError(f'{name}: immutable gain/router requires FP32')
    fmt = get_quantization_info(info.type)
    elements = math.prod(shape)
    if any(type(width) is not int or width <= 0 for width in shape) or not fmt['block_size'] or elements % fmt['block_size']:
        raise ValueError(f'{name}: invalid native block geometry')
    owner = info._source_parser or reader.parser
    begin = owner.data_offset + info.offset
    size = elements // fmt['block_size'] * fmt['block_bytes']
    if info.offset < 0 or begin < owner.data_offset or begin + size > owner.file_path.stat().st_size:
        raise ValueError(f'{name}: truncated original native extent')


def bind_dense0_sources(main, predictor):
    """Authenticate explicit original directories under Dense0/PerBranch policy.

    The arithmetic geometry and consumed role inventory are the existing
    Dense0 equation's. The separate artifact contract mirrors the published
    source metadata: main48, source48 of total49, original target globals and
    dormant exported indexer roles. Sparse predictors are intentionally rejected.
    """
    target, side = main.parser.metadata, predictor.parser.metadata
    validate_qwen4exp_main_metadata(target)
    if target.get('general.architecture') != 'qwen4exp' or side.get('general.architecture') != 'qwen4exp':
        raise ValueError('Dense0 reference requires paired qwen4exp directories')
    if target.get('qwen4exp.block_count') != 48 or target.get('qwen4exp.nextn_predict_layers', 0) != 0:
        raise ValueError('Dense0 reference requires 48 original main blocks and no inline predictor')
    if side.get('qwen4exp.nextn_shared_target_tensors') is not True:
        raise ValueError('Dense0 reference requires explicit shared target ownership')
    if side.get('qwen4exp.block_count') != 49 or side.get('qwen4exp.nextn_predict_layers') != 1:
        raise ValueError('Dense0 reference requires exactly original source 48')
    scalar = ('context_length', 'embedding_length', 'attention.head_count', 'attention.head_count_kv',
              'attention.key_length', 'attention.value_length', 'expert_count', 'expert_used_count',
              'expert_feed_forward_length', 'expert_shared_feed_forward_length', 'ssm.conv_kernel',
              'ssm.state_size', 'ssm.group_count', 'ssm.time_step_rank', 'ssm.inner_size',
              'full_attention_interval', 'rope.dimension_count', 'hyper_connection.count',
              'hyper_connection.low_rank', 'attention.indexer.head_count', 'attention.indexer.key_length',
              'attention.indexer.top_k', 'rope.dimension_sections', 'rope.freq_base',
              'attention.layer_norm_rms_epsilon')
    for suffix in scalar:
        key = 'qwen4exp.' + suffix
        if key not in target or not _typed_equal(target[key], side.get(key)):
            raise ValueError(f'{key}: main/predictor policy differs')
    ratios = target.get('qwen4exp.attention.compress_ratios')
    if not isinstance(ratios, list) or len(ratios) != 48 or side.get('qwen4exp.attention.compress_ratios') != ratios + [0]:
        raise ValueError('Dense0 requires unchanged48 main compression entries +zero predictor entry')
    def lexical(key):
        return key.startswith('tokenizer.') and not (key in ('tokenizer.chat_template', 'tokenizer.chat_templates') or key.startswith('tokenizer.chat_template.'))
    for key in set(filter(lexical, target)) | set(filter(lexical, side)):
        if key not in target or key not in side or not _typed_equal(target[key], side[key]):
            raise ValueError(f'{key}: shared tokenizer policy differs')
    g = Dense0Geometry()
    fixed = {'embedding_length': g.width, 'hyper_connection.count': g.branches,
             'hyper_connection.low_rank': g.low_rank, 'attention.head_count': g.query_heads,
             'attention.head_count_kv': g.kv_heads, 'attention.key_length': g.head_dim,
             'attention.value_length': g.head_dim, 'rope.dimension_count': g.rotary_dim,
             'expert_count': g.experts, 'expert_used_count': g.experts_used,
             'expert_feed_forward_length': g.expert_width, 'expert_shared_feed_forward_length': g.shared_width}
    for suffix, value in fixed.items():
        if type(target['qwen4exp.' + suffix]) is not int or target['qwen4exp.' + suffix] != value:
            raise ValueError(f'qwen4exp.{suffix}: published Dense0 geometry differs')
    tokens = target.get('tokenizer.ggml.tokens')
    if not isinstance(tokens, list) or len(tokens) != 248320 or any(type(token) is not str for token in tokens):
        raise ValueError('target tokenizer must own its original vocabulary')
    for name in ('token_embd.weight', 'output.weight'):
        _native_role(main, name, (len(tokens), g.width))
    h, trunk, rank, q, kv = g.width, g.trunk_width, g.low_rank, g.query_heads*g.head_dim, g.kv_heads*g.head_dim
    roles = {'attn_q': ((2*q, h), False), 'attn_k': ((kv, h), False), 'attn_v': ((kv, h), False),
             'attn_output': ((h, q), False), 'attn_q_norm': ((g.head_dim,), True), 'attn_k_norm': ((g.head_dim,), True),
             'ffn_gate_inp': ((g.experts, h), True), 'ffn_gate_inp_shexp': ((h,), True),
             'ffn_gate_exps': ((g.experts, g.expert_width, h), False),
             'ffn_up_exps': ((g.experts, g.expert_width, h), False),
             'ffn_down_exps': ((g.experts, h, g.expert_width), False),
             'ffn_gate_shexp': ((g.shared_width, h), False), 'ffn_up_shexp': ((g.shared_width, h), False),
             'ffn_down_shexp': ((h, g.shared_width), False),
             'nextn.eh_proj': ((h, 2*h), False), 'nextn.enorm': ((h,), True), 'nextn.hnorm': ((trunk,), True),
             'nextn.hc_head_down': ((rank, trunk), False), 'nextn.hc_head_up': ((trunk, rank), False),
             'nextn.hc_head_norm': ((trunk,), True)}
    for role in ('hc_attn', 'hc_ffn'):
        roles.update({role+'_down': ((rank, trunk), False), role+'_up': ((trunk, rank), False),
                      role+'_inject': ((g.branches, trunk), False), role+'_norm': ((trunk,), True)})
    ih, dim = target['qwen4exp.attention.indexer.head_count'], target['qwen4exp.attention.indexer.key_length']
    roles.update({'indexer.q_proj': ((ih*dim, h), False), 'indexer.k_proj': ((dim, h), False),
                  'indexer.q_norm': ((dim,), True), 'indexer.k_norm': ((dim,), True)})
    expected = {'blk.48.' + role + '.weight' for role in roles}
    if set(predictor.directory) != expected or len(predictor.parser.tensors) != len(expected):
        raise ValueError('predictor contains absent/foreign/duplicate source roles')
    for role, (shape, fp32) in roles.items():
        name = 'blk.48.' + role + '.weight'
        if name in main.directory:
            raise ValueError(f'{name}: main/predictor source collision')
        _native_role(predictor, name, shape, fp32)
    return Qwen4ExpDense0Reference(predictor, main, g, source_layer=SOURCE_LAYER,
        epsilon=target['qwen4exp.attention.layer_norm_rms_epsilon'], rope_base=target['qwen4exp.rope.freq_base'])


def retain_dense0_prefix(cache, rows):
    """Retain an explicit accepted main-prefix frontier, dropping speculative KV.

    QFN's visible seed makes shifted row count equal committed main row count.
    This reference-only operation never derives acceptance from its own logits.
    The canonical/additive driver advances one committed main row per step;
    other tests may author zero/partial/full accepted prefix lengths explicitly.
    """
    if not isinstance(cache, Dense0KV) or type(rows) is not int:
        raise ValueError('Dense0 retained prefix requires an exact cache and row count')
    if (not isinstance(cache.keys, torch.Tensor) or not isinstance(cache.values, torch.Tensor) or
            cache.keys.ndim != 4 or cache.keys.device.type != 'cpu' or cache.keys.dtype != torch.float32 or
            cache.values.device.type != 'cpu' or cache.values.dtype != torch.float32):
        raise ValueError('Dense0 retained KV requires original CPU FP32 keys/values')
    if not 0 <= rows <= cache.keys.shape[1]:
        raise ValueError('Dense0 retained prefix is outside speculative cache')
    n = cache.keys.shape[1]
    if cache.values.shape != cache.keys.shape or cache.positions.shape != cache.keys.shape[:2]:
        raise ValueError('Dense0 KV prefix has incompatible native coordinates')
    if cache.positions.dtype != torch.int64 or cache.positions.device.type != 'cpu' or not torch.equal(
            cache.positions, torch.arange(n, dtype=torch.int64).expand(cache.positions.shape[0], -1)):
        raise ValueError('Dense0 KV must retain contiguous visible-seed positions from zero')
    return Dense0KV(cache.keys[:, :rows].clone(), cache.values[:, :rows].clone(), cache.positions[:, :rows].clone())


@dataclass(frozen=True)
class Dense0MainTrajectory:
    """Immutable independent main rows and canonical conditions, never native state."""
    token_ids: tuple[int, ...]
    decode_tokens: tuple[int, ...]
    prefill_hidden: torch.Tensor
    decode_hidden: tuple[torch.Tensor, ...]

    def require(self, width):
        """Check every row before any predictor arithmetic or snapshot publication."""
        if not self.token_ids or not self.decode_hidden or len(self.decode_tokens) != len(self.decode_hidden) + 1:
            raise ValueError('Dense0 trajectory has incomplete canonical conditions')
        for token in self.token_ids + self.decode_tokens:
            if type(token) is not int or token < 0:
                raise ValueError('Dense0 trajectory tokens must be nonnegative integers')
        for value, rows in [(self.prefill_hidden, len(self.token_ids))] + [(h, 1) for h in self.decode_hidden]:
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32 or tuple(value.shape) != (1, rows, width):
                raise ValueError(f'Dense0 trajectory requires original wide hidden [{1},rows,{width}]')
            if not torch.isfinite(value).all():
                raise ValueError('Dense0 trajectory hidden must be finite')


def _checked_replay_inputs(model, trajectory, maximum_depth, branch):
    """Validate a complete independent replay request before payload or publication."""
    trajectory.require(model.g.trunk_width)
    if type(maximum_depth) is not int or not 1 <= maximum_depth <= 15:
        raise ValueError('Dense0 snapshot depth must be in[1,15]')
    overrides = {} if branch is None else normalize_mtp_branch_override_batches(branch)[0]
    if branch is not None and len(normalize_mtp_branch_override_batches(branch)) != 1:
        raise ValueError('each additive pass owns one independent branch')
    for step, values in overrides.items():
        if type(step) is not int or not 0 <= step < len(trajectory.decode_hidden) or len(values['draft_tokens']) >= maximum_depth:
            raise ValueError('Dense0 branch step/depth is outside trajectory')
    vocabulary = model.target_weights.directory['token_embd.weight'].shape[0]
    for token in trajectory.token_ids + trajectory.decode_tokens:
        if token >= vocabulary:
            raise ValueError('Dense0 trajectory token exceeds original target vocabulary')
    for values in overrides.values():
        if any(token >= vocabulary for token in values['draft_tokens']) or (values['condition_token'] is not None and values['condition_token'] >= vocabulary):
            raise ValueError('Dense0 branch token exceeds original target vocabulary')
    return overrides


def replay_dense0_trajectory(model, trajectory, maximum_depth, branch=None):
    """Yield canonical/additive checkpoints using the existing Dense0 equation.

    Every invocation constructs an independent reference-owned shifted cache.
    Only depth0 joins the next canonical main step; deeper speculative suffixes
    never leak into a later branch. Recursive hidden is the full returned trunk,
    explicitly distinct from the compute-width terminal head.
    """
    overrides = _checked_replay_inputs(model, trajectory, maximum_depth, branch)
    # Visible seed: token0 with ZERO wide hidden at position0. Prompt token0
    # is deliberately NOT substituted for this publisher-defined bootstrap.
    hidden = trajectory.prefill_hidden
    seed_hidden = torch.zeros((1, 1, model.g.trunk_width), dtype=torch.float32)
    seed_tokens = torch.zeros((1, 1), dtype=torch.int64)
    projected, _ = model.mix_inputs(seed_tokens, seed_hidden)
    read, _, _ = model.hc_read(projected, 'hc_attn')
    _, cache, _ = model.dense_attention(read, torch.zeros((1, 1), dtype=torch.int64))
    if len(trajectory.token_ids) > 1:
        tokens = torch.tensor([trajectory.token_ids[1:]], dtype=torch.int64)
        projected, _ = model.mix_inputs(tokens, hidden[:, :-1])
        read, _, _ = model.hc_read(projected, 'hc_attn')
        positions = torch.arange(1, len(trajectory.token_ids), dtype=torch.int64)[None]
        _, cache, _ = model.dense_attention(read, positions, cache)
    for step, canonical_condition in enumerate(trajectory.decode_tokens[:-1]):
        previous = cache.keys.shape[1]
        override = overrides.get(step)
        condition = canonical_condition if override is None or override['condition_token'] is None else override['condition_token']
        trunk = hidden[:, -1:]
        consumed = []
        for depth in range(mtp_sidecar_replay_depth(step, maximum_depth, overrides)):
            if depth and override is not None and depth-1 < len(override['draft_tokens']):
                condition = override['draft_tokens'][depth-1]
            if depth:
                consumed.append(condition)
            positions = torch.tensor([[len(trajectory.token_ids) + step + depth]], dtype=torch.int64)
            result = model.forward(torch.tensor([[condition]], dtype=torch.int64), trunk, positions, cache=cache)
            qualifier = mtp_branch_qualifier(override, consumed)
            if not overrides or qualifier:
                yield step, depth, qualifier, trunk.clone(), result
            cache = result['past_key_values']
            trunk = result['hidden_trunk']  # Wide4H, NEVER result['head'].
            if tuple(trunk.shape) != (1, 1, model.g.trunk_width):
                raise ValueError('Dense0 recursive producer lost its original wide trunk')
            condition = int(result['logits'][0, -1].argmax())
        cache = retain_dense0_prefix(cache, previous + 1)
        if override is not None:
            break
        hidden = trajectory.decode_hidden[step]


def _hidden(path, rows):
    """Load one canonical wide main checkpoint, preserving exact FP32 shape."""
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (1, rows, 10240) or not np.isfinite(value).all():
        raise ValueError(f'{path}: requires finite original FP32 wide main trajectory')
    return torch.from_numpy(value)


def load_main_trajectory(pack, model_path, prompt, token_ids, decode_steps):
    """Authenticate the original main pack before opening any predictor payload."""
    pack = Path(pack)
    fields = read_metadata(pack/'metadata.txt')
    available = int(fields.get('decode_steps', '-1'))
    if type(decode_steps) is not int or not 0 < decode_steps <= available:
        raise ValueError('Dense0 snapshot request exceeds canonical decode trajectory')
    identity = build_reference_identity(model_path, prompt, token_ids, available)
    required = dict(line.split(': ', 1) for line in identity.metadata_lines())
    required.update({'n_layers': '48', 'd_model': '2560', 'qwen4exp_reference_schema': '1',
                     'qwen4exp_checkpoint_revision': CHECKPOINT_REVISION,
                     'qwen4exp_transformers_revision': TRANSFORMERS_REVISION,
                     'qwen4exp_converter_revision': CONVERTER_REVISION,
                     'qwen4exp_residual_width': '10240', 'qwen4exp_terminal_head_width': '2560',
                     'qwen4exp_integer_checkpoint_dtype': 'int64', 'qwen4exp_gdn_output_gate': 'sigmoid'})
    for key, value in required.items():
        if fields.get(key) != value:
            raise ValueError(f'Dense0 trajectory identity differs: {key}')
    observed = tuple(int(v) for v in fields.get('token_ids', '').split(','))
    if observed != tuple(token_ids):
        raise ValueError('Dense0 canonical prompt token identity differs')
    conditions = tuple(int(v) for v in fields.get('decode_tokens', '').split(','))
    if len(conditions) != available + 1:
        raise ValueError('Dense0 canonical conditions are incomplete')
    return Dense0MainTrajectory(observed, conditions[:decode_steps+1],
        _hidden(pack/'layer47_FFN_RESIDUAL.npy', len(observed)),
        tuple(_hidden(pack/f'decode_step{step}_layer47_FFN_RESIDUAL.npy', 1) for step in range(decode_steps)))


# This explicit observable cut names actual generic predictor boundaries. Extra
# attention probabilities/HC gate diagnostics remain reference-equation tests.
SNAPSHOTS = {'mtp_tok_embd': 'EMBEDDING', 'mtp_hnorm': 'NORM_HIDDEN',
             'mtp_concat': 'CONCAT', 'mtp_eh_proj': 'FC',
             'mtp_hc_attn_pre': 'ATTENTION_NORM', 'mtp_attn_out': 'ATTENTION_OUTPUT',
             'mtp_hc_attn_post': 'ATTENTION_RESIDUAL', 'mtp_hc_ffn_pre': 'FFN_NORM',
             'mtp_router': 'MOE_ROUTER_OUTPUT', 'mtp_expert_ids': 'MOE_ROUTING_INDICES',
             'mtp_expert_weights': 'MOE_ROUTING_WEIGHTS', 'mtp_routed': 'MOE_EXPERT_OUTPUT',
             'mtp_shared_ungated': 'MOE_SHARED_EXPERT_OUTPUT', 'mtp_shared': 'MOE_SHARED_GATE_OUTPUT',
             'mtp_ffn_out': 'MOE_COMBINED_OUTPUT', 'mtp_hc_ffn_post': 'FFN_RESIDUAL',
             'mtp_hc_head': 'FINAL_NORM', 'result_output': 'LM_HEAD'}


def write_dense0_snapshots(model, trajectory, pack, maximum_depth, overrides=None):
    """Atomically publish only canonical or correctly branch-qualified checkpoints."""
    pack = Path(pack)
    batches = [None] if overrides is None else normalize_mtp_branch_override_batches(overrides)
    # A later malformed branch cannot leave an earlier branch's additive files.
    for branch in batches:
        _checked_replay_inputs(model, trajectory, maximum_depth, branch)
    total = 0
    for branch in batches:
        for step, depth, qualifier, hidden, result in replay_dense0_trajectory(model, trajectory, maximum_depth, branch):
            values = {SNAPSHOTS[key]: result['snapshots'][key] for key in SNAPSHOTS}
            values['TERMINAL_HIDDEN_ROW_SELECT'] = hidden
            for name, value in values.items():
                # The canonical MTP routing comparator consumes exact FP32
                # expert IDs (all <512), just like the existing main oracle.
                # PLE/QSA integer checkpoints are outside this Dense0 block.
                if name == 'MOE_ROUTING_INDICES':
                    if value.dtype != torch.int64 or torch.any(value < 0) or torch.any(value >= model.g.experts):
                        raise ValueError('Dense0 routing IDs lost their original exact expert identity')
                    value = value.to(torch.float32)
                payload = value.detach().numpy().copy()
                if payload.dtype not in (np.dtype('float32'), np.dtype('int64')) or not np.isfinite(payload).all():
                    raise ValueError(f'Dense0 {name}: requires finite original FP32/int64 snapshot')
                if payload.ndim == 4:
                    payload = payload.reshape(*payload.shape[:2], -1)
                save_mtp_snapshot_atomic(pack/f'decode_step{step}{qualifier}_MTP{depth}_{name}.npy', payload)
                total += 1
    return total


def dense0_metadata_fields(predictor_path, maximum_depth, decode_steps):
    """Project exact source/schema fields without claiming generated snapshots."""
    predictor = Path(predictor_path).resolve(strict=True)
    if (not predictor.is_file() or type(maximum_depth) is not int or not 1 <= maximum_depth <= 15 or
            type(decode_steps) is not int or decode_steps <= 0 or predictor.stat().st_size <= 0):
        raise ValueError('Dense0 metadata requires exact positive source/depth/trajectory bounds')
    return {'qwen4exp_mtp_reference_schema': str(SCHEMA), 'qwen4exp_mtp_source_layer': '48',
            'qwen4exp_mtp_attention': 'dense0', 'qwen4exp_mtp_hidden_norm': 'per_branch',
            'qwen4exp_mtp_cache_layer': '0',
            'qwen4exp_mtp_shifted_origin': ORIGIN, 'qwen4exp_mtp_recursive_width': '10240',
            'qwen4exp_mtp_decode_steps': str(decode_steps),
            'qwen4exp_mtp_predictor_filename': predictor.name,
            'qwen4exp_mtp_predictor_size_bytes': str(predictor.stat().st_size),
            'mtp_sidecar_max_draft_depth': str(maximum_depth)}


def authenticate_dense0_metadata(pack, predictor_path, required_depth, decode_steps):
    """Bind additive work to the predictor that owns existing canonical MTP files.

    Compatibility of two GGUF headers is insufficient: existing canonical
    sidecar files retain their own exact original-source/schema/origin identity.
    Authenticate that identity before any branch output, preserving the shared
    framework's metadata-only filename/native-size policy without payload hashes.
    """
    fields = read_metadata(Path(pack)/'metadata.txt')
    try:
        depth = int(fields['mtp_sidecar_max_draft_depth'])
        steps = int(fields['qwen4exp_mtp_decode_steps'])
    except (KeyError, ValueError) as error:
        raise ValueError('additive QFN MTP pack has no valid canonical capacity') from error
    if (type(required_depth) is not int or not 1 <= required_depth <= depth or
            type(decode_steps) is not int or not 0 < decode_steps <= steps):
        raise ValueError('additive QFN MTP request exceeds existing canonical capacity')
    expected = dense0_metadata_fields(predictor_path, depth, steps)
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(f'additive QFN MTP source identity differs: {key}')
    return fields


def publish_dense0_metadata(pack, predictor_path, maximum_depth, decode_steps=None):
    """Publish source/schema/origin only after the canonical NPY inventory exists.

    This commit point is called only after the paired-source generator returns.
    Tiny equation tests do not fabricate a complete published-source pack.
    """
    pack = Path(pack)
    original_fields = read_metadata(pack/'metadata.txt')
    decode_steps = int(original_fields.get('decode_steps', '-1')) if decode_steps is None else decode_steps
    fields = dense0_metadata_fields(predictor_path, maximum_depth, decode_steps)
    if decode_steps > int(original_fields.get('decode_steps', '-1')):
        raise ValueError('Dense0 metadata trajectory exceeds the canonical main pack')
    for step in range(decode_steps):
        for depth in range(maximum_depth):
            for suffix in [*SNAPSHOTS.values(), 'TERMINAL_HIDDEN_ROW_SELECT']:
                if not (pack/f'decode_step{step}_MTP{depth}_{suffix}.npy').is_file():
                    raise ValueError(f'Dense0 canonical pack is incomplete at step{step}/depth{depth}/{suffix}')
    original = (pack/'metadata.txt').read_text(encoding='utf-8').splitlines()
    retained = [line for line in original if line.split(':', 1)[0] not in fields]
    write_metadata_atomically(pack/'metadata.txt', retained + [f'{key}: {value}' for key, value in fields.items()])
