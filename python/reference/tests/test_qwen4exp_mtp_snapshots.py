"""Model-free source/trajectory/branch tests of the real QFN reference adapter.

Tiny arithmetic reuses the existing real Dense0 equation fixture. Published
source48 metadata is tested separately without opening model payloads. These
are reference-contract proofs, not native MTP or full-model certification.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from python.reference.generate_qwen4exp_pipeline_snapshots import parse_args, reference_fields
from python.reference.loaders.gguf_parser import GGUFTensorInfo, GGUFTensorType as T
from python.reference.qwen4exp_mtp import Dense0Geometry, Dense0KV, Qwen4ExpDense0Reference
from python.reference.qwen4exp import validate_qwen4exp_main_metadata
from python.reference.qwen4exp_mtp_snapshots import (
    Dense0MainTrajectory, SOURCE_LAYER, bind_dense0_sources, load_main_trajectory,
    publish_dense0_metadata, dense0_metadata_fields, authenticate_dense0_metadata, read_metadata, replay_dense0_trajectory,
    retain_dense0_prefix, write_dense0_snapshots,
)
from python.reference.snapshot_metadata import build_reference_identity, write_metadata_atomically
from python.reference.tests.test_qwen4exp_mtp import fixture


@pytest.fixture(scope='module', autouse=True)
def bounded_reference_workers():
    """Keep tiny equation fixtures inside the ordinary device-free test budget."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def trajectory():
    """Use the existing asymmetric real Dense0 weights, with distinct main rows."""
    model, hidden, _, _ = fixture(batch=1, length=3)
    decode = (hidden[:, :1].clone() * .7, hidden[:, 1:2].clone() * 1.1)
    return model, Dense0MainTrajectory((4, 3, 2), (1, 2, 3), hidden, decode)


def test_visible_token_zero_zero_hidden_seed_and_original_successor_alignment():
    model, main = trajectory()
    recorded = []
    original = model.mix_inputs
    def recording(tokens, hidden):
        recorded.append((tokens.clone(), hidden.clone()))
        return original(tokens, hidden)
    model.mix_inputs = recording
    list(replay_dense0_trajectory(model, main, 1))
    assert torch.equal(recorded[0][0], torch.tensor([[0]]))
    assert torch.count_nonzero(recorded[0][1]) == 0
    assert recorded[0][1].shape[-1] == model.g.trunk_width
    assert torch.equal(recorded[1][0], torch.tensor([[3, 2]]))
    assert torch.equal(recorded[1][1], main.prefill_hidden[:, :-1])
    assert torch.equal(recorded[2][0], torch.tensor([[1]]))
    assert torch.equal(recorded[2][1], main.prefill_hidden[:, -1:])
    assert torch.equal(recorded[3][1], main.decode_hidden[0])


def test_recursion_uses_original_wide_trunk_and_not_terminal_head():
    model, main = trajectory()
    frames = list(replay_dense0_trajectory(model, main, 3))
    first, second = frames[:2]
    assert first[:2] == (0, 0) and second[:2] == (0, 1)
    assert torch.equal(second[3], first[4]['hidden_trunk'])
    assert second[3].shape[-1] == model.g.trunk_width
    assert first[4]['head'].shape[-1] == model.g.width
    with pytest.raises(ValueError, match='wide hidden'):
        malformed = Dense0MainTrajectory(main.token_ids, main.decode_tokens,
            main.prefill_hidden[..., :model.g.width], main.decode_hidden)
        list(replay_dense0_trajectory(model, malformed, 1))
    assert Dense0Geometry().trunk_width == 10240
    published = Dense0MainTrajectory((7,), (1, 2), torch.zeros(1, 1, 10240), (torch.zeros(1, 1, 10240),))
    published.require(10240)
    with pytest.raises(ValueError, match='wide hidden'):
        published.require(2560)


@pytest.mark.parametrize('retained', [0, 1, 2, 3])
def test_explicit_accepted_prefix_retirement_and_continuation(retained):
    model, _ = trajectory()
    generator = torch.Generator().manual_seed(121)
    hidden = torch.randn(1, 4, model.g.trunk_width, generator=generator) * .1
    tokens = torch.tensor([[0, 1, 2, 3]])
    result = model.forward(tokens[:, :3], hidden[:, :3], torch.arange(3)[None])
    full = result['past_key_values']
    saved = full.keys.clone()
    prefix = retain_dense0_prefix(full, retained)
    assert prefix.keys.shape[1] == retained
    assert torch.equal(full.keys, saved)
    assert torch.equal(prefix.positions, torch.arange(retained)[None])
    continuation = model.forward(tokens[:, retained:retained+1], hidden[:, retained:retained+1],
        torch.tensor([[retained]]), cache=prefix)
    fresh = model.forward(tokens[:, :retained+1], hidden[:, :retained+1], torch.arange(retained+1)[None])
    torch.testing.assert_close(continuation['logits'], fresh['logits'][:, -1:], rtol=3e-5, atol=1e-6)
    # Returned accepted-prefix storage is independent of discarded speculative rows.
    if retained:
        prefix.keys.zero_()
        assert torch.equal(full.keys, saved)


def test_branch_continuation_and_reset_preserve_canonical_cache():
    model, main = trajectory()
    canonical = list(replay_dense0_trajectory(model, main, 3))
    branch = {1: {'condition_token': 5, 'draft_tokens': [6, 7]}}
    changed = list(replay_dense0_trajectory(model, main, 3, branch))
    assert len(changed) == 3
    assert all(frame[0] == 1 for frame in changed)
    assert changed[0][2] == '_CONDITION_5'
    assert changed[2][2] == '_CONDITION_5_BRANCH_6_7'
    # Earlier canonical depth0 history is identical; only the named branch differs.
    assert torch.equal(changed[0][3], main.decode_hidden[0])
    repeated = list(replay_dense0_trajectory(model, main, 3))
    for left, right in zip(canonical, repeated):
        assert left[:3] == right[:3]
        assert torch.equal(left[4]['logits'], right[4]['logits'])
        assert torch.equal(left[4]['past_key_values'].keys, right[4]['past_key_values'].keys)


def test_additive_branches_leave_main_canonical_predictor_and_metadata_bytes_unchanged(tmp_path):
    model, main = trajectory()
    original_main = b'original canonical main trajectory\n'
    (tmp_path/'metadata.txt').write_bytes(original_main)
    (tmp_path/'layer47_FFN_RESIDUAL.npy').write_bytes(b'canonical main checkpoint')
    write_dense0_snapshots(model, main, tmp_path, 3)
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    branches = [{'1': {'condition_token': 5, 'draft_tokens': [6]}},
                {'0': {'condition_token': None, 'draft_tokens': [4]}}]
    count = write_dense0_snapshots(model, main, tmp_path, 3, branches)
    assert count > 0
    for name, payload in original.items():
        assert (tmp_path/name).read_bytes() == payload
    assert (tmp_path/'decode_step1_CONDITION_5_MTP0_LM_HEAD.npy').is_file()
    assert (tmp_path/'decode_step1_CONDITION_5_BRANCH_6_MTP1_LM_HEAD.npy').is_file()
    assert (tmp_path/'decode_step0_BRANCH_4_MTP1_LM_HEAD.npy').is_file()
    assert not list(tmp_path.glob('*.tmp'))
    combined = np.load(tmp_path/'decode_step0_MTP0_MOE_COMBINED_OUTPUT.npy')
    expert = np.load(tmp_path/'decode_step0_MTP0_MOE_EXPERT_OUTPUT.npy')
    gated_shared = np.load(tmp_path/'decode_step0_MTP0_MOE_SHARED_GATE_OUTPUT.npy')
    np.testing.assert_array_equal(combined, expert + gated_shared)
    ids = np.load(tmp_path/'decode_step0_MTP0_MOE_ROUTING_INDICES.npy')
    assert ids.dtype == np.float32
    np.testing.assert_array_equal(ids, ids.astype(np.int64))


@pytest.mark.parametrize('branch', [
    {'2': {'condition_token': 1, 'draft_tokens': [2]}},
    {'0': {'condition_token': -1, 'draft_tokens': [2]}},
    {'0': {'condition_token': 99, 'draft_tokens': [2]}},
    {'0': {'condition_token': None, 'draft_tokens': [2, 3, 4]}},
])
def test_invalid_branch_rejects_before_predictor_payload_or_publication(branch, tmp_path):
    model, main = trajectory()
    def forbidden(*args, **kwargs):
        raise AssertionError('predictor payload was touched before branch validation')
    model.mix_inputs = forbidden
    with pytest.raises(ValueError):
        write_dense0_snapshots(model, main, tmp_path, 3, branch)
    assert not list(tmp_path.iterdir())


def metadata_pair():
    """Published geometry with header-only descriptors; no payload API exists."""
    class MetadataPath:
        def stat(self):
            return SimpleNamespace(st_size=1 << 50)
    meta = {'general.architecture': 'qwen4exp', 'qwen4exp.block_count': 48,
            'qwen4exp.nextn_predict_layers': 0, 'tokenizer.ggml.tokens': ['token'] * 248320,
            'tokenizer.ggml.model': 'test-tokenizer',
            'qwen4exp.attention.compress_ratios': [4 if (i+1)%4 == 0 else 0 for i in range(48)],
            'qwen4exp.ple.layers': [1], 'qwen4exp.ple.ngram_size': 3,
            'qwen4exp.ple.heads_per_ngram': 8, 'qwen4exp.ple.conv_kernel': 4,
            'qwen4exp.embedding_length_per_layer_input': 160, 'qwen4exp.ple.eos_token_id': 248044}
    values = {'context_length': 4096, 'embedding_length': 2560, 'attention.head_count': 24,
              'attention.head_count_kv': 2, 'attention.key_length': 256, 'attention.value_length': 256,
              'expert_count': 512, 'expert_used_count': 10, 'expert_feed_forward_length': 640,
              'expert_shared_feed_forward_length': 640, 'ssm.conv_kernel': 4, 'ssm.state_size': 128,
              'ssm.group_count': 16, 'ssm.time_step_rank': 48, 'ssm.inner_size': 6144,
              'full_attention_interval': 4, 'rope.dimension_count': 64, 'hyper_connection.count': 4,
              'hyper_connection.low_rank': 320, 'attention.indexer.head_count': 4,
              'attention.indexer.key_length': 128, 'attention.indexer.top_k': 2048,
              'rope.dimension_sections': [16, 24, 24], 'rope.freq_base': 1e7,
              'attention.layer_norm_rms_epsilon': 1e-6}
    meta.update({'qwen4exp.'+key: value for key, value in values.items()})
    owner = SimpleNamespace(metadata=meta, tensors=[], data_offset=128, file_path=MetadataPath())
    main = SimpleNamespace(parser=owner, directory={})
    side_meta = deepcopy(meta)
    side_meta.update({'qwen4exp.block_count': 49, 'qwen4exp.nextn_predict_layers': 1,
                      'qwen4exp.nextn_shared_target_tensors': True,
                      'qwen4exp.attention.compress_ratios': meta['qwen4exp.attention.compress_ratios']+[0]})
    side_owner = SimpleNamespace(metadata=side_meta, tensors=[], data_offset=128, file_path=MetadataPath())
    side = SimpleNamespace(parser=side_owner, directory={})
    def add(reader, name, shape, fmt=T.Q8_0):
        info = GGUFTensorInfo(name, list(shape), fmt, 0, reader.parser)
        reader.directory[name] = info
        reader.parser.tensors.append(info)
    for name in ('token_embd.weight', 'output.weight'):
        add(main, name, (248320, 2560))
    roles = {'attn_q': (12288, 2560), 'attn_k': (512, 2560), 'attn_v': (512, 2560),
             'attn_output': (2560, 6144), 'attn_q_norm': (256,), 'attn_k_norm': (256,),
             'ffn_gate_inp': (512, 2560), 'ffn_gate_inp_shexp': (2560,),
             'ffn_gate_exps': (512, 640, 2560), 'ffn_up_exps': (512, 640, 2560),
             'ffn_down_exps': (512, 2560, 640), 'ffn_gate_shexp': (640, 2560),
             'ffn_up_shexp': (640, 2560), 'ffn_down_shexp': (2560, 640),
             'nextn.eh_proj': (2560, 5120), 'nextn.enorm': (2560,), 'nextn.hnorm': (10240,),
             'nextn.hc_head_down': (320, 10240), 'nextn.hc_head_up': (10240, 320),
             'nextn.hc_head_norm': (10240,), 'indexer.q_proj': (512, 2560),
             'indexer.k_proj': (128, 2560), 'indexer.q_norm': (128,), 'indexer.k_norm': (128,)}
    for role in ('hc_attn', 'hc_ffn'):
        roles.update({role+'_down': (320, 10240), role+'_up': (10240, 320),
                      role+'_inject': (4, 10240), role+'_norm': (10240,)})
    fp32 = {'attn_q_norm', 'attn_k_norm', 'ffn_gate_inp', 'ffn_gate_inp_shexp',
            'nextn.enorm', 'nextn.hnorm', 'nextn.hc_head_norm', 'indexer.q_norm', 'indexer.k_norm',
            'hc_attn_norm', 'hc_ffn_norm'}
    for name, shape in roles.items():
        add(side, 'blk.48.'+name+'.weight', shape, T.F32 if name in fp32 else T.Q8_0)
    return main, side


def test_metadata_binding_retains_exact_sidecar48_and_shared_main_globals_without_payload():
    main, side = metadata_pair()
    actual = bind_dense0_sources(main, side)
    assert actual.weights is side and actual.target_weights is main
    assert actual.prefix == 'blk.48.' and SOURCE_LAYER == 48
    assert actual.g.trunk_width == 10240
    assert 'token_embd.weight' not in side.directory and 'output.weight' not in side.directory
    assert not hasattr(main, 'tensor') and not hasattr(side, 'tensor')


@pytest.mark.parametrize('fault', ['source', 'shared', 'norm', 'sparse', 'tokenizer', 'global', 'shape', 'gain', 'duplicate', 'foreign'])
def test_metadata_rejects_wrong_source_policy_global_owner_shape_or_format(fault):
    main, side = metadata_pair()
    role = 'blk.48.nextn.hnorm.weight'
    if fault == 'source': side.parser.metadata['qwen4exp.block_count'] = 48
    elif fault == 'shared': side.parser.metadata['qwen4exp.nextn_shared_target_tensors'] = 1
    elif fault == 'norm': side.parser.metadata['qwen4exp.hyper_connection.count'] = 1
    elif fault == 'sparse': side.parser.metadata['qwen4exp.attention.compress_ratios'][-1] = 4
    elif fault == 'tokenizer': side.parser.metadata['tokenizer.ggml.model'] = 'foreign'
    elif fault == 'global': side.directory['output.weight'] = main.directory['output.weight']
    elif fault == 'shape': side.directory[role].dimensions = [2560]
    elif fault == 'gain': side.directory[role].type = T.Q8_0
    elif fault == 'duplicate': side.parser.tensors.append(side.directory[role])
    elif fault == 'foreign': side.directory['blk.49.nextn.hnorm.weight'] = side.directory[role]
    with pytest.raises(ValueError): bind_dense0_sources(main, side)


def main_pack(tmp_path):
    """Author metadata/native-size fixtures only, without a GGUF forward or payload."""
    source = tmp_path/'main.gguf'
    source.write_bytes(b'original-source-descriptor')
    pack = tmp_path/'pack'
    pack.mkdir()
    tokens, prompt, steps = (7,), 'exact request', 1
    fields = build_reference_identity(source, prompt, tokens, steps).metadata_lines()
    write_metadata_atomically(pack/'metadata.txt', fields + reference_fields() + [
        'n_layers: 48', 'd_model: 2560', 'token_ids: 7', 'decode_tokens: 1,2'])
    for name in ('layer47_FFN_RESIDUAL.npy', 'decode_step0_layer47_FFN_RESIDUAL.npy'):
        np.save(pack/name, np.zeros((1, 1, 10240), dtype=np.float32))
    return source, pack, tokens, prompt


def test_canonical_trajectory_and_sidecar_metadata_bind_exact_source_and_width(tmp_path):
    source, pack, tokens, prompt = main_pack(tmp_path)
    actual = load_main_trajectory(pack, source, prompt, tokens, 1)
    assert actual.prefill_hidden.shape == (1, 1, 10240)
    predictor = tmp_path/'original-predictor.gguf'
    predictor.write_bytes(b'original-sidecar-metadata-fixture')
    original = (pack/'metadata.txt').read_bytes()
    # This pure projection authenticates source metadata only. It deliberately
    # does not invent a completed published-source MTP corpus in a tiny test.
    fields = dense0_metadata_fields(predictor, 15, 1)
    assert (pack/'metadata.txt').read_bytes() == original
    assert fields['qwen4exp_mtp_source_layer'] == '48'
    assert fields['qwen4exp_mtp_cache_layer'] == '0'
    assert fields['qwen4exp_mtp_recursive_width'] == '10240'
    assert fields['qwen4exp_mtp_predictor_filename'] == predictor.name
    assert fields['qwen4exp_mtp_predictor_size_bytes'] == str(predictor.stat().st_size)
    assert fields['qwen4exp_mtp_shifted_origin'] == 'visible_zero_hidden_token_zero'
    assert fields['mtp_sidecar_max_draft_depth'] == '15'


@pytest.mark.parametrize('fault', ['prompt', 'tokens', 'source', 'pin', 'dtype', 'width', 'duplicate'])
def test_trajectory_authentication_rejects_drift_before_sidecar_work(fault, tmp_path):
    source, pack, tokens, prompt = main_pack(tmp_path)
    if fault == 'prompt': prompt = 'foreign request'
    elif fault == 'tokens': tokens = (6,)
    elif fault == 'source': source.write_bytes(b'changed source size')
    elif fault == 'pin':
        text = (pack/'metadata.txt').read_text().replace('qwen4exp_reference_schema: 1', 'qwen4exp_reference_schema: 0')
        (pack/'metadata.txt').write_text(text)
    elif fault == 'dtype': np.save(pack/'layer47_FFN_RESIDUAL.npy', np.zeros((1, 1, 10240), dtype=np.float16))
    elif fault == 'width': np.save(pack/'layer47_FFN_RESIDUAL.npy', np.zeros((1, 1, 2560), dtype=np.float32))
    elif fault == 'duplicate':
        with (pack/'metadata.txt').open('a') as out: out.write('decode_steps: 1\n')
    with pytest.raises(ValueError): load_main_trajectory(pack, source, prompt, tokens, 1)


def test_thin_existing_generator_arguments_preserve_explicit_pair_and_branch_identity():
    args = parse_args(['--model', '/main.gguf', '--output', '/pack', '--shared-predictor', '/predictor.gguf',
        '--mtp-sidecar-snapshots', '--mtp-sidecar-only', '--mtp-max-draft-depth', '15',
        '--mtp-branch-overrides', '/branch.json'])
    assert args.model == '/main.gguf' and args.shared_predictor == Path('/predictor.gguf')
    assert args.mtp_sidecar_only and args.mtp_max_draft_depth == 15
    assert args.mtp_branch_overrides == Path('/branch.json')
    ordinary = parse_args(['--model', '/main.gguf', '--output', '/pack'])
    assert ordinary.shared_predictor is None and not ordinary.mtp_sidecar_snapshots
    for extra in (['--shared-predictor', '/predictor.gguf'], ['--mtp-sidecar-snapshots'],
                  ['--mtp-sidecar-only'], ['--mtp-max-draft-depth', '16']):
        with pytest.raises(SystemExit): parse_args(['--model', '/main.gguf', '--output', '/pack'] + extra)


def test_shared_main_header_validation_matches_normal_and_sidecar_only_contracts():
    main, side = metadata_pair()
    assert validate_qwen4exp_main_metadata(main.parser.metadata) == (48, 0)
    assert bind_dense0_sources(main, side).prefix == 'blk.48.'
    for key, bad in [('qwen4exp.ple.layers', [0]), ('qwen4exp.ssm.group_count', 8),
                     ('qwen4exp.attention.compress_ratios', [0]*48),
                     ('qwen4exp.embedding_length', 10240)]:
        original = main.parser.metadata[key]
        main.parser.metadata[key] = bad
        with pytest.raises(ValueError) as normal:
            validate_qwen4exp_main_metadata(main.parser.metadata)
        with pytest.raises(ValueError) as sidecar_only:
            bind_dense0_sources(main, side)
        assert str(normal.value) == str(sidecar_only.value)
        main.parser.metadata[key] = original


def test_later_malformed_branch_cannot_publish_an_earlier_valid_branch(tmp_path):
    model, main = trajectory()
    def forbidden(*args, **kwargs):
        raise AssertionError('branch frame was not completely validated')
    model.mix_inputs = forbidden
    branches = [{'0': {'condition_token': 4, 'draft_tokens': [5]}},
                {'2': {'condition_token': 4, 'draft_tokens': [5]}}]
    with pytest.raises(ValueError):
        write_dense0_snapshots(model, main, tmp_path, 3, branches)
    assert not list(tmp_path.iterdir())


def test_incomplete_pack_cannot_publish_depth_or_replace_main_metadata(tmp_path):
    _, pack, _, _ = main_pack(tmp_path)
    predictor = tmp_path/'predictor.gguf'
    predictor.write_bytes(b'original-predictor-descriptor')
    original = (pack/'metadata.txt').read_bytes()
    with pytest.raises(ValueError, match='incomplete'):
        publish_dense0_metadata(pack, predictor, 15)
    assert (pack/'metadata.txt').read_bytes() == original


@pytest.mark.parametrize('rows', [-1, 4, True])
def test_invalid_retained_prefix_rejects_without_cache_mutation(rows):
    cache = Dense0KV(torch.ones(1, 3, 2, 6), torch.ones(1, 3, 2, 6), torch.arange(3)[None])
    original = cache.keys.clone()
    with pytest.raises(ValueError): retain_dense0_prefix(cache, rows)
    assert torch.equal(cache.keys, original)
    malformed = Dense0KV(cache.keys, cache.values, torch.arange(17, 20)[None])
    with pytest.raises(ValueError, match='visible-seed'): retain_dense0_prefix(malformed, 1)


def canonical_mtp_metadata_fixture(tmp_path):
    """Build source metadata only; do not claim a generated predictor corpus."""
    _, pack, _, _ = main_pack(tmp_path)
    predictor = tmp_path/'canonical-predictor.gguf'
    predictor.write_bytes(b'original canonical predictor source identity')
    fields = dense0_metadata_fields(predictor, 15, 1)
    # The guard is metadata-only. No hand-authored MTP NPY is needed or trusted.
    lines = (pack/'metadata.txt').read_text().splitlines()
    write_metadata_atomically(pack/'metadata.txt', lines + [f'{k}: {v}' for k, v in fields.items()])
    return pack, predictor


def test_additive_guard_authenticates_same_original_predictor_metadata(tmp_path):
    pack, predictor = canonical_mtp_metadata_fixture(tmp_path)
    original = {p.name: p.read_bytes() for p in pack.iterdir()}
    observed = authenticate_dense0_metadata(pack, predictor, 3, 1)
    assert observed['qwen4exp_mtp_predictor_filename'] == predictor.name
    assert {p.name: p.read_bytes() for p in pack.iterdir()} == original


@pytest.mark.parametrize('field,value', [
    ('qwen4exp_mtp_predictor_filename', 'foreign-predictor.gguf'),
    ('qwen4exp_mtp_predictor_size_bytes', '1'),
    ('qwen4exp_mtp_reference_schema', '0'),
    ('qwen4exp_mtp_source_layer', '49'),
    ('qwen4exp_mtp_cache_layer', '1'),
    ('qwen4exp_mtp_attention', 'sparse'),
    ('qwen4exp_mtp_hidden_norm', 'flattened'),
    ('qwen4exp_mtp_shifted_origin', 'successor_pairs_only'),
    ('qwen4exp_mtp_recursive_width', '2560'),
    ('mtp_sidecar_max_draft_depth', '1'),
    ('qwen4exp_mtp_decode_steps', '0'),
])
def test_additive_source_schema_origin_or_capacity_drift_preserves_every_file(field, value, tmp_path):
    pack, predictor = canonical_mtp_metadata_fixture(tmp_path)
    lines = (pack/'metadata.txt').read_text().splitlines()
    write_metadata_atomically(pack/'metadata.txt', [f'{field}: {value}' if line.startswith(field+':') else line for line in lines])
    original = {p.name: p.read_bytes() for p in pack.iterdir()}
    with pytest.raises(ValueError): authenticate_dense0_metadata(pack, predictor, 3, 1)
    assert {p.name: p.read_bytes() for p in pack.iterdir()} == original


def test_actual_entrypoint_rejects_wrong_predictor_before_setup_or_branch_writes(tmp_path, monkeypatch):
    from python.reference import generate_qwen4exp_pipeline_snapshots as entry
    pack, _ = canonical_mtp_metadata_fixture(tmp_path)
    wrong = tmp_path/'other-predictor.gguf'
    wrong.write_bytes(b'compatible geometry alone is not canonical source ownership')
    request = tmp_path/'branch.json'
    request.write_text(json.dumps({'0': {'condition_token': 2, 'draft_tokens': [3]}}))
    original = {p.name: p.read_bytes() for p in pack.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError('wrong source reached model/tokenizer/payload setup')
    monkeypatch.setattr(entry, 'NativeSliceWeights', forbidden)
    monkeypatch.setattr(entry, 'create_reference_model', forbidden)
    monkeypatch.setattr(entry, 'load_main_trajectory', forbidden)
    with pytest.raises(ValueError, match='source identity'):
        entry.main(['--model', str(tmp_path/'main.gguf'), '--output', str(pack),
            '--shared-predictor', str(wrong), '--mtp-sidecar-snapshots', '--mtp-sidecar-only',
            '--mtp-max-draft-depth', '3', '--decode-steps', '1', '--mtp-branch-overrides', str(request)])
    assert {p.name: p.read_bytes() for p in pack.iterdir()} == original
