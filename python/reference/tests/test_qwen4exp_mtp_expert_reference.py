"""Tiny real Dense0 expert equations and exact canonical bank/source contracts.

No production activations, external GGUF payloads, models or accelerators are
loaded. The existing real Dense0 equation fixture supplies explicit tiny F32
source tensors; metadata-only file identities never pretend to be valid GGUFs.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from python.reference.generate_moe_route_conditioned_reference import authenticate_canonical_sum
from python.reference.generate_qwen4exp_mtp_expert_reference import (
    authenticate_pack, expert_bank, load_operands, prefix_identity,
)
from python.reference.qwen4exp import CHECKPOINT_REVISION, TRANSFORMERS_REVISION, CONVERTER_REVISION
from python.reference.qwen4exp_mtp_snapshots import dense0_metadata_fields
from python.reference.snapshot_metadata import build_reference_identity, write_metadata_atomically
from python.reference.tests.test_qwen4exp_mtp import fixture

PREFIX = 'decode_step0_MTP0_'


@pytest.fixture(scope='module', autouse=True)
def bounded_workers():
    """Keep tiny CPU equations inside the canonical device-free budget."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def equation_case():
    """Use the canonical Dense0 equation, not a substitute expert implementation."""
    model, hidden, tokens, positions = fixture(batch=1, length=1)
    result = model.forward(tokens, hidden, positions)
    return model, hidden, result


def write_case(directory, model, hidden, result):
    """Publish only the six exact physical singleton planes consumed by the bank."""
    names = {'FFN_NORM': 'mtp_hc_ffn_pre', 'MOE_ROUTING_INDICES': 'mtp_expert_ids',
             'MOE_ROUTING_WEIGHTS': 'mtp_expert_weights', 'MOE_ROUTER_OUTPUT': 'mtp_router',
             'MOE_EXPERT_OUTPUT': 'mtp_routed'}
    for name, source in names.items():
        np.save(Path(directory)/f'{PREFIX}{name}.npy', result['snapshots'][source].float().numpy())
    np.save(Path(directory)/f'{PREFIX}TERMINAL_HIDDEN_ROW_SELECT.npy', hidden.numpy())


def test_real_swiglu_bank_matches_independent_scalar_equation_and_canonical_sum(tmp_path):
    model, hidden, result = equation_case()
    write_case(tmp_path, model, hidden, result)
    logical, physical = load_operands(tmp_path, PREFIX, model.g)
    ids = list(range(model.g.experts))
    model.weights.calls.clear()
    model.target_weights.calls.clear()
    bank = expert_bank(model, logical['FFN_NORM'], ids)
    x = logical['FFN_NORM']
    for expert in ids:
        weights = model.weights.data
        gate = x @ weights['blk.48.ffn_gate_exps.weight'][expert].numpy().T
        up = x @ weights['blk.48.ffn_up_exps.weight'][expert].numpy().T
        expected = ((gate/(1+np.exp(-gate))) * up) @ weights['blk.48.ffn_down_exps.weight'][expert].numpy().T
        np.testing.assert_allclose(bank[expert], expected, rtol=2e-5, atol=1e-7)
    authenticate_canonical_sum(bank, ids, logical)
    assert bank.dtype == np.float32 and bank.shape == (model.g.experts, 1, model.g.width)
    assert physical['FFN_NORM'].shape == (1, 1, model.g.width)
    assert physical['TERMINAL_HIDDEN_ROW_SELECT'].shape == (1, 1, model.g.trunk_width)
    assert not model.target_weights.calls
    assert {name for kind, name, expert in model.weights.calls} == {
        'blk.48.ffn_gate_exps.weight', 'blk.48.ffn_up_exps.weight', 'blk.48.ffn_down_exps.weight'}
    assert all(kind == 'linear' for kind, name, expert in model.weights.calls)
    corrupt = bank.copy() * 2
    with pytest.raises(ValueError, match='does not reproduce'):
        authenticate_canonical_sum(corrupt, ids, logical)


@pytest.mark.parametrize('mutation', ['rank2', 'batch2', 'wide_instead_of_compute', 'collapsed_terminal',
                                     'dtype', 'fortran', 'truncated', 'trailing'])
def test_exact_physical_schema_rejects_other_views_before_expert_math(tmp_path, mutation):
    model, hidden, result = equation_case()
    write_case(tmp_path, model, hidden, result)
    path = tmp_path/f'{PREFIX}FFN_NORM.npy'
    if mutation == 'rank2':
        np.save(path, np.zeros((1, model.g.width), np.float32))
    elif mutation == 'batch2':
        np.save(path, np.zeros((2, 1, model.g.width), np.float32))
    elif mutation == 'wide_instead_of_compute':
        np.save(path, hidden.numpy())
    elif mutation == 'collapsed_terminal':
        path = tmp_path/f'{PREFIX}TERMINAL_HIDDEN_ROW_SELECT.npy'
        np.save(path, np.zeros((1, 1, model.g.width), np.float32))
    elif mutation == 'dtype':
        np.save(path, np.zeros((1, 1, model.g.width), np.float64))
    elif mutation == 'fortran':
        raw = path.read_bytes()
        assert b"'fortran_order': False" in raw
        path.write_bytes(raw.replace(b"'fortran_order': False", b"'fortran_order': True ", 1))
    elif mutation == 'truncated':
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError):
        load_operands(tmp_path, PREFIX, model.g)


@pytest.mark.parametrize('ids', [[1, 0], [0, 0], [-1], [3], [True], [0.0]])
def test_exact_original_expert_coordinates_are_mandatory(ids):
    model, _, result = equation_case()
    hidden = result['snapshots']['mtp_hc_ffn_pre'].numpy()[0]
    with pytest.raises(ValueError):
        expert_bank(model, hidden, ids)


@pytest.mark.parametrize('prefix', ['layer48_', 'decode_step0_MTP15_', 'decode_step0_CONDITION_1_MTP1_',
                                    'decode_step0_BRANCH_1_MTP2_', '../decode_step0_MTP0_'])
def test_checkpoint_identity_rejects_foreign_or_incomplete_branches(prefix):
    with pytest.raises(ValueError):
        prefix_identity(prefix)


def metadata_case(directory):
    """Create non-GGUF files solely for the canonical metadata-only stat contract."""
    main, predictor = directory/'main.gguf', directory/'predictor.gguf'
    main.write_bytes(b'main descriptor only')
    predictor.write_bytes(b'predictor descriptor only')
    fields = dict(line.split(': ', 1) for line in build_reference_identity(main, 'prompt', [1, 2], 2).metadata_lines())
    fields.update({'prompt': 'prompt', 'token_ids': '1,2', 'n_layers': '48', 'd_model': '2560',
        'qwen4exp_reference_schema': '1', 'qwen4exp_checkpoint_revision': CHECKPOINT_REVISION,
        'qwen4exp_transformers_revision': TRANSFORMERS_REVISION, 'qwen4exp_converter_revision': CONVERTER_REVISION,
        'qwen4exp_residual_width': '10240', 'qwen4exp_terminal_head_width': '2560',
        'qwen4exp_integer_checkpoint_dtype': 'int64', 'qwen4exp_gdn_output_gate': 'sigmoid'})
    fields.update(dense0_metadata_fields(predictor, 3, 2))
    write_metadata_atomically(directory/'metadata.txt', [f'{key}: {value}' for key, value in fields.items()])
    return main, predictor, fields


def test_original_pair_and_branch_metadata_are_authenticated_without_payload(tmp_path):
    main, predictor, _ = metadata_case(tmp_path)
    assert authenticate_pack(main, predictor, tmp_path, PREFIX)['qwen4exp_mtp_recursive_width'] == '10240'
    assert prefix_identity('decode_step1_CONDITION_5_BRANCH_6_7_MTP2_') == (1, 2)
    other = tmp_path/'foreign.gguf'
    other.write_bytes(predictor.read_bytes())
    with pytest.raises(ValueError, match='identity differs'):
        authenticate_pack(main, other, tmp_path, PREFIX)
    with pytest.raises(ValueError, match='capacity'):
        authenticate_pack(main, predictor, tmp_path, 'decode_step2_MTP0_')
    predictor.write_bytes(b'changed size')
    with pytest.raises(ValueError, match='identity differs'):
        authenticate_pack(main, predictor, tmp_path, PREFIX)


def test_main_wide_contract_cannot_be_relabelled_as_terminal_head(tmp_path):
    main, predictor, fields = metadata_case(tmp_path)
    fields['qwen4exp_residual_width'] = '2560'
    write_metadata_atomically(tmp_path/'metadata.txt', [f'{key}: {value}' for key, value in fields.items()])
    with pytest.raises(ValueError, match='authority differs'):
        authenticate_pack(main, predictor, tmp_path, PREFIX)
