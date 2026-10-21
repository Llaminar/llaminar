#!/usr/bin/env python3
"""Independent QFN Dense0 expert equations on an authenticated canonical HF row.

Only validated selected expert IDs cross from production. Original paired GGUF
readers and the exact saved HF FFN input supply arithmetic; native activations
and outputs are not accepted. No ordinary-RMS/HC suffix or logits are generated.
The canonical expert sum must reconstruct before this additive bank is published.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python.reference.generate_moe_route_conditioned_reference import (
    atomic_publish, authenticate_canonical_sum, model_identity,
)
from python.reference.qwen4exp import (
    NativeSliceWeights, CHECKPOINT_REVISION, TRANSFORMERS_REVISION, CONVERTER_REVISION,
)
from python.reference.qwen4exp_mtp_snapshots import (
    authenticate_dense0_metadata, bind_dense0_sources,
)
from python.reference.snapshot_metadata import build_reference_identity

SCHEMA = 1
OPERANDS = ('FFN_NORM', 'MOE_ROUTING_INDICES', 'MOE_ROUTING_WEIGHTS',
            'MOE_ROUTER_OUTPUT', 'MOE_EXPERT_OUTPUT', 'TERMINAL_HIDDEN_ROW_SELECT')


def prefix_identity(prefix):
    """Authenticate one canonical or consumed-input-qualified scalar MTP identity."""
    match = re.fullmatch(
        r'decode_step(0|[1-9][0-9]*)((?:_CONDITION_(?:0|[1-9][0-9]*))?'
        r'(?:_BRANCH_(?:0|[1-9][0-9]*)(?:_(?:0|[1-9][0-9]*))*)?)_MTP(0|[1-9][0-9]*)_', prefix)
    if match is None:
        raise ValueError('QFN expert reference requires an exact MTP checkpoint prefix')
    step, qualifier, depth = int(match[1]), match[2], int(match[3])
    if depth >= 15:
        raise ValueError('QFN expert reference exceeds original predictor depth')
    if '_BRANCH_' in qualifier:
        consumed = qualifier.split('_BRANCH_', 1)[1].split('_')
        if len(consumed) != depth:
            raise ValueError('QFN branch qualifier must name exactly the consumed recursive inputs')
    elif qualifier and depth:
        raise ValueError('QFN recursive condition override requires its consumed branch inputs')
    return step, depth


def load_plane(path, shape):
    """Bound exact little-endian C-order FP32 NPY geometry before reading payload."""
    with Path(path).open('rb') as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            raise ValueError(f'{path}: requires canonical NumPy1.0')
        observed, order, dtype = np.lib.format.read_array_header_1_0(stream)
        if tuple(observed) != tuple(shape) or order or dtype.str != '<f4':
            raise ValueError(f'{path}: requires original C-order FP32 shape{shape}')
        count = math.prod(shape)
        if stream.tell() + count * 4 != os.fstat(stream.fileno()).st_size:
            raise ValueError(f'{path}: original snapshot payload extent differs')
        value = np.fromfile(stream, dtype='<f4', count=count).reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f'{path}: original HF checkpoint is nonfinite')
    return value


def load_operands(directory, prefix, geometry):
    """Keep physical singleton-batch schemas separate from the exact logical expert row."""
    prefix_identity(prefix)
    shapes = {'FFN_NORM': (1, 1, geometry.width),
              'MOE_ROUTING_INDICES': (1, 1, geometry.experts_used),
              'MOE_ROUTING_WEIGHTS': (1, 1, geometry.experts_used),
              'MOE_ROUTER_OUTPUT': (1, 1, geometry.experts),
              'MOE_EXPERT_OUTPUT': (1, 1, geometry.width),
              'TERMINAL_HIDDEN_ROW_SELECT': (1, 1, geometry.trunk_width)}
    physical = {name: load_plane(Path(directory)/f'{prefix}{name}.npy', shape)
                for name, shape in shapes.items()}
    # Exactly the authenticated singleton batch is projected; no arbitrary
    # squeeze/flatten can turn a different request/branch geometry into this row.
    logical = {name: value[0] for name, value in physical.items() if name != 'TERMINAL_HIDDEN_ROW_SELECT'}
    router, ids, weights = (logical[name] for name in
                           ('MOE_ROUTER_OUTPUT', 'MOE_ROUTING_INDICES', 'MOE_ROUTING_WEIGHTS'))
    if (not np.equal(ids, np.floor(ids)).all() or (ids < 0).any() or
            (ids >= geometry.experts).any() or (router < 0).any() or (weights < 0).any() or
            any(len(set(row)) != len(row) for row in ids)):
        raise ValueError('QFN HF routing identity is invalid')
    selected = np.take_along_axis(router, ids.astype(np.int64), axis=1).astype(np.float64)
    mass = selected.sum(axis=1, keepdims=True)
    if (mass <= 0).any() or not np.allclose(weights, selected/mass, rtol=0, atol=2e-6):
        raise ValueError('QFN HF weights differ from normalized selected probabilities')
    return logical, physical


def expert_bank(model, hidden, ids):
    """Reuse the canonical Dense0 original-weight SwiGLU equation for every selected expert."""
    if (not ids or ids != sorted(set(ids)) or any(type(expert) is not int or
            not 0 <= expert < model.g.experts for expert in ids) or
            hidden.dtype != np.float32 or hidden.shape != (1, model.g.width) or
            not np.isfinite(hidden).all()):
        raise ValueError('QFN independent bank requires exact scalar HF input and sorted original experts')
    x = torch.from_numpy(hidden.copy())
    with torch.inference_mode():
        outputs = [model.expert_equation(x, expert).numpy().copy() for expert in ids]
    result = np.stack(outputs)
    if result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError('QFN independent expert equation is invalid')
    return result


def authenticate_pack(model_path, predictor_path, directory, prefix):
    """Retain canonical main/sidecar identity without reading their weight payloads."""
    step, depth = prefix_identity(prefix)
    fields = authenticate_dense0_metadata(directory, predictor_path, depth+1, step+1)
    tokens = tuple(int(token) for token in fields['token_ids'].split(','))
    available = int(fields['decode_steps'])
    identity = build_reference_identity(model_path, fields['prompt'], tokens, available)
    required = dict(line.split(': ', 1) for line in identity.metadata_lines())
    required.update({'n_layers': '48', 'd_model': '2560', 'qwen4exp_reference_schema': '1',
        'qwen4exp_checkpoint_revision': CHECKPOINT_REVISION,
        'qwen4exp_transformers_revision': TRANSFORMERS_REVISION,
        'qwen4exp_converter_revision': CONVERTER_REVISION,
        'qwen4exp_residual_width': '10240', 'qwen4exp_terminal_head_width': '2560',
        'qwen4exp_integer_checkpoint_dtype': 'int64', 'qwen4exp_gdn_output_gate': 'sigmoid'})
    if any(fields.get(key) != value for key, value in required.items()):
        raise ValueError('QFN expert reference main checkpoint authority differs')
    return fields


def generate(model_path, predictor_path, directory, prefix, ids, output):
    """Publish a source-authenticated expert-only bank; no suffix evidence is synthesized."""
    authenticate_pack(model_path, predictor_path, directory, prefix)
    main = NativeSliceWeights(model_path, 0)
    predictor = None
    try:
        predictor = NativeSliceWeights(predictor_path, 0)
        model = bind_dense0_sources(main, predictor)
        logical, physical = load_operands(directory, prefix, model.g)
        if (not ids or len(ids) > 2*model.g.experts_used or ids != sorted(set(ids)) or
                ids[0] < 0 or ids[-1] >= model.g.experts or
                not set(logical['MOE_ROUTING_INDICES'].astype(int).flat) <= set(ids)):
            raise ValueError('QFN expert union must include every original canonical selected expert')
        identity = dict(schema=SCHEMA, equation='qwen4exp-dense0-expert-only',
            main=model_identity(Path(model_path)), predictor=model_identity(Path(predictor_path)),
            checkpoint=prefix, source_layer=48, suffix_contract='canonical_comparisons', expert_ids=ids,
            operands={name: dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name, value in physical.items()})
        output = Path(output)
        metadata = output.with_suffix('.json')
        if metadata.exists() and json.loads(metadata.read_text())['identity'] == identity:
            stored = json.loads(metadata.read_text())
            bank = load_plane(output, (len(ids), 1, model.g.width))
            if hashlib.sha256(bank.tobytes()).hexdigest() != stored['bank_sha256']:
                raise ValueError('QFN committed expert bank checksum mismatch')
            authenticate_canonical_sum(bank, ids, logical)
            return True
        bank = expert_bank(model, logical['FFN_NORM'], ids)
        authenticate_canonical_sum(bank, ids, logical)
        if model_identity(Path(model_path)) != identity['main'] or model_identity(Path(predictor_path)) != identity['predictor']:
            raise ValueError('QFN original sources changed during independent expert evaluation')
        atomic_publish(output, lambda stream: np.save(stream, bank, allow_pickle=False))
        receipt = dict(identity=identity, bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest())
        atomic_publish(metadata, lambda stream: stream.write(json.dumps(receipt, sort_keys=True).encode()))
        return False
    finally:
        if predictor is not None:
            predictor.close()
        main.close()


def main(argv=None):
    """Load only the explicitly paired original sources and named canonical expert input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--shared-predictor', required=True, type=Path)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--reference-prefix', required=True)
    parser.add_argument('--expert-ids', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    # Match the existing canonical independent reference's available physical-core budget.
    import psutil
    torch.set_num_threads(min(len(os.sched_getaffinity(0)), psutil.cpu_count(logical=False) or 1))
    reused = generate(args.model, args.shared_predictor, args.reference_dir, args.reference_prefix,
                      [int(expert) for expert in args.expert_ids.split(',')], args.output)
    print(f'qwen4exp_mtp_expert_reference schema={SCHEMA} reused={int(reused)} suffix=canonical_comparisons')


if __name__ == '__main__':
    main()
