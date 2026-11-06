#!/usr/bin/env python3
"""Record exact W4A4 intermediates from the pinned upstream CPU implementation.

This is an explicit fixture-authoring command, never a test-time regeneration.
It refuses an existing destination and altered/unpinned source. Expected values
come only from the compiled upstream functions, not the PyTorch implementation.
The retained trace contains complete input bytes, policy, geometry, native A4
operands and outputs; ordinary tests need no compiler, network, model or GPU.
"""

import argparse
import base64
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import zlib

import numpy as np


REVISION = "bfc4c5dc3c96df652f42924370ef49b728d585ad"
VERSION = "ruquant-w4a4-g64-shift4-v1"
SOURCE_FILES = (
    'ggml/include/ggml.h', 'ggml/src/ggml-common.h',
    'ggml/src/ggml-quants.c', 'ggml/src/ggml-cpu/quants.c',
)


def array_record(values: np.ndarray) -> dict:
    """Store complete little-endian array bytes with bounded geometry and digest."""
    values = np.ascontiguousarray(values)
    payload = values.tobytes()
    return {'dtype': values.dtype.str, 'shape': list(values.shape),
            'sha256': hashlib.sha256(payload).hexdigest(),
            'zlib_base64': base64.b64encode(zlib.compress(payload, 9)).decode('ascii')}


def source_payload(codes: np.ndarray, mode: str, rotation: int, clip: float, unit: float) -> bytes:
    """Author exact native source blocks; no reference quantizer creates weights."""
    result = bytearray()
    for row in codes:
        for begin in range(0, len(row), 256):
            if mode == 'group64':
                descriptor = 0x60004321
            elif mode == 'group256':
                descriptor = 0x40000000 | (begin // 256 % 15 + 1)
            elif mode == 'row':
                descriptor = int(row.sum()) & 0xFFFFFFFF
            else:
                raise ValueError(mode)
            result.extend(struct.pack('<fIif', unit, descriptor, rotation, clip))
            block = row[begin:begin + 256]
            result.extend(bytes((int(block[i]) & 15) | ((int(block[i + 1]) & 15) << 4)
                                for i in range(0, 256, 2)))
    return bytes(result)


def cases():
    """Provide bounded source inputs; all bytes are saved, not regenerated in tests."""
    designs = [
        ('grid_epilogue', 44, 256, 1, 1, 256, 'group64', 0.0, 0.125),
        ('negative_shift', 44, 256, 1, 1, 256, 'row', 0.0, 1.0),
        ('rounding_ties', 44, 256, 2, 2, 256, 'group64', 1.0, 0.1),
        ('zero_and_skew', 44, 512, 2, 3, 512, 'group64', 0.0, 0.125),
        ('block256_weights', 44, 512, 2, 2, 512, 'group256', 0.75, 0.0625),
        ('plain_row_weights', 44, 512, 2, 2, 512, 'row', 0.0, 0.1),
        ('sylvester_blocks', 45, 512, 2, 3, 128, 'group64', 0.0, 0.125),
        ('target_hidden', 44, 5120, 1, 2, 5120, 'group64', 0.0, 0.125),
        ('target_attention_hadamard', 45, 6144, 1, 2, 1536, 'group64', 0.0, 0.125),
        ('target_ffn_hadamard', 45, 17408, 1, 2, 4352, 'group64', 0.0, 0.125),
        ('whole_row_paley', 45, 5120, 1, 2, 5120, 'row', 1.0, 0.125),
    ]
    for name, type_id, width, outputs, tokens, rotation, mode, clip, unit in designs:
        codes = ((np.arange(outputs * width).reshape(outputs, width) * 7 + 3) % 16 - 8).astype(np.int8)
        values = (((np.arange(tokens * width).reshape(tokens, width) * 11 + 5) % 97) - 48).astype(np.float32) * np.float32(0.03125)
        if name == 'grid_epilogue':
            values = np.tile(np.arange(16, dtype=np.float32), 16).reshape(1, width)
            codes = (values - 8).astype(np.int8)
        elif name == 'negative_shift':
            values.fill(1)
            codes.fill(0)
            codes[:, ::64] = -8
        elif name == 'rounding_ties':
            values[0] = np.resize(np.arange(31, dtype=np.float32) * np.float32(0.5), width)
            values[1] = -values[0]
        elif name == 'zero_and_skew':
            values[0].fill(0)
            values[1, :64] *= np.float32(0.0001)
            values[2, -64:] *= np.float32(10000)
        yield name, type_id, values, source_payload(codes, mode, rotation, clip, unit), outputs, rotation, clip


def main() -> None:
    """Authenticate build/source identity and record native functions once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Recorded expectations are immutable; author a new reviewed destination')
    source, build = args.source.resolve(), args.build.resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'ggml'], text=True).strip()
    if head != REVISION or dirty:
        raise ValueError('Oracle source must be the exact unmodified pinned revision')
    if any(key.startswith('RQ_') for key in os.environ):
        raise ValueError('Experimental RQ_ environment overrides are forbidden during recording')
    base_path = build / 'upstream/src/libggml-base.so'
    cpu_path = build / 'upstream/src/libggml-cpu.so'
    base, cpu = ct.CDLL(str(base_path)), ct.CDLL(str(cpu_path))
    base.quantize_row_a4_clip.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_float]
    base.quantize_row_a4h_clip_n.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int64, ct.c_float, ct.c_int64]
    base.ggml_rq_hadamard.argtypes = [ct.c_void_p, ct.c_int64, ct.c_bool]
    cpu.ggml_vec_dot_w4a4_a4.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_int]
    for function in [base.quantize_row_a4_clip, base.quantize_row_a4h_clip_n,
                     base.ggml_rq_hadamard, cpu.ggml_vec_dot_w4a4_a4]:
        function.restype = None
    base.ggml_rq_act_group.restype = ct.c_int
    if base.ggml_rq_act_group() != 64:
        raise ValueError('Native oracle did not select group-64 arithmetic')
    records = []
    for name, type_id, values, payload, outputs, rotation, clip in cases():
        tokens, width = values.shape
        row_bytes = width // 256 * 144
        weights = ct.create_string_buffer(payload)
        activation_bytes = np.zeros((tokens, row_bytes), dtype=np.uint8)
        rotated = values.copy()
        result = np.empty((tokens, outputs), dtype=np.float32)
        for token in range(tokens):
            if type_id == 45:
                for start in range(0, width, rotation):
                    base.ggml_rq_hadamard(rotated[token, start:].ctypes.data, rotation, False)
                base.quantize_row_a4h_clip_n(values[token].ctypes.data, activation_bytes[token].ctypes.data, width, clip or 1.0, rotation)
            else:
                base.quantize_row_a4_clip(values[token].ctypes.data, activation_bytes[token].ctypes.data, width, clip or 1.0)
            for output in range(outputs):
                cpu.ggml_vec_dot_w4a4_a4(width, result[token, output:].ctypes.data, 0,
                                       ct.byref(weights, output * row_bytes), 0,
                                       activation_bytes[token].ctypes.data, 0, 1)
        records.append({'name': name, 'type_id': type_id, 'weight_shape': [outputs, width],
                        'rotation_length': rotation, 'clip': clip or 1.0,
                        'weights': array_record(np.frombuffer(payload, dtype=np.uint8)),
                        'input': array_record(values), 'rotated': array_record(rotated),
                        'a4': array_record(activation_bytes), 'output': array_record(result)})
    commands = json.loads((build / 'compile_commands.json').read_text())
    commands = [item['command'].replace(str(source), '${ORACLE_SOURCE}').replace(str(build), '${ORACLE_BUILD}')
                for item in commands if item['file'].endswith(('ggml-quants.c', 'ggml-cpu/quants.c'))]
    if len(commands) != 2:
        raise ValueError('Oracle compile database must bind both native arithmetic translation units')
    document = {'schema': 'llaminar-w4a4-native-trace-v1', 'arithmetic_version': VERSION,
                'oracle_revision': REVISION, 'activation_group': 64, 'term_shift': 4,
                'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
                'library_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (base_path, cpu_path)},
                'compile_commands': commands, 'cases': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')
    print(f'Recorded {len(records)} native cases to {args.output}')


if __name__ == '__main__':
    main()
