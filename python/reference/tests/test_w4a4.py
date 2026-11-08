"""Device-free contracts for the independent ruquant arithmetic reference.

Synthetic cases are named contracts, not replay. Exact source-recorded replay
fixtures additionally bind exporter policy and expected bit patterns. These
checks do not load models, initialize accelerators or use production kernels.
"""

import base64
import copy
import hashlib
import json
from pathlib import Path
import struct
import unittest
import zlib

import numpy as np
import torch

from python.reference.quantization.w4a4 import (
    ARITHMETIC_VERSION, KERNEL_REVISION, A4Operands, W4A4Weights,
    hadamard, hadamard_factor, quantize_activations,
)


def source_rows(codes: np.ndarray, *, unit: float = 0.125, rotation: int | None = None,
                multipliers: tuple[int, int, int, int] | None = (1, 2, 3, 4), clip: float = 0.0) -> bytes:
    """Author native little-endian rows independently of the reference decoder."""
    rows, width = codes.shape
    payload = bytearray()
    for row in range(rows):
        descriptor = (0x60000000 | sum(value << (4 * group) for group, value in enumerate(multipliers))) if multipliers else int(codes[row].sum())
        for start in range(0, width, 256):
            payload.extend(struct.pack('<fIif', unit, descriptor & 0xFFFFFFFF, rotation or width, clip))
            block = codes[row, start:start + 256]
            payload.extend(bytes((int(block[j]) & 15) | ((int(block[j + 1]) & 15) << 4) for j in range(0, 256, 2)))
    return bytes(payload)


def patterned_codes(rows: int, width: int) -> np.ndarray:
    """Provide signed nibble extremes without a random-number implementation."""
    return ((np.arange(rows * width).reshape(rows, width) * 7 + 3) % 16 - 8).astype(np.int8)


class W4A4ReferenceContract(unittest.TestCase):
    """Test stored policy, exact arithmetic order, geometry and row invariance."""

    def assert_bytes_equal(self, left: torch.Tensor, right: torch.Tensor) -> None:
        """Compare FP32 representations, including signed zero and rounding bits."""
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.detach().numpy().tobytes(), right.detach().numpy().tobytes())

    def test_pinned_arithmetic_identity(self):
        self.assertEqual(ARITHMETIC_VERSION, 'ruquant-w4a4-g64-shift4-v1')

    def test_source_codes_units_and_groups(self):
        codes = patterned_codes(3, 512)
        weights = W4A4Weights.from_bytes(source_rows(codes), (3, 512), 44)
        self.assertTrue(np.array_equal(weights.codes.numpy(), codes))
        self.assertTrue(torch.equal(weights.units, torch.full((3,), 0.125)))
        self.assertEqual(weights.multipliers[0].tolist(), [1, 2, 3, 4] * 2)
        self.assertEqual(weights.clip, 1.0)
        self.assertEqual(weights.rotation_length, 512)

    def test_plain_row_source_has_unit_multipliers(self):
        codes = patterned_codes(2, 256)
        weights = W4A4Weights.from_bytes(source_rows(codes, multipliers=None), (2, 256), 44)
        self.assertTrue(torch.equal(weights.multipliers, torch.ones((2, 4), dtype=torch.int32)))

    def test_group256_source_repeats_one_multiplier(self):
        raw = bytearray(source_rows(patterned_codes(1, 256)))
        struct.pack_into('<I', raw, 4, 0x40000005)
        weights = W4A4Weights.from_bytes(raw, (1, 256), 44)
        self.assertEqual(weights.multipliers.tolist(), [[5, 5, 5, 5]])

    def test_all_zero_activation_policy(self):
        operands = quantize_activations(torch.zeros((2, 256)))
        self.assertIsInstance(operands, A4Operands)
        self.assertTrue(torch.equal(operands.codes, torch.full((2, 4, 64), -8, dtype=torch.int8)))
        self.assertTrue(torch.equal(operands.zero_points, torch.zeros((2, 4), dtype=torch.int32)))
        self.assertTrue(torch.equal(operands.multipliers, torch.ones((2, 4), dtype=torch.int32)))
        self.assert_bytes_equal(operands.unit, torch.full((2,), np.float32(2) / np.float32(3825)))

    def test_exact_integer_activation_grid(self):
        values = torch.arange(16, dtype=torch.float32).repeat(16).reshape(1, 256)
        operands = quantize_activations(values)
        self.assertTrue(torch.equal(operands.codes.reshape_as(values), values.to(torch.int8) - 8))
        self.assertEqual(operands.zero_points.tolist(), [[0] * 4])
        self.assertEqual(operands.multipliers.tolist(), [[255] * 4])
        weights = W4A4Weights.from_bytes(source_rows((values.numpy() - 8).astype(np.int8)), (1, 256), 44)
        # The pinned native CPU gives 0x44af0001, not mathematical 1400:
        # the rounded FP32 1/255 unit is multiplied before the weight unit.
        expected = torch.from_numpy(np.array([[0x44AF0001]], dtype=np.uint32).view(np.float32).copy())
        self.assert_bytes_equal(weights.linear(values), expected)

    def test_negative_group_term_shifts_before_sum(self):
        codes = np.zeros((1, 256), dtype=np.int8)
        codes[:, ::64] = -8
        weights = W4A4Weights.from_bytes(source_rows(codes, unit=1, multipliers=(1, 1, 1, 1)), (1, 256), 44)
        # Each corrected group is -120, times 255 = -30600. Arithmetic >>4
        # gives -1913 (not -1912). Four independent shifts therefore give -7652.
        expected = ((torch.tensor([[-7652]], dtype=torch.float32) * 16) * (torch.tensor(1, dtype=torch.float32) / 3825))
        actual = weights.linear(torch.ones((1, 256)))
        self.assert_bytes_equal(actual, expected)
        self.assertNotEqual(actual.item(), -32.0)

    def test_weight_zero_yields_zero(self):
        weights = W4A4Weights.from_bytes(source_rows(np.zeros((2, 256), dtype=np.int8)), (2, 256), 44)
        self.assert_bytes_equal(weights.linear(torch.ones((3, 256))), torch.zeros((3, 2)))

    def test_hadamard_factorization_identity(self):
        expected = {256: (1, 256), 1280: (20, 64), 1536: (12, 128), 4352: (68, 64), 5120: (20, 256)}
        for length, factor in expected.items():
            self.assertEqual(hadamard_factor(length), factor)
        for length in [0, -1, 7, 15, 129]:
            with self.assertRaises(ValueError):
                hadamard_factor(length)

    def test_hadamard_butterfly_and_block_boundaries(self):
        source = torch.tensor([[1., 2., 3., 4., -4., -3., -2., -1.]])
        original = source.clone()
        self.assert_bytes_equal(hadamard(source, 4), torch.tensor([[5., -1., -2., 0., -5., -1., -2., 0.]]))
        self.assert_bytes_equal(source, original)
        self.assert_bytes_equal(hadamard(hadamard(source, 4), 4, transpose=True), original)

    def test_paley_inverse_at_actual_export_geometries(self):
        for rotation in [1280, 1536, 4352, 5120]:
            source = (torch.arange(rotation, dtype=torch.float32) % 31 - 15).reshape(1, -1)
            rotated = hadamard(source, rotation)
            restored = hadamard(rotated, rotation, transpose=True)
            torch.testing.assert_close(restored, source, atol=4e-5, rtol=0)
            self.assert_bytes_equal(hadamard(source.repeat(3, 1), rotation), rotated.repeat(3, 1))

    def test_grouped_rows_match_serial_and_repeated_partial_work(self):
        for type_id, width, rotation in [(44, 512, 512), (45, 6144, 1536), (45, 17408, 4352)]:
            weights = W4A4Weights.from_bytes(source_rows(patterned_codes(3, width), rotation=rotation), (3, width), type_id)
            values = ((torch.arange(8 * width, dtype=torch.float32).reshape(8, width) % 97) - 48) * 0.03125
            serial = torch.cat([weights.linear(row.unsqueeze(0)) for row in values])
            for tokens in [8, 1, 3, 8]:
                self.assert_bytes_equal(weights.linear(values[:tokens], output_tile=2), serial[:tokens])
                self.assert_bytes_equal(weights.linear(values[:tokens], output_tile=1), serial[:tokens])

    def test_source_payload_geometry_and_policy_rejections(self):
        raw = source_rows(patterned_codes(2, 512))
        for invalid_raw, shape, type_id in [(raw[:-1], (2, 512), 44), (raw + b'x', (2, 512), 44), (raw, (2, 512), 39), (raw, (2, 255), 44), (raw, (0, 512), 44), (b'', (1, 131072), 44)]:
            with self.assertRaises(ValueError):
                W4A4Weights.from_bytes(invalid_raw, shape, type_id)
        mutations = [(0, '<f', float('nan')), (144, '<f', 0.25), (4, '<I', 0x60001230), (8, '<i', 128), (12, '<f', -1), (12, '<f', float('inf')), (144 + 12, '<f', 0.5)]
        for offset, kind, value in mutations:
            changed = bytearray(raw)
            struct.pack_into(kind, changed, offset, value)
            with self.assertRaises(ValueError):
                W4A4Weights.from_bytes(changed, (2, 512), 44)

    def test_reference_input_policy_rejections(self):
        for values in [torch.ones(256, dtype=torch.float64), torch.full((256,), float('nan')), torch.full((256,), float('inf')), torch.ones(63)]:
            with self.assertRaises(ValueError):
                quantize_activations(values)
        for clip in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                quantize_activations(torch.ones(256), clip)
        with self.assertRaises(ValueError):
            hadamard(torch.ones(256), 12)


def decode_recorded_array(record: dict) -> np.ndarray:
    """Reject missing, oversized, truncated or corrupted recorded array payloads."""
    try:
        if record['dtype'] not in ('|u1', '<f4'):
            raise ValueError('Unsupported recorded array precision')
        dtype = np.dtype(record['dtype'])
        shape = tuple(record['shape'])
        if not shape or len(shape) > 2 or any(not isinstance(value, int) or value <= 0 for value in shape):
            raise ValueError('Invalid recorded shape')
        size = int(np.prod(shape)) * dtype.itemsize
        if size > 1024 * 1024:
            raise ValueError('Recorded fixture array exceeds its bounded size')
        compressed = base64.b64decode(record['zlib_base64'], validate=True)
        decoder = zlib.decompressobj()
        payload = decoder.decompress(compressed, size + 1)
        if len(payload) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Incomplete or oversized recorded payload')
        if hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('Recorded payload digest mismatch')
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    except (KeyError, TypeError, zlib.error) as error:
        raise ValueError('Malformed recorded array') from error


def decode_trace(document: dict) -> list[dict]:
    """Admit complete traces only for the exact pinned arithmetic/source policy."""
    try:
        expected = {'schema': 'llaminar-w4a4-native-trace-v1',
                    'arithmetic_version': ARITHMETIC_VERSION,
                    'oracle_revision': KERNEL_REVISION,
                    'activation_group': 64, 'term_shift': 4}
        if any(document[key] != value for key, value in expected.items()):
            raise ValueError('Incompatible W4A4 trace policy or source revision')
        if len(document['source_sha256']) != 4 or len(document['library_sha256']) != 2 or len(document['compile_commands']) != 2:
            raise ValueError('Incomplete native provenance')
        if len(document['cases']) != 11 or len({case['name'] for case in document['cases']}) != 11:
            raise ValueError('Incomplete or duplicate trace cases')
        cases = []
        for record in document['cases']:
            case = {**record, **{key: decode_recorded_array(record[key])
                               for key in ('weights', 'input', 'rotated', 'a4', 'output')}}
            rows, width = case['weight_shape']
            inputs = case['input']
            weights = W4A4Weights.from_bytes(case['weights'].tobytes(), (rows, width), case['type_id'])
            if case['rotation_length'] != weights.rotation_length or case['clip'] != weights.clip:
                raise ValueError('Trace metadata disagrees with recorded weight policy')
            if inputs.dtype != np.float32 or inputs.ndim != 2 or inputs.shape[1] != width:
                raise ValueError('Invalid recorded input geometry')
            if case['rotated'].dtype != np.float32 or case['rotated'].shape != inputs.shape:
                raise ValueError('Invalid recorded rotation geometry')
            if case['a4'].dtype != np.uint8 or case['a4'].shape != (inputs.shape[0], width // 256 * 144):
                raise ValueError('Invalid recorded A4 geometry')
            if case['output'].dtype != np.float32 or case['output'].shape != (inputs.shape[0], rows):
                raise ValueError('Invalid recorded output geometry')
            case['prepared_weights'] = weights
            cases.append(case)
        return cases
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete recorded W4A4 trace') from error


class W4A4RecordedNativeReplay(unittest.TestCase):
    """Replay recorded native inputs through the independent PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        """Load the checked-in trace without a compiler, accelerator or network."""
        cls.document = json.loads((Path(__file__).parent / 'fixtures/w4a4/native_trace_v1.json').read_text())

    def test_native_intermediates_and_outputs(self):
        for case in decode_trace(self.document):
            with self.subTest(case=case['name']):
                weights = case['prepared_weights']
                values = torch.from_numpy(case['input'])
                rotated = hadamard(values, weights.rotation_length) if weights.type_id == 45 else values
                self.assertEqual(rotated.numpy().tobytes(), case['rotated'].tobytes())
                operands = quantize_activations(rotated, weights.clip)
                tokens, width = values.shape
                raw = case['a4'].reshape(tokens, width // 256, 144)
                headers = np.ascontiguousarray(raw[:, :, :16]).view('<u4').reshape(tokens, width // 256, 4)
                expected_units = headers[:, 0, 0].copy().view('<f4')
                expected_zero_points = np.stack([(headers[:, :, 1] >> (4 * group)) & 15 for group in range(4)], axis=-1).reshape(tokens, -1)
                expected_multipliers = np.stack([(headers[:, :, 2] >> (8 * group)) & 255 for group in range(4)], axis=-1).reshape(tokens, -1)
                packed = raw[:, :, 16:]
                nibbles = np.stack((packed & 15, packed >> 4), axis=-1).reshape(tokens, width).astype(np.int8)
                expected_codes = np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.int8)
                self.assertEqual(operands.unit.numpy().tobytes(), expected_units.tobytes())
                self.assertTrue(np.array_equal(operands.zero_points.numpy(), expected_zero_points))
                self.assertTrue(np.array_equal(operands.multipliers.numpy(), expected_multipliers))
                self.assertEqual(operands.codes.numpy().tobytes(), expected_codes.tobytes())
                for count in [tokens, 1, tokens]:
                    actual = weights.linear(values[:count], output_tile=1)
                    self.assertEqual(actual.numpy().tobytes(), case['output'][:count].tobytes())

    def test_rejects_incompatible_and_incomplete_traces(self):
        for key, value in [('oracle_revision', '0' * 40), ('arithmetic_version', 'unknown'), ('activation_group', 128), ('term_shift', 0)]:
            changed = copy.deepcopy(self.document)
            changed[key] = value
            with self.assertRaises(ValueError):
                decode_trace(changed)
        for mutation in ['missing-output', 'missing-case', 'corrupt-digest', 'truncated-data', 'policy-mismatch']:
            changed = copy.deepcopy(self.document)
            if mutation == 'missing-output':
                del changed['cases'][0]['output']
            elif mutation == 'missing-case':
                changed['cases'].pop()
            elif mutation == 'corrupt-digest':
                changed['cases'][0]['input']['sha256'] = '0' * 64
            elif mutation == 'truncated-data':
                changed['cases'][0]['input']['zlib_base64'] = base64.b64encode(b'bad').decode()
            else:
                changed['cases'][0]['clip'] = 0.5
            with self.assertRaises(ValueError):
                decode_trace(changed)


if __name__ == '__main__':
    unittest.main()
