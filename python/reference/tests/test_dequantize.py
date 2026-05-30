"""
Unit Tests for GGUF Dequantization Correctness

Tests for quantization format dequantization including:
- Q4_0, Q4_1, Q5_0, Q6_K, Q8_0 quantized formats
- F16, F32 unquantized formats
- Numerical accuracy and properties
- Edge cases (zeros, small values, large values)
- Block alignment and padding

@author David Sanftenberg
"""

import unittest
import struct
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from python.reference.loaders.dequantize import (
    dequantize_q4_0,
    dequantize_q4_1,
    dequantize_q5_0,
    dequantize_q6_k,
    dequantize_iq3_s,
    dequantize_iq3_xxs,
    dequantize_q8_0,
    dequantize_f16,
    dequantize_f32,
    dequantize,
    get_quantization_info,
)
from python.reference.loaders.gguf_parser import GGUFParser, GGUFTensorInfo, GGUFTensorType


class TestQ40Dequantization(unittest.TestCase):
    """Test Q4_0 dequantization (4-bit symmetric quantization)."""
    
    def test_single_block_zeros(self):
        """Test dequantization of a block of zeros."""
        # Q4_0 block: 2 bytes scale + 16 bytes data (32 4-bit values)
        # Scale = 0.0 (FP16), all values = 0
        scale_bytes = struct.pack('<e', 0.0)  # FP16 zero
        data_bytes = b'\x00' * 16  # All zeros (32 x 4-bit)
        block = scale_bytes + data_bytes
        
        result = dequantize_q4_0(block, 32)
        
        self.assertEqual(result.shape, (32,))
        self.assertTrue(np.allclose(result, 0.0))
    
    def test_single_block_ones(self):
        """Test dequantization with scale and known nibble values."""
        # Scale = 1.0, nibbles = 0x1
        scale_bytes = struct.pack('<e', 1.0)
        # Pack nibble value 1 into each half-byte: 0x11 = low=1, high=1
        data_bytes = bytes([0x11] * 16)
        block = scale_bytes + data_bytes
        
        result = dequantize_q4_0(block, 32)
        
        # Q4_0 subtracts 8 from all nibble values: 1 - 8 = -7
        # Expected: scale * (nibble - 8) = 1.0 * -7 = -7.0
        self.assertEqual(result.shape, (32,))
        expected_value = 1.0 * (1 - 8)
        self.assertTrue(np.allclose(result, expected_value))
    
    def test_multiple_blocks(self):
        """Test dequantization with multiple blocks."""
        # 2 blocks = 64 elements
        # Block 1: scale=1.0, nibbles=8 (neutral value: 8-8=0)
        block1_scale = struct.pack('<e', 1.0)
        block1_data = bytes([0x88] * 16)  # Nibble 8 → signed 0
        
        # Block 2: scale=2.0, nibbles=10 (signed: 10-8=+2)
        block2_scale = struct.pack('<e', 2.0)
        block2_data = bytes([0xAA] * 16)  # Nibble 0xA=10 → signed +2
        
        data = block1_scale + block1_data + block2_scale + block2_data
        result = dequantize_q4_0(data, 64)
        
        self.assertEqual(result.shape, (64,))
        # First block: scale * (8 - 8) = 0.0
        self.assertTrue(np.allclose(result[:32], 0.0))
        # Second block: scale * (10 - 8) = 2.0 * 2 = 4.0
        self.assertTrue(np.allclose(result[32:], 2.0 * 2.0))
    
    def test_block_alignment(self):
        """Test that n_elements is properly aligned to block size."""
        # 1 block (32 elements)
        scale = struct.pack('<e', 1.0)
        data = bytes([0x00] * 16)
        block = scale + data
        
        # Should work with exact block size
        result = dequantize_q4_0(block, 32)
        self.assertEqual(result.shape, (32,))


class TestQ41Dequantization(unittest.TestCase):
    """Test Q4_1 dequantization (4-bit asymmetric with min offset)."""
    
    def test_single_block_range(self):
        """Test Q4_1 with scale and min offset."""
        # Q4_1 block: 2 bytes scale + 2 bytes min + 16 bytes data
        # Scale controls range, min controls offset
        scale_bytes = struct.pack('<e', 1.0)
        min_bytes = struct.pack('<e', 0.0)
        data_bytes = bytes([0x00] * 16)
        block = scale_bytes + min_bytes + data_bytes
        
        result = dequantize_q4_1(block, 32)
        
        self.assertEqual(result.shape, (32,))
        # With scale=1.0, min=0.0, nibble 0x0 = 0
        # Result = min + scale * value = 0.0 + 1.0 * 0 = 0.0
        self.assertTrue(np.allclose(result, 0.0))
    
    def test_asymmetric_quantization(self):
        """Test that Q4_1 properly handles min offset."""
        # Scale = 0.5, min = -1.0
        scale_bytes = struct.pack('<e', 0.5)
        min_bytes = struct.pack('<e', -1.0)
        # All nibbles = 0xF (max value = 15)
        data_bytes = bytes([0xFF] * 16)
        block = scale_bytes + min_bytes + data_bytes
        
        result = dequantize_q4_1(block, 32)
        
        # Result = min + scale * value = -1.0 + 0.5 * 15 = 6.5
        expected = -1.0 + 0.5 * 15
        self.assertTrue(np.allclose(result, expected))


class TestQ50Dequantization(unittest.TestCase):
    """Test Q5_0 dequantization (5-bit symmetric quantization)."""
    
    def test_single_block_zeros(self):
        """Test Q5_0 with all zeros."""
        # Q5_0 block: 2 bytes scale + 4 bytes high bits + 16 bytes low bits
        scale_bytes = struct.pack('<e', 0.0)
        high_bits = struct.pack('<I', 0)  # uint32 for high bits
        low_bits = bytes([0x00] * 16)
        block = scale_bytes + high_bits + low_bits
        
        result = dequantize_q5_0(block, 32)
        
        self.assertEqual(result.shape, (32,))
        self.assertTrue(np.allclose(result, 0.0))
    
    def test_5bit_range(self):
        """Test Q5_0 uses full 5-bit range."""
        # 5-bit signed values range from -16 to 15
        scale_bytes = struct.pack('<e', 1.0)
        # High bits all set (contributes 1 << 4 = 16 to each value)
        high_bits = struct.pack('<I', 0xFFFFFFFF)
        # Low 4 bits all 0xF (contributes 15)
        low_bits = bytes([0xFF] * 16)
        block = scale_bytes + high_bits + low_bits
        
        result = dequantize_q5_0(block, 32)
        
        # 5-bit value = (high_bit << 4) | low_bits = (1 << 4) | 15 = 16 + 15 = 31
        # With conversion to signed: 31 >= 16, so 31 - 32 = -1
        expected = 1.0 * (-1)
        self.assertTrue(np.allclose(result, expected))


class TestQ6KDequantization(unittest.TestCase):
    """Test Q6_K dequantization (6-bit K-quantization)."""
    
    def test_single_block_zeros(self):
        """Test Q6_K with all zeros."""
        # Q6_K block: 210 bytes total
        # Structure: half + 16*uint8 (scales) + 128 bytes (qs) + 64 bytes (qh)
        block = bytes([0] * 210)
        
        result = dequantize_q6_k(block, 256)
        
        self.assertEqual(result.shape, (256,))
        # With zero scale, all outputs should be zero
        self.assertTrue(np.allclose(result, 0.0, atol=1e-5))
    
    def test_block_size(self):
        """Test Q6_K uses 256-element blocks."""
        # 2 blocks = 512 elements
        block1 = bytes([0] * 210)
        block2 = bytes([0] * 210)
        data = block1 + block2
        
        result = dequantize_q6_k(data, 512)
        
        self.assertEqual(result.shape, (512,))
    
    def test_numerical_stability(self):
        """Test Q6_K handles numerical edge cases."""
        # Block with small non-zero scale
        block = bytearray(210)
        # Set a very small scale (FP16 format)
        small_scale = struct.pack('<e', 1e-3)
        block[0:2] = small_scale
        
        result = dequantize_q6_k(bytes(block), 256)
        
        # Should not produce NaN or Inf
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))


class TestQ80Dequantization(unittest.TestCase):
    """Test Q8_0 dequantization (8-bit symmetric quantization)."""
    
    def test_single_block_zeros(self):
        """Test Q8_0 with all zeros."""
        # Q8_0 block: 2 bytes scale + 32 bytes data
        scale_bytes = struct.pack('<e', 0.0)
        data_bytes = bytes([0] * 32)
        block = scale_bytes + data_bytes
        
        result = dequantize_q8_0(block, 32)
        
        self.assertEqual(result.shape, (32,))
        self.assertTrue(np.allclose(result, 0.0))
    
    def test_full_8bit_range(self):
        """Test Q8_0 uses full int8 range."""
        # Scale = 1.0, max positive value = 127
        scale_bytes = struct.pack('<e', 1.0)
        data_bytes = bytes([127] * 32)
        block = scale_bytes + data_bytes
        
        result = dequantize_q8_0(block, 32)
        
        self.assertEqual(result.shape, (32,))
        self.assertTrue(np.allclose(result, 127.0))
    
    def test_negative_values(self):
        """Test Q8_0 handles negative int8 values."""
        scale_bytes = struct.pack('<e', 1.0)
        # -128 as signed int8
        data_bytes = bytes([128] * 32)  # 0x80 = -128 in signed int8
        block = scale_bytes + data_bytes
        
        result = dequantize_q8_0(block, 32)
        
        # Should be -128 (as int8)
        self.assertTrue(np.allclose(result, -128.0))


class TestIQ3SDequantization(unittest.TestCase):
    """Test IQ3_S dequantization (3-bit small importance quantization)."""

    def _block(self, scale=1.0, qs=None, qh=None, signs=None, scales=None):
        qs = bytes(qs if qs is not None else [0] * 64)
        qh = bytes(qh if qh is not None else [0] * 8)
        signs = bytes(signs if signs is not None else [0] * 32)
        scales = bytes(scales if scales is not None else [0] * 4)
        return struct.pack('<e', scale) + qs + qh + signs + scales

    def test_single_block_grid_zero(self):
        """Test a simple IQ3_S block using grid index zero and positive signs."""
        result = dequantize_iq3_s(self._block(scale=1.0), (256,))

        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, 1.0))

    def test_scales_and_direct_sign_bits(self):
        """Test scale nibbles and the direct sign mask for the first sub-block."""
        signs = [0] * 32
        signs[0] = 0x01
        result = dequantize_iq3_s(
            self._block(scale=2.0, signs=signs, scales=[0x21, 0, 0, 0]),
            (256,),
        )

        self.assertAlmostEqual(result[0], -6.0)
        self.assertTrue(np.allclose(result[1:32], 6.0))
        self.assertTrue(np.allclose(result[32:64], 10.0))

    def test_row_padding(self):
        """Test row-aware decoding when the final dimension is not a 256 multiple."""
        data = b''.join(self._block(scale=scale) for scale in [1.0, 2.0, 3.0, 4.0])
        result = dequantize_iq3_s(data, (2, 300))

        self.assertEqual(result.shape, (600,))
        self.assertTrue(np.allclose(result[:256], 1.0))
        self.assertTrue(np.allclose(result[256:300], 2.0))
        self.assertTrue(np.allclose(result[300:556], 3.0))
        self.assertTrue(np.allclose(result[556:600], 4.0))


class TestIQ3XXSDequantization(unittest.TestCase):
    """Test IQ3_XXS dequantization (3-bit extra-extra-small importance quantization)."""

    def _block(self, scale=1.0, qs=None, aux_words=None):
        qs = bytes(qs if qs is not None else [0] * 64)
        aux_words = aux_words if aux_words is not None else [0] * 8
        aux = b''.join(struct.pack('<I', word) for word in aux_words)
        return struct.pack('<e', scale) + qs + aux

    def test_single_block_grid_zero(self):
        """Test a simple IQ3_XXS block using grid index zero and positive signs."""
        result = dequantize_iq3_xxs(self._block(scale=1.0), (256,))

        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.allclose(result, 1.0))

    def test_scale_nibble_and_packed_sign_code(self):
        """Test aux-word scale nibbles and ksigns_iq2xs sign decoding."""
        aux_words = [0] * 8
        aux_words[0] = (2 << 28) | 1
        result = dequantize_iq3_xxs(self._block(scale=2.0, aux_words=aux_words), (256,))

        self.assertAlmostEqual(result[0], -10.0)
        self.assertTrue(np.allclose(result[1:7], 10.0))
        self.assertAlmostEqual(result[7], -10.0)
        self.assertTrue(np.allclose(result[32:64], 2.0))

    def test_row_padding(self):
        """Test row-aware decoding when the final dimension is not a 256 multiple."""
        data = b''.join(self._block(scale=scale) for scale in [1.0, 2.0, 3.0, 4.0])
        result = dequantize_iq3_xxs(data, (2, 300))

        self.assertEqual(result.shape, (600,))
        self.assertTrue(np.allclose(result[:256], 1.0))
        self.assertTrue(np.allclose(result[256:300], 2.0))
        self.assertTrue(np.allclose(result[300:556], 3.0))
        self.assertTrue(np.allclose(result[556:600], 4.0))


class TestF16Dequantization(unittest.TestCase):
    """Test F16 (FP16) dequantization."""
    
    def test_simple_values(self):
        """Test F16 with simple known values."""
        # FP16 encoding of [0.0, 1.0, -1.0, 2.0]
        values = [0.0, 1.0, -1.0, 2.0]
        data = b''.join(struct.pack('<e', v) for v in values)
        
        result = dequantize_f16(data, 4)
        
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.allclose(result, values))
    
    def test_precision(self):
        """Test F16 maintains expected precision."""
        # FP16 has ~3 decimal digits of precision
        value = 1.234
        data = struct.pack('<e', value)
        
        result = dequantize_f16(data, 1)
        
        # Should be close within FP16 precision
        self.assertAlmostEqual(result[0], value, places=3)
    
    def test_special_values(self):
        """Test F16 handles special IEEE 754 values."""
        # Test zero, small values, large values
        values = [0.0, 1e-4, 1e4]
        data = b''.join(struct.pack('<e', v) for v in values)
        
        result = dequantize_f16(data, 3)
        
        # Check each value
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 1e-4, places=7)
        self.assertAlmostEqual(result[2], 1e4, places=0)


class TestF32Dequantization(unittest.TestCase):
    """Test F32 (FP32) dequantization (identity operation)."""
    
    def test_simple_values(self):
        """Test F32 passes through values unchanged."""
        values = [0.0, 1.0, -1.0, 2.5, -3.7, 1e6, 1e-6]
        data = b''.join(struct.pack('<f', v) for v in values)
        
        result = dequantize_f32(data, len(values))
        
        self.assertEqual(result.shape, (len(values),))
        self.assertTrue(np.allclose(result, values))
    
    def test_full_precision(self):
        """Test F32 maintains full single-precision."""
        value = 1.23456789
        data = struct.pack('<f', value)
        
        result = dequantize_f32(data, 1)
        
        # Should match exactly (within float precision)
        self.assertAlmostEqual(result[0], value, places=6)


class TestDequantizeDispatch(unittest.TestCase):
    """Test main dequantize() dispatcher function."""
    
    def test_f32_dispatch(self):
        """Test dispatch to F32 dequantization."""
        values = [1.0, 2.0, 3.0, 4.0]
        data = b''.join(struct.pack('<f', v) for v in values)
        shape = (2, 2)
        
        result = dequantize(data, GGUFTensorType.F32, shape)
        
        self.assertEqual(result.shape, shape)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.allclose(result, expected))
    
    def test_f16_dispatch(self):
        """Test dispatch to F16 dequantization."""
        values = [1.0, 2.0, 3.0, 4.0]
        data = b''.join(struct.pack('<e', v) for v in values)
        shape = (4,)
        
        result = dequantize(data, GGUFTensorType.F16, shape)
        
        self.assertEqual(result.shape, shape)
        self.assertTrue(np.allclose(result, values))
    
    def test_q4_0_dispatch(self):
        """Test dispatch to Q4_0 dequantization."""
        # Single block (32 elements)
        scale = struct.pack('<e', 1.0)
        data_bytes = bytes([0x00] * 16)
        data = scale + data_bytes
        shape = (32,)
        
        result = dequantize(data, GGUFTensorType.Q4_0, shape)
        
        self.assertEqual(result.shape, shape)

    def test_iq3_s_dispatch(self):
        """Test dispatch to IQ3_S dequantization."""
        data = struct.pack('<e', 1.0) + bytes([0] * 108)
        shape = (256,)

        result = dequantize(data, GGUFTensorType.IQ3_S, shape)

        self.assertEqual(result.shape, shape)
        self.assertTrue(np.allclose(result, 1.0))

    def test_iq3_xxs_dispatch(self):
        """Test dispatch to IQ3_XXS dequantization."""
        data = struct.pack('<e', 1.0) + bytes([0] * 96)
        shape = (256,)

        result = dequantize(data, GGUFTensorType.IQ3_XXS, shape)

        self.assertEqual(result.shape, shape)
        self.assertTrue(np.allclose(result, 1.0))
    
    def test_shape_reshape(self):
        """Test that result is properly reshaped."""
        # 64 F32 values reshaped to (8, 8)
        values = list(range(64))
        data = b''.join(struct.pack('<f', float(v)) for v in values)
        shape = (8, 8)
        
        result = dequantize(data, GGUFTensorType.F32, shape)
        
        self.assertEqual(result.shape, shape)
        # Verify values are in correct positions
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[0, 7], 7.0)
        self.assertEqual(result[7, 7], 63.0)
    
    def test_unsupported_type(self):
        """Test that unsupported types raise ValueError."""
        data = bytes([0] * 100)
        shape = (10,)
        
        # Use a tensor type that's not implemented (if any)
        # For now, all main types are supported, so this tests the error path
        with self.assertRaises(ValueError):
            # Create an invalid type by using a raw value
            invalid_type = GGUFTensorType(999)  # Non-existent type
            dequantize(data, invalid_type, shape)


class TestQuantizationInfo(unittest.TestCase):
    """Test get_quantization_info() utility function."""
    
    def test_f32_info(self):
        """Test F32 quantization info."""
        info = get_quantization_info(GGUFTensorType.F32)
        
        self.assertEqual(info['block_size'], 1)
        self.assertEqual(info['block_bytes'], 4)
        self.assertEqual(info['bits_per_weight'], 32)
        self.assertIn('FP32', info['name'])
    
    def test_f16_info(self):
        """Test F16 quantization info."""
        info = get_quantization_info(GGUFTensorType.F16)
        
        self.assertEqual(info['block_size'], 1)
        self.assertEqual(info['block_bytes'], 2)
        self.assertEqual(info['bits_per_weight'], 16)
        self.assertIn('FP16', info['name'])
    
    def test_q4_0_info(self):
        """Test Q4_0 quantization info."""
        info = get_quantization_info(GGUFTensorType.Q4_0)
        
        self.assertEqual(info['block_size'], 32)
        self.assertEqual(info['block_bytes'], 18)
        self.assertEqual(info['bits_per_weight'], 4.5)
        self.assertIn('Q4_0', info['name'])
    
    def test_q8_0_info(self):
        """Test Q8_0 quantization info."""
        info = get_quantization_info(GGUFTensorType.Q8_0)
        
        self.assertEqual(info['block_size'], 32)
        self.assertEqual(info['block_bytes'], 34)
        self.assertEqual(info['bits_per_weight'], 8.5)
    
    def test_q6_k_info(self):
        """Test Q6_K quantization info."""
        info = get_quantization_info(GGUFTensorType.Q6_K)
        
        self.assertEqual(info['block_size'], 256)
        self.assertEqual(info['block_bytes'], 210)
        self.assertAlmostEqual(info['bits_per_weight'], 6.5625)

    def test_iq3_s_info(self):
        """Test IQ3_S quantization info."""
        info = get_quantization_info(GGUFTensorType.IQ3_S)

        self.assertEqual(info['block_size'], 256)
        self.assertEqual(info['block_bytes'], 110)
        self.assertAlmostEqual(info['bits_per_weight'], 3.4375)
        self.assertIn('IQ3_S', info['name'])

    def test_iq3_xxs_info(self):
        """Test IQ3_XXS quantization info."""
        info = get_quantization_info(GGUFTensorType.IQ3_XXS)

        self.assertEqual(info['block_size'], 256)
        self.assertEqual(info['block_bytes'], 98)
        self.assertAlmostEqual(info['bits_per_weight'], 3.0625)
        self.assertIn('IQ3_XXS', info['name'])

    def test_iq3_xxs_parser_size_uses_row_padding(self):
        """Test IQ3_XXS raw tensor sizing accounts for row block padding."""
        parser = GGUFParser("unused.gguf")
        tensor_info = GGUFTensorInfo("test.weight", [2, 300], GGUFTensorType.IQ3_XXS, 0)
        calculate_tensor_size = getattr(parser, "_calculate_tensor_size")

        self.assertEqual(calculate_tensor_size(tensor_info), 4 * 98)


class TestNumericalProperties(unittest.TestCase):
    """Test numerical properties and edge cases."""
    
    def test_no_nan_propagation(self):
        """Test that dequantization doesn't produce unexpected NaNs."""
        # Q4_0 with normal scale
        scale = struct.pack('<e', 1.0)
        data = bytes([0xFF] * 16)
        block = scale + data
        
        result = dequantize_q4_0(block, 32)
        
        self.assertFalse(np.any(np.isnan(result)))
    
    def test_no_inf_propagation(self):
        """Test that dequantization doesn't produce unexpected Infs."""
        # Q8_0 with large but valid scale
        scale = struct.pack('<e', 100.0)
        data = bytes([127] * 32)
        block = scale + data
        
        result = dequantize_q8_0(block, 32)
        
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_symmetry(self):
        """Test that positive and negative values are symmetric."""
        # Q8_0 with ±127
        scale = struct.pack('<e', 1.0)
        
        # Positive values
        pos_data = bytes([127] * 32)
        pos_block = scale + pos_data
        pos_result = dequantize_q8_0(pos_block, 32)
        
        # Negative values (0x81 = -127 in int8)
        neg_data = bytes([129] * 32)  # 129 = -127 in two's complement
        neg_block = scale + neg_data
        neg_result = dequantize_q8_0(neg_block, 32)
        
        # Should be symmetric (negatives)
        self.assertTrue(np.allclose(pos_result, -neg_result))


class TestRealWorldPatterns(unittest.TestCase):
    """Test patterns commonly seen in real GGUF files."""
    
    def test_mixed_scales_q4_0(self):
        """Test Q4_0 with varying scales across blocks."""
        # 3 blocks with different scales, nibble=0xA (10 → signed +2)
        blocks = []
        scales = [0.1, 1.0, 10.0]
        
        for scale in scales:
            scale_bytes = struct.pack('<e', scale)
            data_bytes = bytes([0xAA] * 16)  # Nibble 0xA=10, signed=+2
            blocks.append(scale_bytes + data_bytes)
        
        data = b''.join(blocks)
        result = dequantize_q4_0(data, 96)
        
        # Each block should have different magnitude: scale * 2
        self.assertLess(np.abs(result[0:32].mean()), np.abs(result[32:64].mean()))
        self.assertLess(np.abs(result[32:64].mean()), np.abs(result[64:96].mean()))
    
    def test_sparse_pattern(self):
        """Test pattern with mostly zeros (common in quantized weights)."""
        # Q4_0 with small scale and mostly zero nibbles
        scale = struct.pack('<e', 0.01)
        # Most nibbles = 0x8 (which becomes 0 after bias)
        data_bytes = bytes([0x88] * 16)
        block = scale + data_bytes
        
        result = dequantize_q4_0(block, 32)
        
        # Values should be close to zero
        self.assertLess(np.abs(result).max(), 1.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
