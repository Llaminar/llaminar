"""
GGUF Dequantization Utilities

Converts quantized GGUF tensor data to FP32 for PyTorch model load    # Interleave to get correct order (low, high, low, high, ...)
    unpacked = np.empty((full_blocks, 32), dtype=np.uint8)
    unpacked[:, 0::2] = low_nibbles
    unpacked[:, 1::2] = high_nibbles
    
    # Convert unsigned 4-bit to signed: subtract 8 to get range (-8 to 7)
    # This matches llama.cpp's dequantize_row_q4_0: (nibble - 8)
    # Previous code used (nibble > 7 ? nibble - 16 : nibble) which was WRONG
    unpacked_signed = unpacked.astype(np.int16) - 8  # Use int16 to avoid overflow
    
    # Apply scales (broadcast over block elements)
    dequantized = unpacked_signed.astype(np.float32) * scales[:, np.newaxis]ts Q4_0, Q8_0, Q6_K, and F16 quantization formats.

Port of dequantization logic from:
- src/repacker.cpp (Q4_0, Q8_0, F16)
- src/model_loader.cpp (Q6_K via llama.cpp)
- llama.cpp/ggml (K-quantization formats)

Author: David Sanftenberg
"""

import struct
import numpy as np
from typing import Tuple
from .gguf_parser import GGUFTensorType


def fp16_to_fp32(fp16_bytes: bytes) -> float:
    """
    Convert IEEE 754 half-precision (FP16) to single-precision (FP32).
    
    Args:
        fp16_bytes: 2-byte FP16 representation
        
    Returns:
        FP32 float value
    """
    # Unpack as uint16
    bits = struct.unpack('<H', fp16_bytes)[0]
    
    # Extract sign, exponent, mantissa
    sign = (bits >> 15) & 0x1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    
    # Handle special cases
    if exponent == 0:
        if mantissa == 0:
            # Zero
            return -0.0 if sign else 0.0
        else:
            # Denormalized number
            value = mantissa / 1024.0 * (2.0 ** -14)
            return -value if sign else value
    elif exponent == 31:
        if mantissa == 0:
            # Infinity
            return float('-inf') if sign else float('inf')
        else:
            # NaN
            return float('nan')
    else:
        # Normalized number
        value = (1.0 + mantissa / 1024.0) * (2.0 ** (exponent - 15))
        return -value if sign else value


def dequantize_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantize Q4_0 format to FP32 (VECTORIZED).
    
    Q4_0 Format:
    - Block size: 32 elements
    - Block layout: 2 bytes (FP16 scale) + 16 bytes (32 x 4-bit values)
    - Total block size: 18 bytes
    - 4-bit values are signed (-8 to 7)
    
    Args:
        data: Raw quantized data
        n_elements: Total number of elements to dequantize
        
    Returns:
        NumPy array of FP32 values
    """
    BLOCK_SIZE = 32
    BLOCK_BYTES = 18  # 2 (scale) + 16 (data)
    
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Convert entire data to numpy array for vectorized operations
    data_array = np.frombuffer(data, dtype=np.uint8)
    
    # Reshape to blocks (may truncate partial block)
    full_blocks = len(data) // BLOCK_BYTES
    data_blocks = data_array[:full_blocks * BLOCK_BYTES].reshape(full_blocks, BLOCK_BYTES)
    
    # Extract scales (first 2 bytes of each block) - vectorized FP16 conversion
    scale_bytes = data_blocks[:, :2].copy()
    scales = np.frombuffer(scale_bytes.tobytes(), dtype=np.float16).astype(np.float32)
    
    # Extract quantized data (remaining 16 bytes per block)
    quant_data = data_blocks[:, 2:].reshape(full_blocks, 16)
    
    # Unpack 4-bit values vectorized
    # Each byte contains 2 4-bit values (lower 4 bits, upper 4 bits)
    low_nibbles = quant_data & 0x0F  # Lower 4 bits
    high_nibbles = (quant_data >> 4) & 0x0F  # Upper 4 bits
    
    # Layout matches llama.cpp: low nibbles in first half, high nibbles in second half
    # For a block of 32 values from 16 bytes:
    # byte[0] -> values[0] (low) and values[16] (high)
    # byte[1] -> values[1] (low) and values[17] (high)
    # ...
    # byte[15] -> values[15] (low) and values[31] (high)
    unpacked = np.empty((full_blocks, 32), dtype=np.uint8)
    unpacked[:, :16] = low_nibbles   # First half: low nibbles
    unpacked[:, 16:] = high_nibbles  # Second half: high nibbles
    
    # Convert unsigned 4-bit to signed: subtract 8 to get range (-8 to 7)
    # This matches llama.cpp's dequantize_row_q4_0: (nibble - 8)
    unpacked_signed = unpacked.astype(np.int16) - 8  # Use int16 to avoid overflow
    
    # Apply scales (broadcast over block elements)
    dequantized = unpacked_signed.astype(np.float32) * scales[:, np.newaxis]
    
    # Flatten and truncate to actual element count
    result = dequantized.flatten()[:n_elements]
    
    # Handle partial last block if exists
    if len(result) < n_elements:
        # Slow path for remaining elements
        remaining = n_elements - len(result)
        last_block_offset = full_blocks * BLOCK_BYTES
        
        if last_block_offset < len(data):
            scale_bytes = data[last_block_offset:last_block_offset + 2]
            scale = fp16_to_fp32(scale_bytes)
            
            partial_result = np.zeros(remaining, dtype=np.float32)
            for i in range(remaining):
                byte_idx = i // 2
                packed_byte = data[last_block_offset + 2 + byte_idx]
                
                if i % 2 == 0:
                    quantized_val = packed_byte & 0x0F
                else:
                    quantized_val = (packed_byte >> 4) & 0x0F
                
                # Match llama.cpp: subtract 8 from nibble value (not conditional subtraction)
                quantized_val_signed = quantized_val - 8
                
                partial_result[i] = scale * quantized_val_signed
            
            result = np.concatenate([result, partial_result])
    
    return result


def dequantize_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantize Q8_0 format to FP32 (VECTORIZED).
    
    Q8_0 Format:
    - Block size: 32 elements
    - Block layout: 2 bytes (FP16 scale) + 32 bytes (32 x int8 values)
    - Total block size: 34 bytes
    
    Args:
        data: Raw quantized data
        n_elements: Total number of elements to dequantize
        
    Returns:
        NumPy array of FP32 values
    """
    BLOCK_SIZE = 32
    BLOCK_BYTES = 34  # 2 (scale) + 32 (data)
    
    # Convert to numpy array
    data_array = np.frombuffer(data, dtype=np.uint8)
    
    # Reshape to blocks
    full_blocks = len(data) // BLOCK_BYTES
    data_blocks = data_array[:full_blocks * BLOCK_BYTES].reshape(full_blocks, BLOCK_BYTES)
    
    # Extract scales (first 2 bytes of each block) - vectorized FP16 conversion
    scale_bytes = data_blocks[:, :2].copy()
    scales = np.frombuffer(scale_bytes.tobytes(), dtype=np.float16).astype(np.float32)
    
    # Extract quantized data (remaining 32 bytes per block)
    quant_data = data_blocks[:, 2:].copy()
    
    # Reinterpret as signed int8
    quant_signed = quant_data.view(np.int8)
    
    # Apply scales (broadcast over block elements)
    dequantized = quant_signed.astype(np.float32) * scales[:, np.newaxis]
    
    # Flatten and truncate to actual element count
    result = dequantized.flatten()[:n_elements]
    
    # Handle partial last block if exists
    if len(result) < n_elements:
        remaining = n_elements - len(result)
        last_block_offset = full_blocks * BLOCK_BYTES
        
        if last_block_offset < len(data):
            scale_bytes = data[last_block_offset:last_block_offset + 2]
            scale = fp16_to_fp32(scale_bytes)
            
            partial_data = data[last_block_offset + 2:last_block_offset + 2 + remaining]
            partial_signed = np.frombuffer(partial_data, dtype=np.int8)
            partial_result = partial_signed.astype(np.float32) * scale
            
            result = np.concatenate([result, partial_result])
    
    return result


def dequantize_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantize Q6_K (6-bit K-quant) to FP32 (VECTORIZED).

    block_q6_K layout (210 bytes per 256 elements):
      - ql[128] (128 bytes, offset 0): lower 4 bits of each quant
      - qh[64] (64 bytes, offset 128): upper 2 bits of each quant (4 per byte)
      - scales[16] (16 bytes, offset 192, int8): per-sub-block scales
      - d (FP16, 2 bytes, offset 208): super-block scale

    Dequant formula: value = d * scales[sub_block] * (6bit_quant - 32)
    """
    QK_K = 256
    BLOCK_BYTES = 210

    full_blocks = n_elements // QK_K
    result = np.zeros(n_elements, dtype=np.float32)

    if full_blocks == 0:
        return result

    data_array = np.frombuffer(data[:full_blocks * BLOCK_BYTES], dtype=np.uint8)
    data_blocks = data_array.reshape(full_blocks, BLOCK_BYTES)

    # Extract fields per block_q6_K struct layout
    ql = data_blocks[:, 0:128]       # (num_blocks, 128) lower 4 bits
    qh = data_blocks[:, 128:192]     # (num_blocks, 64) upper 2 bits
    sc = data_blocks[:, 192:208].copy().view(np.int8)  # (num_blocks, 16) signed
    d = np.frombuffer(data_blocks[:, 208:210].copy().tobytes(), dtype=np.float16).astype(np.float32)

    # Build 6-bit quants as uint8 (N, 256) in the llama.cpp Q6_K element order:
    #   Each half (128 elems) uses ql[64] for low 4 bits, qh[32] for upper 2 bits.
    #   Output element order per half: (ql_lo & 0xF), (ql_hi & 0xF), (ql_lo >> 4), (ql_hi >> 4)
    quants = np.empty((full_blocks, QK_K), dtype=np.uint8)
    for half in range(2):
        ql_lo = ql[:, half * 64:half * 64 + 32]
        ql_hi = ql[:, half * 64 + 32:(half + 1) * 64]
        qh_h  = qh[:, half * 32:(half + 1) * 32]
        off = half * 128
        quants[:, off:off+32]       = (ql_lo & 0xF) | (((qh_h >> 0) & 3) << 4)
        quants[:, off+32:off+64]    = (ql_hi & 0xF) | (((qh_h >> 2) & 3) << 4)
        quants[:, off+64:off+96]    = (ql_lo >> 4)  | (((qh_h >> 4) & 3) << 4)
        quants[:, off+96:off+128]   = (ql_hi >> 4)  | (((qh_h >> 6) & 3) << 4)

    # 3D reshape: 16 groups of 16 elements, each group has one scale value.
    # Broadcasting (N,16,1) scale × (N,16,16) quants avoids materializing (N,256) scale array.
    q_3d = quants.reshape(full_blocks, 16, 16).astype(np.float32)
    q_3d -= 32
    sc_3d = sc.astype(np.float32).reshape(full_blocks, 16, 1)
    q_3d *= d[:, np.newaxis, np.newaxis] * sc_3d

    result[:full_blocks * QK_K] = q_3d.reshape(-1)
    return result


def dequantize_q4_1(data: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantize Q4_1 (4-bit with per-block scale and minimum) to FP32 (VECTORIZED).
    
    Q4_1 format:
    - Scale (FP16, 2 bytes)
    - Minimum (FP16, 2 bytes)  
    - Quantized values (32 x 4-bit, packed in 16 bytes)
    - Block size: 32 elements
    - Total block size: 20 bytes
    
    Formula: value = scale * quant + minimum
    
    Args:
        data: Raw Q4_1 data
        n_elements: Total number of elements
        
    Returns:
        NumPy array of FP32 values
    """
    QK = 32
    BLOCK_BYTES = 20  # 2 + 2 + 16
    
    full_blocks = n_elements // QK
    remainder = n_elements % QK
    
    # Pre-allocate
    result = np.zeros(n_elements, dtype=np.float32)
    
    if full_blocks == 0:
        if remainder > 0:
            # Handle partial block - low nibbles for first 16, high nibbles for next 16
            scale = fp16_to_fp32(data[0:2])
            minimum = fp16_to_fp32(data[2:4])
            for i in range(remainder):
                if i < 16:
                    q = data[4 + i] & 0x0F
                else:
                    q = (data[4 + i - 16] >> 4) & 0x0F
                result[i] = scale * q + minimum
        return result
    
    # Vectorized processing for full blocks
    data_array = np.frombuffer(data[:full_blocks * BLOCK_BYTES], dtype=np.uint8)
    data_blocks = data_array.reshape(full_blocks, BLOCK_BYTES)
    
    # Extract scales and minimums - VECTORIZED
    scales_bytes = data_blocks[:, :2].tobytes()
    mins_bytes = data_blocks[:, 2:4].tobytes()
    scales = np.frombuffer(scales_bytes, dtype=np.float16).astype(np.float32)
    mins = np.frombuffer(mins_bytes, dtype=np.float16).astype(np.float32)
    
    # Extract quantized data
    quants_packed = data_blocks[:, 4:]  # Shape: (num_blocks, 16)
    
    # Unpack 4-bit values - VECTORIZED
    # Layout matches llama.cpp: low nibbles in first half, high nibbles in second half
    # byte[j] -> element j (low nibble) and element j+16 (high nibble)
    quants = np.empty((full_blocks, QK), dtype=np.uint8)
    quants[:, :16] = quants_packed & 0x0F
    quants[:, 16:] = (quants_packed >> 4) & 0x0F
    
    # Dequantize: scale * quant + minimum - FULLY VECTORIZED
    scales_expanded = scales[:, np.newaxis]
    mins_expanded = mins[:, np.newaxis]
    result_blocks = scales_expanded * quants.astype(np.float32) + mins_expanded
    
    # Flatten
    result[:full_blocks * QK] = result_blocks.reshape(-1)
    
    # Handle remainder
    if remainder > 0:
        offset = full_blocks * BLOCK_BYTES
        scale = fp16_to_fp32(data[offset:offset+2])
        minimum = fp16_to_fp32(data[offset+2:offset+4])
        for i in range(remainder):
            if i < 16:
                q = data[offset + 4 + i] & 0x0F
            else:
                q = (data[offset + 4 + i - 16] >> 4) & 0x0F
            result[full_blocks * QK + i] = scale * q + minimum
    
    return result


def dequantize_q5_0(data: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantize Q5_0 (5-bit) to FP32 (VECTORIZED).
    
    Q5_0 format:
    - Scale (FP16, 2 bytes)
    - High bits (32 bits packed in 4 bytes)
    - Low bits (32 x 4-bit, packed in 16 bytes)
    - Block size: 32 elements
    - Total block size: 22 bytes
    
    Each 5-bit value is split: 1 high bit + 4 low bits
    
    Args:
        data: Raw Q5_0 data
        n_elements: Total number of elements
        
    Returns:
        NumPy array of FP32 values
    """
    QK = 32
    BLOCK_BYTES = 22  # 2 + 4 + 16
    
    full_blocks = n_elements // QK
    remainder = n_elements % QK
    
    result = np.zeros(n_elements, dtype=np.float32)
    
    if full_blocks == 0 and remainder > 0:
        # Single partial block fallback
        scale = fp16_to_fp32(data[0:2])
        high_bits = int.from_bytes(data[2:6], byteorder='little')
        quants = []
        for i in range(remainder):
            byte_idx = i // 2
            if i % 2 == 0:
                low = data[6 + byte_idx] & 0x0F
            else:
                low = (data[6 + byte_idx] >> 4) & 0x0F
            high = (high_bits >> i) & 1
            q = (high << 4) | low
            if q >= 16:
                q -= 32
            result[i] = scale * q
        return result
    
    if full_blocks == 0:
        return result
    
    # Vectorized processing
    data_array = np.frombuffer(data[:full_blocks * BLOCK_BYTES], dtype=np.uint8)
    data_blocks = data_array.reshape(full_blocks, BLOCK_BYTES)
    
    # Extract scales - VECTORIZED
    scales_bytes = data_blocks[:, :2].tobytes()
    scales = np.frombuffer(scales_bytes, dtype=np.float16).astype(np.float32)
    
    # Extract high bits (4 bytes = 32 bits per block)
    high_bits_packed = data_blocks[:, 2:6]
    high_bits = np.unpackbits(high_bits_packed, axis=1, bitorder='little')[:, :QK]
    
    # Extract low bits (16 bytes = 32 x 4-bit per block)
    low_bits_packed = data_blocks[:, 6:]
    low_bits = np.empty((full_blocks, QK), dtype=np.uint8)
    low_bits[:, 0::2] = low_bits_packed & 0x0F
    low_bits[:, 1::2] = (low_bits_packed >> 4) & 0x0F
    
    # Combine high + low bits - VECTORIZED
    quants = ((high_bits.astype(np.uint8) << 4) | low_bits).astype(np.int8)
    
    # Convert to signed - VECTORIZED
    quants = np.where(quants >= 16, quants - 32, quants)
    
    # Dequantize - VECTORIZED
    scales_expanded = scales[:, np.newaxis]
    result_blocks = scales_expanded * quants.astype(np.float32)
    
    # Flatten
    result[:full_blocks * QK] = result_blocks.reshape(-1)
    
    # Handle remainder
    if remainder > 0:
        offset = full_blocks * BLOCK_BYTES
        scale = fp16_to_fp32(data[offset:offset+2])
        high_bits = int.from_bytes(data[offset+2:offset+6], byteorder='little')
        for i in range(remainder):
            byte_idx = i // 2
            if i % 2 == 0:
                low = data[offset + 6 + byte_idx] & 0x0F
            else:
                low = (data[offset + 6 + byte_idx] >> 4) & 0x0F
            high = (high_bits >> i) & 1
            q = (high << 4) | low
            if q >= 16:
                q -= 32
            result[full_blocks * QK + i] = scale * q
    
    return result


def _dequantize_q6_k_fallback(data: bytes, n_elements: int) -> np.ndarray:
    """Scalar fallback for Q6_K (matches llama.cpp dequantize_row_q6_K exactly)."""
    QK_K = 256
    BLOCK_BYTES = 210

    result = np.zeros(n_elements, dtype=np.float32)
    num_blocks = (n_elements + QK_K - 1) // QK_K

    for block_idx in range(num_blocks):
        block_offset = block_idx * BLOCK_BYTES
        if block_offset + BLOCK_BYTES > len(data):
            break

        d = fp16_to_fp32(data[block_offset + 208:block_offset + 210])
        ql = data[block_offset + 0:block_offset + 128]
        qh = data[block_offset + 128:block_offset + 192]
        sc_raw = data[block_offset + 192:block_offset + 208]
        sc = np.frombuffer(sc_raw, dtype=np.int8)

        ql_off = 0
        qh_off = 0
        sc_off = 0
        y_off = block_idx * QK_K

        for _half in range(2):
            for l in range(32):
                is_idx = l // 16
                q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32

                idx = y_off + l
                if idx < n_elements:
                    result[idx] = d * sc[sc_off + is_idx + 0] * q1
                if idx + 32 < n_elements:
                    result[idx + 32] = d * sc[sc_off + is_idx + 2] * q2
                if idx + 64 < n_elements:
                    result[idx + 64] = d * sc[sc_off + is_idx + 4] * q3
                if idx + 96 < n_elements:
                    result[idx + 96] = d * sc[sc_off + is_idx + 6] * q4

            ql_off += 64
            qh_off += 32
            sc_off += 8
            y_off += 128

    return result


def dequantize_q5_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantize Q5_K (5-bit K-quantization) to FP32 (VECTORIZED).

    Q5_K format (super-block of 256 elements, 176 bytes):
      - d (FP16, 2 bytes): super-block scale
      - dmin (FP16, 2 bytes): super-block min offset
      - scales (12 bytes): packed 6-bit scale/min for 8 sub-blocks
      - qh (32 bytes): high bits
      - qs (128 bytes): low 4-bit quants

    Args:
        data: Raw Q5_K data
        n_elements: Total elements

    Returns:
        FP32 numpy array
    """
    QK_K = 256
    BLOCK_BYTES = 176  # 2+2+12+32+128
    n_blocks = (n_elements + QK_K - 1) // QK_K

    data_arr = np.frombuffer(data, dtype=np.uint8)
    full_blocks = min(n_blocks, len(data) // BLOCK_BYTES)
    if full_blocks == 0:
        return np.zeros(n_elements, dtype=np.float32)

    blocks = data_arr[:full_blocks * BLOCK_BYTES].reshape(full_blocks, BLOCK_BYTES)

    # --- Extract d and dmin (FP16) ---
    d = np.frombuffer(blocks[:, 0:2].copy().tobytes(), dtype=np.float16).astype(np.float32)
    dmin = np.frombuffer(blocks[:, 2:4].copy().tobytes(), dtype=np.float16).astype(np.float32)

    # --- Unpack 6-bit scales and mins from 12 bytes → 8 pairs each ---
    sr = blocks[:, 4:16]  # (n_blocks, 12)

    sc = np.empty((full_blocks, 8), dtype=np.float32)
    m_arr = np.empty((full_blocks, 8), dtype=np.float32)

    # First 4 sub-blocks
    sc[:, 0:4] = (sr[:, 0:4] & 0x3F).astype(np.float32)
    m_arr[:, 0:4] = (sr[:, 4:8] & 0x3F).astype(np.float32)

    # Next 4 sub-blocks
    sc[:, 4:8] = ((sr[:, 8:12] & 0x0F) | ((sr[:, 0:4] >> 6) << 4)).astype(np.float32)
    m_arr[:, 4:8] = ((sr[:, 8:12] >> 4) | ((sr[:, 4:8] >> 6) << 4)).astype(np.float32)

    # --- Extract qh (32 bytes) and qs (128 bytes) ---
    qh = blocks[:, 16:48]   # (n_blocks, 32)
    qs = blocks[:, 48:176]  # (n_blocks, 128)

    # --- Unpack all 256 elements per block ---
    # Process 4 iterations of 64 elements: each uses 32 qs bytes
    # Iteration j: elements [j*64 .. j*64+63]
    #   First 32:  low nibble of qs[j*32+l] + high bit from qh[l] bit (2*j)
    #   Next 32:   high nibble of qs[j*32+l] + high bit from qh[l] bit (2*j+1)
    result_blocks = np.empty((full_blocks, QK_K), dtype=np.float32)

    for j in range(4):
        qs_chunk = qs[:, j*32:(j+1)*32]  # (n_blocks, 32)

        # Low nibbles → first 32 elements
        low_nibs = (qs_chunk & 0x0F).astype(np.uint8)       # (n_blocks, 32)
        # High nibbles → next 32 elements
        high_nibs = ((qs_chunk >> 4) & 0x0F).astype(np.uint8)  # (n_blocks, 32)

        # High bits from qh: bit (2*j) for low half, bit (2*j+1) for high half
        hb_low = ((qh >> (2 * j)) & 1).astype(np.uint8)         # (n_blocks, 32)
        hb_high = ((qh >> (2 * j + 1)) & 1).astype(np.uint8)    # (n_blocks, 32)

        # Combine: 5-bit quant = low_nib + 16*high_bit
        q_low = low_nibs.astype(np.float32) + 16.0 * hb_low.astype(np.float32)
        q_high = high_nibs.astype(np.float32) + 16.0 * hb_high.astype(np.float32)

        # Scale indices: sub-block 2*j and 2*j+1
        d1 = (d * sc[:, 2*j])[:, np.newaxis]       # (n_blocks, 1)
        m1 = (dmin * m_arr[:, 2*j])[:, np.newaxis]
        d2 = (d * sc[:, 2*j+1])[:, np.newaxis]
        m2 = (dmin * m_arr[:, 2*j+1])[:, np.newaxis]

        result_blocks[:, j*64:j*64+32] = d1 * q_low - m1
        result_blocks[:, j*64+32:j*64+64] = d2 * q_high - m2

    result = result_blocks.reshape(-1)[:n_elements]
    return result


def dequantize_f16(data: bytes, n_elements: int) -> np.ndarray:
    """
    Convert FP16 (half precision) to FP32.
    
    Args:
        data: Raw FP16 data (2 bytes per element)
        n_elements: Total number of elements
        
    Returns:
        NumPy array of FP32 values
    """
    # NumPy has built-in FP16 support
    fp16_array = np.frombuffer(data, dtype=np.float16, count=n_elements)
    return fp16_array.astype(np.float32)


def dequantize_f32(data: bytes, n_elements: int) -> np.ndarray:
    """
    Read FP32 data directly (no dequantization needed).
    
    Args:
        data: Raw FP32 data (4 bytes per element)
        n_elements: Total number of elements
        
    Returns:
        NumPy array of FP32 values
    """
    return np.frombuffer(data, dtype=np.float32, count=n_elements)


def dequantize(data: bytes, tensor_type: GGUFTensorType, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Dequantize tensor data based on its type.
    
    Main entry point for dequantization. Routes to appropriate
    dequantization function based on tensor type.
    
    Args:
        data: Raw quantized tensor data
        tensor_type: GGUF tensor type (Q4_0, Q8_0, Q6_K, F16, F32)
        shape: Tensor shape (used to calculate n_elements)
        
    Returns:
        NumPy array of FP32 values with the given shape
        
    Raises:
        ValueError: If tensor type is not supported
    """
    # Calculate total number of elements
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    
    # Route to appropriate dequantization function
    if tensor_type == GGUFTensorType.F32:
        result = dequantize_f32(data, n_elements)
    elif tensor_type == GGUFTensorType.F16:
        result = dequantize_f16(data, n_elements)
    elif tensor_type == GGUFTensorType.Q4_0:
        result = dequantize_q4_0(data, n_elements)
    elif tensor_type == GGUFTensorType.Q4_1:
        result = dequantize_q4_1(data, n_elements)
    elif tensor_type == GGUFTensorType.Q5_0:
        result = dequantize_q5_0(data, n_elements)
    elif tensor_type == GGUFTensorType.Q8_0:
        result = dequantize_q8_0(data, n_elements)
    elif tensor_type == GGUFTensorType.Q6_K:
        result = dequantize_q6_k(data, n_elements)
    elif tensor_type == GGUFTensorType.Q5_K:
        result = dequantize_q5_k(data, n_elements)
    else:
        raise ValueError(f"Unsupported tensor type for dequantization: {tensor_type.name}")
    
    # Reshape to match tensor shape
    return result.reshape(shape)



def get_quantization_info(tensor_type: GGUFTensorType) -> dict:
    """
    Get information about a quantization format.
    
    Args:
        tensor_type: GGUF tensor type
        
    Returns:
        Dictionary with:
        - block_size: Number of elements per quantization block
        - block_bytes: Bytes per quantization block
        - bits_per_weight: Effective bits per weight
        - name: Human-readable name
    """
    info_map = {
        GGUFTensorType.F32: {
            'block_size': 1,
            'block_bytes': 4,
            'bits_per_weight': 32,
            'name': 'FP32 (unquantized)',
        },
        GGUFTensorType.F16: {
            'block_size': 1,
            'block_bytes': 2,
            'bits_per_weight': 16,
            'name': 'FP16 (half precision)',
        },
        GGUFTensorType.Q4_0: {
            'block_size': 32,
            'block_bytes': 18,
            'bits_per_weight': 4.5,  # (18*8)/32
            'name': 'Q4_0 (4-bit symmetric)',
        },
        GGUFTensorType.Q8_0: {
            'block_size': 32,
            'block_bytes': 34,
            'bits_per_weight': 8.5,  # (34*8)/32
            'name': 'Q8_0 (8-bit symmetric)',
        },
        GGUFTensorType.Q6_K: {
            'block_size': 256,
            'block_bytes': 210,
            'bits_per_weight': 6.5625,  # (210*8)/256
            'name': 'Q6_K (6-bit K-quantization)',
        },
    }
    
    return info_map.get(tensor_type, {
        'block_size': 0,
        'block_bytes': 0,
        'bits_per_weight': 0,
        'name': f'Unknown ({tensor_type.name})',
    })
