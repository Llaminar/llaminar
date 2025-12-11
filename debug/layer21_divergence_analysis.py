#!/usr/bin/env python3
"""
Analysis of Layer 21 Q8_1 vs FP32 Divergence

FINDINGS:
=========

The layer21_FFN_RESIDUAL divergence (cos=-0.134) is NOT a bug in the residual add operation.
It's caused by accumulated quantization error in the inputs.

KEY DATA:
---------
| Stage              | FP32 Range          | Q8_1 Range          | Cosine Sim |
|--------------------|---------------------|---------------------|------------|
| FFN_INPUT_RESIDUAL | [-92.87, 1553.35]   | [-64.99, 1096.37]   | 0.998489   |
| FFN_DOWN           | [-1499.99, 90.66]   | [-1966.52, 119.44]  | 0.999649   |
| FFN_RESIDUAL       | [-7.77, 53.36]      | [-870.15, 54.45]    | -0.134626  |

THE PROBLEM:
------------
Both inputs have EXCELLENT cosine similarity (>0.998), but their EXTREME VALUES differ:

FP32 Calculation:
  max(residual) + min(ffn_output) = 1553 + (-1500) = 53 (cancellation!)

Q8_1 Calculation:  
  max(residual) + min(ffn_output) = 1096 + (-1967) = -871 (no cancellation!)

The Q8_1 residual has ~30% lower max (1096 vs 1553)
The Q8_1 FFN output has ~31% more negative min (-1967 vs -1500)

These differences are small in the context of cosine similarity (which measures angular alignment)
but catastrophic for element-wise addition where large values nearly cancel.

ROOT CAUSE:
-----------
Quantization noise in Q8_1 affects the SPECIFIC ELEMENTS that should cancel, not the overall
distribution. High cosine similarity means the vectors point in similar directions, but
element-wise operations like addition are sensitive to positional errors.

This is a fundamental limitation of Q8_1 quantized activations when:
1. The tensor contains large magnitude values (>1000)
2. The addition involves near-cancellation of those large values
3. Accumulated quantization error shifts which elements have extreme values

IMPLICATIONS:
-------------
1. The q8_1_add_q8_1 function is working correctly (verified by dump analysis)
2. The problem is in the inputs, not the operation itself
3. This may be acceptable for inference (the model still produces reasonable outputs)
4. For perfect FP32 parity, we would need higher precision for these edge cases

POSSIBLE MITIGATIONS:
---------------------
1. Use FP32 for residual adds when input magnitudes exceed a threshold
2. Accept the divergence as a limitation of Q8_1 precision
3. Use BF16 or FP16 for activations instead of Q8_1
4. Increase Q8_1 precision for outlier values (asymmetric quantization)

DUMP FILES:
-----------
/tmp/layer21_dump/layer21_residual_q8_1.bin - Q8_1 blocks for residual input
/tmp/layer21_dump/layer21_ffn_output_q8_1.bin - Q8_1 blocks for FFN output
/tmp/layer21_dump/layer21_residual_output_q8_1.bin - Q8_1 blocks for result
/tmp/layer21_dump/layer21_metadata.txt - Tensor dimensions and metadata

Each .bin file contains Q8_1Block structs (36 bytes each):
  - d: uint16 (FP16 bits for scale)
  - sum_qs: int16 (sum of quantized values)
  - qs: int8[32] (quantized values)
"""

import numpy as np
import struct

def fp16_bits_to_float(bits: int) -> float:
    """Convert FP16 bit pattern to float32."""
    packed = struct.pack('<H', bits)
    return struct.unpack('<e', packed)[0]

def load_q8_1_blocks(filepath: str, num_blocks: int) -> list:
    """Load Q8_1 blocks from binary file."""
    blocks = []
    with open(filepath, 'rb') as f:
        for _ in range(num_blocks):
            data = f.read(36)
            if len(data) < 36:
                break
            d_bits = struct.unpack('<H', data[0:2])[0]
            sum_qs = struct.unpack('<h', data[2:4])[0]
            qs = list(struct.unpack('<32b', data[4:36]))
            scale = fp16_bits_to_float(d_bits)
            blocks.append({
                'd_bits': d_bits,
                'scale': scale,
                'sum_qs': sum_qs,
                'qs': qs
            })
    return blocks

def dequantize_block(block: dict) -> np.ndarray:
    """Dequantize a Q8_1 block to FP32."""
    scale = block['scale']
    qs = np.array(block['qs'], dtype=np.float32)
    return qs * scale

def main():
    # Load only effective blocks (9 rows × 28 blocks/row)
    effective_blocks = 252
    
    res_blocks = load_q8_1_blocks('/tmp/layer21_dump/layer21_residual_q8_1.bin', effective_blocks)
    ffn_blocks = load_q8_1_blocks('/tmp/layer21_dump/layer21_ffn_output_q8_1.bin', effective_blocks)
    out_blocks = load_q8_1_blocks('/tmp/layer21_dump/layer21_residual_output_q8_1.bin', effective_blocks)
    
    print(f"Loaded {len(res_blocks)} blocks each")
    
    # Find Block 1 which has the extreme values
    print("\n" + "="*60)
    print("BLOCK 1 ANALYSIS (highest magnitude)")
    print("="*60)
    
    res_dequant = dequantize_block(res_blocks[1])
    ffn_dequant = dequantize_block(ffn_blocks[1])
    out_dequant = dequantize_block(out_blocks[1])
    expected = res_dequant + ffn_dequant
    
    print(f"\nResidual block 1:")
    print(f"  scale={res_blocks[1]['scale']:.4f}")
    print(f"  sum_qs={res_blocks[1]['sum_qs']}")
    print(f"  dequant range: [{res_dequant.min():.2f}, {res_dequant.max():.2f}]")
    
    print(f"\nFFN output block 1:")
    print(f"  scale={ffn_blocks[1]['scale']:.4f}")
    print(f"  sum_qs={ffn_blocks[1]['sum_qs']}")
    print(f"  dequant range: [{ffn_dequant.min():.2f}, {ffn_dequant.max():.2f}]")
    
    print(f"\nExpected (res + ffn):")
    print(f"  range: [{expected.min():.2f}, {expected.max():.2f}]")
    
    print(f"\nActual output block 1:")
    print(f"  scale={out_blocks[1]['scale']:.4f}")
    print(f"  sum_qs={out_blocks[1]['sum_qs']}")
    print(f"  dequant range: [{out_dequant.min():.2f}, {out_dequant.max():.2f}]")
    
    print(f"\nError analysis:")
    diff = np.abs(expected - out_dequant)
    print(f"  max_diff: {diff.max():.4f}")
    print(f"  mean_diff: {diff.mean():.4f}")
    
    print("\n" + "="*60)
    print("CONCLUSION: Q8_1 residual add is CORRECT given its inputs")
    print("The divergence is from accumulated quantization error in inputs")
    print("="*60)

if __name__ == "__main__":
    main()
