#!/usr/bin/env python3
"""
Analyze layer 21 Q8_1 tensor dumps to debug residual add divergence.

Q8_1Block format:
- d: uint16 (FP16 bits for scale)
- sum_qs: int16 (sum of quantized values)
- qs: int8[32] (quantized values)

Total: 36 bytes per block (2 + 2 + 32)
"""

import numpy as np
import struct
import sys
from pathlib import Path


def fp16_bits_to_float(bits: int) -> float:
    """Convert FP16 bit pattern to float32."""
    # Unpack the 16-bit value as FP16 and convert to float32
    packed = struct.pack('<H', bits)
    return struct.unpack('<e', packed)[0]


def load_q8_1_blocks(filepath: str, num_blocks: int) -> list:
    """Load Q8_1 blocks from binary file."""
    blocks = []
    with open(filepath, 'rb') as f:
        for _ in range(num_blocks):
            data = f.read(36)  # 2 + 2 + 32 = 36 bytes per block
            if len(data) < 36:
                break
            
            d_bits = struct.unpack('<H', data[0:2])[0]
            sum_qs = struct.unpack('<h', data[2:4])[0]  # signed int16
            qs = list(struct.unpack('<32b', data[4:36]))  # signed int8[32]
            
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


def analyze_blocks(name: str, blocks: list, max_blocks: int = 252):
    """Analyze Q8_1 blocks and print statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")
    
    n = min(len(blocks), max_blocks)
    
    # Collect statistics
    scales = []
    sum_qs_list = []
    all_dequant = []
    
    for i, block in enumerate(blocks[:n]):
        scales.append(block['scale'])
        sum_qs_list.append(block['sum_qs'])
        dequant = dequantize_block(block)
        all_dequant.extend(dequant)
    
    scales = np.array(scales)
    sum_qs_arr = np.array(sum_qs_list)
    all_dequant = np.array(all_dequant)
    
    print(f"Blocks analyzed: {n}")
    print(f"\nScale statistics:")
    print(f"  min: {scales.min():.6f}")
    print(f"  max: {scales.max():.6f}")
    print(f"  mean: {scales.mean():.6f}")
    print(f"  std: {scales.std():.6f}")
    
    print(f"\nSum_qs statistics:")
    print(f"  min: {sum_qs_arr.min()}")
    print(f"  max: {sum_qs_arr.max()}")
    print(f"  mean: {sum_qs_arr.mean():.2f}")
    
    print(f"\nDequantized value statistics:")
    print(f"  min: {all_dequant.min():.4f}")
    print(f"  max: {all_dequant.max():.4f}")
    print(f"  mean: {all_dequant.mean():.4f}")
    print(f"  std: {all_dequant.std():.4f}")
    
    # Find extreme blocks
    print(f"\nExtreme scale blocks:")
    max_scale_idx = np.argmax(scales)
    min_scale_idx = np.argmin(np.abs(scales) + (scales == 0) * 1e10)  # Find smallest non-zero
    
    print(f"  Max scale block {max_scale_idx}: scale={scales[max_scale_idx]:.6f}, sum_qs={sum_qs_arr[max_scale_idx]}")
    print(f"    qs sample: {blocks[max_scale_idx]['qs'][:8]}")
    
    print(f"  Min non-zero scale block {min_scale_idx}: scale={scales[min_scale_idx]:.6f}, sum_qs={sum_qs_arr[min_scale_idx]}")
    print(f"    qs sample: {blocks[min_scale_idx]['qs'][:8]}")
    
    return scales, sum_qs_arr, all_dequant


def verify_residual_add_manual(res_blocks: list, ffn_blocks: list, out_blocks: list, n: int = 252):
    """Manually verify residual add: out = res + ffn."""
    print(f"\n{'='*60}")
    print("Manual residual add verification (dequant → add → requant)")
    print(f"{'='*60}")
    
    errors = []
    for i in range(min(n, len(res_blocks), len(ffn_blocks), len(out_blocks))):
        # Dequantize inputs
        res_dequant = dequantize_block(res_blocks[i])
        ffn_dequant = dequantize_block(ffn_blocks[i])
        out_dequant = dequantize_block(out_blocks[i])
        
        # Expected: res + ffn
        expected = res_dequant + ffn_dequant
        
        # Compare
        diff = np.abs(expected - out_dequant)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        if max_diff > 10.0:  # Significant error
            errors.append({
                'block': i,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'res_scale': res_blocks[i]['scale'],
                'ffn_scale': ffn_blocks[i]['scale'],
                'out_scale': out_blocks[i]['scale'],
                'res_sum_qs': res_blocks[i]['sum_qs'],
                'ffn_sum_qs': ffn_blocks[i]['sum_qs'],
                'out_sum_qs': out_blocks[i]['sum_qs'],
                'expected_max': expected.max(),
                'expected_min': expected.min(),
                'actual_max': out_dequant.max(),
                'actual_min': out_dequant.min(),
            })
    
    if errors:
        print(f"\nFound {len(errors)} blocks with significant errors (max_diff > 10):")
        for e in errors[:10]:  # Show first 10
            print(f"\n  Block {e['block']}:")
            print(f"    res_scale={e['res_scale']:.4f}, ffn_scale={e['ffn_scale']:.4f}, out_scale={e['out_scale']:.4f}")
            print(f"    res_sum_qs={e['res_sum_qs']}, ffn_sum_qs={e['ffn_sum_qs']}, out_sum_qs={e['out_sum_qs']}")
            print(f"    expected range: [{e['expected_min']:.2f}, {e['expected_max']:.2f}]")
            print(f"    actual range:   [{e['actual_min']:.2f}, {e['actual_max']:.2f}]")
            print(f"    max_diff={e['max_diff']:.2f}, mean_diff={e['mean_diff']:.2f}")
    else:
        print("\nNo significant errors found - residual add looks correct!")
    
    return errors


def find_problem_blocks(res_blocks: list, ffn_blocks: list, out_blocks: list, n: int = 252):
    """Find specific blocks where large magnitude cancellation happens."""
    print(f"\n{'='*60}")
    print("Looking for large magnitude cancellation scenarios")
    print(f"{'='*60}")
    
    problematic = []
    for i in range(min(n, len(res_blocks), len(ffn_blocks), len(out_blocks))):
        res_scale = res_blocks[i]['scale']
        ffn_scale = ffn_blocks[i]['scale']
        out_scale = out_blocks[i]['scale']
        
        # Large scale inputs but small output scale = cancellation happened
        max_input_scale = max(abs(res_scale), abs(ffn_scale))
        
        # If either input has large scale AND output has much smaller scale
        # This is where numerical issues can occur
        if max_input_scale > 10.0:
            res_dequant = dequantize_block(res_blocks[i])
            ffn_dequant = dequantize_block(ffn_blocks[i])
            out_dequant = dequantize_block(out_blocks[i])
            expected = res_dequant + ffn_dequant
            
            diff = np.abs(expected - out_dequant)
            max_diff = diff.max()
            
            problematic.append({
                'block': i,
                'res_scale': res_scale,
                'ffn_scale': ffn_scale,
                'out_scale': out_scale,
                'scale_ratio': max_input_scale / (abs(out_scale) + 1e-10),
                'max_diff': max_diff,
                'res_max': res_dequant.max(),
                'res_min': res_dequant.min(),
                'ffn_max': ffn_dequant.max(),
                'ffn_min': ffn_dequant.min(),
                'expected_max': expected.max(),
                'expected_min': expected.min(),
                'actual_max': out_dequant.max(),
                'actual_min': out_dequant.min(),
            })
    
    if problematic:
        # Sort by max_diff
        problematic.sort(key=lambda x: -x['max_diff'])
        
        print(f"\nFound {len(problematic)} blocks with large input scales (>10):")
        print(f"Top 10 by error:")
        for p in problematic[:10]:
            print(f"\n  Block {p['block']}:")
            print(f"    res_scale={p['res_scale']:.4f} (range [{p['res_min']:.2f}, {p['res_max']:.2f}])")
            print(f"    ffn_scale={p['ffn_scale']:.4f} (range [{p['ffn_min']:.2f}, {p['ffn_max']:.2f}])")
            print(f"    out_scale={p['out_scale']:.4f} (range [{p['actual_min']:.2f}, {p['actual_max']:.2f}])")
            print(f"    expected range: [{p['expected_min']:.2f}, {p['expected_max']:.2f}]")
            print(f"    max_diff={p['max_diff']:.2f}")
    
    return problematic


def main():
    dump_dir = Path("/tmp/layer21_dump")
    
    # Read metadata
    meta = {}
    with open(dump_dir / "layer21_metadata.txt") as f:
        for line in f:
            k, v = line.strip().split('=')
            meta[k] = int(v) if v.isdigit() else v
    
    print(f"Metadata: {meta}")
    
    # Effective blocks: effective_seq_len * blocks_per_row
    effective_blocks = meta['effective_seq_len'] * meta['blocks_per_row']
    print(f"Effective blocks: {effective_blocks} (9 rows * 28 blocks/row)")
    
    # Load blocks (only load effective blocks)
    print("\nLoading blocks...")
    res_blocks = load_q8_1_blocks(dump_dir / "layer21_residual_q8_1.bin", effective_blocks)
    ffn_blocks = load_q8_1_blocks(dump_dir / "layer21_ffn_output_q8_1.bin", effective_blocks)
    out_blocks = load_q8_1_blocks(dump_dir / "layer21_residual_output_q8_1.bin", effective_blocks)
    
    print(f"Loaded: {len(res_blocks)} residual, {len(ffn_blocks)} ffn_output, {len(out_blocks)} output blocks")
    
    # Analyze each tensor
    res_scales, res_sum_qs, res_dequant = analyze_blocks("Residual (before FFN)", res_blocks)
    ffn_scales, ffn_sum_qs, ffn_dequant = analyze_blocks("FFN Output", ffn_blocks)
    out_scales, out_sum_qs, out_dequant = analyze_blocks("Residual Output (after add)", out_blocks)
    
    # Verify the residual add operation
    errors = verify_residual_add_manual(res_blocks, ffn_blocks, out_blocks)
    
    # Find problematic cancellation scenarios  
    problematic = find_problem_blocks(res_blocks, ffn_blocks, out_blocks)
    
    # If we found errors, dump a specific problematic block for unit testing
    if errors:
        worst = errors[0]
        block_idx = worst['block']
        
        print(f"\n{'='*60}")
        print(f"WORST BLOCK FOR UNIT TEST: Block {block_idx}")
        print(f"{'='*60}")
        
        print(f"\nResidual block {block_idx}:")
        print(f"  d_bits=0x{res_blocks[block_idx]['d_bits']:04x}")
        print(f"  scale={res_blocks[block_idx]['scale']}")
        print(f"  sum_qs={res_blocks[block_idx]['sum_qs']}")
        print(f"  qs={res_blocks[block_idx]['qs']}")
        
        print(f"\nFFN output block {block_idx}:")
        print(f"  d_bits=0x{ffn_blocks[block_idx]['d_bits']:04x}")
        print(f"  scale={ffn_blocks[block_idx]['scale']}")
        print(f"  sum_qs={ffn_blocks[block_idx]['sum_qs']}")
        print(f"  qs={ffn_blocks[block_idx]['qs']}")
        
        print(f"\nOutput block {block_idx}:")
        print(f"  d_bits=0x{out_blocks[block_idx]['d_bits']:04x}")
        print(f"  scale={out_blocks[block_idx]['scale']}")
        print(f"  sum_qs={out_blocks[block_idx]['sum_qs']}")
        print(f"  qs={out_blocks[block_idx]['qs']}")


if __name__ == "__main__":
    main()
