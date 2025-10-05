"""
Test script for GGUF dequantization.

Tests Q4_0, Q8_0, Q6_K, and F16 dequantization against real GGUF data.

Usage:
    python -m python.reference.loaders.test_dequantize models/qwen2.5-0.5b-instruct-q4_0.gguf
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from python.reference.loaders import GGUFParser, GGUFTensorType, dequantize


def test_dequantization(gguf_path: str):
    """Test dequantization on real GGUF tensors"""
    print(f"\n{'='*80}")
    print(f"Testing GGUF Dequantization on: {gguf_path}")
    print(f"{'='*80}\n")
    
    with GGUFParser(gguf_path) as parser:
        parser.parse()
        
        # Group tensors by type
        tensors_by_type = {}
        for tensor in parser.tensors:
            tensor_type = tensor.type
            if tensor_type not in tensors_by_type:
                tensors_by_type[tensor_type] = []
            tensors_by_type[tensor_type].append(tensor)
        
        print(f"{'='*80}")
        print("TENSOR TYPE DISTRIBUTION")
        print(f"{'='*80}")
        for tensor_type in sorted(tensors_by_type.keys(), key=lambda t: t.value):
            count = len(tensors_by_type[tensor_type])
            info = dequantize.get_quantization_info(tensor_type)
            print(f"{tensor_type.name:10s} ({info['name']:30s}): {count:3d} tensors")
        
        print(f"\n{'='*80}")
        print("DEQUANTIZATION TESTS")
        print(f"{'='*80}\n")
        
        # Test each supported format
        test_types = [
            GGUFTensorType.F32,
            GGUFTensorType.F16,
            GGUFTensorType.Q4_0,
            GGUFTensorType.Q8_0,
            GGUFTensorType.Q6_K,
        ]
        
        for tensor_type in test_types:
            if tensor_type not in tensors_by_type:
                print(f"⊘ {tensor_type.name:10s} - No tensors of this type in file")
                continue
            
            # Test first tensor of this type
            tensor = tensors_by_type[tensor_type][0]
            
            print(f"Testing {tensor_type.name:10s}: {tensor.name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Elements: {tensor.n_elements:,}")
            
            try:
                # Read raw data
                raw_data = parser.read_tensor_data(tensor)
                print(f"  Raw data size: {len(raw_data):,} bytes")
                
                # Dequantize
                fp32_array = dequantize.dequantize(raw_data, tensor.type, tensor.shape)
                print(f"  Dequantized shape: {fp32_array.shape}")
                print(f"  Dequantized dtype: {fp32_array.dtype}")
                
                # Statistics
                print(f"  Statistics:")
                print(f"    Min:  {fp32_array.min():.6f}")
                print(f"    Max:  {fp32_array.max():.6f}")
                print(f"    Mean: {fp32_array.mean():.6f}")
                print(f"    Std:  {fp32_array.std():.6f}")
                
                # Sample values
                flat = fp32_array.flatten()
                sample_size = min(10, len(flat))
                sample_indices = np.linspace(0, len(flat) - 1, sample_size, dtype=int)
                samples = flat[sample_indices]
                print(f"  Sample values ({sample_size} evenly spaced):")
                print(f"    {samples}")
                
                # Sanity checks
                if np.isnan(fp32_array).any():
                    print(f"  ⚠️  WARNING: Contains NaN values!")
                if np.isinf(fp32_array).any():
                    print(f"  ⚠️  WARNING: Contains Inf values!")
                if np.all(fp32_array == 0):
                    print(f"  ⚠️  WARNING: All zeros!")
                
                print(f"  ✓ SUCCESS\n")
                
            except Exception as e:
                print(f"  ✗ FAILED: {e}\n")
                import traceback
                traceback.print_exc()
        
        print(f"{'='*80}")
        print("DETAILED TEST: Compare small tensor end-to-end")
        print(f"{'='*80}\n")
        
        # Find a small Q4_0 tensor for detailed inspection
        small_q4_tensors = [
            t for t in tensors_by_type.get(GGUFTensorType.Q4_0, [])
            if t.n_elements < 1000
        ]
        
        if small_q4_tensors:
            tensor = small_q4_tensors[0]
            print(f"Tensor: {tensor.name}")
            print(f"Shape: {tensor.shape}")
            print(f"Type: {tensor.type.name}")
            print(f"Elements: {tensor.n_elements}")
            
            raw_data = parser.read_tensor_data(tensor)
            fp32_array = dequantize.dequantize(raw_data, tensor.type, tensor.shape)
            
            # Show first block details for Q4_0
            print(f"\nFirst Q4_0 block (32 elements):")
            print(f"  Raw bytes (first 18): {raw_data[:18].hex()}")
            
            # Decode first block manually for verification
            scale_bytes = raw_data[0:2]
            scale = dequantize.fp16_to_fp32(scale_bytes)
            print(f"  Scale (FP16): {scale:.6f}")
            
            print(f"  First 32 dequantized values:")
            for i in range(min(32, tensor.n_elements)):
                print(f"    [{i:2d}] = {fp32_array.flat[i]:10.6f}", end="")
                if (i + 1) % 4 == 0:
                    print()
            print()
        
        print(f"{'='*80}")
        print("DEQUANTIZATION TEST COMPLETE ✓")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m python.reference.loaders.test_dequantize <gguf_file>")
        print("\nExample:")
        print("  python -m python.reference.loaders.test_dequantize models/qwen2.5-0.5b-instruct-q4_0.gguf")
        sys.exit(1)
    
    test_dequantization(sys.argv[1])
