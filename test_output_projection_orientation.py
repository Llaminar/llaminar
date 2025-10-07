#!/usr/bin/env python3
"""
Direct test of output projection: Compare PyTorch vs manual matmul.
This will help us understand the correct transpose/orientation.
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, 'python/reference')

from loaders.gguf_loader import GGUFLoader

# Load model
model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"
loader = GGUFLoader(model_path, verbose=False)
model = loader.load_model()

# Get layer 0 output projection weight
o_proj_weight = model.model.layers[0].self_attn.o_proj.weight  # Shape: [d_model, total_head_dim] = [896, 896]

print("=" * 80)
print("OUTPUT PROJECTION WEIGHT ANALYSIS")
print("=" * 80)
print(f"o_proj.weight.shape: {o_proj_weight.shape}")
print(f"o_proj.weight dtype: {o_proj_weight.dtype}")
print()

# Create dummy input (attended output before projection)
batch_size = 1
seq_len = 5
total_head_dim = 896
x = torch.randn(batch_size, seq_len, total_head_dim)

print(f"Input x.shape: {x.shape}")
print()

# Method 1: PyTorch F.linear (what model actually uses)
print("Method 1: PyTorch F.linear")
y_pytorch = F.linear(x, o_proj_weight, bias=None)
print(f"  Result shape: {y_pytorch.shape}")
print(f"  First 5 values: {y_pytorch[0, 0, :5]}")
print()

# Method 2: Manual x @ weight.T (equivalent to F.linear)
print("Method 2: Manual x @ weight.T")
y_manual_transpose = x @ o_proj_weight.T
print(f"  Result shape: {y_manual_transpose.shape}")
print(f"  First 5 values: {y_manual_transpose[0, 0, :5]}")
print(f"  Matches Method 1: {torch.allclose(y_pytorch, y_manual_transpose, atol=1e-6)}")
print()

# Method 3: Manual x @ weight (NO transpose - what llaminar does with transpose_B=false)
print("Method 3: Manual x @ weight (NO transpose)")
try:
    y_manual_no_transpose = x @ o_proj_weight
    print(f"  ERROR: This should fail! Dimension mismatch expected.")
except RuntimeError as e:
    print(f"  ✓ Correctly fails with: {e}")
print()

# Method 4: Simulating cblas_sgemm with CblasTrans for B
print("Method 4: Simulating BLAS with transpose_B=true")
# BLAS with transpose_B computes: C = A @ B.T
# So if we have weight in [896, 896] shape and transpose it, we get [896, 896].T = [896, 896]
# For square matrices, dimension doesn't change but data layout does!
y_blas_transpose = x @ o_proj_weight.T  # This is what BLAS does with transpose_B=true
print(f"  Result shape: {y_blas_transpose.shape}")
print(f"  First 5 values: {y_blas_transpose[0, 0, :5]}")
print(f"  Matches PyTorch: {torch.allclose(y_pytorch, y_blas_transpose, atol=1e-6)}")
print()

# Now test with GGUF weight orientation
print("="* 80)
print("TESTING GGUF WEIGHT ORIENTATION")
print("="* 80)

# Load GGUF weight directly
from loaders.gguf_parser import GGUFParser
from loaders.dequantize import dequantize_tensor

parser = GGUFParser(model_path)
parser._parse_header()
parser._parse_metadata()
parser._parse_tensor_info()

# Find blk.0.attn_output.weight
for tensor_info in parser.tensors:
    if tensor_info.name == 'blk.0.attn_output.weight':
        print(f"Found tensor: {tensor_info.name}")
        print(f"  Dimensions (after Python reversal): {tensor_info.dimensions}")
        
        # Read and dequantize
        parser.file.seek(parser.data_offset + tensor_info.offset)
        quantized_data = parser.file.read(parser._get_tensor_size_bytes(tensor_info))
        weight_gguf = dequantize_tensor(quantized_data, tensor_info.tensor_type, tensor_info.dimensions)
        weight_gguf_tensor = torch.from_numpy(weight_gguf).float()
        
        print(f"  GGUF weight shape: {weight_gguf_tensor.shape}")
        print(f"  PyTorch weight shape: {o_proj_weight.shape}")
        print()
        
        # Compare orientations
        print("Orientation check:")
        corr_row = torch.corrcoef(torch.stack([o_proj_weight[0], weight_gguf_tensor[0]]))[0, 1]
        corr_col = torch.corrcoef(torch.stack([o_proj_weight[0], weight_gguf_tensor[:, 0]]))[0, 1]
        print(f"  PyTorch row 0 vs GGUF row 0: {corr_row:.4f}")
        print(f"  PyTorch row 0 vs GGUF col 0: {corr_col:.4f}")
        print()
        
        if corr_row > 0.9:
            print("✓ SAME ORIENTATION: GGUF and PyTorch store weight identically")
            print("  This means llaminar should use transpose_B=true in matmul")
            print("  to match PyTorch's x @ weight.T computation")
        else:
            print("✗ DIFFERENT ORIENTATION: GGUF stores transposed weight")
            print("  This means llaminar should use transpose_B=false")
        print()
        
        # Test matmul with GGUF weight
        print("Testing matmul with GGUF weight:")
        y_gguf_no_transpose = x @ weight_gguf_tensor.T  # With transpose (transpose_B=true)
        diff = (y_pytorch - y_gguf_no_transpose).abs()
        print(f"  With transpose_B=true: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
        
        # Compute relative L2
        rel_l2 = (diff.pow(2).sum() / y_pytorch.pow(2).sum()).sqrt()
        print(f"  Relative L2 error: {rel_l2:.6f}")
        print()
        
        if rel_l2 < 0.1:
            print("✓ EXCELLENT: transpose_B=true gives good match (<10% error)")
        elif rel_l2 < 0.5:
            print("⚠ OK: transpose_B=true gives reasonable match but quantization error")
        else:
            print("✗ BAD: transpose_B=true doesn't fix the issue!")
            print("   This suggests a deeper problem beyond just transpose.")
        
        break

parser.file.close()
