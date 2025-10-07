#!/usr/bin/env python3
"""
Deep dive an    # Get tensor info dict
    tensor_info_dict = {t.name: t for t in parser.tensors}
    
    print(f"Total tensors: {len(parser.tensors)}")is of GGUF weight storage format for Qwen models.

This script examines:
1. How GGUF stores weight tensors (dimension order)
2. How llama.cpp interprets these weights
3. What PyTorch expects for linear layers
4. What conventions Llaminar should adopt
"""

import numpy as np
import sys
import os

# Add python/reference to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python', 'reference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python', 'reference', 'loaders'))

from gguf_parser import GGUFParser

def analyze_weight_conventions():
    """
    Analyze weight storage and usage conventions across GGUF, llama.cpp, PyTorch, and Llaminar.
    """
    print("=" * 80)
    print("GGUF WEIGHT STORAGE CONVENTIONS ANALYSIS")
    print("=" * 80)
    
    # Load Qwen model
    model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"\nLoading model: {model_path}")
    parser = GGUFParser(model_path)
    parser.parse()
    
    # Get tensor info dict  
    tensor_info_dict = {t.name: t for t in parser.tensors}
    
    print(f"Total tensors: {len(parser.tensors)}")
    
    print("\n" + "=" * 80)
    print("1. GGUF RAW STORAGE FORMAT")
    print("=" * 80)
    
    # Check key weight tensors
    key_tensors = [
        "token_embd.weight",      # Embedding: special case
        "blk.0.attn_q.weight",    # Query projection
        "blk.0.attn_k.weight",    # Key projection  
        "blk.0.attn_v.weight",    # Value projection
        "blk.0.attn_output.weight", # Output projection
        "blk.0.ffn_gate.weight",  # FFN gate
        "blk.0.ffn_up.weight",    # FFN up
        "blk.0.ffn_down.weight",  # FFN down
        "output.weight",          # LM head: special case
    ]
    
    storage_info = {}
    
    for tensor_name in key_tensors:
        if tensor_name in tensor_info_dict:
            tensor_info = tensor_info_dict[tensor_name]
            shape = tensor_info.shape
            
            # Store for later analysis
            storage_info[tensor_name] = {
                'gguf_raw_shape': shape,
                'quantization': tensor_info.type.name,
            }
            
            print(f"\n{tensor_name}:")
            print(f"  GGUF raw shape: {shape}")
            print(f"  Quantization: {tensor_info.type.name}")
            
            # Determine expected mathematical usage
            if "attn_q" in tensor_name or "attn_k" in tensor_name or "attn_v" in tensor_name:
                print(f"  Mathematical usage: Input[seq, d_model=896] @ W -> Output[seq, ?]")
                print(f"  Expected W shape for x @ W: [d_model=896, out_features]")
                print(f"  Expected W shape for x @ W^T: [out_features, d_model=896]")
                
                if shape[0] == 896:
                    print(f"  ✓ GGUF stores as [896, {shape[1]}] → Suitable for x @ W^T")
                elif shape[1] == 896:
                    print(f"  ✓ GGUF stores as [{shape[0]}, 896] → Suitable for x @ W")
                    
            elif "attn_output" in tensor_name:
                print(f"  Mathematical usage: Input[seq, n_heads*head_dim] @ W -> Output[seq, d_model=896]")
                print(f"  Expected W shape for x @ W: [n_heads*head_dim, 896]")
                print(f"  Expected W shape for x @ W^T: [896, n_heads*head_dim]")
                
                if shape[0] == 896:
                    print(f"  ✓ GGUF stores as [896, {shape[1]}] → Suitable for x @ W")
                elif shape[1] == 896:
                    print(f"  ✓ GGUF stores as [{shape[0]}, 896] → Suitable for x @ W^T")
                    
            elif "ffn_down" in tensor_name:
                print(f"  Mathematical usage: Input[seq, n_ff] @ W -> Output[seq, d_model=896]")
                print(f"  Expected W shape for x @ W: [n_ff, 896]")
                print(f"  Expected W shape for x @ W^T: [896, n_ff]")
                
            elif "ffn_gate" in tensor_name or "ffn_up" in tensor_name:
                print(f"  Mathematical usage: Input[seq, d_model=896] @ W -> Output[seq, n_ff]")
                print(f"  Expected W shape for x @ W: [896, n_ff]")
                print(f"  Expected W shape for x @ W^T: [n_ff, 896]")
                
            elif "token_embd" in tensor_name or "output" in tensor_name:
                print(f"  Special case: Embedding/LM head")
                print(f"  PyTorch Embedding: [vocab_size, d_model]")
                print(f"  GGUF stores: {shape}")
        else:
            print(f"\n{tensor_name}: NOT FOUND")
    
    print("\n" + "=" * 80)
    print("2. LLAMA.CPP CONVENTIONS (from llama-model.cpp)")
    print("=" * 80)
    
    print("""
From llama.cpp/src/llama-model.cpp (Qwen2/Granite architecture):

  layer.wq = create_tensor({n_embd, n_embd_head_k * n_head}, ...)
             n_embd=896, n_embd_head_k*n_head=896 → [896, 896]
  
  layer.wk = create_tensor({n_embd, n_embd_k_gqa}, ...)
             n_embd=896, n_embd_k_gqa=128 → [896, 128]
  
  layer.wv = create_tensor({n_embd, n_embd_v_gqa}, ...)
             n_embd=896, n_embd_v_gqa=128 → [896, 128]
  
  layer.wo = create_tensor({n_embd_head_k * n_head, n_embd}, ...)
             n_embd_head_k*n_head=896, n_embd=896 → [896, 896]

From llama.cpp/ggml/src/ggml.c:

  ggml_mul_mat(a, b):
    - a: weight matrix [k, n]  (a->ne[0]=k, a->ne[1]=n)
    - b: input activations [k, m, ...]  (b->ne[0]=k, b->ne[1]=m)
    - output: [n, m, ...]  (result->ne[0]=a->ne[1], result->ne[1]=b->ne[1])
    
  This computes: C = A^T @ B
  
  So ggml_mul_mat implicitly transposes the weight matrix!
  If weight is stored as [k, n], it's used as [n, k] in the multiplication.
""")
    
    print("\n" + "=" * 80)
    print("3. PYTORCH CONVENTIONS")
    print("=" * 80)
    
    print("""
PyTorch nn.Linear:
  - Stores weight as [out_features, in_features]
  - Applies as: output = input @ weight.T
  - So for Linear(896, 128): weight shape is [128, 896]
  - And computes: x[..., 896] @ W.T[896, 128] = output[..., 128]

PyTorch Embedding:
  - Stores as [vocab_size, embedding_dim]
  - Indexes into: embedding[token_ids, :]
""")
    
    print("\n" + "=" * 80)
    print("4. CONVENTION COMPARISON")
    print("=" * 80)
    
    print("""
| Component        | GGUF Storage      | llama.cpp Usage    | PyTorch Storage   |
|------------------|-------------------|--------------------|-------------------|
| Q projection     | [896, 896]        | W^T @ x implicit   | [896, 896]        |
| K projection     | [896, 128]        | W^T @ x implicit   | [128, 896]        |
| V projection     | [896, 128]        | W^T @ x implicit   | [128, 896]        |
| O projection     | [896, 896]        | W^T @ x implicit   | [896, 896]        |
| FFN gate         | [896, n_ff]       | W^T @ x implicit   | [n_ff, 896]       |
| FFN up           | [896, n_ff]       | W^T @ x implicit   | [n_ff, 896]       |
| FFN down         | [n_ff, 896]       | W^T @ x implicit   | [896, n_ff]       |
| Token embedding  | [896, vocab] *    | Index[tokens, :]   | [vocab, 896]      |
| Output (LM head) | [896, vocab] *    | W^T @ x implicit   | [vocab, 896]      |

* Embeddings need dimension reversal

KEY INSIGHT: GGUF stores weights in [k, n] format where:
  - k is the input feature dimension
  - n is the output feature dimension
  - ggml_mul_mat automatically computes W^T @ x
  
PyTorch stores weights as [out_features, in_features] and computes x @ W^T

So: GGUF[k, n] = PyTorch[n, k]
""")
    
    print("\n" + "=" * 80)
    print("5. LLAMINAR WEIGHT ORIENTATION ANALYSIS")
    print("=" * 80)
    
    # Check actual tensor shapes in Llaminar's expected format
    print("\nAnalyzing what Llaminar currently expects:")
    print("\nFor K projection (Qwen 0.5B):")
    print("  Input: [seq_len, d_model=896]")
    print("  Output: [seq_len, n_heads_kv * head_dim = 2 * 64 = 128]")
    print("  Mathematical operation: x @ W")
    print("  Required W shape: [896, 128]")
    print(f"  GGUF provides: {storage_info.get('blk.0.attn_k.weight', {}).get('gguf_raw_shape', 'N/A')}")
    
    gguf_k_shape = storage_info.get('blk.0.attn_k.weight', {}).get('gguf_raw_shape')
    if gguf_k_shape == (896, 128):
        print("  ✓ GGUF orientation [896, 128] MATCHES Llaminar expectation!")
        print("  ✓ NO dimension reversal needed for K weights")
    elif gguf_k_shape == (128, 896):
        print("  ✗ GGUF orientation [128, 896] is TRANSPOSED vs Llaminar expectation")
        print("  ✗ Dimension reversal IS needed")
    
    print("\nFor O projection (output):")
    print("  Input: [seq_len, n_heads * head_dim = 14 * 64 = 896]")
    print("  Output: [seq_len, d_model=896]")
    print("  Mathematical operation: x @ W")
    print("  Required W shape: [896, 896]")
    print(f"  GGUF provides: {storage_info.get('blk.0.attn_output.weight', {}).get('gguf_raw_shape', 'N/A')}")
    
    gguf_o_shape = storage_info.get('blk.0.attn_output.weight', {}).get('gguf_raw_shape')
    if gguf_o_shape == (896, 896):
        print("  ✓ GGUF orientation [896, 896] (symmetric, either way works)")
        print("  ✓ NO dimension reversal needed for O weights")
    
    print("\n" + "=" * 80)
    print("6. PROPOSED LLAMINAR CONVENTIONS")
    print("=" * 80)
    
    print("""
WEIGHT MATRIX CONVENTION:
-------------------------
All weight matrices in Llaminar should be stored as [in_features, out_features]
and applied as: output = input @ weight

This means:
  - Q/K/V projections: W shape [d_model, projection_dim]
  - O projection: W shape [n_heads*head_dim, d_model]
  - FFN gate/up: W shape [d_model, n_ff]
  - FFN down: W shape [n_ff, d_model]

GGUF LOADING RULE:
------------------
For GGUF files following llama.cpp convention:
  1. Embedding tensors (token_embd.weight, output.weight):
     - GGUF stores: [d_model, vocab_size]
     - Llaminar needs: [vocab_size, d_model]
     - ACTION: REVERSE dimensions
  
  2. Weight matrices (Q/K/V/O, FFN):
     - GGUF stores: [in_features, out_features]
     - Llaminar needs: [in_features, out_features]
     - ACTION: NO reversal (keep as-is)

MATMUL CONVENTION:
------------------
Linear layer: output = input @ weight
  - input: [..., in_features]
  - weight: [in_features, out_features]
  - output: [..., out_features]

NO implicit transposes in matmul calls - explicit is better!

KERNEL CONTRACT:
----------------
All kernels should document:
  - Expected input shape
  - Expected weight shape  
  - Applied operation (e.g., "x @ W" not "x @ W^T")
  - No hidden transposes
""")

if __name__ == "__main__":
    analyze_weight_conventions()
