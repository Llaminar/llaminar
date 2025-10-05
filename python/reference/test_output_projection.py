#!/usr/bin/env python3
"""
Quick test to check if output projection (lm_head) is being applied correctly.

The issue: Top-5 predictions are completely different between PyTorch and Llaminar
even though embeddings match perfectly.

This suggests the problem is in the transformer layers or output projection.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add python directory to path
python_dir = Path(__file__).parent.parent
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from python.reference import ModelRegistry
from python.reference.loaders.gguf_loader import GGUFLoader

def test_output_projection():
    """Test if lm_head weight is transposed or applied incorrectly."""
    
    model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"
    
    print("Loading model...")
    model = ModelRegistry.create("qwen", model_path, auto_load=False)
    model.load_model()
    
    print("Loading GGUF...")
    loader = GGUFLoader(model_path)
    config, tensors = loader.load(as_torch=False)
    
    lm_head_weight = tensors['lm_head.weight']
    print(f"\nGGUF lm_head.weight shape: {lm_head_weight.shape}")
    print(f"Expected: (vocab_size={config.vocab_size}, hidden_dim={config.hidden_size})")
    
    # Get PyTorch lm_head weight
    pytorch_lm_head = model.hf_model.lm_head.weight.cpu().numpy()
    print(f"\nPyTorch lm_head shape: {pytorch_lm_head.shape}")
    
    # Compare first few rows
    print(f"\nGGUF lm_head[0, :10]: {lm_head_weight[0, :10]}")
    print(f"PyTorch lm_head[0, :10]: {pytorch_lm_head[0, :10]}")
    
    # Check if they match or if one is transposed
    if lm_head_weight.shape == pytorch_lm_head.shape:
        diff = np.abs(lm_head_weight - pytorch_lm_head)
        print(f"\nDirect comparison - Mean abs diff: {diff.mean():.6f}")
        if diff.mean() < 0.01:
            print("✓ Weights match directly!")
        else:
            print("❌ Weights DON'T match directly")
    
    if lm_head_weight.shape == pytorch_lm_head.T.shape:
        diff = np.abs(lm_head_weight - pytorch_lm_head.T)
        print(f"\nTranspose comparison - Mean abs diff: {diff.mean():.6f}")
        if diff.mean() < 0.01:
            print("⚠ Weights match when PyTorch is TRANSPOSED!")
            print("This means Llaminar may be using the weight incorrectly")
    
    # Test actual projection
    # Create a fake hidden state
    hidden_state = np.random.randn(1, config.hidden_size).astype(np.float32)
    print(f"\nTest hidden state shape: {hidden_state.shape}")
    
    # PyTorch way: logits = hidden @ lm_head.T
    pytorch_logits = hidden_state @ pytorch_lm_head.T
    print(f"PyTorch logits shape: {pytorch_logits.shape}")
    print(f"PyTorch top-5 tokens: {np.argsort(pytorch_logits[0])[-5:][::-1]}")
    
    # GGUF way (if used as-is): logits = hidden @ lm_head.T
    gguf_logits = hidden_state @ lm_head_weight.T
    print(f"\nGGUF (as loaded) logits shape: {gguf_logits.shape}")
    print(f"GGUF top-5 tokens: {np.argsort(gguf_logits[0])[-5:][::-1]}")
    
    # Check if they match
    diff = np.abs(pytorch_logits - gguf_logits)
    print(f"\nLogits diff (mean abs): {diff.mean():.6f}")
    if diff.mean() < 0.001:
        print("✓ Logits match! Output projection is correct.")
    else:
        print("❌ Logits DON'T match! Output projection has an issue.")

if __name__ == "__main__":
    test_output_projection()
