#!/usr/bin/env python3
"""
Layer-by-layer divergence investigation tool

Compares intermediate activations between Llaminar C++ and PyTorch reference
to pinpoint where numerical divergence occurs.

@author David Sanftenberg
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import torch

# Add python directory to path
python_dir = Path(__file__).parent.parent
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

from python.reference import ModelRegistry
from python.reference.loaders.gguf_loader import GGUFLoader


def compare_tensors(
    name: str,
    pytorch_tensor: np.ndarray,
    llaminar_tensor: np.ndarray,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compare two tensors and return detailed metrics.
    
    Args:
        name: Name of the tensor being compared
        pytorch_tensor: Reference tensor from PyTorch
        llaminar_tensor: Tensor from Llaminar
        verbose: Print detailed comparison info
        
    Returns:
        Dictionary with comparison metrics
    """
    if pytorch_tensor.shape != llaminar_tensor.shape:
        print(f"❌ {name}: Shape mismatch! PyTorch: {pytorch_tensor.shape}, Llaminar: {llaminar_tensor.shape}")
        return {"error": "shape_mismatch"}
    
    # Flatten for easier comparison
    pt_flat = pytorch_tensor.flatten()
    ll_flat = llaminar_tensor.flatten()
    
    # Compute metrics
    abs_diff = np.abs(pt_flat - ll_flat)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Relative error (avoiding division by zero)
    mask = np.abs(pt_flat) > 1e-8
    if np.any(mask):
        rel_errors = np.abs((pt_flat[mask] - ll_flat[mask]) / pt_flat[mask])
        mean_rel_error = np.mean(rel_errors)
        max_rel_error = np.max(rel_errors)
    else:
        mean_rel_error = 0.0
        max_rel_error = 0.0
    
    # Cosine similarity
    pt_norm = np.linalg.norm(pt_flat)
    ll_norm = np.linalg.norm(ll_flat)
    if pt_norm > 0 and ll_norm > 0:
        cosine_sim = np.dot(pt_flat, ll_flat) / (pt_norm * ll_norm)
    else:
        cosine_sim = 1.0 if pt_norm == ll_norm == 0 else 0.0
    
    # L2 norm difference
    l2_diff = np.linalg.norm(pt_flat - ll_flat)
    
    metrics = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_error": mean_rel_error,
        "max_rel_error": max_rel_error,
        "cosine_similarity": cosine_sim,
        "l2_diff": l2_diff,
        "shape": pytorch_tensor.shape,
        "num_elements": len(pt_flat)
    }
    
    if verbose or mean_abs_diff > 0.1:
        status = "✓" if mean_abs_diff < 0.01 else "⚠" if mean_abs_diff < 0.1 else "❌"
        print(f"{status} {name}:")
        print(f"    Shape: {pytorch_tensor.shape}")
        print(f"    Max abs diff: {max_abs_diff:.6f}")
        print(f"    Mean abs diff: {mean_abs_diff:.6f}")
        print(f"    Mean rel error: {mean_rel_error:.6f}")
        print(f"    Cosine similarity: {cosine_sim:.6f}")
        
        if mean_abs_diff > 0.01:
            # Show where the largest differences are
            worst_idx = np.argmax(abs_diff)
            print(f"    Worst element: idx={worst_idx}, PyTorch={pt_flat[worst_idx]:.6f}, Llaminar={ll_flat[worst_idx]:.6f}")
    
    return metrics


def extract_pytorch_layer_activations(
    model,
    token_ids: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    Extract intermediate activations from PyTorch model.
    
    Returns dict with keys like:
    - 'embedding': Token embeddings
    - 'layer_0_attn_out': Layer 0 attention output
    - 'layer_0_ffn_out': Layer 0 FFN output
    - 'final_norm': Final RMSNorm output
    - 'logits': Final logits
    """
    activations = {}
    
    with torch.no_grad():
        # Get embedding
        result = model.forward(token_ids, capture_intermediate=True)
        
        # Note: This requires modifying the PyTorch model to capture intermediates
        # For now, we'll just capture what we can from the forward pass
        
        if "logits" in result:
            logits = result["logits"]
            if isinstance(logits, torch.Tensor):
                activations["logits"] = logits.cpu().numpy()
            else:
                activations["logits"] = logits
        
        # TODO: Modify PyTorch models to capture layer-by-layer activations
        # This would require adding hooks or modifying the forward pass
    
    return activations


def investigate_embedding_layer(
    model_path: str,
    token_ids: List[int]
) -> Dict[str, any]:
    """
    Investigate just the embedding layer to see if divergence starts there.
    """
    print("\n" + "="*80)
    print("INVESTIGATING EMBEDDING LAYER")
    print("="*80)
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    model = ModelRegistry.create("qwen", model_path, auto_load=False)
    model.load_model()
    
    # Load GGUF directly to access raw embeddings
    print("Loading GGUF file...")
    loader = GGUFLoader(model_path)
    config_dict, tensors = loader.load(as_torch=False)
    
    gguf_embeddings = None
    
    # Get embedding weights - try both naming conventions
    emb_key = None
    for key in ['model.embed_tokens.weight', 'token_embd.weight']:
        if key in tensors:
            emb_key = key
            break
    
    if emb_key:
        embedding_weight = tensors[emb_key]
        print(f"\nEmbedding weight shape: {embedding_weight.shape}")
        print(f"Token IDs: {token_ids}")
        
        # Extract embeddings for our tokens from GGUF
        gguf_emb_list = []
        for token_id in token_ids:
            if token_id < embedding_weight.shape[0]:
                gguf_emb_list.append(embedding_weight[token_id])
            else:
                print(f"Warning: Token ID {token_id} out of range!")
        
        if gguf_emb_list:
            gguf_embeddings = np.array(gguf_emb_list)
            print(f"GGUF embeddings shape: {gguf_embeddings.shape}")
            print(f"GGUF embeddings sample (first token, first 10 dims): {gguf_embeddings[0, :10]}")
    else:
        print("Warning: embedding weight not found in GGUF tensors!")
    
    # Get PyTorch embeddings
    token_tensor = torch.tensor([token_ids])
    with torch.no_grad():
        # Access embedding layer directly
        if hasattr(model.hf_model, 'model') and hasattr(model.hf_model.model, 'embed_tokens'):
            pytorch_embeddings = model.hf_model.model.embed_tokens(token_tensor)
            if isinstance(pytorch_embeddings, torch.Tensor):
                pytorch_embeddings = pytorch_embeddings[0].cpu().numpy()
            else:
                pytorch_embeddings = pytorch_embeddings[0]
            
            print(f"\nPyTorch embeddings shape: {pytorch_embeddings.shape}")
            print(f"PyTorch embeddings sample (first token, first 10 dims): {pytorch_embeddings[0, :10]}")
            
            # Compare
            if gguf_embeddings is not None and gguf_embeddings.shape == pytorch_embeddings.shape:
                metrics = compare_tensors("Embeddings", pytorch_embeddings, gguf_embeddings, verbose=True)
                return metrics
            elif gguf_embeddings is not None:
                print(f"Shape mismatch: PyTorch {pytorch_embeddings.shape} vs GGUF {gguf_embeddings.shape}")
            else:
                print("Could not extract GGUF embeddings for comparison")
    
    return {}


def investigate_full_forward_pass(
    model_path: str,
    token_ids: List[int]
) -> Dict[str, any]:
    """
    Run full forward pass and compare final logits.
    """
    print("\n" + "="*80)
    print("INVESTIGATING FULL FORWARD PASS")
    print("="*80)
    
    print("\nLoading PyTorch model...")
    model = ModelRegistry.create("qwen", model_path, auto_load=False)
    model.load_model()
    
    print(f"\nToken IDs: {token_ids}")
    token_tensor = torch.tensor([token_ids])
    
    with torch.no_grad():
        result = model.forward(token_tensor)
        pytorch_logits = result["logits"]
        
        if isinstance(pytorch_logits, torch.Tensor):
            pytorch_logits = pytorch_logits[0, -1, :].cpu().numpy()
        else:
            pytorch_logits = pytorch_logits[0, -1, :]
        
        print(f"\nPyTorch logits shape: {pytorch_logits.shape}")
        print(f"PyTorch logits range: [{pytorch_logits.min():.3f}, {pytorch_logits.max():.3f}]")
        
        # Top-5 predictions
        top5_indices = np.argsort(pytorch_logits)[-5:][::-1]
        print(f"\nPyTorch top-5 tokens: {top5_indices.tolist()}")
        print(f"PyTorch top-5 logits: {pytorch_logits[top5_indices].tolist()}")
    
    # Note: Llaminar logits would need to be loaded from JSON output
    # This function shows what we can get from PyTorch side
    
    return {"pytorch_logits": pytorch_logits}


def investigate_weight_loading(model_path: str) -> Dict[str, any]:
    """
    Compare weight loading between GGUF and PyTorch to check for quantization issues.
    """
    print("\n" + "="*80)
    print("INVESTIGATING WEIGHT LOADING")
    print("="*80)
    
    print("\nLoading GGUF file...")
    loader = GGUFLoader(model_path)
    config_dict, tensors = loader.load(as_torch=False)
    
    print("\nLoading PyTorch model...")
    model = ModelRegistry.create("qwen", model_path, auto_load=False)
    model.load_model()
    
    # Compare a few key weights
    comparisons = []
    
    # Check embedding weights
    if 'token_embd.weight' in tensors:
        gguf_emb = tensors['token_embd.weight']
        pytorch_emb = None
        
        if hasattr(model.hf_model, 'model') and hasattr(model.hf_model.model, 'embed_tokens'):
            pytorch_emb = model.hf_model.model.embed_tokens.weight.cpu().numpy()
        
        if pytorch_emb is not None:
            print(f"\n📊 Embedding weights:")
            metrics = compare_tensors("token_embd.weight", pytorch_emb, gguf_emb, verbose=True)
            comparisons.append(("embedding", metrics))
    
    # Check first layer weights
    layer_weights = [
        ('blk.0.attn_q.weight', 'layers.0.self_attn.q_proj.weight'),
        ('blk.0.attn_k.weight', 'layers.0.self_attn.k_proj.weight'),
        ('blk.0.attn_v.weight', 'layers.0.self_attn.v_proj.weight'),
        ('blk.0.attn_output.weight', 'layers.0.self_attn.o_proj.weight'),
    ]
    
    for gguf_name, pytorch_name in layer_weights:
        if gguf_name in tensors:
            gguf_weight = tensors[gguf_name]
            
            # Navigate PyTorch model structure
            pytorch_weight = model.hf_model
            for attr in pytorch_name.split('.'):
                if attr.isdigit():
                    pytorch_weight = pytorch_weight[int(attr)]
                else:
                    pytorch_weight = getattr(pytorch_weight, attr, None)
                    if pytorch_weight is None:
                        break
            
            if pytorch_weight is not None and hasattr(pytorch_weight, 'cpu'):
                pytorch_weight = pytorch_weight.cpu().numpy()
                print(f"\n📊 {gguf_name}:")
                metrics = compare_tensors(gguf_name, pytorch_weight, gguf_weight, verbose=True)
                comparisons.append((gguf_name, metrics))
    
    return {"comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description="Investigate numerical divergence between Llaminar and PyTorch")
    parser.add_argument("-m", "--model", default="models/qwen2.5-0.5b-instruct-q4_0.gguf",
                       help="Path to GGUF model file")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1639, 266, 285, 17, 10, 17, 30],
                       help="Token IDs to test with")
    parser.add_argument("--mode", choices=["embedding", "weights", "forward", "all"], default="all",
                       help="Investigation mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return 1
    
    print(f"Investigating divergence with model: {args.model}")
    print(f"Token IDs: {args.tokens}")
    
    results = {}
    
    if args.mode in ["embedding", "all"]:
        results["embedding"] = investigate_embedding_layer(args.model, args.tokens)
    
    if args.mode in ["weights", "all"]:
        results["weights"] = investigate_weight_loading(args.model)
    
    if args.mode in ["forward", "all"]:
        results["forward"] = investigate_full_forward_pass(args.model, args.tokens)
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
