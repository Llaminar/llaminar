#!/usr/bin/env python3
"""
Layer-by-layer PyTorch reference capture for parity testing.

This module enables capturing intermediate layer outputs from PyTorch models
to compare against Llaminar's C++ implementation for debugging divergence.

@author David Sanftenberg
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from collections import OrderedDict

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

# Add to path if not already there
for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

# Now import from reference module
from reference import ModelRegistry


class LayerCaptureHook:
    """Hook to capture intermediate layer outputs from PyTorch model."""
    
    def __init__(self):
        self.captures = OrderedDict()
        self.hooks = []
        
    def register_hook(self, module, name):
        """Register a forward hook on a module."""
        def hook_fn(module, input, output):
            # Convert to numpy and store
            if isinstance(output, torch.Tensor):
                self.captures[name] = output.detach().cpu().numpy()
            elif isinstance(output, tuple) and len(output) > 0:
                # For models that return tuples, capture first element
                self.captures[name] = output[0].detach().cpu().numpy()
        
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle
    
    def clear(self):
        """Clear all captures and remove hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.captures.clear()
    
    def get_captures(self) -> Dict[str, np.ndarray]:
        """Get all captured tensors."""
        return dict(self.captures)


class QwenLayerCapture:
    """
    Capture layer-by-layer outputs from Qwen PyTorch model.
    
    Captures:
    - Input embeddings
    - Each layer's:
      - Attention input (after input norm)
      - Attention output (before residual)
      - FFN input (after post-attention norm)
      - FFN output (before residual)
      - Layer output (after both residuals)
    - Final norm output
    - Logits
    """
    
    def __init__(self, model_path: str):
        """Initialize with GGUF model path."""
        self.model_path = model_path
        self.model = None
        self.hook_manager = LayerCaptureHook()
        
    def load_model(self):
        """Load PyTorch model."""
        if self.model is None:
            print(f"Loading PyTorch model from {self.model_path}...")
            self.model = ModelRegistry.create("qwen", self.model_path)
            self.model.load_model()
            print("✓ Model loaded")
    
    def setup_hooks(self):
        """Set up hooks to capture all layer outputs."""
        self.load_model()
        self.hook_manager.clear()
        
        # Access the underlying HF model
        if not hasattr(self.model, 'hf_model') or self.model.hf_model is None:
            raise RuntimeError("Model does not have hf_model attribute")
            
        hf_model = self.model.hf_model
        
        # Hook embedding layer
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'embed_tokens'):
            self.hook_manager.register_hook(
                hf_model.model.embed_tokens, 
                'embeddings'
            )
        
        # Hook each transformer layer
        if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            layers = hf_model.model.layers
            for i, layer in enumerate(layers):
                # Input LayerNorm (before attention)
                if hasattr(layer, 'input_layernorm'):
                    self.hook_manager.register_hook(
                        layer.input_layernorm,
                        f'layer_{i}_input_norm_out'
                    )
                
                # Self-attention
                if hasattr(layer, 'self_attn'):
                    self.hook_manager.register_hook(
                        layer.self_attn,
                        f'layer_{i}_attn_out'
                    )
                
                # Post-attention LayerNorm (before FFN)
                if hasattr(layer, 'post_attention_layernorm'):
                    self.hook_manager.register_hook(
                        layer.post_attention_layernorm,
                        f'layer_{i}_post_attn_norm_out'
                    )
                
                # MLP/FFN
                if hasattr(layer, 'mlp'):
                    self.hook_manager.register_hook(
                        layer.mlp,
                        f'layer_{i}_ffn_out'
                    )
                
                # Entire layer output
                self.hook_manager.register_hook(
                    layer,
                    f'layer_{i}_out'
                )
        
        # Final norm
        if hasattr(hf_model.model, 'norm'):
            self.hook_manager.register_hook(
                hf_model.model.norm,
                'final_norm_out'
            )
        
        # LM head (logits)
        if hasattr(hf_model, 'lm_head'):
            self.hook_manager.register_hook(
                hf_model.lm_head,
                'logits'
            )
        
        print(f"✓ Registered {len(self.hook_manager.hooks)} hooks")
    
    def capture_forward_pass(self, token_ids: List[int]) -> Dict[str, np.ndarray]:
        """
        Run forward pass and capture all intermediate outputs.
        
        Args:
            token_ids: Input token IDs
            
        Returns:
            Dictionary mapping layer names to numpy arrays
        """
        self.load_model()
        self.setup_hooks()
        
        # Run forward pass
        token_tensor = torch.tensor([token_ids])
        with torch.no_grad():
            _ = self.model.forward(token_tensor)
        
        # Get captures
        captures = self.hook_manager.get_captures()
        
        print(f"\n✓ Captured {len(captures)} layer outputs:")
        for name in captures.keys():
            shape = captures[name].shape
            print(f"  - {name}: {shape}")
        
        return captures
    
    def save_captures_npz(self, captures: Dict[str, np.ndarray], output_path: str):
        """Save captures to NPZ file for C++ comparison."""
        np.savez_compressed(output_path, **captures)
        print(f"\n✓ Saved captures to {output_path}")
        
        # Print file info
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
    
    def cleanup(self):
        """Clean up hooks."""
        self.hook_manager.clear()


def compare_layer_outputs(
    pytorch_captures: Dict[str, np.ndarray],
    llaminar_captures: Dict[str, np.ndarray],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare layer outputs between PyTorch and Llaminar.
    
    Args:
        pytorch_captures: Dictionary of PyTorch layer outputs
        llaminar_captures: Dictionary of Llaminar layer outputs  
        verbose: Print detailed comparison
        
    Returns:
        Dictionary of metrics for each layer
    """
    results = {}
    
    # Find common keys
    pt_keys = set(pytorch_captures.keys())
    ll_keys = set(llaminar_captures.keys())
    common_keys = pt_keys & ll_keys
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"LAYER-BY-LAYER COMPARISON")
        print(f"{'='*80}")
        print(f"PyTorch layers: {len(pt_keys)}")
        print(f"Llaminar layers: {len(ll_keys)}")
        print(f"Common layers: {len(common_keys)}")
        
        if pt_keys - ll_keys:
            print(f"\n⚠ Only in PyTorch: {pt_keys - ll_keys}")
        if ll_keys - pt_keys:
            print(f"\n⚠ Only in Llaminar: {ll_keys - pt_keys}")
    
    # Compare each common layer
    for key in sorted(common_keys):
        pt_data = pytorch_captures[key]
        ll_data = llaminar_captures[key]
        
        # Handle shape mismatches
        if pt_data.shape != ll_data.shape:
            if verbose:
                print(f"\n❌ {key}: Shape mismatch!")
                print(f"   PyTorch: {pt_data.shape}")
                print(f"   Llaminar: {ll_data.shape}")
            results[key] = {
                'error': 'shape_mismatch',
                'pt_shape': pt_data.shape,
                'll_shape': ll_data.shape
            }
            continue
        
        # Flatten for comparison
        pt_flat = pt_data.flatten()
        ll_flat = ll_data.flatten()
        
        # Compute metrics
        abs_diff = np.abs(pt_flat - ll_flat)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        # Relative error
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
        
        metrics = {
            'max_abs_diff': float(max_abs_diff),
            'mean_abs_diff': float(mean_abs_diff),
            'mean_rel_error': float(mean_rel_error),
            'max_rel_error': float(max_rel_error),
            'cosine_similarity': float(cosine_sim)
        }
        
        results[key] = metrics
        
        # Print if verbose or if there's divergence
        if verbose or mean_abs_diff > 0.01:
            status = "✓" if mean_abs_diff < 0.001 else "⚠" if mean_abs_diff < 0.1 else "❌"
            print(f"\n{status} {key}:")
            print(f"   Shape: {pt_data.shape}")
            print(f"   Max abs diff: {max_abs_diff:.6f}")
            print(f"   Mean abs diff: {mean_abs_diff:.6f}")
            print(f"   Mean rel error: {mean_rel_error:.6f}")
            print(f"   Cosine similarity: {cosine_sim:.6f}")
            
            # If this is the first diverging layer, highlight it
            if mean_abs_diff > 0.1:
                print(f"\n   🔍 DIVERGENCE DETECTED!")
                worst_idx = np.argmax(abs_diff)
                print(f"   Worst element: idx={worst_idx}")
                print(f"     PyTorch: {pt_flat[worst_idx]:.6f}")
                print(f"     Llaminar: {ll_flat[worst_idx]:.6f}")
                print(f"     Diff: {abs_diff[worst_idx]:.6f}")
    
    return results


def main():
    """Main entry point for layer-by-layer capture and comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Capture and compare layer-by-layer outputs between PyTorch and Llaminar"
    )
    parser.add_argument(
        "-m", "--model",
        default="models/qwen2.5-0.5b-instruct-q4_0.gguf",
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1639, 266, 285, 17, 10, 17, 30],
        help="Token IDs to test with"
    )
    parser.add_argument(
        "-o", "--output",
        default="pytorch_layer_captures.npz",
        help="Output NPZ file for PyTorch captures"
    )
    parser.add_argument(
        "--llaminar-captures",
        help="NPZ file with Llaminar captures for comparison"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return 1
    
    print(f"Model: {args.model}")
    print(f"Tokens: {args.tokens}")
    print()
    
    # Capture PyTorch layers
    capturer = QwenLayerCapture(args.model)
    pytorch_captures = capturer.capture_forward_pass(args.tokens)
    capturer.save_captures_npz(pytorch_captures, args.output)
    capturer.cleanup()
    
    # If Llaminar captures provided, compare
    if args.llaminar_captures:
        if not os.path.exists(args.llaminar_captures):
            print(f"\nWarning: Llaminar captures not found at {args.llaminar_captures}")
        else:
            print(f"\nLoading Llaminar captures from {args.llaminar_captures}...")
            llaminar_data = np.load(args.llaminar_captures)
            llaminar_captures = {k: llaminar_data[k] for k in llaminar_data.files}
            
            # Compare
            results = compare_layer_outputs(pytorch_captures, llaminar_captures, verbose=True)
            
            # Find first diverging layer
            diverging_layers = []
            for key, metrics in results.items():
                if 'error' not in metrics and metrics['mean_abs_diff'] > 0.1:
                    diverging_layers.append((key, metrics['mean_abs_diff']))
            
            if diverging_layers:
                diverging_layers.sort(key=lambda x: x[0])  # Sort by layer name
                print(f"\n{'='*80}")
                print(f"FIRST DIVERGING LAYER: {diverging_layers[0][0]}")
                print(f"Mean abs diff: {diverging_layers[0][1]:.6f}")
                print(f"{'='*80}")
            else:
                print(f"\n✓ All layers match within tolerance!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
