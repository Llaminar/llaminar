#!/usr/bin/env python3
"""
Generate PyTorch reference snapshots for parity testing.

This script generates comprehensive stage-by-stage snapshots from PyTorch
that match Llaminar's pipeline stages for direct comparison.

@author David Sanftenberg
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

from reference import ModelRegistry


class PipelineStageCapture:
    """
    Capture PyTorch model outputs at stages matching Llaminar's pipeline.
    
    Stages captured (matching Llaminar's PipelineStage enum):
    - EMBEDDING: After token embedding
    - Per layer (for each transformer layer):
      - ATTENTION_NORM: After input norm (before attention)
      - ATTENTION_OUTPUT: After attention projection (before residual)
      - ATTENTION_RESIDUAL: After attention residual add
      - FFN_NORM: After post-attention norm (before FFN)
      - FFN_DOWN: After FFN down projection (before residual)
      - FFN_RESIDUAL: After FFN residual add
    - FINAL_NORM: After final layer norm
    - LM_HEAD: After LM head projection (logits)
    """
    
    def __init__(self, model_path: str, verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self.captures = {}
        
    def load_model(self):
        """Load PyTorch model."""
        if self.model is None:
            if self.verbose:
                print(f"Loading PyTorch model from {self.model_path}...")
            self.model = ModelRegistry.create("qwen", self.model_path)
            self.model.load_model()
            if self.verbose:
                print("✓ Model loaded")
    
    def capture_stages(self, token_ids: List[int]) -> Dict[str, np.ndarray]:
        """
        Run forward pass and capture all pipeline stages.
        
        Args:
            token_ids: Input token IDs
            
        Returns:
            Dictionary mapping stage names to numpy arrays
        """
        self.load_model()
        self.captures = {}
        
        # Use the model's forward method which handles all the complexity
        token_tensor = torch.tensor([token_ids])
        
        with torch.no_grad():
            hf_model = self.model.hf_model
            
            # EMBEDDING stage
            if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'embed_tokens'):
                hidden_states = hf_model.model.embed_tokens(token_tensor)
                self.captures['EMBEDDING'] = hidden_states.detach().cpu().numpy()
            else:
                raise RuntimeError("Cannot find embedding layer")
            
            # Create attention mask and position IDs
            seq_length = len(token_ids)
            batch_size = 1
            
            # Causal attention mask (lower triangular)
            attention_mask = torch.tril(torch.ones((batch_size, 1, seq_length, seq_length), dtype=torch.float32))
            # Convert to additive mask (0 for attend, -inf for mask)
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
            
            position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
            
            # Get RoPE embeddings once for all layers
            if hasattr(hf_model.model, 'rotary_emb'):
                # Call rotary_emb with just hidden_states (it will extract seq_len)
                cos, sin = hf_model.model.rotary_emb(hidden_states, position_ids)
                position_embeddings = (cos, sin)
            else:
                raise RuntimeError("Cannot find rotary embeddings")
            
            # Process through transformer layers
            if not hasattr(hf_model.model, 'layers'):
                raise RuntimeError("Cannot find transformer layers")
                
            layers = hf_model.model.layers
            
            for i, layer in enumerate(layers):
                # ATTENTION_NORM: After input norm
                if hasattr(layer, 'input_layernorm'):
                    normed = layer.input_layernorm(hidden_states)
                    self.captures[f'ATTENTION_NORM_layer{i}'] = normed.detach().cpu().numpy()
                else:
                    normed = hidden_states
                
                # Attention computation with detailed intermediate captures
                if hasattr(layer, 'self_attn'):
                    attn_layer = layer.self_attn
                    
                    # Get architecture parameters from config
                    config = hf_model.config
                    num_heads = config.num_attention_heads
                    num_kv_heads = config.num_key_value_heads
                    head_dim = attn_layer.head_dim
                    num_key_value_groups = attn_layer.num_key_value_groups
                    
                    # Capture Q, K, V projections (before RoPE)
                    bsz, q_len, _ = normed.shape
                    
                    # Q projection
                    query_states = attn_layer.q_proj(normed)
                    self.captures[f'Q_PROJECTION_layer{i}'] = query_states.detach().cpu().numpy()
                    
                    # K projection
                    key_states = attn_layer.k_proj(normed)
                    self.captures[f'K_PROJECTION_layer{i}'] = key_states.detach().cpu().numpy()
                    
                    # V projection
                    value_states = attn_layer.v_proj(normed)
                    self.captures[f'V_PROJECTION_layer{i}'] = value_states.detach().cpu().numpy()
                    
                    # Reshape for multi-head attention
                    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
                    
                    # Apply RoPE
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                    
                    # Capture post-RoPE Q and K combined as ROPE_APPLICATION
                    # Flatten Q and K: shape (bsz, q_len, hidden_size)
                    q_rope_flat = query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1).detach().cpu().numpy()
                    k_rope_flat = key_states.transpose(1, 2).contiguous().view(bsz, q_len, -1).detach().cpu().numpy()
                    
                    # DEBUG: Show shapes and layout
                    if i == 0:
                        print(f"[PYTORCH_ROPE_DEBUG] Layer {i}:")
                        print(f"  query_states shape after RoPE (before transpose): {query_states.shape}")
                        print(f"  q_rope_flat shape: {q_rope_flat.shape}")
                        print(f"  k_rope_flat shape: {k_rope_flat.shape}")
                        print(f"  q_rope_flat[0,0,:10]: {q_rope_flat[0,0,:10]}")
                        print(f"  k_rope_flat[0,0,:10]: {k_rope_flat[0,0,:10]}")
                        print(f"  num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
                        print(f"  Expected q_rope_flat last dim: {num_heads * head_dim}")
                        print(f"  Expected k_rope_flat last dim: {num_kv_heads * head_dim}")
                    
                    # Concatenate Q and K along feature dimension: [Q | K]
                    # This matches Llaminar's ROPE_APPLICATION which includes both
                    rope_combined = np.concatenate([q_rope_flat, k_rope_flat], axis=-1)
                    
                    if i == 0:
                        print(f"  rope_combined shape: {rope_combined.shape}")
                        print(f"  rope_combined[0,0,:10] (Q start): {rope_combined[0,0,:10]}")
                        print(f"  rope_combined[0,0,{num_heads*head_dim}:{num_heads*head_dim+10}] (K start): {rope_combined[0,0,num_heads*head_dim:num_heads*head_dim+10]}")
                    
                    self.captures[f'ROPE_APPLICATION_layer{i}'] = rope_combined
                    
                    # Repeat K/V for GQA if needed
                    key_states = repeat_kv(key_states, num_key_value_groups)
                    value_states = repeat_kv(value_states, num_key_value_groups)
                    
                    # Compute attention scores (Q @ K^T / sqrt(d))
                    sqrt_head_dim = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
                    attn_weights_unscaled = torch.matmul(query_states, key_states.transpose(2, 3))
                    attn_weights = attn_weights_unscaled / sqrt_head_dim
                    
                    # DEBUG: Verify scaling
                    if i == 0:
                        print(f"[DEBUG] Layer {i} ATTENTION_SCORES:")
                        print(f"  sqrt_head_dim: {sqrt_head_dim}")
                        print(f"  Before scaling: min={attn_weights_unscaled.min():.2f} max={attn_weights_unscaled.max():.2f}")
                        print(f"  After scaling: min={attn_weights.min():.2f} max={attn_weights.max():.2f}")
                    
                    # Capture attention scores (before softmax)
                    # Reshape from [batch=1, heads, seq_len, seq_len] to [heads*seq_len, seq_len]
                    # This matches Llaminar's flattened 2D layout for easier comparison
                    self.captures[f'ATTENTION_SCORES_layer{i}'] = attn_weights[0].reshape(-1, attn_weights.shape[-1]).detach().cpu().numpy()
                    
                    # Apply attention mask
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    
                    # Apply softmax
                    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    
                    # Capture attention weights (after softmax)
                    # Also reshape to 2D for consistency
                    self.captures[f'ATTENTION_SOFTMAX_layer{i}'] = attn_weights[0].reshape(-1, attn_weights.shape[-1]).detach().cpu().numpy()
                    
                    # Compute context (attention @ V)
                    attn_output = torch.matmul(attn_weights, value_states)
                    
                    # Reshape back
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, -1)
                    
                    # Capture context (before output projection)
                    self.captures[f'ATTENTION_CONTEXT_layer{i}'] = attn_output.detach().cpu().numpy()
                    
                    # Apply output projection
                    attn_out = attn_layer.o_proj(attn_output)
                    
                    # ATTENTION_OUTPUT: After attention projection (before residual)
                    self.captures[f'ATTENTION_OUTPUT_layer{i}'] = attn_out.detach().cpu().numpy()
                    
                    # ATTENTION_RESIDUAL: After residual add
                    hidden_states = hidden_states + attn_out
                    self.captures[f'ATTENTION_RESIDUAL_layer{i}'] = hidden_states.detach().cpu().numpy()
                else:
                    raise RuntimeError(f"Layer {i} missing self_attn")
                
                # FFN_NORM: After post-attention norm
                if hasattr(layer, 'post_attention_layernorm'):
                    normed = layer.post_attention_layernorm(hidden_states)
                    self.captures[f'FFN_NORM_layer{i}'] = normed.detach().cpu().numpy()
                else:
                    normed = hidden_states
                
                # FFN computation - break down MLP internals for detailed snapshots
                if hasattr(layer, 'mlp'):
                    # Qwen2MLP structure: gate_proj, up_proj, act_fn (silu), down_proj
                    mlp = layer.mlp
                    
                    # FFN_GATE: Gate projection output
                    if hasattr(mlp, 'gate_proj'):
                        gate_out = mlp.gate_proj(normed)
                        self.captures[f'FFN_GATE_layer{i}'] = gate_out.detach().cpu().numpy()
                    else:
                        gate_out = None
                    
                    # FFN_UP: Up projection output
                    if hasattr(mlp, 'up_proj'):
                        up_out = mlp.up_proj(normed)
                        self.captures[f'FFN_UP_layer{i}'] = up_out.detach().cpu().numpy()
                    else:
                        up_out = None
                    
                    # FFN_SWIGLU: SwiGLU activation output (silu(gate) * up)
                    if gate_out is not None and up_out is not None:
                        if hasattr(mlp, 'act_fn'):
                            activated_gate = mlp.act_fn(gate_out)
                        else:
                            # Fallback to manual silu if act_fn not available
                            import torch.nn.functional as F
                            activated_gate = F.silu(gate_out)
                        swiglu_out = activated_gate * up_out
                        self.captures[f'FFN_SWIGLU_layer{i}'] = swiglu_out.detach().cpu().numpy()
                    else:
                        # Fallback: use the full MLP forward to get intermediate
                        swiglu_out = None
                    
                    # FFN_DOWN: Down projection output
                    if swiglu_out is not None and hasattr(mlp, 'down_proj'):
                        ffn_out = mlp.down_proj(swiglu_out)
                    else:
                        # Fallback: use full MLP forward
                        ffn_out = mlp(normed)
                    
                    self.captures[f'FFN_DOWN_layer{i}'] = ffn_out.detach().cpu().numpy()
                    
                    # FFN_RESIDUAL: After residual add
                    hidden_states = hidden_states + ffn_out
                    self.captures[f'FFN_RESIDUAL_layer{i}'] = hidden_states.detach().cpu().numpy()
                else:
                    raise RuntimeError(f"Layer {i} missing mlp")
            
            # FINAL_NORM: After final layer norm
            if hasattr(hf_model.model, 'norm'):
                normed = hf_model.model.norm(hidden_states)
                self.captures['FINAL_NORM'] = normed.detach().cpu().numpy()
            else:
                normed = hidden_states
            
            # LM_HEAD: After LM head projection (logits)
            if hasattr(hf_model, 'lm_head'):
                logits = hf_model.lm_head(normed)
                self.captures['LM_HEAD'] = logits.detach().cpu().numpy()
            else:
                raise RuntimeError("Cannot find lm_head")
        
        if self.verbose:
            print(f"\n✓ Captured {len(self.captures)} pipeline stages:")
            for name in sorted(self.captures.keys()):
                shape = self.captures[name].shape
                print(f"  - {name}: {shape}")
        
        return self.captures
    
    def save_snapshots(self, output_dir: str):
        """
        Save captures to NPZ files in format expected by C++ tests.
        
        Creates:
        - <output_dir>/EMBEDDING_-1.npy
        - <output_dir>/ATTENTION_NORM_<layer>.npy
        - <output_dir>/ATTENTION_OUTPUT_<layer>.npy
        - <output_dir>/ATTENTION_RESIDUAL_<layer>.npy
        - <output_dir>/FFN_NORM_<layer>.npy
        - <output_dir>/FFN_DOWN_<layer>.npy
        - <output_dir>/FFN_RESIDUAL_<layer>.npy
        - <output_dir>/FINAL_NORM_-1.npy
        - <output_dir>/LM_HEAD_-1.npy
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\n✓ Captured {len(self.captures)} pipeline stages:")
            for name, data in sorted(self.captures.items()):
                print(f"  - {name}: {data.shape}")
        
        for name, data in self.captures.items():
            # Parse layer index from name (e.g., "ATTENTION_OUTPUT_layer5" -> 5)
            # Global stages like "EMBEDDING", "FINAL_NORM", "LM_HEAD" -> -1
            layer_index = -1
            if '_layer' in name:
                # Extract layer number from names like "ATTENTION_OUTPUT_layer5"
                parts = name.split('_layer')
                if len(parts) == 2:
                    try:
                        layer_index = int(parts[1])
                        # Remove the _layer suffix from the stage name
                        stage_name = parts[0]
                    except ValueError:
                        stage_name = name
                else:
                    stage_name = name
            else:
                stage_name = name
            
            # Format: STAGE_layer.npy (e.g., EMBEDDING_-1.npy, ATTENTION_OUTPUT_0.npy)
            filename = f"{stage_name}_{layer_index}.npy"
            output_path = os.path.join(output_dir, filename)
            
            # Save as .npy (not .npz) to match C++ loader expectations
            np.save(output_path, data)
        
        if self.verbose:
            print(f"\n✓ Saved {len(self.captures)} snapshot files to {output_dir}/")
    
    def cleanup(self):
        """Clean up model and captures."""
        self.captures.clear()
        self.model = None


def main():
    """Main entry point for snapshot generation."""
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference snapshots for parity testing"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--tokens",
        required=True,
        help="Comma-separated token IDs (e.g., '1,2,3,4,5')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for snapshot NPZ files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}", file=sys.stderr)
        return 1
    
    # Parse token IDs
    try:
        token_ids = [int(t.strip()) for t in args.tokens.split(',')]
    except ValueError as e:
        print(f"Error: Invalid token IDs: {e}", file=sys.stderr)
        return 1
    
    if not token_ids:
        print("Error: Empty token sequence", file=sys.stderr)
        return 1
    
    if args.verbose:
        print(f"Model: {args.model}")
        print(f"Tokens: {token_ids} ({len(token_ids)} tokens)")
        print(f"Output: {args.output_dir}/")
        print()
    
    # Generate snapshots
    try:
        capturer = PipelineStageCapture(args.model, verbose=args.verbose)
        capturer.capture_stages(token_ids)
        capturer.save_snapshots(args.output_dir)
        capturer.cleanup()
        
        if args.verbose:
            print("\n✓ Snapshot generation complete!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
