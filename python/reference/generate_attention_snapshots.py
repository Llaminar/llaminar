#!/usr/bin/env python3
"""
Generate PyTorch attention reference snapshots for V2 parity testing.

This script generates minimal attention-focused snapshots from PyTorch
for validating Llaminar V2's FP32 and INT8 attention kernels.

Unlike the comprehensive V1 framework, this captures only attention-specific
intermediate states needed for kernel validation.

@author David Sanftenberg
@date 2025-11-06
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)


class AttentionSnapshotCapture:
    """
    Capture PyTorch attention intermediate states for V2 kernel validation.
    
    Stages captured (single layer):
    - Q_projection: Query projection output (before RoPE)
    - K_projection: Key projection output (before RoPE)  
    - V_projection: Value projection output
    - Q_rope: Query after RoPE
    - K_rope: Key after RoPE
    - attention_scores: Q @ K^T / sqrt(d_head) (pre-softmax)
    - attention_weights: Softmax(scores) (attention probabilities)
    - attention_output: attn_weights @ V (pre-projection)
    """
    
    def __init__(self, model_path: str, layer_idx: int = 0, verbose: bool = False):
        """
        Args:
            model_path: Path to HuggingFace model or GGUF file
            layer_idx: Which transformer layer to capture (default: 0)
            verbose: Enable detailed logging
        """
        self.model_path = model_path
        self.layer_idx = layer_idx
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.captures = {}
        
    def load_model(self):
        """Load PyTorch model from HuggingFace."""
        if self.model is None:
            if self.verbose:
                print(f"Loading model from {self.model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # FP32 for ground truth
                device_map="cpu"
            )
            self.model.eval()
            
            if self.verbose:
                config = self.model.config
                print(f"✓ Model loaded:")
                print(f"  n_layers: {config.num_hidden_layers}")
                print(f"  n_heads: {config.num_attention_heads}")
                print(f"  n_kv_heads: {config.num_key_value_heads}")
                print(f"  d_model: {config.hidden_size}")
                print(f"  d_head: {config.hidden_size // config.num_attention_heads}")
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE (Rotary Position Embeddings).
        
        Args:
            q: Query tensor [batch, n_heads, seq_len, d_head]
            k: Key tensor [batch, n_kv_heads, seq_len, d_head]
            position_ids: Position IDs [batch, seq_len]
            
        Returns:
            (q_rope, k_rope) with RoPE applied
        """
        # Get RoPE embeddings from model
        layer = self.model.model.layers[self.layer_idx]
        
        # Qwen2's RoPE uses (cos, sin) embeddings
        if hasattr(self.model.model, 'rotary_emb'):
            # Create dummy hidden_states to get cos/sin
            bsz, n_heads, seq_len, d_head = q.shape
            dummy_hidden = torch.zeros(bsz, seq_len, self.model.config.hidden_size)
            cos, sin = self.model.model.rotary_emb(dummy_hidden, position_ids)
            
            # Apply rotary embeddings
            # NOTE: transformers 5.x changed apply_rotary_pos_emb signature:
            #   Old (4.x): apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            #   New (5.x): apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            # In 5.x, cos/sin are already position-indexed by rotary_emb().
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
            
            return q_rope, k_rope
        else:
            raise RuntimeError("Cannot find rotary embeddings")
    
    def capture_attention_forward(
        self,
        prompt: str,
        use_causal_mask: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run single-layer attention forward pass and capture all intermediate states.
        
        Args:
            prompt: Input text prompt
            use_causal_mask: Apply causal masking (for autoregressive)
            
        Returns:
            Dictionary mapping stage names to numpy arrays:
            - 'input': Attention input (after pre-norm) [batch, seq_len, d_model]
            - 'Q_projection': Q projection [batch, seq_len, n_heads * d_head]
            - 'K_projection': K projection [batch, seq_len, n_kv_heads * d_head]
            - 'V_projection': V projection [batch, seq_len, n_kv_heads * d_head]
            - 'Q_rope': Q after RoPE [batch, seq_len, n_heads * d_head]
            - 'K_rope': K after RoPE [batch, seq_len, n_kv_heads * d_head]
            - 'attention_scores': Q @ K^T / sqrt(d_head) [batch, n_heads, seq_len, seq_len]
            - 'attention_weights': Softmax(scores) [batch, n_heads, seq_len, seq_len]
            - 'attention_output': Final output [batch, seq_len, d_model]
        """
        self.load_model()
        self.captures = {}
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        batch_size = 1
        
        if self.verbose:
            print(f"\nProcessing prompt: '{prompt}'")
            print(f"Tokens: {input_ids.tolist()[0]}")
            print(f"Sequence length: {seq_len}")
        
        with torch.no_grad():
            # Get embeddings
            hidden_states = self.model.model.embed_tokens(input_ids)
            
            # Process through layers up to target layer
            for i in range(self.layer_idx + 1):
                layer = self.model.model.layers[i]
                
                if i < self.layer_idx:
                    # Skip layers before target
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                    attn_output = layer.self_attn(
                        hidden_states,
                        attention_mask=None,
                        position_ids=torch.arange(seq_len).unsqueeze(0)
                    )[0]
                    hidden_states = residual + attn_output
                    
                    residual = hidden_states
                    hidden_states = layer.post_attention_layernorm(hidden_states)
                    mlp_output = layer.mlp(hidden_states)
                    hidden_states = residual + mlp_output
                    continue
                
                # Target layer - capture everything
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                
                # Capture input (post-norm)
                self.captures['input'] = hidden_states.numpy()
                
                attn = layer.self_attn
                config = self.model.config
                n_heads = config.num_attention_heads
                n_kv_heads = config.num_key_value_heads
                d_head = config.hidden_size // n_heads
                
                # Q, K, V projections
                bsz, q_len, _ = hidden_states.shape
                
                Q = attn.q_proj(hidden_states)  # [bsz, q_len, n_heads * d_head]
                K = attn.k_proj(hidden_states)  # [bsz, q_len, n_kv_heads * d_head]
                V = attn.v_proj(hidden_states)  # [bsz, q_len, n_kv_heads * d_head]
                
                self.captures['Q_projection'] = Q.numpy()
                self.captures['K_projection'] = K.numpy()
                self.captures['V_projection'] = V.numpy()
                
                # Reshape for multi-head
                Q_heads = Q.view(bsz, q_len, n_heads, d_head).transpose(1, 2)
                K_heads = K.view(bsz, q_len, n_kv_heads, d_head).transpose(1, 2)
                V_heads = V.view(bsz, q_len, n_kv_heads, d_head).transpose(1, 2)
                
                # Apply RoPE
                position_ids = torch.arange(q_len).unsqueeze(0)
                Q_rope, K_rope = self._apply_rope(Q_heads, K_heads, position_ids)
                
                # Flatten back to [bsz, q_len, hidden]
                Q_rope_flat = Q_rope.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                K_rope_flat = K_rope.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                
                self.captures['Q_rope'] = Q_rope_flat.numpy()
                self.captures['K_rope'] = K_rope_flat.numpy()
                
                # Expand K/V for GQA if needed
                num_key_value_groups = n_heads // n_kv_heads
                if num_key_value_groups > 1:
                    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
                    K_rope = repeat_kv(K_rope, num_key_value_groups)
                    V_heads = repeat_kv(V_heads, num_key_value_groups)
                
                # Compute attention scores: Q @ K^T / sqrt(d_head)
                scores = torch.matmul(Q_rope, K_rope.transpose(2, 3)) / torch.sqrt(
                    torch.tensor(d_head, dtype=torch.float32)
                )
                
                self.captures['attention_scores'] = scores.numpy()
                
                # Apply causal mask if requested
                if use_causal_mask:
                    causal_mask = torch.tril(torch.ones(q_len, q_len)).unsqueeze(0).unsqueeze(0)
                    scores = scores.masked_fill(causal_mask == 0, float('-inf'))
                
                # Apply softmax
                attn_weights = F.softmax(scores, dim=-1)
                
                self.captures['attention_weights'] = attn_weights.numpy()
                
                # Compute context: attn_weights @ V
                context = torch.matmul(attn_weights, V_heads)
                
                # Reshape back: [bsz, n_heads, q_len, d_head] -> [bsz, q_len, hidden]
                context = context.transpose(1, 2).contiguous().view(bsz, q_len, -1)
                
                # Save context before output projection
                self.captures['attention_context'] = context.numpy()
                
                # Output projection
                output = attn.o_proj(context)
                
                self.captures['attention_output'] = output.numpy()
                
                if self.verbose:
                    print(f"\n✓ Captured {len(self.captures)} attention stages:")
                    for name, arr in self.captures.items():
                        print(f"  {name:20s}: shape {arr.shape}, dtype {arr.dtype}")
        
        return self.captures
    
    def save_snapshots(self, output_dir: str):
        """Save all captured snapshots as .npy files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, arr in self.captures.items():
            filepath = output_path / f"{name}.npy"
            np.save(filepath, arr)
            if self.verbose:
                print(f"Saved: {filepath}")
        
        if self.verbose:
            print(f"\n✓ Saved {len(self.captures)} snapshots to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch attention reference snapshots")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello world",
        help="Input prompt for attention"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Transformer layer to capture (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pytorch_attention_snapshots",
        help="Output directory for .npy files"
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Apply causal masking"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create snapshot capturer
    capturer = AttentionSnapshotCapture(
        model_path=args.model,
        layer_idx=args.layer,
        verbose=args.verbose
    )
    
    # Capture attention forward pass
    print(f"Capturing attention snapshots from layer {args.layer}...")
    capturer.capture_attention_forward(
        prompt=args.prompt,
        use_causal_mask=args.causal
    )
    
    # Save to disk
    capturer.save_snapshots(args.output)
    
    print(f"\n✅ Done! Snapshots saved to: {args.output}/")
    print(f"   Use these in C++ tests with NpzLoader::load_npy()")


if __name__ == "__main__":
    main()
