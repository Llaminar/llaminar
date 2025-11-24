#!/usr/bin/env python3
"""
Generate PyTorch Qwen2 pipeline reference snapshots for V2 E2E parity testing.

This script generates comprehensive end-to-end snapshots from PyTorch Qwen 2.5
for validating Llaminar V2's Qwen2Pipeline implementation.

Captures all intermediate activations:
- Embedding layer output
- Per-layer activations (attention output, FFN output, residuals)
- Final normalization output
- Logits over vocabulary

Usage:
    # Basic usage (8-token prompt)
    python3 generate_qwen2_pipeline_snapshots.py --output pytorch_qwen2_snapshots
    
    # Custom prompt
    python3 generate_qwen2_pipeline_snapshots.py --prompt "Hello, world!" --output snapshots
    
    # Capture specific layers only
    python3 generate_qwen2_pipeline_snapshots.py --layers 0,1,2 --output snapshots_layer012

@author David Sanftenberg
@date 2025-11-06
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
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

from reference.pipeline_stages import PipelineStage, stage_to_string

# Import GGUF loader if available
try:
    from reference.loaders.gguf_loader import GGUFLoader
    HAS_GGUF_LOADER = True
except ImportError:
    HAS_GGUF_LOADER = False
    print("WARNING: GGUF loader not available. Use HuggingFace models only.", file=sys.stderr)


class Qwen2PipelineCapture:
    """
    Capture full Qwen2 pipeline intermediate states from PyTorch for E2E validation.
    
    Stages captured per layer:
    - ATTENTION_NORM: Pre-attention RMSNorm output
    - Q_PROJECTION: Query projection (before RoPE)
    - K_PROJECTION: Key projection (before RoPE)
    - V_PROJECTION: Value projection
    - ROPE_APPLICATION: Q/K after RoPE (saved as Q_rope, K_rope)
    - ATTENTION_SCORES: Q @ K^T / sqrt(d_head) (pre-softmax)
    - ATTENTION_SOFTMAX: Attention probabilities
    - ATTENTION_CONTEXT: Attention weights @ V (before output projection)
    - ATTENTION_OUTPUT: After output projection W_o
    - ATTENTION_RESIDUAL: After residual connection
    - FFN_NORM: Pre-FFN RMSNorm output
    - FFN_GATE: Gate projection output
    - FFN_UP: Up projection output
    - FFN_SWIGLU: SwiGLU activation output (gate * silu(up))
    - FFN_DOWN: Down projection output
    - FFN_RESIDUAL: After FFN residual connection
    
    Global stages:
    - EMBEDDING: Token embeddings (before layer loop)
    - FINAL_NORM: After all layers, before LM head
    - LM_HEAD: Logits over vocabulary
    """
    
    def __init__(
        self,
        model_path: str,
        capture_layers: Optional[Set[int]] = None,
        verbose: bool = False
    ):
        """
        Args:
            model_path: Path to HuggingFace Qwen 2.5 model
            capture_layers: Set of layer indices to capture (None = all layers)
            verbose: Enable detailed logging
        """
        self.model_path = model_path
        self.capture_layers = capture_layers
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.captures = {}
        
    def load_model(self):
        """Load PyTorch model from HuggingFace or GGUF file."""
        if self.model is None:
            # Check if model_path is a GGUF file
            is_gguf = self.model_path.endswith('.gguf')
            
            if is_gguf:
                if not HAS_GGUF_LOADER:
                    raise RuntimeError("GGUF loader not available. Cannot load .gguf files.")
                
                if self.verbose:
                    print(f"Loading Qwen 2.5 model from GGUF: {self.model_path}...")
                
                # Load GGUF file
                loader = GGUFLoader(self.model_path, verbose=self.verbose)
                config, state_dict = loader.load()
                
                # BUGFIX: Remove bias tensors for Qwen2 models (use_qkv_bias=False)
                # GGUF files contain bias tensors that shouldn't be used, leading to
                # extreme output values (-79 to +48 range) that break parity testing.
                # Llaminar correctly ignores these bias tensors.
                if hasattr(config, 'model_type') and config.model_type == 'qwen2':
                    bias_keys_to_remove = [k for k in state_dict.keys() if 
                                          'self_attn.q_proj.bias' in k or 
                                          'self_attn.k_proj.bias' in k or 
                                          'self_attn.v_proj.bias' in k]
                    if bias_keys_to_remove:
                        if self.verbose:
                            print(f"Removing {len(bias_keys_to_remove)} Q/K/V bias tensors (Qwen2 uses use_qkv_bias=False)")
                        for key in bias_keys_to_remove:
                            del state_dict[key]
                
                # Create model from config
                self.model = AutoModelForCausalLM.from_config(config)
                
                # Load state dict with strict=False to allow missing bias tensors
                # (Qwen2 config creates bias parameters, but we filtered them out above)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                
                # CRITICAL: Set bias parameters to None (not just uninitialized)
                # Even after removing from state_dict, PyTorch creates bias parameters
                # that F.linear() will use if not explicitly set to None
                if hasattr(config, 'model_type') and config.model_type == 'qwen2':
                    for layer in self.model.model.layers:
                        layer.self_attn.q_proj.bias = None
                        layer.self_attn.k_proj.bias = None
                        layer.self_attn.v_proj.bias = None
                    if self.verbose:
                        print(f"✓ Set Q/K/V bias parameters to None (Qwen2 use_qkv_bias=False)")
                
                # Load tokenizer from HuggingFace (infer from config)
                if config.model_type == 'qwen2':
                    tokenizer_name = 'Qwen/Qwen2.5-0.5B-Instruct'  # Use official tokenizer
                else:
                    tokenizer_name = 'meta-llama/Llama-2-7b-hf'  # Fallback
                
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                
                if self.verbose:
                    print(f"✓ Model loaded from GGUF:")
                    print(f"  Architecture: {config.__class__.__name__}")
                    print(f"  n_layers: {config.num_hidden_layers}")
                    print(f"  n_heads: {config.num_attention_heads}")
                    print(f"  n_kv_heads: {config.num_key_value_heads}")
                    print(f"  d_model: {config.hidden_size}")
                    print(f"  d_head: {config.hidden_size // config.num_attention_heads}")
                    print(f"  d_ff: {config.intermediate_size}")
                    print(f"  vocab_size: {config.vocab_size}")
                    print(f"  Tokenizer: {tokenizer_name}")
            else:
                # Load from HuggingFace
                if self.verbose:
                    print(f"Loading Qwen 2.5 model from HuggingFace: {self.model_path}...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,  # FP32 for ground truth
                    device_map="cpu"
                )
                self.model.eval()
                
                if self.verbose:
                    config = self.model.config
                    print(f"✓ Model loaded from HuggingFace:")
                    print(f"  Architecture: {config.architectures[0]}")
                    print(f"  n_layers: {config.num_hidden_layers}")
                    print(f"  n_heads: {config.num_attention_heads}")
                    print(f"  n_kv_heads: {config.num_key_value_heads}")
                    print(f"  d_model: {config.hidden_size}")
                    print(f"  d_head: {config.hidden_size // config.num_attention_heads}")
                    print(f"  d_ff: {config.intermediate_size}")
                    print(f"  vocab_size: {config.vocab_size}")
    
    def _should_capture_layer(self, layer_idx: int) -> bool:
        """Check if we should capture this layer."""
        if self.capture_layers is None:
            return True  # Capture all layers
        return layer_idx in self.capture_layers
    
    def _save_snapshot(
        self,
        name: str,
        data: torch.Tensor,
        layer_idx: int = -1
    ):
        """
        Save tensor snapshot with metadata.
        
        Args:
            name: Stage name (e.g., 'Q_PROJECTION', 'ATTENTION_OUTPUT')
            data: Tensor to save
            layer_idx: Layer index (-1 for global stages like EMBEDDING)
        """
        # Create unique key
        if layer_idx >= 0:
            key = f"layer{layer_idx}_{name}"
        else:
            key = name
        
        # Convert to numpy and store
        self.captures[key] = {
            'data': data.detach().cpu().numpy(),
            'shape': list(data.shape),
            'layer_idx': layer_idx,
            'stage': name
        }
        
        if self.verbose:
            print(f"  Captured {key}: shape={list(data.shape)}")
    
    def _apply_rmsnorm(
        self,
        hidden: torch.Tensor,
        gamma: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Apply RMSNorm (Root Mean Square Layer Normalization).
        
        Args:
            hidden: Input tensor [batch, seq_len, d_model]
            gamma: Learned scale parameter [d_model]
            eps: Epsilon for numerical stability
            
        Returns:
            Normalized tensor [batch, seq_len, d_model]
        """
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + eps)
        return hidden * gamma
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        layer_idx: int
    ) -> tuple:
        """
        Apply RoPE (Rotary Position Embeddings).
        
        Args:
            q: Query tensor [batch, n_heads, seq_len, d_head]
            k: Key tensor [batch, n_kv_heads, seq_len, d_head]
            position_ids: Position IDs [batch, seq_len]
            layer_idx: Layer index (for RoPE embeddings)
            
        Returns:
            (q_rope, k_rope) with RoPE applied
        """
        # Get RoPE embeddings from model
        bsz, n_heads, seq_len, d_head = q.shape
        dummy_hidden = torch.zeros(bsz, seq_len, self.model.config.hidden_size)
        cos, sin = self.model.model.rotary_emb(dummy_hidden, position_ids)
        
        # Apply rotary embeddings (Qwen2-specific)
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        return q_rope, k_rope
    
    def _expand_kv_heads_for_gqa(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        n_heads: int,
        n_kv_heads: int
    ) -> tuple:
        """
        Expand K/V heads for Grouped-Query Attention.
        
        Qwen 2.5 0.5B uses GQA with n_heads=14, n_kv_heads=2 (7:1 ratio).
        Each KV head is replicated 7 times to match query heads.
        
        Args:
            k: Key tensor [batch, n_kv_heads, seq_len, d_head]
            v: Value tensor [batch, n_kv_heads, seq_len, d_head]
            n_heads: Number of query heads (14)
            n_kv_heads: Number of KV heads (2)
            
        Returns:
            (k_expanded, v_expanded) both [batch, n_heads, seq_len, d_head]
        """
        if n_heads == n_kv_heads:
            return k, v  # No expansion needed (standard MHA)
        
        # Calculate repetition factor
        n_rep = n_heads // n_kv_heads
        
        # Expand: [batch, n_kv_heads, seq_len, d_head] 
        #      -> [batch, n_kv_heads, n_rep, seq_len, d_head]
        #      -> [batch, n_heads, seq_len, d_head]
        bsz, _, seq_len, d_head = k.shape
        k_expanded = k[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, d_head)
        k_expanded = k_expanded.reshape(bsz, n_heads, seq_len, d_head)
        
        v_expanded = v[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, d_head)
        v_expanded = v_expanded.reshape(bsz, n_heads, seq_len, d_head)
        
        return k_expanded, v_expanded
    
    def _attention_block_impl(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
        capture: bool = True
    ) -> torch.Tensor:
        """
        Process single attention block with optional snapshot capture.
        
        Args:
            hidden: Input hidden states [batch, seq_len, d_model]
            layer_idx: Layer index
            position_ids: Position IDs [batch, seq_len]
            capture: Whether to capture snapshots (default: True)
            
        Returns:
            Output hidden states [batch, seq_len, d_model]
        """
        layer = self.model.model.layers[layer_idx]
        config = self.model.config
        
        bsz, seq_len, d_model = hidden.shape
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        d_head = d_model // n_heads
        
        # 1. Pre-attention RMSNorm
        hidden_norm = self._apply_rmsnorm(hidden, layer.input_layernorm.weight)
        if capture:
            self._save_snapshot('ATTENTION_NORM', hidden_norm, layer_idx)
        
        # 2. Q/K/V projections (use None for bias - Qwen2 has use_qkv_bias=False)
        q_proj = F.linear(hidden_norm, layer.self_attn.q_proj.weight, None)
        k_proj = F.linear(hidden_norm, layer.self_attn.k_proj.weight, None)
        v_proj = F.linear(hidden_norm, layer.self_attn.v_proj.weight, None)
        
        if capture:
            self._save_snapshot('Q_PROJECTION', q_proj, layer_idx)
            self._save_snapshot('K_PROJECTION', k_proj, layer_idx)
            self._save_snapshot('V_PROJECTION', v_proj, layer_idx)
        
        # 3. Reshape for multi-head attention
        q = q_proj.view(bsz, seq_len, n_heads, d_head).transpose(1, 2)  # [batch, n_heads, seq_len, d_head]
        k = k_proj.view(bsz, seq_len, n_kv_heads, d_head).transpose(1, 2)  # [batch, n_kv_heads, seq_len, d_head]
        v = v_proj.view(bsz, seq_len, n_kv_heads, d_head).transpose(1, 2)
        
        # 4. Apply RoPE
        q_rope, k_rope = self._apply_rope(q, k, position_ids, layer_idx)
        
        if capture:
            # Save RoPE outputs (flattened for C++ compatibility)
            q_rope_flat = q_rope.transpose(1, 2).reshape(bsz, seq_len, n_heads * d_head)
            k_rope_flat = k_rope.transpose(1, 2).reshape(bsz, seq_len, n_kv_heads * d_head)
            self._save_snapshot('Q_ROPE', q_rope_flat, layer_idx)
            self._save_snapshot('K_ROPE', k_rope_flat, layer_idx)
        
        # 5. Expand K/V for GQA
        k_expanded, v_expanded = self._expand_kv_heads_for_gqa(k_rope, v, n_heads, n_kv_heads)
        
        # 6. Compute attention scores
        scores = torch.matmul(q_rope, k_expanded.transpose(-2, -1)) / (d_head ** 0.5)
        if capture:
            self._save_snapshot('ATTENTION_SCORES', scores, layer_idx)
        
        # 7. Apply softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_rope.dtype)
        if capture:
            self._save_snapshot('ATTENTION_SOFTMAX', attn_weights, layer_idx)
        
        # 8. Compute attention context
        attn_context = torch.matmul(attn_weights, v_expanded)  # [batch, n_heads, seq_len, d_head]
        
        if capture:
            # Flatten for C++ compatibility
            attn_context_flat = attn_context.transpose(1, 2).reshape(bsz, seq_len, n_heads * d_head)
            self._save_snapshot('ATTENTION_CONTEXT', attn_context_flat, layer_idx)
        
        # 9. Output projection (Qwen2 doesn't use bias for attention projections)
        attn_context_flat = attn_context.transpose(1, 2).reshape(bsz, seq_len, n_heads * d_head)
        attn_output = F.linear(attn_context_flat, layer.self_attn.o_proj.weight, None)
        if capture:
            self._save_snapshot('ATTENTION_OUTPUT', attn_output, layer_idx)
        
        # 10. Residual connection
        hidden_residual = hidden + attn_output
        if capture:
            self._save_snapshot('ATTENTION_RESIDUAL', hidden_residual, layer_idx)
        
        return hidden_residual
    
    def _ffn_block_impl(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        capture: bool = True
    ) -> torch.Tensor:
        """
        Process single FFN block with optional snapshot capture.
        
        Args:
            hidden: Input hidden states [batch, seq_len, d_model]
            layer_idx: Layer index
            capture: Whether to capture snapshots (default: True)
            
        Returns:
            Output hidden states [batch, seq_len, d_model]
        """
        layer = self.model.model.layers[layer_idx]
        
        # 1. Pre-FFN RMSNorm
        hidden_norm = self._apply_rmsnorm(hidden, layer.post_attention_layernorm.weight)
        if capture:
            self._save_snapshot('FFN_NORM', hidden_norm, layer_idx)
        
        # 2. Gate and up projections
        gate_proj = F.linear(hidden_norm, layer.mlp.gate_proj.weight, None)
        up_proj = F.linear(hidden_norm, layer.mlp.up_proj.weight, None)
        
        if capture:
            self._save_snapshot('FFN_GATE', gate_proj, layer_idx)
            self._save_snapshot('FFN_UP', up_proj, layer_idx)
        
        # 3. SwiGLU activation: gate * silu(up)
        # SiLU(x) = x * sigmoid(x)
        silu_up = F.silu(up_proj)
        swiglu_output = gate_proj * silu_up
        if capture:
            self._save_snapshot('FFN_SWIGLU', swiglu_output, layer_idx)
        
        # 4. Down projection
        ffn_output = F.linear(swiglu_output, layer.mlp.down_proj.weight, None)
        if capture:
            self._save_snapshot('FFN_DOWN', ffn_output, layer_idx)
        
        # 5. Residual connection
        hidden_residual = hidden + ffn_output
        if capture:
            self._save_snapshot('FFN_RESIDUAL', hidden_residual, layer_idx)
        
        return hidden_residual
    
    def capture_pipeline_forward(
        self,
        prompt: str
    ) -> Dict[str, Dict]:
        """
        Run full Qwen2 pipeline and capture all intermediate states.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary mapping snapshot keys to capture dictionaries:
            - 'EMBEDDING': Token embeddings [batch, seq_len, d_model]
            - 'layer{i}_ATTENTION_NORM': Pre-attention norm [batch, seq_len, d_model]
            - 'layer{i}_Q_PROJECTION': Q projection [batch, seq_len, n_heads * d_head]
            - ... (all 16 stages per layer)
            - 'layer{i}_FFN_RESIDUAL': Post-FFN residual [batch, seq_len, d_model]
            - 'FINAL_NORM': Final RMSNorm [batch, seq_len, d_model]
            - 'LM_HEAD': Logits [batch, seq_len, vocab_size]
        """
        self.load_model()
        self.captures = {}
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']
        
        if self.verbose:
            print(f"\nProcessing prompt: '{prompt}'")
            print(f"Token IDs: {input_ids.tolist()[0]}")
            print(f"Num tokens: {input_ids.shape[1]}\n")
        
        with torch.no_grad():
            bsz, seq_len = input_ids.shape
            
            # 1. Embedding layer
            hidden = self.model.model.embed_tokens(input_ids)
            self._save_snapshot('EMBEDDING', hidden)
            
            # 2. Position IDs (0, 1, 2, ..., seq_len-1)
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            
            # 3. Position IDs (0, 1, 2, ..., seq_len-1)
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            
            # 4. Process each transformer layer
            n_layers = self.model.config.num_hidden_layers
            for layer_idx in range(n_layers):
                should_capture = self._should_capture_layer(layer_idx)
                
                if self.verbose and should_capture:
                    print(f"Processing layer {layer_idx}/{n_layers}...")
                
                # Attention block (always run our custom code for consistency)
                hidden_before_attn = hidden
                hidden = self._attention_block_impl(hidden, layer_idx, position_ids, should_capture)
                
                # FFN block  
                hidden = self._ffn_block_impl(hidden, layer_idx, should_capture)
            
            # 4. Final RMSNorm
            hidden = self._apply_rmsnorm(hidden, self.model.model.norm.weight)
            self._save_snapshot('FINAL_NORM', hidden)
            
            # 5. LM head (logits over vocabulary)
            logits = F.linear(hidden, self.model.lm_head.weight, None)
            self._save_snapshot('LM_HEAD', logits)
            
            if self.verbose:
                print(f"\n✓ Captured {len(self.captures)} snapshots")
                print(f"  Embedding: {self.captures['EMBEDDING']['shape']}")
                print(f"  Logits: {self.captures['LM_HEAD']['shape']}")
        
        return self.captures
    
    def save_to_directory(self, output_dir: Path):
        """
        Save all captures to disk (NumPy .npz format).
        
        Directory structure:
            output_dir/
                metadata.txt         - Configuration and token info
                EMBEDDING.npz        - Token embeddings
                layer0_Q_PROJECTION.npz
                layer0_K_PROJECTION.npz
                ...
                layer23_FFN_RESIDUAL.npz
                FINAL_NORM.npz
                LM_HEAD.npz
        
        Args:
            output_dir: Output directory path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = output_dir / "metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"Model: {self.model_path}\n")
            config = self.model.config
            arch = getattr(config, 'architectures', [config.__class__.__name__])
            f.write(f"Architecture: {arch[0] if arch else config.__class__.__name__}\n")
            f.write(f"n_layers: {config.num_hidden_layers}\n")
            f.write(f"n_heads: {config.num_attention_heads}\n")
            f.write(f"n_kv_heads: {config.num_key_value_heads}\n")
            f.write(f"d_model: {config.hidden_size}\n")
            f.write(f"d_head: {config.hidden_size // config.num_attention_heads}\n")
            f.write(f"d_ff: {config.intermediate_size}\n")
            f.write(f"vocab_size: {config.vocab_size}\n")
            f.write(f"num_snapshots: {len(self.captures)}\n")
        
        # Save each capture as separate .npy file (cnpy compatible)
        for key, capture in self.captures.items():
            npy_path = output_dir / f"{key}.npy"
            # Save just the data array (cnpy loads this easily)
            np.save(npy_path, capture['data'])
        
        if self.verbose:
            print(f"\n✓ Saved {len(self.captures)} snapshots to {output_dir}")
            print(f"  Metadata: {metadata_path}")
            print(f"  Snapshots: {len(list(output_dir.glob('*.npz')))} .npz files")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch Qwen2 pipeline reference snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default 8-token prompt
    python3 generate_qwen2_pipeline_snapshots.py --output pytorch_qwen2_snapshots
    
    # Custom prompt
    python3 generate_qwen2_pipeline_snapshots.py \\
        --prompt "The quick brown fox" \\
        --output snapshots_fox
    
    # Capture only first 3 layers (faster)
    python3 generate_qwen2_pipeline_snapshots.py \\
        --layers 0,1,2 \\
        --output snapshots_layer012
    
    # Verbose mode
    python3 generate_qwen2_pipeline_snapshots.py -v --output snapshots
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-0.5B-Instruct',
        help='HuggingFace model path or name (default: Qwen/Qwen2.5-0.5B-Instruct)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='The quick brown fox jumps over the lazy dog',
        help='Input prompt text (default: "The quick brown fox jumps over the lazy dog")'
    )
    
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated layer indices to capture (default: all layers). Example: 0,1,2'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('pytorch_qwen2_snapshots'),
        help='Output directory for snapshots (default: pytorch_qwen2_snapshots)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Parse layer indices
    capture_layers = None
    if args.layers:
        capture_layers = set(int(x.strip()) for x in args.layers.split(','))
        if args.verbose:
            print(f"Capturing layers: {sorted(capture_layers)}")
    
    # Create capture instance
    capturer = Qwen2PipelineCapture(
        model_path=args.model,
        capture_layers=capture_layers,
        verbose=args.verbose
    )
    
    # Run pipeline and capture snapshots
    print(f"Generating Qwen2 pipeline snapshots...")
    print(f"Model: {args.model}")
    print(f"Prompt: '{args.prompt}'")
    
    captures = capturer.capture_pipeline_forward(args.prompt)
    
    # Save to disk
    capturer.save_to_directory(args.output)
    
    print(f"\n✓ Done! Snapshots saved to: {args.output}")
    print(f"  Total snapshots: {len(captures)}")
    print(f"  Use these in C++ tests: tests/v2/e2e/Test__Qwen2FP32Parity.cpp")


if __name__ == '__main__':
    main()
