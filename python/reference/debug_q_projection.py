#!/usr/bin/env python3
"""Debug script to print Q projection values for comparison with C++."""

import torch
import sys
import numpy as np
from pathlib import Path

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

from python.reference.loaders.gguf_loader import GGUFLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf"
    prompt = "The capital of France is"
    
    print(f"Loading model: {model_path}")
    
    # Load using our GGUF loader (dequantizes)
    loader = GGUFLoader(model_path, verbose=False)
    config, state_dict = loader.load()
    
    # Create model
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Print Q weight info
    q_weight = model.model.layers[0].self_attn.q_proj.weight
    q_bias = model.model.layers[0].self_attn.q_proj.bias
    print(f"\n=== Q Projection Weight ===")
    print(f"Shape: {q_weight.shape}")
    print(f"First row, first 10 values: {q_weight[0, :10].tolist()}")
    print(f"First 10 rows, first col: {q_weight[:10, 0].tolist()}")
    print(f"Q bias: {q_bias}")
    if q_bias is not None:
        print(f"Q bias shape: {q_bias.shape}")
        print(f"Q bias first 10 values: {q_bias[:10].tolist()}")
    
    # Also print K weight for comparison
    k_weight = model.model.layers[0].self_attn.k_proj.weight
    print(f"\n=== K Projection Weight ===")
    print(f"Shape: {k_weight.shape}")
    print(f"First row, first 10 values: {k_weight[0, :10].tolist()}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    print(f"\nInput tokens: {input_ids.tolist()}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    seq_len = input_ids.shape[1]
    
    # Hook to capture intermediate values
    embeddings = None
    hidden_after_norm = None
    q_projection = None
    k_projection = None
    
    # Hook into embedding layer
    def embed_hook(module, input, output):
        nonlocal embeddings
        embeddings = output.clone().detach()
    
    model.model.embed_tokens.register_forward_hook(embed_hook)
    
    layer0_attn = model.model.layers[0].self_attn
    layer0 = model.model.layers[0]
    original_forward = layer0.forward
    
    def debug_layer_forward(hidden_states, *args, **kwargs):
        nonlocal hidden_after_norm, q_projection, k_projection
        
        # Layer 0 applies input_layernorm first, then attention
        # hidden_states coming in is the residual
        residual = hidden_states
        
        # RMSNorm
        normalized = layer0.input_layernorm(hidden_states)
        hidden_after_norm = normalized.clone().detach()
        
        # Q/K projections
        q_projection = layer0_attn.q_proj(normalized).clone().detach()
        k_projection = layer0_attn.k_proj(normalized).clone().detach()
        
        # Continue with original forward
        result = original_forward(hidden_states, *args, **kwargs)
        
        return result
    
    layer0.forward = debug_layer_forward
    
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits
    
    # Print results
    last_pos = seq_len - 1
    
    print(f"\n=== Embeddings (position {last_pos}, first 10 values) ===")
    emb_vals = embeddings[0, last_pos, :10].tolist()
    print(f"Python: {', '.join([f'{v:.6f}' for v in emb_vals])}")
    
    print(f"\n=== After RMSNorm (position {last_pos}, first 10 values) ===")
    norm_vals = hidden_after_norm[0, last_pos, :10].tolist()
    print(f"Python: {', '.join([f'{v:.6f}' for v in norm_vals])}")
    
    print(f"\n=== Layer 0 Q Projection (seq_len={seq_len}, last_pos={last_pos}, first 10 values) ===")
    q_vals = q_projection[0, last_pos, :10].tolist()
    print(f"Python: {', '.join([f'{v:.6f}' for v in q_vals])}")
    
    print(f"\n=== Layer 0 K Projection (seq_len={seq_len}, last_pos={last_pos}, first 10 values) ===")
    k_vals = k_projection[0, last_pos, :10].tolist()
    print(f"Python: {', '.join([f'{v:.6f}' for v in k_vals])}")
    
    # Manual matmul verification
    print(f"\n=== Manual Q = normalized @ q_weight.T verification ===")
    # Q projection: output[seq, out_dim] = input[seq, in_dim] @ weight[out_dim, in_dim].T
    # So: output[seq, out_dim] = input[seq, in_dim] @ weight.T[in_dim, out_dim]
    manual_q = torch.matmul(hidden_after_norm, q_weight.T)
    manual_q_vals = manual_q[0, last_pos, :10].tolist()
    print(f"Manual Q: {', '.join([f'{v:.6f}' for v in manual_q_vals])}")
    
    print(f"\n=== Final Logits (last position = {last_pos}, top 5 tokens) ===")
    last_logits = logits[0, last_pos, :]
    top_vals, top_indices = torch.topk(last_logits, 5)
    for idx, val in zip(top_indices.tolist(), top_vals.tolist()):
        print(f"  Token {idx}: {tokenizer.decode([idx])!r} = {val:.4f}")

if __name__ == "__main__":
    main()

