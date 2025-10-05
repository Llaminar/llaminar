"""
GGUF to HuggingFace Tensor Name Mapping

Maps GGUF tensor names to HuggingFace transformers naming conventions.
Supports Qwen2 and LLaMA model families.

GGUF Naming (llama.cpp):
- token_embd.weight
- output.weight  
- blk.{i}.attn_q.weight
- blk.{i}.attn_k.weight
- blk.{i}.attn_v.weight
- blk.{i}.attn_output.weight
- blk.{i}.attn_norm.weight
- blk.{i}.ffn_gate.weight
- blk.{i}.ffn_up.weight
- blk.{i}.ffn_down.weight
- blk.{i}.ffn_norm.weight

HuggingFace Naming (transformers):
- model.embed_tokens.weight
- lm_head.weight
- model.layers.{i}.self_attn.q_proj.weight
- model.layers.{i}.self_attn.k_proj.weight
- model.layers.{i}.self_attn.v_proj.weight
- model.layers.{i}.self_attn.o_proj.weight
- model.layers.{i}.input_layernorm.weight
- model.layers.{i}.mlp.gate_proj.weight
- model.layers.{i}.mlp.up_proj.weight
- model.layers.{i}.mlp.down_proj.weight
- model.layers.{i}.post_attention_layernorm.weight

Author: David Sanftenberg
"""

import re
from typing import Dict, Optional, Tuple


# Qwen2 GGUF -> HuggingFace mapping
QWEN2_TENSOR_MAP = {
    # Embedding and output
    'token_embd.weight': 'model.embed_tokens.weight',
    'output.weight': 'lm_head.weight',
    'output_norm.weight': 'model.norm.weight',
    
    # Per-layer patterns (will be formatted with layer index)
    # Attention
    'blk.{}.attn_norm.weight': 'model.layers.{}.input_layernorm.weight',
    'blk.{}.attn_q.weight': 'model.layers.{}.self_attn.q_proj.weight',
    'blk.{}.attn_q.bias': 'model.layers.{}.self_attn.q_proj.bias',
    'blk.{}.attn_k.weight': 'model.layers.{}.self_attn.k_proj.weight',
    'blk.{}.attn_k.bias': 'model.layers.{}.self_attn.k_proj.bias',
    'blk.{}.attn_v.weight': 'model.layers.{}.self_attn.v_proj.weight',
    'blk.{}.attn_v.bias': 'model.layers.{}.self_attn.v_proj.bias',
    'blk.{}.attn_output.weight': 'model.layers.{}.self_attn.o_proj.weight',
    
    # FFN (MLP)
    'blk.{}.ffn_norm.weight': 'model.layers.{}.post_attention_layernorm.weight',
    'blk.{}.ffn_gate.weight': 'model.layers.{}.mlp.gate_proj.weight',
    'blk.{}.ffn_up.weight': 'model.layers.{}.mlp.up_proj.weight',
    'blk.{}.ffn_down.weight': 'model.layers.{}.mlp.down_proj.weight',
}


# LLaMA GGUF -> HuggingFace mapping (very similar to Qwen2)
LLAMA_TENSOR_MAP = {
    # Embedding and output
    'token_embd.weight': 'model.embed_tokens.weight',
    'output.weight': 'lm_head.weight',
    'output_norm.weight': 'model.norm.weight',
    
    # Per-layer patterns
    # Attention
    'blk.{}.attn_norm.weight': 'model.layers.{}.input_layernorm.weight',
    'blk.{}.attn_q.weight': 'model.layers.{}.self_attn.q_proj.weight',
    'blk.{}.attn_k.weight': 'model.layers.{}.self_attn.k_proj.weight',
    'blk.{}.attn_v.weight': 'model.layers.{}.self_attn.v_proj.weight',
    'blk.{}.attn_output.weight': 'model.layers.{}.self_attn.o_proj.weight',
    
    # FFN (MLP)
    'blk.{}.ffn_norm.weight': 'model.layers.{}.post_attention_layernorm.weight',
    'blk.{}.ffn_gate.weight': 'model.layers.{}.mlp.gate_proj.weight',
    'blk.{}.ffn_up.weight': 'model.layers.{}.mlp.up_proj.weight',
    'blk.{}.ffn_down.weight': 'model.layers.{}.mlp.down_proj.weight',
}


class TensorNameMapper:
    """
    Maps GGUF tensor names to HuggingFace naming conventions.
    
    Supports automatic model type detection and flexible name mapping.
    """
    
    def __init__(self, model_type: str = 'qwen2'):
        """
        Initialize tensor name mapper.
        
        Args:
            model_type: Model architecture ('qwen2', 'llama', etc.)
        """
        self.model_type = model_type.lower()
        
        # Select appropriate mapping table
        if self.model_type in ('qwen', 'qwen2'):
            self.mapping = QWEN2_TENSOR_MAP
        elif self.model_type in ('llama', 'llama2', 'llama3'):
            self.mapping = LLAMA_TENSOR_MAP
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported: qwen2, llama")
    
    def map_name(self, gguf_name: str) -> str:
        """
        Map a GGUF tensor name to HuggingFace naming convention.
        
        Args:
            gguf_name: GGUF tensor name (e.g., 'blk.0.attn_q.weight')
            
        Returns:
            HuggingFace tensor name (e.g., 'model.layers.0.self_attn.q_proj.weight')
            
        Raises:
            ValueError: If name cannot be mapped
        """
        # Direct mapping (non-layered tensors)
        if gguf_name in self.mapping:
            return self.mapping[gguf_name]
        
        # Extract layer index for layered tensors
        # Pattern: blk.{layer_idx}.{rest}
        match = re.match(r'blk\.(\d+)\.(.+)', gguf_name)
        if match:
            layer_idx = match.group(1)
            rest = match.group(2)
            
            # Try to find pattern in mapping
            pattern = f'blk.{{}}.{rest}'
            if pattern in self.mapping:
                hf_pattern = self.mapping[pattern]
                return hf_pattern.format(layer_idx)
        
        # If no mapping found, return original name with warning
        # This allows partial loading to continue
        # Skip warning for known ignorable tensors (computed at runtime)
        ignorable_tensors = {'rope_freqs.weight', 'rope.freqs', 'pos_embd.weight'}
        if gguf_name not in ignorable_tensors:
            print(f"Warning: No mapping found for GGUF tensor '{gguf_name}', using as-is")
        return gguf_name
    
    def map_all(self, gguf_names: list) -> Dict[str, str]:
        """
        Map multiple GGUF names to HuggingFace names.
        
        Args:
            gguf_names: List of GGUF tensor names
            
        Returns:
            Dictionary mapping GGUF names to HuggingFace names
        """
        return {name: self.map_name(name) for name in gguf_names}
    
    def get_unmapped_names(self, gguf_names: list) -> list:
        """
        Find GGUF names that don't have mappings.
        
        Args:
            gguf_names: List of GGUF tensor names
            
        Returns:
            List of names without mappings
        """
        unmapped = []
        for name in gguf_names:
            try:
                mapped = self.map_name(name)
                # Check if it was actually mapped or just returned as-is
                if mapped == name and name not in self.mapping:
                    unmapped.append(name)
            except ValueError:
                unmapped.append(name)
        return unmapped


def detect_model_type_from_metadata(metadata: dict) -> str:
    """
    Detect model type from GGUF metadata.
    
    Args:
        metadata: GGUF metadata dictionary
        
    Returns:
        Model type string ('qwen2', 'llama', etc.)
    """
    # Check general.architecture key
    if 'general.architecture' in metadata:
        arch = metadata['general.architecture'].lower()
        if 'qwen' in arch:
            return 'qwen2'
        elif 'llama' in arch:
            return 'llama'
    
    # Fallback: check for model-specific keys
    if any(k.startswith('qwen') for k in metadata):
        return 'qwen2'
    elif any(k.startswith('llama') for k in metadata):
        return 'llama'
    
    # Default to qwen2 (most common for now)
    return 'qwen2'


def create_mapper_from_metadata(metadata: dict) -> TensorNameMapper:
    """
    Create a tensor name mapper by auto-detecting model type from metadata.
    
    Args:
        metadata: GGUF metadata dictionary
        
    Returns:
        Configured TensorNameMapper instance
    """
    model_type = detect_model_type_from_metadata(metadata)
    return TensorNameMapper(model_type)


def verify_mapping_coverage(gguf_names: list, model_type: str = 'qwen2') -> Tuple[int, int, list]:
    """
    Check mapping coverage for a list of GGUF tensor names.
    
    Args:
        gguf_names: List of GGUF tensor names
        model_type: Model architecture
        
    Returns:
        Tuple of (mapped_count, total_count, unmapped_names)
    """
    mapper = TensorNameMapper(model_type)
    unmapped = mapper.get_unmapped_names(gguf_names)
    
    mapped_count = len(gguf_names) - len(unmapped)
    total_count = len(gguf_names)
    
    return mapped_count, total_count, unmapped


if __name__ == '__main__':
    # Test mapping
    print("Testing Tensor Name Mapping")
    print("=" * 80)
    
    mapper = TensorNameMapper('qwen2')
    
    # Test cases
    test_names = [
        'token_embd.weight',
        'output.weight',
        'output_norm.weight',
        'blk.0.attn_q.weight',
        'blk.0.attn_q.bias',
        'blk.5.ffn_down.weight',
        'blk.23.attn_norm.weight',
    ]
    
    print("\nQwen2 Mapping Examples:")
    for gguf_name in test_names:
        hf_name = mapper.map_name(gguf_name)
        print(f"  {gguf_name:35s} -> {hf_name}")
