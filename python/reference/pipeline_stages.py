"""
Pipeline Stage Enumeration

Synchronized with C++ enum class PipelineStage in src/pipeline_stages.h
Must be kept in sync manually until we have automated code generation.

@author David Sanftenberg
"""

from enum import Enum, auto
from typing import Dict


class PipelineStage(Enum):
    """
    Stages of the transformer pipeline where snapshots/instrumentation can occur.
    
    Synchronized with llaminar::PipelineStage in src/pipeline_stages.h
    
    Stages are ordered roughly in execution sequence within a transformer layer.
    Non-layer stages (embedding, final norm, LM head) use layer_index = -1.
    """
    
    # === Input Processing ===
    EMBEDDING = auto()              # Token embedding lookup (before layer loop)
    
    # === Attention Block ===
    ATTENTION_NORM = auto()         # RMSNorm/LayerNorm before attention
    QKV_PROJECTION = auto()         # Combined Q, K, V linear projections
    Q_PROJECTION = auto()           # Query projection only (if separate)
    FA_GATE = auto()                # Qwen3.5 full-attention sigmoid gate split from Q projection
    K_PROJECTION = auto()           # Key projection only (if separate)
    V_PROJECTION = auto()           # Value projection only (if separate)
    Q_NORM = auto()                 # Per-head query RMSNorm (Qwen3/Qwen3.5)
    K_NORM = auto()                 # Per-head key RMSNorm (Qwen3/Qwen3.5)
    Q_ROPE = auto()                 # Query after rotary position embedding
    K_ROPE = auto()                 # Key after rotary position embedding
    ROPE_APPLICATION = auto()       # Rotary position embeddings applied to Q and K
    ATTENTION_SCORES = auto()       # Q @ K^T attention scores (before softmax)
    ATTENTION_SOFTMAX = auto()      # Softmax over attention scores
    ATTENTION_CONTEXT = auto()      # Attention weights @ V (context vectors)
    ATTENTION_CONTEXT_GATED = auto()  # Qwen3.5 context after sigmoid gate, before output projection
    ATTENTION_OUTPUT = auto()       # Output projection W_o (after context)
    ATTENTION_RESIDUAL = auto()     # After attention residual connection
    
    # === Feed-Forward Block ===
    FFN_NORM = auto()               # RMSNorm/LayerNorm before FFN/MLP
    FFN_GATE = auto()               # Gate projection (SwiGLU gate or first linear)
    FFN_UP = auto()                 # Up projection (SwiGLU up or hidden expansion)
    FFN_SWIGLU = auto()             # SwiGLU activation (gate * silu(up))
    FFN_DOWN = auto()               # Down projection (back to hidden dimension)
    FFN_RESIDUAL = auto()           # After FFN residual connection
    
    # === Output Processing ===
    FINAL_NORM = auto()             # Final RMSNorm/LayerNorm (after all layers)
    LM_HEAD = auto()                # Language model head output (logits)
    
    # === Gated Delta Net (Linear Attention) ===
    GDN_CONV1D_OUTPUT = auto()      # Output of causal conv1d (after SiLU activation)
    GDN_Z_PROJECTION = auto()       # Output of Z gate projection (in_proj_z)
    GDN_ALPHA = auto()              # Output of alpha projection (in_proj_a)
    GDN_BETA = auto()               # Output of beta projection (in_proj_b)
    GDN_DELTA_RULE_OUTPUT = auto()  # Output of chunk/recurrent gated delta rule kernel
    GDN_NORM_GATE_OUTPUT = auto()   # Output of RMSNormGated (norm + SiLU gate with z)

    # === Mixture of Experts ===
    MOE_ROUTER_OUTPUT = auto()      # Router logits [seq_len, num_experts]
    MOE_ROUTING_INDICES = auto()    # Selected expert IDs [seq_len, top_k] (int as float)
    MOE_ROUTING_WEIGHTS = auto()    # Normalized top-k routing weights [seq_len, top_k]
    MOE_EXPERT_OUTPUT = auto()      # Combined routed expert output [seq_len, d_model]
    MOE_SHARED_EXPERT_OUTPUT = auto()  # Shared expert FFN output (before gate)
    MOE_SHARED_GATE_OUTPUT = auto()    # Gated shared expert output (after sigmoid gate)
    MOE_COMBINED_OUTPUT = auto()    # Final MoE output (routed + shared expert)

    # === Extensibility ===
    CUSTOM = auto()                 # Custom stage for architecture-specific extensions


# Mapping for string conversion (matches C++ stage_to_string)
_STAGE_TO_STRING: Dict[PipelineStage, str] = {
    PipelineStage.EMBEDDING: "EMBEDDING",
    PipelineStage.ATTENTION_NORM: "ATTENTION_NORM",
    PipelineStage.QKV_PROJECTION: "QKV_PROJECTION",
    PipelineStage.Q_PROJECTION: "Q_PROJECTION",
    PipelineStage.FA_GATE: "FA_GATE",
    PipelineStage.K_PROJECTION: "K_PROJECTION",
    PipelineStage.V_PROJECTION: "V_PROJECTION",
    PipelineStage.Q_NORM: "Q_NORM",
    PipelineStage.K_NORM: "K_NORM",
    PipelineStage.Q_ROPE: "Q_ROPE",
    PipelineStage.K_ROPE: "K_ROPE",
    PipelineStage.ROPE_APPLICATION: "ROPE_APPLICATION",
    PipelineStage.ATTENTION_SCORES: "ATTENTION_SCORES",
    PipelineStage.ATTENTION_SOFTMAX: "ATTENTION_SOFTMAX",
    PipelineStage.ATTENTION_CONTEXT: "ATTENTION_CONTEXT",
    PipelineStage.ATTENTION_CONTEXT_GATED: "ATTENTION_CONTEXT_GATED",
    PipelineStage.ATTENTION_OUTPUT: "ATTENTION_OUTPUT",
    PipelineStage.ATTENTION_RESIDUAL: "ATTENTION_RESIDUAL",
    PipelineStage.FFN_NORM: "FFN_NORM",
    PipelineStage.FFN_GATE: "FFN_GATE",
    PipelineStage.FFN_UP: "FFN_UP",
    PipelineStage.FFN_SWIGLU: "FFN_SWIGLU",
    PipelineStage.FFN_DOWN: "FFN_DOWN",
    PipelineStage.FFN_RESIDUAL: "FFN_RESIDUAL",
    PipelineStage.FINAL_NORM: "FINAL_NORM",
    PipelineStage.LM_HEAD: "LM_HEAD",
    PipelineStage.GDN_CONV1D_OUTPUT: "GDN_CONV1D_OUTPUT",
    PipelineStage.GDN_Z_PROJECTION: "GDN_Z_PROJECTION",
    PipelineStage.GDN_ALPHA: "GDN_ALPHA",
    PipelineStage.GDN_BETA: "GDN_BETA",
    PipelineStage.GDN_DELTA_RULE_OUTPUT: "GDN_DELTA_RULE_OUTPUT",
    PipelineStage.GDN_NORM_GATE_OUTPUT: "GDN_NORM_GATE_OUTPUT",
    PipelineStage.MOE_ROUTER_OUTPUT: "MOE_ROUTER_OUTPUT",
    PipelineStage.MOE_ROUTING_INDICES: "MOE_ROUTING_INDICES",
    PipelineStage.MOE_ROUTING_WEIGHTS: "MOE_ROUTING_WEIGHTS",
    PipelineStage.MOE_EXPERT_OUTPUT: "MOE_EXPERT_OUTPUT",
    PipelineStage.MOE_SHARED_EXPERT_OUTPUT: "MOE_SHARED_EXPERT_OUTPUT",
    PipelineStage.MOE_SHARED_GATE_OUTPUT: "MOE_SHARED_GATE_OUTPUT",
    PipelineStage.MOE_COMBINED_OUTPUT: "MOE_COMBINED_OUTPUT",
    PipelineStage.CUSTOM: "CUSTOM",
}

_STRING_TO_STAGE: Dict[str, PipelineStage] = {v: k for k, v in _STAGE_TO_STRING.items()}


def stage_to_string(stage: PipelineStage) -> str:
    """Convert PipelineStage enum to string (matches C++ implementation)."""
    return _STAGE_TO_STRING[stage]


def string_to_stage(s: str) -> PipelineStage:
    """Convert string to PipelineStage enum (matches C++ implementation)."""
    return _STRING_TO_STAGE[s]
