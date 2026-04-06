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
    K_PROJECTION = auto()           # Key projection only (if separate)
    V_PROJECTION = auto()           # Value projection only (if separate)
    ROPE_APPLICATION = auto()       # Rotary position embeddings applied to Q and K
    ATTENTION_SCORES = auto()       # Q @ K^T attention scores (before softmax)
    ATTENTION_SOFTMAX = auto()      # Softmax over attention scores
    ATTENTION_CONTEXT = auto()      # Attention weights @ V (context vectors)
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
    GDN_DELTA_RULE_OUTPUT = auto()  # Output of chunk/recurrent gated delta rule kernel
    GDN_NORM_GATE_OUTPUT = auto()   # Output of RMSNormGated (norm + SiLU gate with z)

    # === Extensibility ===
    CUSTOM = auto()                 # Custom stage for architecture-specific extensions


# Mapping for string conversion (matches C++ stage_to_string)
_STAGE_TO_STRING: Dict[PipelineStage, str] = {
    PipelineStage.EMBEDDING: "EMBEDDING",
    PipelineStage.ATTENTION_NORM: "ATTENTION_NORM",
    PipelineStage.QKV_PROJECTION: "QKV_PROJECTION",
    PipelineStage.Q_PROJECTION: "Q_PROJECTION",
    PipelineStage.K_PROJECTION: "K_PROJECTION",
    PipelineStage.V_PROJECTION: "V_PROJECTION",
    PipelineStage.ROPE_APPLICATION: "ROPE_APPLICATION",
    PipelineStage.ATTENTION_SCORES: "ATTENTION_SCORES",
    PipelineStage.ATTENTION_SOFTMAX: "ATTENTION_SOFTMAX",
    PipelineStage.ATTENTION_CONTEXT: "ATTENTION_CONTEXT",
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
    PipelineStage.GDN_DELTA_RULE_OUTPUT: "GDN_DELTA_RULE_OUTPUT",
    PipelineStage.GDN_NORM_GATE_OUTPUT: "GDN_NORM_GATE_OUTPUT",
    PipelineStage.CUSTOM: "CUSTOM",
}

_STRING_TO_STAGE: Dict[str, PipelineStage] = {v: k for k, v in _STAGE_TO_STRING.items()}


def stage_to_string(stage: PipelineStage) -> str:
    """Convert PipelineStage enum to string (matches C++ implementation)."""
    return _STAGE_TO_STRING[stage]


def string_to_stage(s: str) -> PipelineStage:
    """Convert string to PipelineStage enum (matches C++ implementation)."""
    return _STRING_TO_STAGE[s]
