"""
PyTorch Reference Implementation for Llaminar Parity Testing

This package provides reference implementations of transformer models using
PyTorch and HuggingFace transformers. It captures intermediate pipeline states
for comparison with Llaminar's C++ implementation.

Architecture:
- AbstractReferenceModel: Base class defining the interface
- Model-specific implementations: QwenReferenceModel, LlamaReferenceModel, etc.
- Factory pattern via ModelRegistry for easy instantiation
- PipelineStage enum synchronized with C++ (src/pipeline_stages.h)

Usage:
    from python.reference import create_reference_model, PipelineStage
    
    model = create_reference_model("qwen", checkpoint="Qwen/Qwen2-0.5B")
    snapshots = model.forward([1, 2, 3], capture_stages=[
        PipelineStage.EMBEDDING,
        PipelineStage.ATTENTION_OUTPUT,
        PipelineStage.LM_HEAD
    ])
"""

__version__ = "0.1.0"
__author__ = "David Sanftenberg"

# Always available (no dependencies)
from .pipeline_stages import PipelineStage

# Lazy imports for torch-dependent modules
def _import_base():
    from .base import AbstractReferenceModel
    return AbstractReferenceModel

def _import_registry():
    from .registry import ModelRegistry, create_reference_model
    return ModelRegistry, create_reference_model

# Try to import torch-dependent modules, but don't fail if unavailable
try:
    from .base import AbstractReferenceModel
    from .registry import ModelRegistry, create_reference_model
    
    # Register available models (done in respective modules)
    from . import qwen
    from . import llama  # Stub implementation for demonstration
    
    _torch_available = True
except ImportError as e:
    _torch_available = False
    # These will be None if torch not available
    AbstractReferenceModel = None
    ModelRegistry = None
    create_reference_model = None

__all__ = [
    "PipelineStage",
    "AbstractReferenceModel", 
    "ModelRegistry",
    "create_reference_model",
]
