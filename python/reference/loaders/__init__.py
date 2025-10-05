"""
GGUF Loaders Module

Provides utilities for loading GGUF (GPT-Generated Unified Format) model files
into PyTorch reference models. Enables using the same GGUF files with both
Llaminar and PyTorch for parity testing.

Components:
- gguf_parser: Low-level GGUF file format parser
- dequantize: Quantization format handlers (Q4_0, Q6_K, Q8_0, F16 -> FP32)
- gguf_loader: Main loader orchestrator

Usage:
    from python.reference.loaders import GGUFLoader
    
    loader = GGUFLoader("models/qwen2.5-0.5b-instruct-q4_0.gguf")
    config, state_dict = loader.load()

Author: David Sanftenberg
"""

from .gguf_parser import GGUFParser, GGUFTensorType
from .gguf_loader import GGUFLoader
from . import dequantize

__all__ = ["GGUFParser", "GGUFLoader", "GGUFTensorType", "dequantize"]
