"""
LLaMA Reference Implementation

PyTorch reference implementation for LLaMA/Llama2/Llama3 models using HuggingFace transformers.
Captures intermediate pipeline states for parity testing with Llaminar.

@author David Sanftenberg
"""

from typing import Optional, Union
from pathlib import Path

import torch

from .base import HuggingFaceReferenceModel
from .registry import ModelRegistry


class LlamaReferenceModel(HuggingFaceReferenceModel):
    """
    PyTorch reference implementation for LLaMA models.

    Inherits all shared loading, hook-registration, and forward-pass logic
    from :class:`HuggingFaceReferenceModel`.  Only the GGUF config→model
    mapping and tokenizer fallback list are model-specific.
    """

    def _create_model_from_gguf_config(
        self,
        config_dict: dict,
        torch_dtype: Optional[torch.dtype],
    ) -> tuple:
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(**config_dict)
        if torch_dtype:
            cfg.torch_dtype = torch_dtype
        return cfg, LlamaForCausalLM(cfg)

    def _tokenizer_fallbacks(self) -> list[str]:
        return [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "HuggingFaceH4/zephyr-7b-beta",
            "meta-llama/Llama-2-7b-hf",
        ]


# Register this implementation
ModelRegistry.register("llama", LlamaReferenceModel)
