"""
Qwen Reference Implementation

PyTorch reference implementation for Qwen/Qwen2/Qwen3 models using HuggingFace transformers.
Captures intermediate pipeline states for parity testing with Llaminar.

@author David Sanftenberg
"""

from typing import Optional, Union
from pathlib import Path

import torch

from .base import HuggingFaceReferenceModel
from .registry import ModelRegistry


class QwenReferenceModel(HuggingFaceReferenceModel):
    """
    PyTorch reference implementation for Qwen/Qwen2/Qwen3 models.

    Inherits all shared loading, hook-registration, and forward-pass logic
    from :class:`HuggingFaceReferenceModel`.  Only the GGUF config→model
    mapping and tokenizer fallback list are model-specific.
    """

    def _create_model_from_gguf_config(
        self,
        config_dict: dict,
        torch_dtype: Optional[torch.dtype],
    ) -> tuple:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        # Try Qwen3 classes (newer transformers)
        try:
            from transformers import Qwen3Config, Qwen3ForCausalLM
            HAS_QWEN3 = True
        except ImportError:
            HAS_QWEN3 = False

        model_type = config_dict.get("model_type", "qwen2")

        if model_type == "qwen3" and HAS_QWEN3:
            print("Detected Qwen3 architecture")
            cfg = Qwen3Config(**config_dict)
            cfg._attn_implementation = "eager"
            if torch_dtype:
                cfg.torch_dtype = torch_dtype
            return cfg, Qwen3ForCausalLM(cfg)

        if model_type == "qwen3":
            print("WARNING: Qwen3 detected but Qwen3ForCausalLM unavailable, falling back to Qwen2")
            config_dict["model_type"] = "qwen2"

        print("Using Qwen2 architecture")
        cfg = Qwen2Config(**config_dict)
        cfg._attn_implementation = "eager"
        if torch_dtype:
            cfg.torch_dtype = torch_dtype
        return cfg, Qwen2ForCausalLM(cfg)

    def _tokenizer_fallbacks(self) -> list[str]:
        return ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2-0.5B-Instruct"]


# Register this implementation
ModelRegistry.register("qwen", QwenReferenceModel)
