"""
Qwen Reference Implementation

PyTorch reference implementation for Qwen/Qwen2 models using HuggingFace transformers.
Captures intermediate pipeline states for parity testing with Llaminar.

@author David Sanftenberg
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import warnings

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base import AbstractReferenceModel
from .pipeline_stages import PipelineStage
from .registry import ModelRegistry


class QwenReferenceModel(AbstractReferenceModel):
    """
    PyTorch reference implementation for Qwen/Qwen2 models.
    
    Uses HuggingFace transformers library and monkey-patches forward hooks
    to capture intermediate pipeline stages.
    
    Supports:
    - Qwen2 architecture (Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B, etc.)
    - Quantization via bitsandbytes (Q4, Q8)
    - Stage-by-stage snapshot capture
    - Export to C++ compatible formats
    
    Example:
        model = QwenReferenceModel("qwen", "Qwen/Qwen2-0.5B-Instruct")
        model.load_model()
        result = model.forward([1, 2, 3], capture_stages=[
            PipelineStage.EMBEDDING,
            PipelineStage.ATTENTION_OUTPUT,
            PipelineStage.LM_HEAD
        ])
    """
    
    def __init__(self, model_name: str, checkpoint_path: Union[str, Path], **kwargs):
        super().__init__(model_name, checkpoint_path, **kwargs)
        
        # HuggingFace model and config
        self.hf_model: Optional[Any] = None
        self.hf_config: Optional[Any] = None
        
        # Forward hook handles (for cleanup)
        self._hook_handles = []
        
        # Which stages to capture (None = all)
        self._capture_stages: Optional[List[PipelineStage]] = None
        
        # Verbose logging
        self.verbose: bool = kwargs.get('verbose', False)
    
    def load_model(
        self,
        quantization: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> None:
        """
        Load Qwen model from HuggingFace checkpoint or GGUF file.
        
        Automatically detects GGUF files by extension (.gguf) and uses
        the GGUFLoader for loading. Falls back to HuggingFace transformers
        for standard checkpoints.
        
        Args:
            quantization: Quantization mode ("4bit", "8bit", or None)
                         Ignored for GGUF files (already quantized)
            torch_dtype: Override default dtype (e.g., torch.bfloat16)
            **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained()
        
        Raises:
            RuntimeError: If model fails to load
        """
        try:
            checkpoint_path_str = str(self.checkpoint_path)
            
            # Auto-detect GGUF files
            if checkpoint_path_str.endswith('.gguf'):
                self._load_from_gguf(checkpoint_path_str, torch_dtype, **kwargs)
            else:
                self._load_from_huggingface(quantization, torch_dtype, **kwargs)
            
            # Register forward hooks for stage capture
            self._register_hooks()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen model: {e}") from e
    
    def _load_from_gguf(
        self,
        gguf_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> None:
        """
        Load Qwen model from GGUF file.
        
        Uses GGUFLoader to extract config and state dict, then creates
        a transformers model and loads the weights.
        
        Args:
            gguf_path: Path to GGUF file
            torch_dtype: Override default dtype
            **kwargs: Additional arguments (mostly ignored for GGUF)
        """
        from .loaders import GGUFLoader
        from transformers import Qwen2Config, Qwen2ForCausalLM
        
        print(f"Loading GGUF file: {gguf_path}")
        
        # Load GGUF file
        loader = GGUFLoader(gguf_path, verbose=self.verbose)
        config_dict, state_dict = loader.load(
            as_transformers_config=False,  # Get dict first
            as_torch=True,  # Convert to torch tensors
            show_progress=self.verbose
        )
        
        if self.verbose:
            print(f"Config extracted: {list(config_dict.keys())}")
        
        # Create Qwen2Config from dict
        self.hf_config = Qwen2Config(**config_dict)
        
        # Set dtype if specified
        if torch_dtype:
            self.hf_config.torch_dtype = torch_dtype
        
        # Create model from config
        print(f"Creating Qwen2 model from config...")
        self.hf_model = Qwen2ForCausalLM(self.hf_config)
        
        # Load state dict
        print(f"Loading {len(state_dict)} tensors into model...")
        missing_keys, unexpected_keys = self.hf_model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            warnings.warn(f"Missing keys when loading GGUF: {missing_keys}")
        if unexpected_keys:
            warnings.warn(f"Unexpected keys when loading GGUF: {unexpected_keys}")
        
        # Move to device and set to eval mode
        self.hf_model = self.hf_model.to(self.device)
        self.hf_model.eval()
        
        # Try to load tokenizer from GGUF directory or use model name fallback
        tokenizer_path = Path(gguf_path).parent
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        except Exception:
            # Fallback: try to infer tokenizer from model name in config
            model_name = config_dict.get('_name_or_path', 'Qwen/Qwen2-0.5B-Instruct')
            print(f"Tokenizer not found locally, trying {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"✓ GGUF model loaded successfully")
    
    def _load_from_huggingface(
        self,
        quantization: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> None:
        """
        Load Qwen model from HuggingFace checkpoint (standard path).
        
        Args:
            quantization: Quantization mode ("4bit", "8bit", or None)
            torch_dtype: Override default dtype
            **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained()
        """
        # Load config first
        self.hf_config = AutoConfig.from_pretrained(self.checkpoint_path)
        
        # Set up quantization config if requested
        model_kwargs = kwargs.copy()
        if quantization:
            from transformers import BitsAndBytesConfig
            
            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype or self.dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                raise ValueError(f"Unsupported quantization: {quantization}")
        
        # Load model
        model_kwargs.setdefault("torch_dtype", torch_dtype or self.dtype)
        model_kwargs.setdefault("device_map", str(self.device))
        
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            **model_kwargs
        )
        self.hf_model.eval()  # Inference mode
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
    
    def _register_hooks(self) -> None:
        """
        Register forward hooks on model layers to capture intermediate states.
        
        This is where the magic happens - we intercept layer outputs during
        forward pass and store them as snapshots.
        """
        if not self.hf_model:
            return
        
        model = self.hf_model.model  # Access underlying Qwen2Model
        
        # Hook 1: Embedding layer
        def embedding_hook(module, input, output):
            if self._should_capture(PipelineStage.EMBEDDING):
                self.capture_stage(PipelineStage.EMBEDDING, output, layer_idx=-1)
        
        handle = model.embed_tokens.register_forward_hook(embedding_hook)
        self._hook_handles.append(handle)
        
        # Hook 2-N: Per-layer hooks
        for layer_idx, layer in enumerate(model.layers):
            # Attention norm (input layernorm)
            def attn_norm_hook(module, input, output, layer_idx=layer_idx):
                if self._should_capture(PipelineStage.ATTENTION_NORM):
                    self.capture_stage(PipelineStage.ATTENTION_NORM, output, layer_idx)
            
            handle = layer.input_layernorm.register_forward_hook(attn_norm_hook)
            self._hook_handles.append(handle)
            
            # Attention output (after self_attn)
            def attn_output_hook(module, input, output, layer_idx=layer_idx):
                if self._should_capture(PipelineStage.ATTENTION_OUTPUT):
                    # output is tuple (hidden_states, attention_weights, ...)
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    self.capture_stage(PipelineStage.ATTENTION_OUTPUT, hidden_states, layer_idx)
            
            handle = layer.self_attn.register_forward_hook(attn_output_hook)
            self._hook_handles.append(handle)
            
            # Attention residual (after first residual connection)
            # This is trickier - we need to hook the layer itself and look at intermediate values
            # For now, we'll skip this and rely on FFN_NORM which is equivalent
            
            # FFN norm (post_attention_layernorm)
            def ffn_norm_hook(module, input, output, layer_idx=layer_idx):
                if self._should_capture(PipelineStage.FFN_NORM):
                    self.capture_stage(PipelineStage.FFN_NORM, output, layer_idx)
            
            handle = layer.post_attention_layernorm.register_forward_hook(ffn_norm_hook)
            self._hook_handles.append(handle)
            
            # FFN output (after MLP)
            def ffn_output_hook(module, input, output, layer_idx=layer_idx):
                if self._should_capture(PipelineStage.FFN_DOWN):
                    self.capture_stage(PipelineStage.FFN_DOWN, output, layer_idx)
            
            handle = layer.mlp.register_forward_hook(ffn_output_hook)
            self._hook_handles.append(handle)
        
        # Final layer norm
        def final_norm_hook(module, input, output):
            if self._should_capture(PipelineStage.FINAL_NORM):
                self.capture_stage(PipelineStage.FINAL_NORM, output, layer_idx=-1)
        
        handle = model.norm.register_forward_hook(final_norm_hook)
        self._hook_handles.append(handle)
        
        # LM head is captured in forward() method directly
    
    def _should_capture(self, stage: PipelineStage) -> bool:
        """Check if we should capture this stage."""
        return self._capture_stages is None or stage in self._capture_stages
    
    def forward(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        capture_stages: Optional[List[PipelineStage]] = None,
        clear_snapshots: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run forward pass and capture intermediate states.
        
        Args:
            token_ids: Input token IDs
            capture_stages: Which stages to capture (None = all)
            clear_snapshots: Clear previous snapshots
            **kwargs: Additional forward pass options
        
        Returns:
            Dictionary with "logits", "hidden_states", "snapshots"
        """
        if not self.hf_model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if clear_snapshots:
            self.clear_snapshots()
        
        # Set which stages to capture
        self._capture_stages = capture_stages
        
        # Convert input to tensor
        if isinstance(token_ids, np.ndarray):
            token_ids = torch.from_numpy(token_ids)
        elif isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)
        
        # Ensure proper shape: (batch_size, seq_len)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        token_ids = token_ids.to(self.device)
        
        # Run forward pass with no_grad (inference only)
        with torch.no_grad():
            outputs = self.hf_model(
                input_ids=token_ids,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
        
        # Capture LM head output (logits)
        if self._should_capture(PipelineStage.LM_HEAD):
            self.capture_stage(PipelineStage.LM_HEAD, outputs.logits, layer_idx=-1)
        
        # Extract results
        result = {
            "logits": outputs.logits.detach().cpu().float().numpy(),
            "hidden_states": outputs.hidden_states[-1].detach().cpu().float().numpy(),
            "snapshots": self.get_snapshots()
        }
        
        return result
    
    def __del__(self):
        """Cleanup forward hooks on deletion."""
        for handle in self._hook_handles:
            handle.remove()


# Register this implementation
ModelRegistry.register("qwen", QwenReferenceModel)
