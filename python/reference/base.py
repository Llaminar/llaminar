"""
Abstract Base Class for Reference Model Implementations

Defines the interface that all model-specific implementations must follow.
This ensures consistency and makes it easy to add support for new architectures.

@author David Sanftenberg
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import numpy as np
import torch

from .pipeline_stages import PipelineStage


class AbstractReferenceModel(ABC):
    """
    Base class for PyTorch reference implementations of transformer models.
    
    All model-specific implementations (Qwen, LLaMA, etc.) should inherit from this
    class and implement the required abstract methods.
    
    The primary purpose is to run forward passes while capturing intermediate
    pipeline states for comparison with Llaminar's C++ implementation.
    
    Attributes:
        model_name: Identifier for the model architecture (e.g., "qwen", "llama")
        checkpoint_path: Path to model checkpoint or HuggingFace model ID
        device: PyTorch device (cpu, cuda, etc.)
        dtype: Default data type for computations
        snapshots: Captured pipeline stage outputs
    """
    
    def __init__(
        self,
        model_name: str,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        """
        Initialize the reference model.
        
        Args:
            model_name: Architecture identifier (e.g., "qwen", "llama")
            checkpoint_path: Path to checkpoint or HuggingFace model ID
            device: PyTorch device string
            dtype: Default tensor dtype
            **kwargs: Model-specific configuration options
        """
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config_kwargs = kwargs
        
        # Snapshot storage: {(stage, layer_idx): np.ndarray}
        # layer_idx = -1 for non-layer stages (embedding, final_norm, lm_head)
        self.snapshots: Dict[tuple[PipelineStage, int], np.ndarray] = {}
        
        # Model and tokenizer (set by subclasses)
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
    
    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """
        Load the model and tokenizer from checkpoint.
        
        Subclasses should:
        1. Load the model architecture
        2. Load weights from checkpoint_path
        3. Set up tokenizer
        4. Apply any quantization or optimization
        5. Register forward hooks for stage capture
        
        Args:
            **kwargs: Model-specific loading options
        
        Raises:
            RuntimeError: If model fails to load
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        capture_stages: Optional[List[PipelineStage]] = None,
        clear_snapshots: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run forward pass and optionally capture intermediate states.
        
        Args:
            token_ids: Input token IDs (will be converted to tensor)
            capture_stages: Which stages to capture (None = all stages)
            clear_snapshots: Whether to clear previous snapshots before running
            **kwargs: Model-specific forward options
        
        Returns:
            Dictionary containing:
                - "logits": Final output logits (np.ndarray)
                - "hidden_states": Final hidden state before LM head (np.ndarray)
                - "snapshots": Dict of captured stages
        
        Raises:
            RuntimeError: If forward pass fails
        """
        pass
    
    def capture_stage(
        self,
        stage: PipelineStage,
        tensor: torch.Tensor,
        layer_idx: int = -1
    ) -> None:
        """
        Capture a pipeline stage output for later comparison.
        
        Args:
            stage: Which pipeline stage this tensor represents
            tensor: The tensor to capture (will be copied to CPU and converted to numpy)
            layer_idx: Layer index (0-based), or -1 for non-layer stages
        """
        # Detach from computation graph, move to CPU, convert to numpy
        snapshot = tensor.detach().cpu().float().numpy()
        
        # Store with (stage, layer) key
        key = (stage, layer_idx)
        self.snapshots[key] = snapshot
    
    def get_snapshots(self) -> Dict[tuple[PipelineStage, int], np.ndarray]:
        """
        Get all captured snapshots.
        
        Returns:
            Dictionary mapping (stage, layer_idx) to numpy arrays
        """
        return self.snapshots.copy()
    
    def clear_snapshots(self) -> None:
        """Clear all captured snapshots."""
        self.snapshots.clear()
    
    def export_snapshots(
        self,
        output_path: Union[str, Path],
        format: str = "npz"
    ) -> None:
        """
        Export captured snapshots to file for C++ test integration.
        
        Args:
            output_path: Path to save snapshots
            format: Export format ("npz" or "json")
        
        Raises:
            ValueError: If format is unsupported
            RuntimeError: If no snapshots to export
        """
        if not self.snapshots:
            raise RuntimeError("No snapshots to export. Run forward() first.")
        
        output_path = Path(output_path)
        
        if format == "npz":
            # NumPy .npz format: efficient for large arrays
            # Keys: "stage_layer" (e.g., "EMBEDDING_-1", "ATTENTION_OUTPUT_0")
            arrays = {}
            for (stage, layer_idx), array in self.snapshots.items():
                from .pipeline_stages import stage_to_string
                key = f"{stage_to_string(stage)}_{layer_idx}"
                arrays[key] = array
            
            np.savez_compressed(output_path, **arrays)
            
        elif format == "json":
            # JSON format: human-readable but less efficient
            import json
            from .pipeline_stages import stage_to_string
            
            data = {
                "model_name": self.model_name,
                "checkpoint": self.checkpoint_path,
                "snapshots": {}
            }
            
            for (stage, layer_idx), array in self.snapshots.items():
                key = f"{stage_to_string(stage)}_{layer_idx}"
                data["snapshots"][key] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "data": array.flatten().tolist()
                }
            
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"checkpoint='{self.checkpoint_path}', "
            f"device='{self.device}', "
            f"snapshots={len(self.snapshots)})"
        )


class HuggingFaceReferenceModel(AbstractReferenceModel):
    """
    Concrete base class for HuggingFace transformers-based reference models.

    Provides shared infrastructure used by all decoder-only transformer models
    that follow the standard HF pattern (embed_tokens → layers → norm → lm_head):
      - GGUF / HuggingFace checkpoint loading dispatch
      - Forward-hook registration for pipeline stage capture
      - Unified ``forward()`` implementation
      - HuggingFace checkpoint loading (``from_pretrained``)

    Subclasses (Qwen, LLaMA, future Qwen 3.5 Gated Delta Net, etc.) only need
    to implement :meth:`_create_model_from_gguf_config` which returns the
    model-specific ``(Config, Model)`` pair from a GGUF config dict, and
    optionally override :meth:`_tokenizer_fallbacks` to customise tokenizer
    resolution order.

    Example subclass::

        class Qwen35GatedDeltaNetReferenceModel(HuggingFaceReferenceModel):
            def _create_model_from_gguf_config(self, config_dict, torch_dtype):
                from transformers import Qwen3Config, Qwen3ForCausalLM
                cfg = Qwen3Config(**config_dict)
                cfg._attn_implementation = 'eager'
                if torch_dtype:
                    cfg.torch_dtype = torch_dtype
                return cfg, Qwen3ForCausalLM(cfg)

            def _tokenizer_fallbacks(self):
                return ["Qwen/Qwen3-0.6B"]

        ModelRegistry.register("qwen3.5_gated_delta_net",
                               Qwen35GatedDeltaNetReferenceModel)
    """

    def __init__(self, model_name: str, checkpoint_path, **kwargs):
        super().__init__(model_name, checkpoint_path, **kwargs)
        self.hf_model = None
        self.hf_config = None
        self._hook_handles: list = []
        self._capture_stages = None
        self.verbose: bool = kwargs.get("verbose", False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(
        self,
        quantization=None,
        torch_dtype=None,
        **kwargs,
    ) -> None:
        """Load from GGUF (auto-detected) or HuggingFace checkpoint."""
        try:
            if str(self.checkpoint_path).endswith(".gguf"):
                self._load_from_gguf(str(self.checkpoint_path), torch_dtype, **kwargs)
            else:
                self._load_from_huggingface(quantization, torch_dtype, **kwargs)
            self._register_hooks()
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name} model: {e}") from e

    def forward(self, token_ids, capture_stages=None, clear_snapshots=True,
                past_key_values=None, use_cache=False, **kwargs):
        """Run forward pass and capture intermediate states.

        When ``use_cache=True``, returns the updated ``past_key_values`` in the
        result dict so the caller can chain incremental decode steps without
        recomputing K/V (and recurrent state for GDN layers) for the prompt.
        For Qwen3.5's hybrid architecture, HF's ``DynamicCache`` handles both
        attention K/V and the gated-delta-net recurrent state transparently.
        """
        if not self.hf_model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if clear_snapshots:
            self.clear_snapshots()

        self._capture_stages = capture_stages

        if isinstance(token_ids, np.ndarray):
            token_ids = torch.from_numpy(token_ids)
        elif isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        token_ids = token_ids.to(self.device)

        # When resuming from a cache, we must pass position_ids that pick up
        # where the cache left off. HF will derive these from cache length if
        # we don't pass attention_mask, but being explicit keeps things clear.
        forward_kwargs = dict(kwargs)
        if past_key_values is not None and "position_ids" not in forward_kwargs:
            seen = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else 0
            new_len = token_ids.shape[1]
            forward_kwargs["position_ids"] = torch.arange(
                seen, seen + new_len, device=self.device
            ).unsqueeze(0)

        with torch.no_grad():
            outputs = self.hf_model(
                input_ids=token_ids,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **forward_kwargs,
            )

        if self._should_capture(PipelineStage.LM_HEAD):
            self.capture_stage(PipelineStage.LM_HEAD, outputs.logits, layer_idx=-1)

        result = {
            "logits": outputs.logits.detach().cpu().float().numpy(),
            "hidden_states": outputs.hidden_states[-1].detach().cpu().float().numpy(),
            "snapshots": self.get_snapshots(),
        }
        if use_cache:
            result["past_key_values"] = outputs.past_key_values
        return result

    def __del__(self):
        for handle in self._hook_handles:
            handle.remove()

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_model_from_gguf_config(
        self,
        config_dict: dict,
        torch_dtype,
    ) -> tuple:
        """
        Create a ``(config, model)`` pair from GGUF config dict.

        Returns:
            (hf_config, hf_model) — both un-moved to device (caller handles that).
        """
        ...

    def _tokenizer_fallbacks(self) -> list[str]:
        """Return a list of HuggingFace model IDs to try as tokenizer fallbacks."""
        return []

    # ------------------------------------------------------------------
    # Shared hook registration (standard decoder-only transformer layout)
    # ------------------------------------------------------------------

    def _should_capture(self, stage: PipelineStage) -> bool:
        return self._capture_stages is None or stage in self._capture_stages

    def _register_hooks(self) -> None:
        """Register forward hooks on the standard decoder-only layout."""
        if not self.hf_model:
            return

        model = self.hf_model.model  # underlying transformer

        # Embedding
        def _emb(module, inp, out):
            if self._should_capture(PipelineStage.EMBEDDING):
                self.capture_stage(PipelineStage.EMBEDDING, out, layer_idx=-1)

        self._hook_handles.append(model.embed_tokens.register_forward_hook(_emb))

        # Per-layer hooks
        for idx, layer in enumerate(model.layers):
            def _attn_norm(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.ATTENTION_NORM):
                    self.capture_stage(PipelineStage.ATTENTION_NORM, out, i)
            self._hook_handles.append(layer.input_layernorm.register_forward_hook(_attn_norm))

            def _attn_out(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.ATTENTION_OUTPUT):
                    h = out[0] if isinstance(out, tuple) else out
                    self.capture_stage(PipelineStage.ATTENTION_OUTPUT, h, i)
            self._hook_handles.append(layer.self_attn.register_forward_hook(_attn_out))

            def _ffn_norm(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.FFN_NORM):
                    self.capture_stage(PipelineStage.FFN_NORM, out, i)
            self._hook_handles.append(layer.post_attention_layernorm.register_forward_hook(_ffn_norm))

            def _ffn_out(mod, inp, out, i=idx):
                if self._should_capture(PipelineStage.FFN_DOWN):
                    self.capture_stage(PipelineStage.FFN_DOWN, out, i)
            self._hook_handles.append(layer.mlp.register_forward_hook(_ffn_out))

        # Final norm
        def _fnorm(mod, inp, out):
            if self._should_capture(PipelineStage.FINAL_NORM):
                self.capture_stage(PipelineStage.FINAL_NORM, out, layer_idx=-1)
        self._hook_handles.append(model.norm.register_forward_hook(_fnorm))

    # ------------------------------------------------------------------
    # Shared GGUF & HuggingFace loading logic
    # ------------------------------------------------------------------

    def _load_from_gguf(self, gguf_path: str, torch_dtype=None, **kwargs) -> None:
        import warnings
        from .loaders import GGUFLoader
        from transformers.initialization import no_init_weights

        print(f"Loading GGUF file: {gguf_path}")
        loader = GGUFLoader(gguf_path, verbose=self.verbose)
        config_dict, state_dict = loader.load(
            as_transformers_config=False,
            as_torch=True,
            show_progress=self.verbose,
        )

        # PERF: Skip torch's random weight initialization (normal_ / uniform_ /
        # kaiming_uniform_) during model construction. Profiling showed that
        # for a 4B Qwen3.5 model, ~50s of the 82s wall-clock generation time
        # was spent on torch.nn.init.normal_ / kaiming_uniform_ filling tensors
        # that are immediately overwritten by GGUF state_dict load. The
        # transformers `no_init_weights()` context manager patches those init
        # functions to no-ops for the duration of model construction.
        with no_init_weights():
            self.hf_config, self.hf_model = self._create_model_from_gguf_config(
                config_dict, torch_dtype
            )

        print(f"Loading {len(state_dict)} tensors into model...")
        missing, unexpected = self.hf_model.load_state_dict(state_dict, strict=False)

        if "lm_head.weight" in missing or missing == ["lm_head.weight"]:
            print("Tying lm_head.weight to model.embed_tokens.weight (weight sharing)")
            self.hf_model.lm_head.weight = self.hf_model.model.embed_tokens.weight

        if missing and missing != ["lm_head.weight"]:
            warnings.warn(f"Missing keys when loading GGUF: {missing}")
        if unexpected:
            warnings.warn(f"Unexpected keys when loading GGUF: {unexpected}")

        self.hf_model = self.hf_model.to(self.device)
        self.hf_model.eval()

        # Tokenizer resolution: local dir → model-specific fallbacks
        self._resolve_tokenizer(gguf_path)
        print("✓ GGUF model loaded successfully")

    def _resolve_tokenizer(self, gguf_path: str) -> None:
        """Try to load tokenizer from local dir, then HF cache, then network."""
        from transformers import AutoTokenizer as _AT

        local_dir = Path(gguf_path).parent
        try:
            self.tokenizer = _AT.from_pretrained(local_dir)
            print(f"Loaded tokenizer from {local_dir}")
            return
        except Exception:
            pass

        # PERF: First try the local HF cache only (no network) — saves ~2s of
        # SSL handshake + download checks per process when the tokenizer was
        # already cached by a previous run. Only fall back to a real network
        # fetch if the cached lookup fails.
        for fb in self._tokenizer_fallbacks():
            try:
                self.tokenizer = _AT.from_pretrained(fb, local_files_only=True)
                print(f"Loaded tokenizer from {fb} (local cache)")
                return
            except Exception:
                continue

        for fb in self._tokenizer_fallbacks():
            try:
                self.tokenizer = _AT.from_pretrained(fb)
                print(f"Loaded tokenizer from {fb}")
                return
            except Exception:
                continue

        raise RuntimeError(
            "Could not load tokenizer. Provide tokenizer files next to the "
            "GGUF file or add a fallback via _tokenizer_fallbacks()."
        )

    def _load_from_huggingface(self, quantization=None, torch_dtype=None, **kwargs) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.hf_config = AutoConfig.from_pretrained(self.checkpoint_path)

        model_kwargs = kwargs.copy()
        if quantization:
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype or self.dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(f"Unsupported quantization: {quantization}")

        model_kwargs.setdefault("torch_dtype", torch_dtype or self.dtype)
        model_kwargs.setdefault("device_map", str(self.device))

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path, **model_kwargs
        )
        self.hf_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
