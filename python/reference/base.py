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
