"""
Model Registry and Factory Pattern

Provides a clean interface for creating reference model instances without
tight coupling to specific implementations.

@author David Sanftenberg
"""

from typing import Dict, Type, Optional, Any
from pathlib import Path

from .base import AbstractReferenceModel


class ModelRegistry:
    """
    Registry for reference model implementations.
    
    Model-specific modules should register themselves on import:
        ModelRegistry.register("qwen", QwenReferenceModel)
    
    Users can then create models via factory function:
        model = create_reference_model("qwen", checkpoint="...")
    """
    
    _registry: Dict[str, Type[AbstractReferenceModel]] = {}
    
    @classmethod
    def register(
        cls,
        model_name: str,
        model_class: Type[AbstractReferenceModel]
    ) -> None:
        """
        Register a model implementation.
        
        Args:
            model_name: Identifier for this model (e.g., "qwen", "llama")
            model_class: Class that inherits from AbstractReferenceModel
        
        Raises:
            ValueError: If model_name already registered or invalid
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        if not issubclass(model_class, AbstractReferenceModel):
            raise ValueError(
                f"{model_class.__name__} must inherit from AbstractReferenceModel"
            )
        
        if model_name in cls._registry:
            raise ValueError(
                f"Model '{model_name}' already registered "
                f"({cls._registry[model_name].__name__})"
            )
        
        cls._registry[model_name] = model_class
    
    @classmethod
    def create(
        cls,
        model_name: str,
        checkpoint_path: str,
        **kwargs
    ) -> AbstractReferenceModel:
        """
        Create a reference model instance.
        
        Args:
            model_name: Registered model identifier
            checkpoint_path: Path to checkpoint or HuggingFace model ID
            **kwargs: Passed to model constructor
        
        Returns:
            Instantiated model (not yet loaded - call load_model())
        
        Raises:
            ValueError: If model_name not registered
        """
        if model_name not in cls._registry:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available models: {available}"
            )
        
        model_class = cls._registry[model_name]
        return model_class(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
    
    @classmethod
    def list_models(cls) -> list[str]:
        """Get list of registered model names."""
        return sorted(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model is registered."""
        return model_name in cls._registry


def create_reference_model(
    model_name: str,
    checkpoint_path: str,
    auto_load: bool = True,
    **kwargs
) -> AbstractReferenceModel:
    """
    Factory function to create and optionally load a reference model.
    
    This is the primary API for users:
        model = create_reference_model("qwen", "Qwen/Qwen2-0.5B")
    
    Args:
        model_name: Model architecture ("qwen", "llama", etc.)
        checkpoint_path: HuggingFace model ID or local path
        auto_load: Whether to automatically call load_model()
        **kwargs: Passed to model constructor and load_model()
    
    Returns:
        Loaded reference model ready for inference
    
    Raises:
        ValueError: If model_name not registered
        RuntimeError: If auto_load=True and loading fails
    
    Example:
        >>> model = create_reference_model("qwen", "Qwen/Qwen2-0.5B")
        >>> result = model.forward([1, 2, 3])
        >>> print(result["logits"].shape)
    """
    model = ModelRegistry.create(model_name, checkpoint_path, **kwargs)
    
    if auto_load:
        model.load_model(**kwargs)
    
    return model
