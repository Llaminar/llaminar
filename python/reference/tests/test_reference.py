"""
Basic Tests for Reference Implementation

Tests the base architecture without requiring large model downloads.

@author David Sanftenberg
"""

import pytest
import numpy as np

from python.reference import (
    PipelineStage,
    AbstractReferenceModel,
    ModelRegistry,
    create_reference_model
)
from python.reference.pipeline_stages import stage_to_string, string_to_stage


class TestPipelineStage:
    """Test PipelineStage enum and conversion functions."""
    
    def test_stage_to_string(self):
        """Test stage enum to string conversion."""
        assert stage_to_string(PipelineStage.EMBEDDING) == "EMBEDDING"
        assert stage_to_string(PipelineStage.ATTENTION_OUTPUT) == "ATTENTION_OUTPUT"
        assert stage_to_string(PipelineStage.LM_HEAD) == "LM_HEAD"
    
    def test_string_to_stage(self):
        """Test string to stage enum conversion."""
        assert string_to_stage("EMBEDDING") == PipelineStage.EMBEDDING
        assert string_to_stage("ATTENTION_OUTPUT") == PipelineStage.ATTENTION_OUTPUT
        assert string_to_stage("LM_HEAD") == PipelineStage.LM_HEAD
    
    def test_roundtrip(self):
        """Test roundtrip conversion."""
        for stage in PipelineStage:
            string = stage_to_string(stage)
            recovered = string_to_stage(string)
            assert recovered == stage


class TestModelRegistry:
    """Test ModelRegistry registration and factory."""
    
    def test_registry_has_models(self):
        """Test that models are registered."""
        models = ModelRegistry.list_models()
        assert "qwen" in models
        assert "llama" in models  # Even though it's a stub
    
    def test_create_qwen_model(self):
        """Test creating Qwen model (doesn't load weights)."""
        model = ModelRegistry.create(
            "qwen",
            checkpoint_path="Qwen/Qwen2-0.5B",
            auto_load=False
        )
        assert isinstance(model, AbstractReferenceModel)
        assert model.model_name == "qwen"
        assert "Qwen2-0.5B" in model.checkpoint_path
    
    def test_unknown_model_raises(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.create("nonexistent", "some/path")
    
    def test_is_registered(self):
        """Test is_registered() method."""
        assert ModelRegistry.is_registered("qwen")
        assert ModelRegistry.is_registered("llama")
        assert not ModelRegistry.is_registered("nonexistent")


class TestAbstractReferenceModel:
    """Test AbstractReferenceModel base functionality."""
    
    def test_snapshot_capture(self):
        """Test snapshot capture and retrieval."""
        # We'll use a mock since we can't instantiate abstract class directly
        # Instead, create a minimal concrete implementation
        
        class MockModel(AbstractReferenceModel):
            def load_model(self, **kwargs):
                pass
            
            def forward(self, token_ids, **kwargs):
                return {"logits": np.array([1, 2, 3])}
        
        model = MockModel("mock", "mock/path")
        
        # Simulate capturing a stage
        import torch
        dummy_tensor = torch.randn(1, 10, 512)
        model.capture_stage(PipelineStage.EMBEDDING, dummy_tensor, layer_idx=-1)
        
        # Check snapshot was captured
        snapshots = model.get_snapshots()
        assert (PipelineStage.EMBEDDING, -1) in snapshots
        
        # Check it's a numpy array
        snapshot = snapshots[(PipelineStage.EMBEDDING, -1)]
        assert isinstance(snapshot, np.ndarray)
        assert snapshot.shape == (1, 10, 512)
    
    def test_clear_snapshots(self):
        """Test clearing snapshots."""
        class MockModel(AbstractReferenceModel):
            def load_model(self, **kwargs):
                pass
            def forward(self, token_ids, **kwargs):
                return {"logits": np.array([1, 2, 3])}
        
        model = MockModel("mock", "mock/path")
        
        import torch
        model.capture_stage(PipelineStage.EMBEDDING, torch.randn(1, 10), -1)
        assert len(model.get_snapshots()) == 1
        
        model.clear_snapshots()
        assert len(model.get_snapshots()) == 0


class TestHuggingFaceReferenceModel:
    """Test HuggingFaceReferenceModel shared infrastructure."""

    def test_subclass_only_needs_create_and_fallbacks(self):
        """New model implementations only need two small methods."""
        from python.reference.base import HuggingFaceReferenceModel

        class MinimalModel(HuggingFaceReferenceModel):
            def _create_model_from_gguf_config(self, config_dict, torch_dtype):
                return None, None  # placeholder

            def _tokenizer_fallbacks(self):
                return ["some/model"]

        model = MinimalModel("minimal", "some/path")
        assert model.model_name == "minimal"
        # Should NOT raise - it's a fully concrete class
        assert model._tokenizer_fallbacks() == ["some/model"]

    def test_default_tokenizer_fallbacks_empty(self):
        """Default fallback list is empty (must be overridden per-model)."""
        from python.reference.base import HuggingFaceReferenceModel

        class NoFallbackModel(HuggingFaceReferenceModel):
            def _create_model_from_gguf_config(self, config_dict, torch_dtype):
                return None, None

        model = NoFallbackModel("nofb", "some/path")
        assert model._tokenizer_fallbacks() == []

    def test_forward_raises_without_load(self):
        """forward() raises RuntimeError when model not loaded."""
        from python.reference.base import HuggingFaceReferenceModel

        class StubModel(HuggingFaceReferenceModel):
            def _create_model_from_gguf_config(self, config_dict, torch_dtype):
                return None, None

        model = StubModel("stub", "some/path")
        with pytest.raises(RuntimeError, match="not loaded"):
            model.forward([1, 2, 3])

    def test_should_capture_all_when_none(self):
        """_should_capture returns True for every stage when capture_stages is None."""
        from python.reference.base import HuggingFaceReferenceModel

        class StubModel(HuggingFaceReferenceModel):
            def _create_model_from_gguf_config(self, config_dict, torch_dtype):
                return None, None

        model = StubModel("stub", "some/path")
        model._capture_stages = None
        for stage in PipelineStage:
            assert model._should_capture(stage) is True

    def test_should_capture_filters_correctly(self):
        """_should_capture filters by the provided list."""
        from python.reference.base import HuggingFaceReferenceModel

        class StubModel(HuggingFaceReferenceModel):
            def _create_model_from_gguf_config(self, config_dict, torch_dtype):
                return None, None

        model = StubModel("stub", "some/path")
        model._capture_stages = [PipelineStage.EMBEDDING, PipelineStage.LM_HEAD]
        assert model._should_capture(PipelineStage.EMBEDDING) is True
        assert model._should_capture(PipelineStage.LM_HEAD) is True
        assert model._should_capture(PipelineStage.ATTENTION_OUTPUT) is False
        assert model._should_capture(PipelineStage.FFN_DOWN) is False

    def test_qwen_inherits_from_hf_base(self):
        """QwenReferenceModel inherits from HuggingFaceReferenceModel."""
        from python.reference.base import HuggingFaceReferenceModel
        from python.reference.qwen import QwenReferenceModel

        assert issubclass(QwenReferenceModel, HuggingFaceReferenceModel)
        model = QwenReferenceModel("qwen", "some/path")
        assert isinstance(model, HuggingFaceReferenceModel)
        assert isinstance(model, AbstractReferenceModel)

    def test_llama_inherits_from_hf_base(self):
        """LlamaReferenceModel inherits from HuggingFaceReferenceModel."""
        from python.reference.base import HuggingFaceReferenceModel
        from python.reference.llama import LlamaReferenceModel

        assert issubclass(LlamaReferenceModel, HuggingFaceReferenceModel)
        model = LlamaReferenceModel("llama", "some/path")
        assert isinstance(model, HuggingFaceReferenceModel)

    def test_qwen_tokenizer_fallbacks(self):
        """Qwen model provides Qwen-specific tokenizer fallbacks."""
        from python.reference.qwen import QwenReferenceModel

        model = QwenReferenceModel("qwen", "some/path")
        fallbacks = model._tokenizer_fallbacks()
        assert len(fallbacks) >= 1
        assert any("Qwen" in fb for fb in fallbacks)

    def test_llama_tokenizer_fallbacks(self):
        """LLaMA model provides LLaMA-specific tokenizer fallbacks."""
        from python.reference.llama import LlamaReferenceModel

        model = LlamaReferenceModel("llama", "some/path")
        fallbacks = model._tokenizer_fallbacks()
        assert len(fallbacks) >= 1
        assert any("llama" in fb.lower() or "TinyLlama" in fb for fb in fallbacks)

    def test_register_custom_model(self):
        """Demonstrate how easy it is to register a new model implementation."""
        from python.reference.base import HuggingFaceReferenceModel

        class CustomModel(HuggingFaceReferenceModel):
            def _create_model_from_gguf_config(self, config_dict, torch_dtype):
                return None, None

            def _tokenizer_fallbacks(self):
                return ["custom/model-v1"]

        ModelRegistry.register("custom_test", CustomModel)
        assert ModelRegistry.is_registered("custom_test")

        model = ModelRegistry.create("custom_test", "some/path", auto_load=False)
        assert isinstance(model, CustomModel)
        assert isinstance(model, HuggingFaceReferenceModel)
        assert model._tokenizer_fallbacks() == ["custom/model-v1"]

        # Cleanup — remove from registry to avoid polluting other tests
        del ModelRegistry._registry["custom_test"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
