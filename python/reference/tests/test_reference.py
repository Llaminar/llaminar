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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
