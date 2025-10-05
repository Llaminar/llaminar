"""
Test GGUF integration with QwenReferenceModel.

Verifies that QwenReferenceModel can load GGUF files and run inference.

Usage:
    python -m python.reference.test_gguf_qwen models/qwen2.5-0.5b-instruct-q4_0.gguf

Author: David Sanftenberg
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.reference.qwen import QwenReferenceModel
from python.reference.pipeline_stages import PipelineStage


def test_gguf_loading(gguf_path: str):
    """Test loading a GGUF file with QwenReferenceModel"""
    print(f"\n{'='*80}")
    print(f"TEST 1: GGUF File Loading")
    print(f"{'='*80}\n")
    
    print(f"Creating QwenReferenceModel with GGUF path: {gguf_path}")
    
    # Create model (auto-detects .gguf extension)
    model = QwenReferenceModel(
        model_name="qwen2-gguf",
        checkpoint_path=gguf_path,
        verbose=True
    )
    
    # Load model
    print(f"\nLoading model...")
    model.load_model()
    
    print(f"\n✓ Model loaded successfully")
    print(f"  Model type: {type(model.hf_model).__name__}")
    print(f"  Config type: {type(model.hf_config).__name__}")
    print(f"  Device: {model.device}")
    
    return model


def test_model_config(model):
    """Test that model config is correct"""
    print(f"\n{'='*80}")
    print(f"TEST 2: Model Configuration")
    print(f"{'='*80}\n")
    
    config = model.hf_config
    
    print(f"Model configuration:")
    print(f"  hidden_size:              {config.hidden_size}")
    print(f"  num_attention_heads:      {config.num_attention_heads}")
    print(f"  num_hidden_layers:        {config.num_hidden_layers}")
    print(f"  intermediate_size:        {config.intermediate_size}")
    print(f"  max_position_embeddings:  {config.max_position_embeddings}")
    print(f"  vocab_size:               {config.vocab_size}")
    
    # Validate expected values for qwen2.5-0.5b
    expected = {
        'hidden_size': 896,
        'num_attention_heads': 14,
        'num_hidden_layers': 24,
        'vocab_size': 151936
    }
    
    all_match = True
    for key, expected_val in expected.items():
        actual_val = getattr(config, key)
        if actual_val != expected_val:
            print(f"\n  ✗ {key}: expected {expected_val}, got {actual_val}")
            all_match = False
    
    if all_match:
        print(f"\n✓ All config values match expected")
    
    return all_match


def test_tokenizer(model):
    """Test that tokenizer works"""
    print(f"\n{'='*80}")
    print(f"TEST 3: Tokenizer")
    print(f"{'='*80}\n")
    
    test_text = "Hello, world!"
    
    print(f"Test text: '{test_text}'")
    
    # Tokenize
    tokens = model.tokenizer.encode(test_text)
    print(f"Token IDs: {tokens}")
    
    # Decode
    decoded = model.tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")
    
    print(f"\n✓ Tokenizer working")
    
    return True


def test_inference(model):
    """Test basic inference"""
    print(f"\n{'='*80}")
    print(f"TEST 4: Basic Inference")
    print(f"{'='*80}\n")
    
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not available, skipping inference test")
        return False
    
    # Simple prompt
    prompt = "The capital of France is"
    print(f"Prompt: '{prompt}'")
    
    # Tokenize
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model.hf_model(input_ids)
        logits = outputs.logits
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits device: {logits.device}")
    
    # Get top-5 predictions for last token
    last_token_logits = logits[0, -1, :]
    top_k = torch.topk(last_token_logits, k=5)
    
    print(f"\nTop 5 predictions for next token:")
    for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
        token = model.tokenizer.decode([token_id.item()])
        print(f"  {i+1}. '{token}' (score: {score.item():.4f})")
    
    print(f"\n✓ Inference successful")
    
    return True


def test_stage_capture(model):
    """Test capturing pipeline stages"""
    print(f"\n{'='*80}")
    print(f"TEST 5: Pipeline Stage Capture")
    print(f"{'='*80}\n")
    
    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not available, skipping stage capture test")
        return False
    
    # Simple prompt
    prompt = "Hello"
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    # Capture specific stages
    stages = [
        PipelineStage.EMBEDDING,
        PipelineStage.ATTENTION_OUTPUT,
        PipelineStage.FFN_DOWN,
        PipelineStage.LM_HEAD
    ]
    
    print(f"Capturing stages: {[s.name for s in stages]}")
    
    # Run forward with capture
    result = model.forward(
        input_ids.squeeze(0).tolist(),
        capture_stages=stages
    )
    
    print(f"\nCaptured {len(model.snapshots)} snapshots:")
    # Sort snapshots by stage value and layer index for consistent output
    sorted_snapshots = sorted(model.snapshots.items(), key=lambda x: (x[0][0].value, x[0][1]))
    for (stage, layer_idx), snapshot in sorted_snapshots:
        if layer_idx == -1:
            print(f"  {stage.name:25s} (global): {snapshot.shape}")
        else:
            print(f"  {stage.name:25s} (layer {layer_idx:2d}): {snapshot.shape}")
    
    print(f"\n✓ Stage capture successful")
    
    return True


def main(gguf_path: str):
    """Run all tests"""
    print(f"\n{'='*80}")
    print(f"GGUF INTEGRATION TEST FOR QwenReferenceModel")
    print(f"{'='*80}")
    print(f"File: {gguf_path}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Test 1: Loading
    try:
        model = test_gguf_loading(gguf_path)
        results.append(("GGUF Loading", True))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("GGUF Loading", False))
        return False
    
    # Test 2: Config
    try:
        config_ok = test_model_config(model)
        results.append(("Model Config", config_ok))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Model Config", False))
    
    # Test 3: Tokenizer
    try:
        tokenizer_ok = test_tokenizer(model)
        results.append(("Tokenizer", tokenizer_ok))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Tokenizer", False))
    
    # Test 4: Inference
    try:
        inference_ok = test_inference(model)
        results.append(("Basic Inference", inference_ok))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Basic Inference", False))
    
    # Test 5: Stage capture
    try:
        capture_ok = test_stage_capture(model)
        results.append(("Stage Capture", capture_ok))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Stage Capture", False))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:25s} {status}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL: {passed}/{total} tests passed")
    print(f"{'='*80}\n")
    
    return passed == total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m python.reference.test_gguf_qwen <gguf_file>")
        print("\nExample:")
        print("  python -m python.reference.test_gguf_qwen models/qwen2.5-0.5b-instruct-q4_0.gguf")
        sys.exit(1)
    
    success = main(sys.argv[1])
    sys.exit(0 if success else 1)
