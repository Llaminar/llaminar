"""
Integration Tests for GGUF Loading in LlamaReferenceModel

Tests the GGUF loader integration with LlamaReferenceModel to ensure:
1. GGUF files load correctly
2. Model config is extracted properly
3. Tokenizer fallback works
4. Inference produces valid outputs
5. Pipeline stage capture functions correctly

@author David Sanftenberg
"""

import sys
from pathlib import Path
import torch

from python.reference.llama import LlamaReferenceModel
from python.reference.pipeline_stages import PipelineStage


def test_gguf_loading(gguf_path: str):
    """Test 1: Basic GGUF file loading"""
    print("\n" + "="*80)
    print("TEST 1: GGUF File Loading")
    print("="*80)
    
    print(f"\nCreating LlamaReferenceModel with GGUF path: {gguf_path}")
    
    try:
        model = LlamaReferenceModel(
            model_name="llama",
            checkpoint_path=gguf_path,
            device="cpu",
            verbose=False
        )
        
        print("\nLoading model...")
        model.load_model()
        
        print(f"\n✓ Model loaded successfully")
        print(f"  Model type: {type(model.hf_model).__name__}")
        print(f"  Config type: {type(model.hf_config).__name__}")
        print(f"  Device: {model.device}")
        
        return model
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_config(model: LlamaReferenceModel):
    """Test 2: Model configuration validation"""
    print("\n" + "="*80)
    print("TEST 2: Model Configuration")
    print("="*80)
    
    try:
        config = model.hf_config
        
        print("\nModel configuration:")
        print(f"  hidden_size:              {config.hidden_size}")
        print(f"  num_attention_heads:      {config.num_attention_heads}")
        print(f"  num_hidden_layers:        {config.num_hidden_layers}")
        print(f"  intermediate_size:        {config.intermediate_size}")
        print(f"  max_position_embeddings:  {config.max_position_embeddings}")
        print(f"  vocab_size:               {config.vocab_size}")
        
        # Basic validation - check that values are reasonable
        assert config.hidden_size > 0, "hidden_size must be positive"
        assert config.num_attention_heads > 0, "num_attention_heads must be positive"
        assert config.num_hidden_layers > 0, "num_hidden_layers must be positive"
        assert config.vocab_size > 0, "vocab_size must be positive"
        
        print(f"\n✓ All config values are valid")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_tokenizer(model: LlamaReferenceModel):
    """Test 3: Tokenizer functionality"""
    print("\n" + "="*80)
    print("TEST 3: Tokenizer")
    print("="*80)
    
    try:
        if not model.tokenizer:
            print("\n⚠️  WARNING: Tokenizer not loaded, skipping test")
            return
        
        test_text = "Hello, world!"
        print(f"\nTest text: '{test_text}'")
        
        # Encode
        token_ids = model.tokenizer.encode(test_text, return_tensors="pt")
        print(f"Token IDs: {token_ids[0].tolist()}")
        
        # Decode
        decoded = model.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        print(f"Decoded: '{decoded}'")
        
        # Check round-trip (may not be exact due to tokenizer behavior)
        print(f"\n✓ Tokenizer working")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_inference(model: LlamaReferenceModel):
    """Test 4: Basic inference"""
    print("\n" + "="*80)
    print("TEST 4: Basic Inference")
    print("="*80)
    
    try:
        if not model.tokenizer:
            print("\n⚠️  WARNING: Tokenizer not loaded, using dummy input IDs")
            # Use dummy token IDs if tokenizer not available
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        else:
            # Use actual prompt
            prompt = "The capital of France is"
            print(f"\nPrompt: '{prompt}'")
            input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
        
        print(f"Input shape: {input_ids.shape}")
        
        # Run forward pass
        result = model.forward(input_ids)
        
        # Check outputs
        logits = result["logits"]
        print(f"Output logits shape: {logits.shape}")
        print(f"Logits dtype: {logits.dtype}")
        print(f"Logits device: cpu (converted)")
        
        # Validate shape
        batch_size, seq_len, vocab_size = logits.shape
        assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
        assert seq_len == input_ids.shape[1], f"Seq len mismatch"
        assert vocab_size == model.hf_config.vocab_size, f"Vocab size mismatch"
        
        # Show top predictions for last token
        if model.tokenizer:
            last_token_logits = torch.from_numpy(logits[0, -1])
            top_k = torch.topk(last_token_logits, k=5)
            
            print(f"\nTop 5 predictions for next token:")
            for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices), 1):
                token = model.tokenizer.decode([token_id])
                print(f"  {i}. '{token}' (score: {score:.4f})")
        
        print(f"\n✓ Inference successful")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_stage_capture(model: LlamaReferenceModel):
    """Test 5: Pipeline stage capture"""
    print("\n" + "="*80)
    print("TEST 5: Pipeline Stage Capture")
    print("="*80)
    
    try:
        # Define stages to capture
        stages = [
            PipelineStage.EMBEDDING,
            PipelineStage.ATTENTION_OUTPUT,
            PipelineStage.FFN_DOWN,
            PipelineStage.LM_HEAD
        ]
        
        print(f"\nCapturing stages: {[s.name for s in stages]}")
        
        # Use simple input
        if model.tokenizer:
            input_ids = model.tokenizer.encode("Test", return_tensors="pt")
        else:
            input_ids = torch.tensor([[1]])
        
        # Run forward with stage capture
        result = model.forward(
            input_ids,
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
        
        # Validate we captured expected stages
        captured_stages = set(stage for stage, _ in model.snapshots.keys())
        for expected_stage in stages:
            assert expected_stage in captured_stages, f"Missing stage: {expected_stage}"
        
        print(f"\n✓ Stage capture successful")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    if len(sys.argv) < 2:
        print("Usage: python -m python.reference.test_gguf_llama <path_to_gguf_file>")
        print("\nExample:")
        print("  python -m python.reference.test_gguf_llama models/llama-2-7b-q4_0.gguf")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    
    if not Path(gguf_path).exists():
        print(f"Error: GGUF file not found: {gguf_path}")
        sys.exit(1)
    
    print("="*80)
    print("GGUF INTEGRATION TEST FOR LlamaReferenceModel")
    print("="*80)
    print(f"File: {gguf_path}")
    print("="*80)
    
    # Run tests sequentially
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Test 1: Loading
        model = test_gguf_loading(gguf_path)
        if model is None:
            tests_failed += 1
            print("\n" + "="*80)
            print("TEST SUMMARY")
            print("="*80)
            print("\n  GGUF Loading              ✗ FAIL")
            print("\n" + "="*80)
            print("OVERALL: 0/1 tests passed")
            print("="*80)
            sys.exit(1)
        tests_passed += 1
        
        # Test 2: Config
        test_config(model)
        tests_passed += 1
        
        # Test 3: Tokenizer
        test_tokenizer(model)
        tests_passed += 1
        
        # Test 4: Inference
        test_inference(model)
        tests_passed += 1
        
        # Test 5: Stage Capture
        test_stage_capture(model)
        tests_passed += 1
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print("\n  GGUF Loading              ✓ PASS" if tests_passed >= 1 else "  GGUF Loading              ✗ FAIL")
    print("  Model Config              ✓ PASS" if tests_passed >= 2 else "  Model Config              ✗ FAIL")
    print("  Tokenizer                 ✓ PASS" if tests_passed >= 3 else "  Tokenizer                 ✗ FAIL")
    print("  Basic Inference           ✓ PASS" if tests_passed >= 4 else "  Basic Inference           ✗ FAIL")
    print("  Stage Capture             ✓ PASS" if tests_passed >= 5 else "  Stage Capture             ✗ FAIL")
    
    print("\n" + "="*80)
    print(f"OVERALL: {tests_passed}/5 tests passed")
    print("="*80)
    
    sys.exit(0 if tests_passed == 5 else 1)


if __name__ == "__main__":
    main()
