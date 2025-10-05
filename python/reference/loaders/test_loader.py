"""
Test script for GGUFLoader.

Complete end-to-end test of GGUF loading pipeline.

Usage:
    python -m python.reference.loaders.test_loader models/qwen2.5-0.5b-instruct-q4_0.gguf
    
Tests:
1. Config extraction (GGUF metadata → dict/transformers config)
2. State dict loading (all tensors dequantized and mapped)
3. Tensor validation (shapes, values, statistics)
4. Integration readiness (compatible with AutoModelForCausalLM)

Author: David Sanftenberg
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from python.reference.loaders import GGUFLoader


def test_loader_basic(gguf_path: str):
    """Test basic GGUFLoader functionality"""
    print(f"\n{'='*80}")
    print(f"TEST 1: Basic GGUFLoader Functionality")
    print(f"{'='*80}\n")
    
    print(f"Testing file: {gguf_path}\n")
    
    # Test config-only loading
    print("Loading config only...")
    loader = GGUFLoader(gguf_path, verbose=False)
    config = loader.load_config(as_transformers_config=False)
    
    print(f"\n✓ Config loaded successfully")
    print(f"  Config type: {type(config).__name__}")
    print(f"  Config keys: {sorted(config.keys())}")
    print(f"\nConfig values:")
    for key, value in sorted(config.items()):
        print(f"  {key:30s} = {value}")
    
    return config


def test_loader_state_dict(gguf_path: str):
    """Test state dict loading"""
    print(f"\n{'='*80}")
    print(f"TEST 2: State Dict Loading")
    print(f"{'='*80}\n")
    
    loader = GGUFLoader(gguf_path, verbose=False)
    
    # Load as NumPy arrays (works without PyTorch)
    print("Loading state dict (as NumPy arrays)...")
    state_dict = loader.load_state_dict(as_torch=False, show_progress=False)
    
    print(f"\n✓ State dict loaded successfully")
    print(f"  Total tensors: {len(state_dict)}")
    
    # Analyze tensor statistics
    print(f"\nTensor statistics:")
    total_elements = 0
    total_size_mb = 0
    
    for name, array in state_dict.items():
        total_elements += array.size
        total_size_mb += array.nbytes / 1024**2
    
    print(f"  Total parameters: {total_elements:,}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    
    # Show sample tensors
    print(f"\nSample tensors (first 10):")
    for i, (name, array) in enumerate(state_dict.items()):
        if i >= 10:
            print(f"  ... and {len(state_dict) - 10} more")
            break
        print(f"  {name:60s} {str(array.shape):20s} {array.dtype}")
    
    return state_dict


def test_loader_validation(gguf_path: str, state_dict: dict):
    """Validate loaded tensors"""
    print(f"\n{'='*80}")
    print(f"TEST 3: Tensor Validation")
    print(f"{'='*80}\n")
    
    # Check for NaN/Inf
    print("Checking for NaN/Inf values...")
    nan_count = 0
    inf_count = 0
    
    for name, array in state_dict.items():
        if np.isnan(array).any():
            nan_count += 1
            print(f"  WARNING: NaN found in {name}")
        if np.isinf(array).any():
            inf_count += 1
            print(f"  WARNING: Inf found in {name}")
    
    if nan_count == 0 and inf_count == 0:
        print(f"✓ No NaN or Inf values found")
    else:
        print(f"✗ Found {nan_count} tensors with NaN, {inf_count} with Inf")
    
    # Check key tensors exist
    print(f"\nChecking for key tensors...")
    key_patterns = [
        'model.embed_tokens.weight',
        'lm_head.weight',
        'model.norm.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.mlp.gate_proj.weight',
    ]
    
    found = 0
    for pattern in key_patterns:
        if pattern in state_dict:
            found += 1
            tensor = state_dict[pattern]
            print(f"  ✓ {pattern:50s} {str(tensor.shape):20s}")
        else:
            print(f"  ✗ {pattern:50s} NOT FOUND")
    
    print(f"\nKey tensor check: {found}/{len(key_patterns)} found")
    
    # Statistics for a sample tensor
    print(f"\nDetailed statistics for embedding tensor:")
    if 'model.embed_tokens.weight' in state_dict:
        emb = state_dict['model.embed_tokens.weight']
        print(f"  Shape: {emb.shape}")
        print(f"  Dtype: {emb.dtype}")
        print(f"  Min: {emb.min():.6f}")
        print(f"  Max: {emb.max():.6f}")
        print(f"  Mean: {emb.mean():.6f}")
        print(f"  Std: {emb.std():.6f}")
        print(f"  Sample values (first 10): {emb.flatten()[:10]}")
    
    return nan_count == 0 and inf_count == 0 and found == len(key_patterns)


def test_loader_complete(gguf_path: str):
    """Test complete load() method"""
    print(f"\n{'='*80}")
    print(f"TEST 4: Complete Loading (config + state_dict)")
    print(f"{'='*80}\n")
    
    loader = GGUFLoader(gguf_path, verbose=True)
    
    # Load everything
    print("Loading complete GGUF file...\n")
    config, state_dict = loader.load(
        as_transformers_config=False,  # Return dict (works without transformers)
        as_torch=False,  # NumPy arrays (works without PyTorch)
        show_progress=True
    )
    
    print(f"\n✓ Complete loading successful")
    print(f"  Config: {len(config)} keys")
    print(f"  State dict: {len(state_dict)} tensors")
    
    return config, state_dict


def test_name_mapping_coverage(state_dict: dict):
    """Check tensor name mapping coverage"""
    print(f"\n{'='*80}")
    print(f"TEST 5: Name Mapping Coverage")
    print(f"{'='*80}\n")
    
    # Count different tensor categories
    categories = {
        'embedding': 0,
        'lm_head': 0,
        'norm': 0,
        'attention': 0,
        'mlp': 0,
        'other': 0
    }
    
    for name in state_dict.keys():
        if 'embed_tokens' in name:
            categories['embedding'] += 1
        elif 'lm_head' in name:
            categories['lm_head'] += 1
        elif 'norm' in name:
            categories['norm'] += 1
        elif 'self_attn' in name:
            categories['attention'] += 1
        elif 'mlp' in name:
            categories['mlp'] += 1
        else:
            categories['other'] += 1
    
    print("Tensor categories:")
    for category, count in sorted(categories.items()):
        print(f"  {category:20s}: {count:4d} tensors")
    
    # Check if all names follow HuggingFace conventions
    hf_prefix_count = sum(1 for name in state_dict.keys() 
                          if name.startswith('model.') or name.startswith('lm_head'))
    
    coverage = hf_prefix_count / len(state_dict) * 100
    print(f"\nHuggingFace naming coverage: {hf_prefix_count}/{len(state_dict)} ({coverage:.1f}%)")
    
    if coverage >= 99.0:
        print("✓ Excellent name mapping coverage")
    elif coverage >= 90.0:
        print("⚠ Good name mapping coverage (some unmapped tensors)")
    else:
        print("✗ Poor name mapping coverage (many unmapped tensors)")
    
    # Show unmapped tensors
    unmapped = [name for name in state_dict.keys() 
                if not (name.startswith('model.') or name.startswith('lm_head'))]
    if unmapped:
        print(f"\nUnmapped tensors ({len(unmapped)}):")
        for name in unmapped[:20]:
            print(f"  {name}")
        if len(unmapped) > 20:
            print(f"  ... and {len(unmapped) - 20} more")
    
    return coverage >= 99.0


def main(gguf_path: str):
    """Run all tests"""
    print(f"\n{'='*80}")
    print(f"GGUF LOADER TEST SUITE")
    print(f"{'='*80}")
    print(f"File: {gguf_path}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Test 1: Config loading
    try:
        config = test_loader_basic(gguf_path)
        results.append(("Config Loading", True))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        results.append(("Config Loading", False))
        return
    
    # Test 2: State dict loading
    try:
        state_dict = test_loader_state_dict(gguf_path)
        results.append(("State Dict Loading", True))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        results.append(("State Dict Loading", False))
        return
    
    # Test 3: Validation
    try:
        valid = test_loader_validation(gguf_path, state_dict)
        results.append(("Tensor Validation", valid))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        results.append(("Tensor Validation", False))
    
    # Test 4: Complete loading
    try:
        config, state_dict = test_loader_complete(gguf_path)
        results.append(("Complete Loading", True))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        results.append(("Complete Loading", False))
        return
    
    # Test 5: Name mapping
    try:
        coverage_ok = test_name_mapping_coverage(state_dict)
        results.append(("Name Mapping Coverage", coverage_ok))
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        results.append(("Name Mapping Coverage", False))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:30s} {status}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL: {passed}/{total} tests passed")
    print(f"{'='*80}\n")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - GGUFLoader ready for use!")
    else:
        print(f"✗ {total - passed} test(s) failed - review output above")
    
    return passed == total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m python.reference.loaders.test_loader <gguf_file>")
        print("\nExample:")
        print("  python -m python.reference.loaders.test_loader models/qwen2.5-0.5b-instruct-q4_0.gguf")
        sys.exit(1)
    
    success = main(sys.argv[1])
    sys.exit(0 if success else 1)
