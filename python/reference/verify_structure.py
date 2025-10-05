#!/usr/bin/env python3
"""
Quick verification script to check the reference implementation structure.

This script validates the architecture without requiring PyTorch/transformers
to be installed. It checks imports, registry, and basic functionality.

@author David Sanftenberg
"""

import sys
from pathlib import Path

# Add workspace root to path (so we can import python.reference)
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

def check_imports():
    """Check that all modules can be imported (structure is correct)."""
    print("Checking imports...")
    
    try:
        # Core modules (should work without PyTorch)
        from python.reference.pipeline_stages import PipelineStage, stage_to_string, string_to_stage
        print("  ✓ pipeline_stages")
        
        # Try importing modules that require PyTorch
        torch_available = False
        try:
            import torch
            import numpy as np
            torch_available = True
            print("  ✓ torch, numpy available")
        except ImportError:
            print("  ⚠ torch/numpy not installed (expected if deps not yet installed)")
        
        if torch_available:
            from python.reference.base import AbstractReferenceModel
            print("  ✓ base")
            
            from python.reference.registry import ModelRegistry
            print("  ✓ registry")
            
            from python.reference import qwen, llama
            print("  ✓ qwen, llama")
            
            print("  ✓ All imports successful")
        else:
            print("  ℹ Skipping torch-dependent imports")
            print("  ℹ Install deps: pip install -r requirements.txt")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_registry():
    """Check that ModelRegistry works."""
    print("\nChecking ModelRegistry...")
    
    # First check if torch is available
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False
    
    if not torch_available:
        print("  ℹ Skipping (torch not installed)")
        print("  ℹ Install deps: pip install -r requirements.txt")
        return True  # Not a failure, just skipped
    
    try:
        from python.reference.registry import ModelRegistry
        
        # Check registered models
        models = ModelRegistry.list_models()
        print(f"  Registered models: {models}")
        
        if "qwen" in models:
            print("  ✓ Qwen registered")
        else:
            print("  ✗ Qwen not registered")
            return False
        
        if "llama" in models:
            print("  ✓ LLaMA registered")
        else:
            print("  ✗ LLaMA not registered")
            return False
        
        # Check is_registered
        assert ModelRegistry.is_registered("qwen")
        assert ModelRegistry.is_registered("llama")
        assert not ModelRegistry.is_registered("nonexistent")
        print("  ✓ is_registered() works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Registry check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pipeline_stages():
    """Check PipelineStage enum."""
    print("\nChecking PipelineStage enum...")
    
    try:
        from python.reference.pipeline_stages import PipelineStage, stage_to_string, string_to_stage
        
        # Check all stages exist
        expected_stages = [
            "EMBEDDING", "ATTENTION_NORM", "QKV_PROJECTION", "Q_PROJECTION",
            "K_PROJECTION", "V_PROJECTION", "ROPE_APPLICATION",
            "ATTENTION_SCORES", "ATTENTION_SOFTMAX", "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL", "FFN_NORM", "FFN_GATE",
            "FFN_UP", "FFN_SWIGLU", "FFN_DOWN", "FFN_RESIDUAL",
            "FINAL_NORM", "LM_HEAD", "CUSTOM"
        ]
        
        for stage_name in expected_stages:
            stage = getattr(PipelineStage, stage_name)
            assert stage_to_string(stage) == stage_name
            assert string_to_stage(stage_name) == stage
        
        print(f"  ✓ All {len(expected_stages)} stages validated")
        print(f"  ✓ Conversion functions work")
        
        return True
        
    except Exception as e:
        print(f"  ✗ PipelineStage check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_file_structure():
    """Check that all expected files exist."""
    print("\nChecking file structure...")
    
    base_path = Path(__file__).parent
    expected_files = [
        "__init__.py",
        "pipeline_stages.py",
        "base.py",
        "registry.py",
        "qwen.py",
        "llama.py",
        "utils.py",
        "run_reference.py",
        "README.md",
        "IMPLEMENTATION_SUMMARY.md",
        "tests/test_reference.py",
    ]
    
    all_exist = True
    for file in expected_files:
        path = base_path / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} MISSING")
            all_exist = False
    
    return all_exist


def main():
    """Run all checks."""
    print("=" * 60)
    print("PyTorch Reference Implementation - Structure Verification")
    print("=" * 60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Imports", check_imports),
        ("PipelineStage Enum", check_pipeline_stages),
        ("ModelRegistry", check_registry),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All checks passed! Structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: pytest python/reference/tests/ -v")
        print("3. Try CLI: python python/reference/run_reference.py --help")
    else:
        print("\n❌ Some checks failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
