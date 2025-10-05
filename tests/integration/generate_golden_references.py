#!/usr/bin/env python3
"""
Generate golden reference snapshots for integration tests.

This script automates the generation of PyTorch reference snapshots for all
integration test configurations.

Usage:
    python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct
    python3 tests/integration/generate_golden_references.py --all

Author: David Sanftenberg
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Test configurations matching qwen_integration_test.cpp
TEST_CONFIGS = {
    "qwen2.5-0.5b-instruct": {
        "precisions": ["q4_0", "q6_k"],  # FP32 optional
        "token_sets": {
            "standard": [1639, 266, 285, 17, 10, 17, 30],  # "1+1="
            "short": [1639, 266, 285],
            "single": [1639],
        },
        "stages": ["prefill", "decode"],
    }
}

def get_model_path(model_name: str, precision: str) -> str:
    """Construct path to GGUF model file."""
    return f"models/{model_name}-{precision}.gguf"

def get_output_path(model_name: str, precision: str, stage: str, token_key: str) -> str:
    """Construct output path for golden reference NPZ."""
    output_dir = f"tests/golden_references/{model_name}-{precision}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get tokens for filename
    tokens = TEST_CONFIGS[model_name]["token_sets"][token_key]
    tokens_str = "_".join(str(t) for t in tokens)
    
    return f"{output_dir}/{stage}_tokens_{tokens_str}.npz"

def check_model_exists(model_path: str) -> bool:
    """Check if model file exists."""
    return Path(model_path).exists()

def generate_reference(model_name: str, precision: str, stage: str, 
                      token_key: str, tokens: List[int], dry_run: bool = False) -> bool:
    """
    Generate a single golden reference snapshot.
    
    Args:
        model_name: Model identifier (e.g., "qwen2.5-0.5b-instruct")
        precision: Precision (e.g., "q4_0", "q6_k", "fp32")
        stage: Stage ("prefill" or "decode")
        token_key: Token set identifier
        tokens: Token sequence
        dry_run: If True, only print commands without executing
    
    Returns:
        True if successful, False otherwise
    """
    model_path = get_model_path(model_name, precision)
    output_path = get_output_path(model_name, precision, stage, token_key)
    
    # Check if model exists
    if not check_model_exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        print(f"   Skipping {precision} {stage} {token_key}")
        return False
    
    # Check if golden reference already exists
    if Path(output_path).exists():
        print(f"✓ Golden reference already exists: {output_path}")
        return True
    
    # Construct command
    tokens_str = ",".join(str(t) for t in tokens)
    cmd = [
        "python3",
        "python/reference/capture_pytorch_layers.py",
        "-m", model_path,
        "--tokens", tokens_str,
        "--output", output_path,
    ]
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Generating: {output_path}")
    print(f"  Model: {model_path}")
    print(f"  Stage: {stage}")
    print(f"  Tokens: {tokens} ({len(tokens)} tokens)")
    print(f"  Command: {' '.join(cmd)}")
    
    if dry_run:
        return True
    
    # Execute command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Generated: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate: {output_path}")
        print(f"  Error: {e.stderr}")
        return False

def generate_all_references(model_name: str, dry_run: bool = False,
                           precisions: List[str] = None) -> Tuple[int, int]:
    """
    Generate all golden references for a model.
    
    Args:
        model_name: Model identifier
        dry_run: If True, only print commands
        precisions: List of precisions to generate (None = all)
    
    Returns:
        (success_count, total_count)
    """
    if model_name not in TEST_CONFIGS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(TEST_CONFIGS.keys())}")
        return 0, 0
    
    config = TEST_CONFIGS[model_name]
    
    # Filter precisions if specified
    test_precisions = precisions if precisions else config["precisions"]
    
    success_count = 0
    total_count = 0
    
    print(f"=== Generating Golden References for {model_name} ===")
    print(f"Precisions: {test_precisions}")
    print(f"Token sets: {list(config['token_sets'].keys())}")
    print(f"Stages: {config['stages']}")
    
    for precision in test_precisions:
        for stage in config["stages"]:
            for token_key, tokens in config["token_sets"].items():
                total_count += 1
                if generate_reference(model_name, precision, stage, 
                                    token_key, tokens, dry_run):
                    success_count += 1
    
    return success_count, total_count

def list_references(model_name: str = None):
    """List all existing golden references."""
    base_dir = Path("tests/golden_references")
    
    if not base_dir.exists():
        print("No golden references directory found.")
        return
    
    print("=== Existing Golden References ===\n")
    
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        if model_name and not model_dir.name.startswith(model_name):
            continue
        
        print(f"📁 {model_dir.name}/")
        
        npz_files = list(model_dir.glob("*.npz"))
        if not npz_files:
            print("   (empty)")
        else:
            for npz_file in sorted(npz_files):
                size_mb = npz_file.stat().st_size / (1024 * 1024)
                print(f"   • {npz_file.name} ({size_mb:.2f} MB)")
        
        print()

def clean_references(model_name: str = None, precision: str = None, 
                    dry_run: bool = False):
    """Remove golden references."""
    base_dir = Path("tests/golden_references")
    
    if not base_dir.exists():
        print("No golden references to clean.")
        return
    
    removed_count = 0
    
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        if model_name:
            # Filter by model name
            if not model_dir.name.startswith(model_name):
                continue
            
            if precision:
                # Filter by precision
                if not model_dir.name.endswith(f"-{precision}"):
                    continue
        
        print(f"{'[DRY RUN] ' if dry_run else ''}Removing: {model_dir}")
        
        if not dry_run:
            import shutil
            shutil.rmtree(model_dir)
            removed_count += 1
    
    if removed_count > 0:
        print(f"\n✓ Removed {removed_count} reference directories")
    else:
        print("\nNo references removed")

def main():
    parser = argparse.ArgumentParser(
        description="Generate golden reference snapshots for integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all references for Qwen
  python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct
  
  # Generate only Q4_0 references
  python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct --precision q4_0
  
  # Dry run (show commands without executing)
  python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct --dry-run
  
  # List existing references
  python3 tests/integration/generate_golden_references.py --list
  
  # Clean references
  python3 tests/integration/generate_golden_references.py --clean --model qwen2.5-0.5b-instruct
        """
    )
    
    parser.add_argument("--model", "-m", 
                       help="Model name (e.g., qwen2.5-0.5b-instruct)")
    parser.add_argument("--all", action="store_true",
                       help="Generate for all configured models")
    parser.add_argument("--precision", "-p",
                       help="Specific precision (q4_0, q6_k, fp32)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show commands without executing")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List existing golden references")
    parser.add_argument("--clean", action="store_true",
                       help="Remove golden references")
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_references(args.model)
        return 0
    
    # Clean mode
    if args.clean:
        if not args.model and not args.all:
            print("Error: --clean requires --model or --all")
            return 1
        
        clean_references(args.model, args.precision, args.dry_run)
        return 0
    
    # Generate mode
    if not args.model and not args.all:
        print("Error: Must specify --model or --all")
        parser.print_help()
        return 1
    
    # Determine which models to process
    if args.all:
        models = list(TEST_CONFIGS.keys())
    else:
        models = [args.model]
    
    # Filter precisions if specified
    precisions = [args.precision] if args.precision else None
    
    # Generate references
    total_success = 0
    total_count = 0
    
    for model in models:
        success, count = generate_all_references(model, args.dry_run, precisions)
        total_success += success
        total_count += count
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {total_success}/{total_count} references generated")
    
    if total_success < total_count:
        print(f"⚠️  {total_count - total_success} references failed or skipped")
        print("   (may be due to missing model files)")
        return 1
    
    print("✓ All references generated successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
