#!/usr/bin/env python3
"""
CLI Tool for Running PyTorch Reference Inference

Runs reference model inference and exports snapshots for parity testing.

@author David Sanftenberg
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

from python.reference import create_reference_model, PipelineStage
from python.reference.utils import print_snapshot_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run PyTorch reference inference for parity testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Qwen with HuggingFace checkpoint
  %(prog)s --model qwen --checkpoint Qwen/Qwen2-0.5B --tokens 1,2,3 --output snapshots.npz
  
  # Run with GGUF checkpoint (auto-detected by .gguf extension)
  %(prog)s --model qwen --checkpoint models/qwen2.5-0.5b-instruct-q4_0.gguf --tokens 1,2,3
  
  # Run LLaMA with GGUF
  %(prog)s --model llama --checkpoint models/Llama-3.2-1B-Instruct-Q4_0.gguf --tokens 1,2,3
  
  # Force GGUF loader even without .gguf extension
  %(prog)s --model qwen --checkpoint path/to/model --tokens 1,2,3 --force-gguf
  
  # Force HuggingFace loader even with .gguf file
  %(prog)s --model qwen --checkpoint model.gguf --tokens 1,2,3 --force-huggingface
  
  # Capture specific stages only
  %(prog)s --model qwen --checkpoint Qwen/Qwen2-0.5B --tokens 1,2,3 \\
      --stages EMBEDDING,ATTENTION_OUTPUT,LM_HEAD
  
  # Use quantization (HuggingFace checkpoints only)
  %(prog)s --model qwen --checkpoint Qwen/Qwen2-0.5B --tokens 1,2,3 \\
      --quantization 4bit --output snapshots.npz
"""
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model architecture (qwen, llama, etc.)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="HuggingFace model ID, local checkpoint path, or GGUF file (.gguf auto-detected)"
    )
    
    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Comma-separated token IDs (e.g., '1,2,3,4')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="snapshots.npz",
        help="Output path for snapshots (default: snapshots.npz)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "json"],
        default="npz",
        help="Export format (default: npz)"
    )
    
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated stages to capture (default: all)"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode for HuggingFace checkpoints (default: None, ignored for GGUF)"
    )
    
    parser.add_argument(
        "--force-gguf",
        action="store_true",
        help="Force GGUF loader even if file doesn't have .gguf extension"
    )
    
    parser.add_argument(
        "--force-huggingface",
        action="store_true",
        help="Force HuggingFace loader even for .gguf files"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (default: cpu)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed snapshot summary"
    )
    
    return parser.parse_args()


def parse_token_ids(tokens_str: str) -> List[int]:
    """Parse comma-separated token IDs."""
    try:
        return [int(x.strip()) for x in tokens_str.split(",")]
    except ValueError as e:
        print(f"Error parsing token IDs: {e}", file=sys.stderr)
        sys.exit(1)


def parse_stages(stages_str: str) -> List[PipelineStage]:
    """Parse comma-separated stage names."""
    from python.reference.pipeline_stages import string_to_stage
    
    try:
        return [string_to_stage(s.strip()) for s in stages_str.split(",")]
    except KeyError as e:
        print(f"Error parsing stages: {e}", file=sys.stderr)
        print(f"Valid stages: {', '.join(PipelineStage.__members__.keys())}")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate force flags
    if args.force_gguf and args.force_huggingface:
        print("Error: Cannot specify both --force-gguf and --force-huggingface", file=sys.stderr)
        sys.exit(1)
    
    # Auto-detect GGUF files
    is_gguf = args.checkpoint.lower().endswith('.gguf')
    if args.force_gguf:
        is_gguf = True
    elif args.force_huggingface:
        is_gguf = False
    
    # Print loader info
    loader_type = "GGUF" if is_gguf else "HuggingFace"
    print(f"Running {args.model} reference inference")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Loader: {loader_type}")
    if is_gguf and args.quantization:
        print(f"Note: --quantization ignored for GGUF files (quantization embedded in GGUF)")
    print(f"Device: {args.device}")
    
    # Parse inputs
    token_ids = parse_token_ids(args.tokens)
    print(f"Tokens: {token_ids}")
    
    capture_stages = None
    if args.stages:
        capture_stages = parse_stages(args.stages)
        print(f"Capturing stages: {[s.name for s in capture_stages]}")
    else:
        print("Capturing all stages")
    
    # Create and load model
    print("\nLoading model...")
    try:
        model = create_reference_model(
            args.model,
            args.checkpoint,
            device=args.device,
            quantization=args.quantization if not is_gguf else None
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Model loaded successfully")
    
    # Run forward pass
    print("\nRunning forward pass...")
    try:
        result = model.forward(
            token_ids,
            capture_stages=capture_stages
        )
    except Exception as e:
        print(f"Error during forward pass: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print results
    logits = result["logits"]
    print(f"\nLogits shape: {logits.shape}")
    print(f"Captured {len(result['snapshots'])} snapshots")
    
    if args.verbose:
        print_snapshot_summary(result["snapshots"], "Captured Snapshots")
    
    # Export snapshots
    print(f"\nExporting snapshots to {args.output}...")
    try:
        model.export_snapshots(args.output, format=args.format)
        print("Export successful")
    except Exception as e:
        print(f"Error exporting snapshots: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
