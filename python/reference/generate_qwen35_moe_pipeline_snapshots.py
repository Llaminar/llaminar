#!/usr/bin/env python3
"""
Generate PyTorch Qwen 3.5 MoE pipeline reference snapshots for V2 parity testing.

Uses the Qwen35MoEReferenceModel (registry-based) which handles:
  - Heterogeneous GDN + full attention layers (same as dense Qwen3.5)
  - SparseMoeBlock FFN with router, 256 experts (top-8), shared expert

Captures all intermediate activations as individual .npy files, compatible
with the C++ parity test infrastructure (cnpy::npy_load).

Usage:
    python3 generate_qwen35_moe_pipeline_snapshots.py \
        --model /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
        --output pytorch_qwen35_moe_snapshots

    python3 generate_qwen35_moe_pipeline_snapshots.py \
        --model /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
        --prompt "The quick brown fox" \
        --decode-steps 3

@author David Sanftenberg
"""

import sys
import argparse
from pathlib import Path

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

from python.reference import create_reference_model
# Reuse the snapshot save/run infrastructure from the dense Qwen3.5 generator
from python.reference.generate_qwen35_pipeline_snapshots import (
    run_prefill_and_decode,
    write_metadata,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch Qwen 3.5 MoE pipeline reference snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 generate_qwen35_moe_pipeline_snapshots.py \\
        --model /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf

    python3 generate_qwen35_moe_pipeline_snapshots.py \\
        --model /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \\
        --prompt "The quick brown fox" \\
        --decode-steps 3 \\
        --output pytorch_qwen35_moe_snapshots
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Qwen3.5 MoE GGUF model file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog",
        help='Input prompt (default: "The quick brown fox jumps over the lazy dog")',
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=0,
        help="Number of decode steps after prefill (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: pytorch_qwen35_moe_snapshots)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write metadata.txt with prompt/decode tokens without saving .npy snapshots",
    )
    parser.add_argument(
        "--decode-snapshots-only",
        action="store_true",
        help="Save decode-step snapshots but skip prefill snapshots",
    )
    parser.add_argument(
        "--mtp-sidecar-snapshots",
        action="store_true",
        help="Also save decode-step MTP0 sidecar reference snapshots",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = Path("pytorch_qwen35_moe_snapshots")

    print(f"Generating Qwen 3.5 MoE pipeline snapshots...")
    print(f"  Model: {args.model}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Output: {args.output}")
    print(f"  Decode steps: {args.decode_steps}")
    print(f"  Metadata only: {args.metadata_only}")
    print(f"  Decode snapshots only: {args.decode_snapshots_only}")
    print(f"  MTP sidecar snapshots: {args.mtp_sidecar_snapshots}")

    # Create and load model via registry
    print("\nLoading model...")
    model = create_reference_model("qwen35_moe", args.model)
    print("Model loaded successfully")

    # Run inference and save snapshots
    total, token_ids, decode_tokens = run_prefill_and_decode(
        model,
        args.prompt,
        args.decode_steps,
        args.output,
        verbose=args.verbose,
        save_snapshots=not args.metadata_only,
        save_prefill_snapshots=not args.decode_snapshots_only,
        save_decode_snapshots=True,
    )

    if args.mtp_sidecar_snapshots and not args.metadata_only:
        mtp_total = model.generate_mtp_sidecar_decode_snapshots(
            args.prompt,
            args.decode_steps,
            args.output,
            verbose=args.verbose,
        )
        total += mtp_total
        print(f"  Captured {mtp_total} MTP sidecar snapshots")

    # Write metadata
    write_metadata(
        args.output,
        args.model,
        model,
        args.prompt,
        token_ids,
        args.decode_steps,
        decode_tokens,
    )

    print(f"\n✓ Done! {total} snapshots saved to: {args.output}")


if __name__ == "__main__":
    main()
