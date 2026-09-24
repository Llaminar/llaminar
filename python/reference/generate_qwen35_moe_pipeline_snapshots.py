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
import json
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
    QWEN36_MTP_SIDECAR_SNAPSHOT_SCHEMA,
    run_prefill_and_decode,
    write_metadata,
)
from python.reference.mtp_sidecar_reference import (
    normalize_mtp_branch_override_batches,
    promote_mtp_sidecar_metadata,
)


# Increment whenever an existing snapshot key changes semantic meaning.  The
# C++ integration harness authenticates this marker before reusing expensive
# 35B sidecar fixtures, preventing an old-but-present NPY file from silently
# masquerading as the current reference contract.
# Schema 5 makes recursive MTP1/MTP2 consume the preceding predictor's
# shared-head-normalized hidden result. Schema 4 introduced recursive packs
# but incorrectly chained their pre-normalized FFN residual. Schema 3 restores
# MOE_ROUTER_OUTPUT to the production graph's full
# post-softmax expert distribution. Schema 2 incorrectly stored reconstructed
# pre-softmax logits under that established key.
MTP_SIDECAR_SNAPSHOT_SCHEMA = QWEN36_MTP_SIDECAR_SNAPSHOT_SCHEMA

# Schema 1 binds the main-model MOE_ROUTER_OUTPUT key to the complete
# post-softmax probability distribution retained by the live CUDA/ROCm routing
# workspace. Packs without this marker used the retired raw linear projection.
MOE_ROUTER_SNAPSHOT_SCHEMA = 1

# Schema 1 adds weighted per-route expert contributions. They let the C++
# movement witness compare one expert independently of unrelated top-k drift.
MOE_ROUTE_CONTRIBUTION_SNAPSHOT_SCHEMA = 1


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
        help="Also save recursive decode-step MTP sidecar reference snapshots",
    )
    parser.add_argument(
        "--mtp-max-draft-depth",
        type=int,
        default=3,
        help=(
            "Maximum recursive MTP draft depth to materialize when sidecar "
            "snapshots are enabled (default: 3)"
        ),
    )
    parser.add_argument(
        "--mtp-branch-overrides",
        type=Path,
        default=None,
        help=(
            "JSON mapping decode steps to recursive token arrays or objects "
            "with condition_token and draft_tokens; generate independent "
            "additive branches without changing canonical FP32 references"
        ),
    )
    parser.add_argument(
        "--mtp-sidecar-only",
        action="store_true",
        help=(
            "Add canonical recursive MTP snapshots to an existing authenticated "
            "main-model pack using only the bounded sidecar model context"
        ),
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
    print(f"  MTP maximum draft depth: {args.mtp_max_draft_depth}")
    print(f"  MTP branch overrides: {args.mtp_branch_overrides}")
    print(f"  MTP sidecar only: {args.mtp_sidecar_only}")

    if args.mtp_max_draft_depth < 1 or args.mtp_max_draft_depth > 15:
        raise ValueError("--mtp-max-draft-depth must be in [1, 15]")
    if args.mtp_sidecar_only and not args.mtp_sidecar_snapshots:
        raise ValueError("--mtp-sidecar-only requires --mtp-sidecar-snapshots")
    if args.mtp_sidecar_only and args.metadata_only:
        raise ValueError("--mtp-sidecar-only is incompatible with --metadata-only")
    if args.mtp_sidecar_only and args.decode_snapshots_only:
        raise ValueError(
            "--mtp-sidecar-only is incompatible with --decode-snapshots-only"
        )
    if args.mtp_sidecar_only and args.mtp_branch_overrides is not None:
        raise ValueError(
            "--mtp-sidecar-only is canonical generation and cannot be combined "
            "with branch overrides"
        )

    branch_override_batches = None
    if args.mtp_branch_overrides is not None:
        raw_overrides = json.loads(
            args.mtp_branch_overrides.read_text(encoding="utf-8")
        )
        branch_override_batches = normalize_mtp_branch_override_batches(
            raw_overrides
        )
        if not args.mtp_sidecar_snapshots:
            raise ValueError(
                "--mtp-branch-overrides requires --mtp-sidecar-snapshots"
            )

    # Create and load model via registry. Additive branch campaigns consume
    # the authenticated main-model trajectory already in ``args.output`` and
    # therefore load only the graph-external MTP sidecar context. A typed
    # sidecar-only repair may also generate the canonical recursive branch from
    # the exact committed trajectory authenticated by that immutable pack.
    print("\nLoading model...")
    sidecar_context = (
        args.mtp_sidecar_only or branch_override_batches is not None
    )
    reference_kwargs = (
        {"mtp_sidecar_reference_pack": args.output}
        if sidecar_context
        else {}
    )
    model = create_reference_model(
        "qwen35_moe", args.model, **reference_kwargs
    )
    print("Model loaded successfully")

    # Run inference and save snapshots
    if not sidecar_context:
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
    else:
        if not args.output.is_dir() or not (args.output / "metadata.txt").is_file():
            raise ValueError(
                "Sidecar-context generation requires an existing canonical pack"
            )
        total = 0
        token_ids = []
        decode_tokens = []

    if args.mtp_sidecar_snapshots and not args.metadata_only:
        mtp_total = 0
        batches = branch_override_batches or [None]
        for branch_overrides in batches:
            mtp_total += model.generate_mtp_sidecar_decode_snapshots(
                args.prompt,
                args.decode_steps,
                args.output,
                max_draft_depth=args.mtp_max_draft_depth,
                verbose=args.verbose,
                draft_token_overrides=branch_overrides,
                reuse_canonical_main_trajectory=args.mtp_sidecar_only,
            )
        total += mtp_total
        (args.output / "mtp_sidecar_snapshot_schema.txt").write_text(
            f"{MTP_SIDECAR_SNAPSHOT_SCHEMA}\n", encoding="ascii"
        )
        print(f"  Captured {mtp_total} MTP sidecar snapshots")
        if branch_override_batches is not None:
            (args.output / "mtp_sidecar_branch_overrides.json").write_text(
                json.dumps(branch_override_batches, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )

    # Write metadata
    if not sidecar_context:
        write_metadata(
            args.output,
            args.model,
            model,
            args.prompt,
            token_ids,
            args.decode_steps,
            decode_tokens,
            extra_metadata_lines=[
                f"moe_router_snapshot_schema: {MOE_ROUTER_SNAPSHOT_SCHEMA}",
                "moe_route_contribution_snapshot_schema: "
                f"{MOE_ROUTE_CONTRIBUTION_SNAPSHOT_SCHEMA}",
                *(
                    [
                        "mtp_sidecar_max_draft_depth: "
                        f"{args.mtp_max_draft_depth}"
                    ]
                    if args.mtp_sidecar_snapshots
                    else []
                ),
            ],
        )
    elif args.mtp_sidecar_only:
        promote_mtp_sidecar_metadata(
            args.output / "metadata.txt", args.mtp_max_draft_depth
        )

    print(f"\n✓ Done! {total} snapshots saved to: {args.output}")


if __name__ == "__main__":
    main()
