#!/usr/bin/env python3
"""
Generate PyTorch Qwen 3.5 pipeline reference snapshots for V2 parity testing.

Uses the Qwen35ReferenceModel (registry-based) which handles heterogeneous
GDN + full attention layers via HuggingFace forward hooks.

Captures all intermediate activations as individual .npy files, compatible
with the C++ parity test infrastructure (cnpy::npy_load).

Usage:
    python3 generate_qwen35_pipeline_snapshots.py \
        --model models/Qwen3.5-0.8B-Q4_0.gguf \
        --output pytorch_qwen35_snapshots

    python3 generate_qwen35_pipeline_snapshots.py \
        --model models/Qwen3.5-0.8B-Q4_0.gguf \
        --prompt "The quick brown fox" \
        --decode-steps 3

@author David Sanftenberg
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Set

import numpy as np
import torch

# Add parent directories to path
script_dir = Path(__file__).parent.absolute()
python_dir = script_dir.parent.absolute()
workspace_dir = python_dir.parent.absolute()

for path_to_add in [str(python_dir), str(workspace_dir)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

from python.reference import create_reference_model, PipelineStage
from python.reference.pipeline_stages import stage_to_string
from python.reference.snapshot_metadata import (
    build_reference_identity,
    write_metadata_atomically,
)
from python.reference.mtp_sidecar_reference import (
    normalize_mtp_branch_override_batches,
    promote_mtp_sidecar_metadata,
)


# Recursive depth semantics are shared by dense and MoE Qwen3.6 sidecars.
# Schema 5 means MTP1/MTP2 consume the preceding predictor's shared-head-
# normalized hidden result; earlier experimental packs chained the raw decoder
# residual and are not valid production parity oracles.
QWEN36_MTP_SIDECAR_SNAPSHOT_SCHEMA = 5


def save_snapshots_as_npy(
    snapshots: dict,
    output_dir: Path,
    prefix: str = "",
    verbose: bool = False,
):
    """
    Save captured snapshots as individual .npy files.

    File naming matches the existing Qwen2/3 convention:
      - Global stages: EMBEDDING.npy, FINAL_NORM.npy, LM_HEAD.npy
      - Per-layer stages: layer{N}_{STAGE_NAME}.npy
      - Decode steps: decode_step{S}_{key}.npy
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build (path, data) pairs first so the parallel save loop has nothing
    # to compute beyond filesystem writes.
    items: list = []
    for (stage, layer_idx), data in snapshots.items():
        stage_name = stage_to_string(stage)
        if layer_idx >= 0:
            key = f"layer{layer_idx}_{stage_name}"
        else:
            key = stage_name
        if prefix:
            key = f"{prefix}_{key}"
        items.append((output_dir / f"{key}.npy", key, data))

    # PERF: ``np.save`` releases the GIL during the actual write, and on
    # NVMe the bottleneck is per-file syscall latency rather than
    # bandwidth. A small thread pool overlaps those syscalls and cuts the
    # snapshot save phase by ~Nx for large layer counts (27B → 64 layers ×
    # ~12 stages = 770 files).
    from concurrent.futures import ThreadPoolExecutor
    n_workers = min(os.cpu_count() or 4, 16)

    def _save_one(item):
        npy_path, _key, payload = item
        np.save(npy_path, payload)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # Drain to surface any exception.
        for _ in ex.map(_save_one, items):
            pass

    if verbose:
        for _path, key, payload in items:
            print(f"  Saved {key}: shape={list(payload.shape)}")

    return len(items)


def write_metadata(
    output_dir: Path,
    model_path: str,
    model,
    prompt: str,
    token_ids: list,
    decode_steps: int,
    decode_tokens: Optional[list] = None,
    extra_metadata_lines: Optional[list[str]] = None,
):
    """Publish metadata binding this pack to its exact inference inputs."""
    config = model.hf_model.config
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.txt"
    identity = build_reference_identity(
        model_path, prompt, token_ids, decode_steps
    )
    arch = getattr(config, "architectures", [config.__class__.__name__])
    n_kv_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    # MoE configs use moe_intermediate_size instead of intermediate_size.
    d_ff = getattr(config, "intermediate_size", None) or getattr(
        config, "moe_intermediate_size", 0
    )
    lines = [
        # v4 emits GDN alpha/beta projections for recurrence diagnostics.
        "snapshot_version: 4",
        f"Model: {model_path}",
        f"Architecture: {arch[0] if arch else config.__class__.__name__}",
        f"n_layers: {config.num_hidden_layers}",
        f"n_heads: {config.num_attention_heads}",
        f"n_kv_heads: {n_kv_heads}",
        f"d_model: {config.hidden_size}",
        f"d_head: {head_dim}",
        f"d_ff: {d_ff}",
        f"vocab_size: {config.vocab_size}",
        # Keep token_ids immediately after prompt for the legacy multiline
        # prompt reader; SHA-256 is the production identity authority.
        f"prompt: {prompt}",
        f"token_ids: {','.join(map(str, token_ids))}",
        *identity.metadata_lines(),
        *(extra_metadata_lines or []),
    ]
    if decode_tokens:
        lines.append(f"decode_tokens: {','.join(map(str, decode_tokens))}")
    write_metadata_atomically(metadata_path, lines)


def run_prefill_and_decode(
    model,
    prompt: str,
    decode_steps: int,
    output_dir: Path,
    verbose: bool = False,
    save_snapshots: bool = True,
    save_prefill_snapshots: bool = True,
    save_decode_snapshots: bool = True,
    snapshot_decode_steps: Optional[Set[int]] = None,
):
    """
    Run prefill + optional decode steps with snapshot capture.

    Returns (total_snapshot_count, token_ids, decode_tokens).
    """
    # Tokenize
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded. model.load_model() should initialize it.")
    encoding = tokenizer(prompt, return_tensors="pt")
    token_ids = encoding["input_ids"][0].tolist()
    print(f"  Token IDs ({len(token_ids)}): {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")

    # ---- Prefill ----
    print("\n[Prefill] Running forward pass...")
    # PERF: Run prefill with use_cache=True so subsequent decode steps only
    # need to forward the new token through cached K/V (and GDN recurrent
    # state). Without this, each decode step re-runs the full prompt + all
    # prior decoded tokens — O(S²) attention work that produces snapshots
    # we then throw away (the C++ parity test only compares the LAST row).
    capture_prefill_stages = None if save_snapshots and save_prefill_snapshots else []
    result = model.forward(
        token_ids,
        clear_snapshots=True,
        use_cache=True,
        capture_stages=capture_prefill_stages,
    )
    total = 0
    if save_snapshots and save_prefill_snapshots:
        prefill_snaps = model.get_snapshots()
        total = save_snapshots_as_npy(prefill_snaps, output_dir, prefix="", verbose=verbose)
        print(f"  Captured {total} prefill snapshots")
    else:
        model.clear_snapshots()
        print("  Prefill snapshot capture disabled")

    # Get next token (greedy)
    logits = result["logits"]
    next_token = int(np.argmax(logits[0, -1, :]))
    print(f"  Next token (greedy): {next_token}")

    decode_tokens = []
    if decode_steps <= 0:
        return total, token_ids, decode_tokens

    # ---- Decode steps ----
    # Cache from prefill carries K/V for the prompt + GDN recurrent state.
    cache = result.get("past_key_values")
    decode_tokens.append(next_token)

    for step in range(decode_steps):
        prefix = f"decode_step{step}"
        print(f"\n[Decode step {step}] token={next_token}")
        capture_this_decode_step = (
            snapshot_decode_steps is None or step in snapshot_decode_steps
        )

        # Single-token forward using cache. Snapshots from this step have
        # shape [1, 1, H] (one new position), matching what Llaminar's
        # incremental decode produces.
        result = model.forward(
            [next_token],
            clear_snapshots=True,
            past_key_values=cache,
            use_cache=True,
            capture_stages=None
            if save_snapshots and save_decode_snapshots and capture_this_decode_step
            else [],
        )
        cache = result.get("past_key_values")
        if save_snapshots and save_decode_snapshots and capture_this_decode_step:
            step_snaps = model.get_snapshots()
            n = save_snapshots_as_npy(step_snaps, output_dir, prefix=prefix, verbose=verbose)
            total += n
            print(f"  Captured {n} decode snapshots for step {step}")
        else:
            model.clear_snapshots()

        # Pick next token
        logits = result["logits"]
        next_token = int(np.argmax(logits[0, -1, :]))
        decode_tokens.append(next_token)
        print(f"  Next token (greedy): {next_token}")

    return total, token_ids, decode_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch Qwen 3.5 pipeline reference snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 generate_qwen35_pipeline_snapshots.py \\
        --model models/Qwen3.5-0.8B-Q4_0.gguf

    python3 generate_qwen35_pipeline_snapshots.py \\
        --model models/Qwen3.5-0.8B-Q4_0.gguf \\
        --prompt "The quick brown fox" \\
        --decode-steps 3 \\
        --output pytorch_qwen35_snapshots
""",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Qwen3.5 GGUF model file",
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
        help="Output directory (default: pytorch_qwen35_snapshots)",
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
        help=(
            "Also save recursive Qwen3.6 MTP checkpoints from the "
            "GGUF's real trailing nextn weights"
        ),
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
            "JSON mapping decode steps to production recursive MTP condition "
            "tokens; generate additive branch-qualified sidecar snapshots"
        ),
    )
    parser.add_argument(
        "--mtp-sidecar-only",
        action="store_true",
        help=(
            "Add canonical recursive MTP snapshots to an authenticated dense "
            "main-model pack using only the bounded sidecar context"
        ),
    )
    parser.add_argument(
        "--snapshot-decode-steps",
        type=str,
        default="",
        help=(
            "Comma-separated decode step indices to save when decode snapshots "
            "are enabled. Forward still runs through all decode steps so cache "
            "state is correct, but only selected steps are written."
        ),
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = Path("pytorch_qwen35_snapshots")

    print(f"Generating Qwen 3.5 pipeline snapshots...")
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
    print(f"  Snapshot decode steps: {args.snapshot_decode_steps or '<all>'}")

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

    snapshot_decode_steps: Optional[Set[int]] = None
    if args.snapshot_decode_steps:
        snapshot_decode_steps = set()
        for raw_part in args.snapshot_decode_steps.split(","):
            part = raw_part.strip()
            if not part:
                continue
            try:
                step = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid --snapshot-decode-steps entry: {part!r}"
                ) from exc
            if step < 0 or step >= args.decode_steps:
                raise ValueError(
                    f"Snapshot decode step {step} is outside decode range "
                    f"[0, {args.decode_steps})"
                )
            snapshot_decode_steps.add(step)

    # Additive branches and canonical depth repair need only the graph-external
    # sidecar allocation plus the authenticated main trajectory already stored
    # in the output pack.
    print("\nLoading model...")
    sidecar_context = (
        args.mtp_sidecar_only or branch_override_batches is not None
    )
    reference_kwargs = (
        {"mtp_sidecar_reference_pack": args.output}
        if sidecar_context
        else {}
    )
    model = create_reference_model("qwen35", args.model, **reference_kwargs)
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
            snapshot_decode_steps=snapshot_decode_steps,
        )
    else:
        if not args.output.is_dir() or not (args.output / "metadata.txt").is_file():
            raise ValueError(
                "Dense sidecar-context generation requires an existing "
                "canonical pack"
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
            f"{QWEN36_MTP_SIDECAR_SNAPSHOT_SCHEMA}\n", encoding="ascii"
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
            extra_metadata_lines=(
                [
                    "mtp_sidecar_max_draft_depth: "
                    f"{args.mtp_max_draft_depth}"
                ]
                if args.mtp_sidecar_snapshots
                else []
            ),
        )
    elif args.mtp_sidecar_only:
        promote_mtp_sidecar_metadata(
            args.output / "metadata.txt", args.mtp_max_draft_depth
        )

    print(f"\n✓ Done! {total} snapshots saved to: {args.output}")


if __name__ == "__main__":
    main()
