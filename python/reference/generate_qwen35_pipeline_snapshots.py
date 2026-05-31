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
):
    """Write metadata.txt compatible with parity test loader."""
    config = model.hf_model.config
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        # Snapshot version: bumped when the snapshot format or V-head
        # reversal semantics change. The C++ parity test framework checks
        # this version and regenerates snapshots automatically when stale.
        #   v1: original format
        #   v2: MoE-only V-head reversal (dense models skip reversal)
        #   v3: Qwen3.5 prefill GDN conv and Q/K norm snapshots match C++ layout
        f.write(f"snapshot_version: 3\n")
        f.write(f"Model: {model_path}\n")
        arch = getattr(config, "architectures", [config.__class__.__name__])
        f.write(f"Architecture: {arch[0] if arch else config.__class__.__name__}\n")
        f.write(f"n_layers: {config.num_hidden_layers}\n")
        f.write(f"n_heads: {config.num_attention_heads}\n")
        n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        f.write(f"n_kv_heads: {n_kv_heads}\n")
        f.write(f"d_model: {config.hidden_size}\n")
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        f.write(f"d_head: {head_dim}\n")
        # MoE configs use moe_intermediate_size instead of intermediate_size
        d_ff = getattr(config, "intermediate_size", None) or getattr(config, "moe_intermediate_size", 0)
        f.write(f"d_ff: {d_ff}\n")
        f.write(f"vocab_size: {config.vocab_size}\n")
        f.write(f"prompt: {prompt}\n")
        f.write(f"token_ids: {','.join(map(str, token_ids))}\n")
        f.write(f"decode_steps: {decode_steps}\n")
        if decode_tokens:
            f.write(f"decode_tokens: {','.join(map(str, decode_tokens))}\n")


def run_prefill_and_decode(
    model,
    prompt: str,
    decode_steps: int,
    output_dir: Path,
    verbose: bool = False,
    save_snapshots: bool = True,
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
    capture_stages = None if save_snapshots else []
    result = model.forward(
        token_ids,
        clear_snapshots=True,
        use_cache=True,
        capture_stages=capture_stages,
    )
    total = 0
    if save_snapshots:
        prefill_snaps = model.get_snapshots()
        total = save_snapshots_as_npy(prefill_snaps, output_dir, prefix="", verbose=verbose)
        print(f"  Captured {total} prefill snapshots")
    else:
        model.clear_snapshots()
        print("  Snapshot capture disabled; metadata only")

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

        # Single-token forward using cache. Snapshots from this step have
        # shape [1, 1, H] (one new position), matching what Llaminar's
        # incremental decode produces.
        result = model.forward(
            [next_token],
            clear_snapshots=True,
            past_key_values=cache,
            use_cache=True,
            capture_stages=capture_stages,
        )
        cache = result.get("past_key_values")
        if save_snapshots:
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

    args = parser.parse_args()

    if args.output is None:
        args.output = Path("pytorch_qwen35_snapshots")

    print(f"Generating Qwen 3.5 pipeline snapshots...")
    print(f"  Model: {args.model}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Output: {args.output}")
    print(f"  Decode steps: {args.decode_steps}")
    print(f"  Metadata only: {args.metadata_only}")

    # Create and load model via registry
    print("\nLoading model...")
    model = create_reference_model("qwen35", args.model)
    print("Model loaded successfully")

    # Run inference and save snapshots
    total, token_ids, decode_tokens = run_prefill_and_decode(
        model,
        args.prompt,
        args.decode_steps,
        args.output,
        verbose=args.verbose,
        save_snapshots=not args.metadata_only,
    )

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
