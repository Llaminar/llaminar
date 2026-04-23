#!/usr/bin/env python3
"""
Verify HuggingFace Qwen3.5 MoE reference model output against llama.cpp.

Runs both systems with greedy decoding on the same prompt and compares
token-by-token output. This validates that the HF reference model
(used for parity tests) is itself correct.

Usage:
    python -m python.reference.verify_moe_vs_llamacpp \
        --model /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
        --llama-cli /tmp/llama.cpp/build/bin/llama-completion \
        --prompt "The capital of France is" \
        --n-tokens 20
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile

import torch


def run_llamacpp(binary: str, model: str, prompt: str, n_tokens: int, threads: int = 28) -> str:
    """Run llama-completion and return generated text."""
    print(f"\n{'='*60}")
    print(f"Running llama.cpp ({n_tokens} tokens, greedy)...")
    print(f"{'='*60}")

    cmd = [
        binary,
        "-m", model,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", "0",
        "--in-prefix", "",
        "--in-suffix", "",
        "-t", str(threads),
    ]

    # Use a timeout and /dev/null stdin to prevent interactive blocking
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        stdout_file = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        stderr_file = f.name

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=open(stdout_file, 'w'),
            stderr=open(stderr_file, 'w'),
            preexec_fn=os.setsid,
        )

        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            # Kill the process group (handles any child processes)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()

        with open(stdout_file, 'r') as f:
            stdout = f.read()
        with open(stderr_file, 'r') as f:
            stderr = f.read()

        # Extract the generated text (everything after the prompt)
        full_text = stdout.strip()
        if full_text.startswith(prompt):
            generated = full_text[len(prompt):]
        else:
            generated = full_text

        # Check if interactive mode garbled the output
        if '>' in generated and 'Interrupted' in generated:
            # Trim at the interactive prompt
            generated = generated.split('\n>')[0]

        print(f"llama.cpp full output: {repr(full_text)}")
        print(f"llama.cpp generated:   {repr(generated.strip())}")

        # Extract timing info from stderr
        for line in stderr.split('\n'):
            if 'eval time' in line or 'prompt eval time' in line:
                print(f"  {line.strip()}")

        return generated.strip()
    finally:
        os.unlink(stdout_file)
        os.unlink(stderr_file)


def run_hf_model(model_path: str, prompt: str, n_tokens: int) -> tuple[str, list[int]]:
    """Run HF reference model with greedy decoding and return generated text + token IDs."""
    print(f"\n{'='*60}")
    print(f"Running HuggingFace reference model ({n_tokens} tokens, greedy)...")
    print(f"{'='*60}")

    from python.reference.registry import ModelRegistry
    import python.reference.qwen35_moe  # noqa: F401 - registers model

    model_ref = ModelRegistry.create("qwen35_moe", model_path, verbose=True)
    model_ref.load_model(torch_dtype=torch.float32)

    tokenizer = model_ref.tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model_ref.device)

    print(f"Input tokens ({input_ids.shape[1]}): {input_ids[0].tolist()}")
    print(f"Input decoded: {repr(tokenizer.decode(input_ids[0]))}")

    # Greedy decode token by token
    generated_ids = []
    current_ids = input_ids

    with torch.no_grad():
        for step in range(n_tokens):
            outputs = model_ref.hf_model(current_ids)
            logits = outputs.logits[:, -1, :]  # [1, vocab]

            # Greedy: pick argmax
            next_token = logits.argmax(dim=-1).item()
            generated_ids.append(next_token)

            # Top-5 for debugging
            top5_vals, top5_ids = logits.topk(5, dim=-1)
            top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids[0].tolist()]
            top5_probs = torch.softmax(top5_vals, dim=-1)[0].tolist()

            decoded = tokenizer.decode([next_token])
            print(f"  Step {step:2d}: token={next_token:6d} {repr(decoded):20s}  "
                  f"top5={list(zip(top5_tokens, [f'{p:.3f}' for p in top5_probs]))}")

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                print(f"  -> EOS at step {step}")
                break

            # Append for next step (full context, no KV cache for simplicity)
            next_tensor = torch.tensor([[next_token]], device=model_ref.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nHF generated: {repr(generated_text)}")
    return generated_text, generated_ids


def compare_outputs(prompt: str, llama_text: str, hf_text: str, tokenizer=None):
    """Compare llama.cpp and HF outputs."""
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"Prompt:     {repr(prompt)}")
    print(f"llama.cpp:  {repr(llama_text)}")
    print(f"HF model:   {repr(hf_text)}")
    print()

    # Simple word-level comparison
    llama_words = llama_text.split()
    hf_words = hf_text.split()

    max_len = max(len(llama_words), len(hf_words))
    matches = 0
    for i in range(max_len):
        lw = llama_words[i] if i < len(llama_words) else "<END>"
        hw = hf_words[i] if i < len(hf_words) else "<END>"
        match = "✓" if lw == hw else "✗"
        if lw == hw:
            matches += 1
        print(f"  Word {i:2d}: {match} llama={repr(lw):20s}  hf={repr(hw):20s}")

    print()
    pct = matches / max_len * 100 if max_len > 0 else 0
    print(f"Word match: {matches}/{max_len} ({pct:.1f}%)")

    if llama_text == hf_text:
        print("\n✓ EXACT MATCH - HF reference model matches llama.cpp perfectly!")
        return True
    elif pct >= 80:
        print(f"\n~ CLOSE MATCH ({pct:.0f}%) - Minor divergence expected due to quantization dequant differences")
        return True
    else:
        print(f"\n✗ MISMATCH ({pct:.0f}%) - Significant divergence, investigate")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify HF MoE model vs llama.cpp")
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model file")
    parser.add_argument("--llama-cli", default="/tmp/llama.cpp/build/bin/llama-completion",
                        help="Path to llama-completion binary")
    parser.add_argument("--prompt", "-p", default="The capital of France is",
                        help="Test prompt")
    parser.add_argument("--n-tokens", "-n", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--threads", "-t", type=int, default=28, help="CPU threads for llama.cpp")
    parser.add_argument("--skip-llama", action="store_true",
                        help="Skip llama.cpp run, use cached output from --llama-output")
    parser.add_argument("--llama-output", default=None,
                        help="Use pre-captured llama.cpp output instead of running it")
    args = parser.parse_args()

    # Step 1: Get llama.cpp output
    if args.llama_output:
        print(f"Using pre-captured llama.cpp output: {repr(args.llama_output)}")
        llama_text = args.llama_output
    elif args.skip_llama:
        llama_text = ""
        print("Skipping llama.cpp (--skip-llama)")
    else:
        llama_text = run_llamacpp(args.llama_cli, args.model, args.prompt, args.n_tokens, args.threads)

    # Step 2: Run HF model
    hf_text, hf_ids = run_hf_model(args.model, args.prompt, args.n_tokens)

    # Step 3: Compare
    if llama_text:
        success = compare_outputs(args.prompt, llama_text, hf_text)
        sys.exit(0 if success else 1)
    else:
        print(f"\nHF model output: {repr(hf_text)}")
        print("(no llama.cpp reference to compare against)")


if __name__ == "__main__":
    main()
