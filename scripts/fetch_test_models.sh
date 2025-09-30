#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main"
MODEL_DIR="models"

# List of quantized variants to attempt. Some may not exist upstream; missing ones are warned & skipped.
MODELS=(
  "qwen2.5-0.5b-instruct-q2_k.gguf"
  "qwen2.5-0.5b-instruct-q3_k.gguf"
  "qwen2.5-0.5b-instruct-q4_0.gguf"
  "qwen2.5-0.5b-instruct-q4_1.gguf"
  "qwen2.5-0.5b-instruct-q5_0.gguf"
  "qwen2.5-0.5b-instruct-q5_1.gguf"
  "qwen2.5-0.5b-instruct-q6_k.gguf"
  "qwen2.5-0.5b-instruct-q8_0.gguf"
  "qwen2.5-0.5b-instruct-q4_k_m.gguf"
  "qwen2.5-0.5b-instruct-q8_1.gguf"
)

echo "[fetch_test_models] Ensuring test models present in '$MODEL_DIR'"
mkdir -p "$MODEL_DIR"

have_any=0
for f in "${MODELS[@]}"; do
  if [[ -s "$MODEL_DIR/$f" ]]; then
    echo "[fetch_test_models] Found existing $f (skip)"
    have_any=1
    continue
  fi
  if [[ -n "${LLAMINAR_SKIP_MODEL_DOWNLOAD:-}" ]]; then
    echo "[fetch_test_models] Skipping download for $f due to LLAMINAR_SKIP_MODEL_DOWNLOAD"
    continue
  fi
  url="$BASE_URL/$f"
  echo "[fetch_test_models] Downloading $f from $url"
  if curl -L --fail --retry 3 --retry-delay 2 -o "$MODEL_DIR/$f.part" "$url" 2>&1; then
    mv "$MODEL_DIR/$f.part" "$MODEL_DIR/$f"
    echo "[fetch_test_models] Downloaded $f ($(du -h "$MODEL_DIR/$f" | cut -f1))"
    have_any=1
  else
    echo "[fetch_test_models][WARN] Failed to fetch $f (will skip this variant)" >&2
    rm -f "$MODEL_DIR/$f.part"
  fi
done

if [[ $have_any -eq 0 ]]; then
  echo "[fetch_test_models][WARN] No models present after fetch attempts." >&2
  if [[ -n "${LLAMINAR_ENFORCE_MODELS:-}" ]]; then
    echo "[fetch_test_models][ERROR] LLAMINAR_ENFORCE_MODELS set; failing." >&2
    exit 1
  fi
fi

echo "[fetch_test_models] Done."