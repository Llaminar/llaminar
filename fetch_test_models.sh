#!/bin/bash
# Fetch test models for V2 dequantization equivalency tests

set -e

MODELS_DIR="/workspaces/llaminar/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "Fetching IQ2_XS test model..."
wget -c https://huggingface.co/bartowski/ReWiz-Qwen-2.5-14B-GGUF/resolve/main/ReWiz-Qwen-2.5-14B-IQ2_XS.gguf

echo "Fetching IQ4_XS test model..."
wget -c https://huggingface.co/bartowski/ReWiz-Qwen-2.5-14B-GGUF/resolve/main/ReWiz-Qwen-2.5-14B-IQ4_XS.gguf

echo "Fetching BF16 test model..."
wget -c https://huggingface.co/Mungert/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-bf16.gguf

echo "Done! Downloaded models:"
ls -lh ReWiz-Qwen-2.5-14B-IQ*.gguf Qwen2.5-1.5B-Instruct-bf16.gguf 2>/dev/null || true
