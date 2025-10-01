#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main"
MODEL_DIR="models"

# Stable variants confirmed to exist upstream (reduce noise by default)
STABLE_MODELS=(
  "qwen2.5-0.5b-instruct-q2_k.gguf"
  "qwen2.5-0.5b-instruct-q4_0.gguf"
  "qwen2.5-0.5b-instruct-q5_0.gguf"
  "qwen2.5-0.5b-instruct-q6_k.gguf"
  "qwen2.5-0.5b-instruct-q8_0.gguf"
  "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

# Experimental / legacy variants that produced 404s during CI runs.
# Enable by setting LLAMINAR_FETCH_EXPERIMENTAL=1 to probe for them.
EXPERIMENTAL_MODELS=(
  "qwen2.5-0.5b-instruct-q3_k.gguf"   # Currently 404
  "qwen2.5-0.5b-instruct-q4_1.gguf"   # Legacy format (restored loader support) – 404
  "qwen2.5-0.5b-instruct-q5_1.gguf"   # Legacy format (restored loader support) – 404
  "qwen2.5-0.5b-instruct-q8_1.gguf"   # Unsupported upstream – 404
)

MODELS=("${STABLE_MODELS[@]}")
if [[ -n "${LLAMINAR_FETCH_EXPERIMENTAL:-}" ]]; then
  MODELS+=("${EXPERIMENTAL_MODELS[@]}")
  echo "[fetch_test_models] Experimental variant probing enabled (LLAMINAR_FETCH_EXPERIMENTAL=1)" >&2
fi

echo "[fetch_test_models] Ensuring test models present in '$MODEL_DIR'"
mkdir -p "$MODEL_DIR"

have_any=0
downloaded=()
skipped_existing=()
missing=()
attempted=()

preflight_check() {
  local url="$1"
  # Fast HEAD probe to avoid long curl output for known-missing variants
  if curl -s -I -L "$url" | head -n1 | grep -q " 200"; then
    return 0
  fi
  return 1
}
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
  attempted+=("$f")
  if ! preflight_check "$url"; then
    echo "[fetch_test_models][INFO] Skipping $f (HTTP 404 preflight)" >&2
    missing+=("$f")
    continue
  fi
  echo "[fetch_test_models] Downloading $f from $url"
  if curl -L --fail --retry 3 --retry-delay 2 -o "$MODEL_DIR/$f.part" "$url" 2>&1; then
    mv "$MODEL_DIR/$f.part" "$MODEL_DIR/$f"
    size=$(du -h "$MODEL_DIR/$f" | cut -f1)
    echo "[fetch_test_models] Downloaded $f ($size)"
    downloaded+=("$f:$size")
    have_any=1
  else
    echo "[fetch_test_models][WARN] Failed to fetch $f after preflight success (possible transient)" >&2
    rm -f "$MODEL_DIR/$f.part"
    missing+=("$f")
  fi
done

if [[ $have_any -eq 0 ]]; then
  echo "[fetch_test_models][WARN] No models present after fetch attempts." >&2
  if [[ -n "${LLAMINAR_ENFORCE_MODELS:-}" ]]; then
    echo "[fetch_test_models][ERROR] LLAMINAR_ENFORCE_MODELS set; failing." >&2
    exit 1
  fi
fi

echo "[fetch_test_models] Summary:"
echo "  Present (pre-existing): ${#skipped_existing[@]}"${skipped_existing:+" -> ${skipped_existing[*]}"}
echo "  Downloaded: ${#downloaded[@]}"${downloaded:+" -> ${downloaded[*]}"}
echo "  Missing (404): ${#missing[@]}"${missing:+" -> ${missing[*]}"}
echo "  Attempted: ${#attempted[@]}"${attempted:+" -> ${attempted[*]}"}
if [[ -n "${LLAMINAR_FETCH_EXPERIMENTAL:-}" && ${#missing[@]} -gt 0 ]]; then
  echo "[fetch_test_models][NOTE] Some experimental variants are currently unavailable upstream; this is informational only." >&2
fi
echo "[fetch_test_models] Done."