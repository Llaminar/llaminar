#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main"
MODEL_DIR="models"

# Stable variants confirmed to exist upstream (reduce noise by default)
STABLE_MODELS=(
  "qwen2.5-0.5b-instruct-fp16.gguf"
  "qwen2.5-0.5b-instruct-q2_k.gguf"
  "qwen2.5-0.5b-instruct-q4_0.gguf"
  "qwen2.5-0.5b-instruct-q5_0.gguf"
  "qwen2.5-0.5b-instruct-q6_k.gguf"
  "qwen2.5-0.5b-instruct-q8_0.gguf"
  "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

# Large 7B models for IQ format testing (downloaded separately with specific URLs)
# These require LLAMINAR_FETCH_7B_IQ_MODELS=1 to enable due to size (~1-3GB each)
LARGE_7B_IQ_MODELS=(
  # Legacy 0.5B models (RichardErkhov)
  "https://huggingface.co/RichardErkhov/unsloth_-_Qwen2-0.5B-gguf/resolve/main/Qwen2-0.5B.IQ3_M.gguf"
  "https://huggingface.co/RichardErkhov/unsloth_-_Qwen2-0.5B-gguf/resolve/main/Qwen2-0.5B.IQ3_XS.gguf"
  "https://huggingface.co/RichardErkhov/unsloth_-_Qwen2-0.5B-gguf/resolve/main/Qwen2-0.5B.IQ3_S.gguf"
  "https://huggingface.co/RichardErkhov/unsloth_-_Qwen2-0.5B-gguf/resolve/main/Qwen2-0.5B.IQ4_NL.gguf"
  "https://huggingface.co/RichardErkhov/unsloth_-_Qwen2-0.5B-gguf/resolve/main/Qwen2-0.5B.IQ4_XS.gguf"
  "https://huggingface.co/unsloth/Qwen2.5-Omni-7B-GGUF/resolve/main/Qwen2.5-Omni-7B-BF16.gguf"
  
  # Qwen2.5-7B models (bartowski) - for missing format coverage
  "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q2_K.gguf"      # Q2_K format
  "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q3_K_M.gguf"    # Q3_K format
  "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_0.gguf"      # Q4_1 format (mislabeled upstream)
  "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-IQ4_XS.gguf"    # IQ4_XS format
  "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-IQ3_XS.gguf"    # IQ3_XXS, IQ3_S formats
  "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-IQ2_M.gguf"     # IQ2_S format
)

# Qwen3 models for Qwen3 architecture testing.
# Always fetched when available – essential for parity tests.
QWEN3_MODELS=(
  "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
  "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
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
  # Fast HEAD probe first (may fail on some hosts that don't support HEAD correctly for large objects)
  if curl -s -I -L "$url" | head -n1 | grep -q " 200"; then
    [[ -n "${LLAMINAR_FETCH_VERBOSE:-}" ]] && echo "[fetch_test_models][preflight] HEAD 200 $url" >&2
    return 0
  fi
  # Fallback: perform a ranged GET (first byte) which often succeeds even if HEAD blocked.
  if curl -L -s --fail -H 'Range: bytes=0-0' -o /dev/null "$url"; then
    [[ -n "${LLAMINAR_FETCH_VERBOSE:-}" ]] && echo "[fetch_test_models][preflight] RANGED GET OK (HEAD failed) $url" >&2
    return 0
  fi
  [[ -n "${LLAMINAR_FETCH_VERBOSE:-}" ]] && echo "[fetch_test_models][preflight] UNAVAILABLE $url" >&2
  return 1
}
for f in "${MODELS[@]}"; do
  if [[ -s "$MODEL_DIR/$f" ]]; then
    echo "[fetch_test_models] Found existing $f (skip)"
    have_any=1
    skipped_existing+=("$f")
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

# Qwen3 models (always attempted - small and essential for parity tests)
echo "[fetch_test_models] Checking Qwen3 models" >&2
for url in "${QWEN3_MODELS[@]}"; do
  fname=$(basename "$url")
  if [[ -s "$MODEL_DIR/$fname" ]]; then
    echo "[fetch_test_models] Found existing Qwen3 model $fname (skip)"
    skipped_existing+=("$fname")
    continue
  fi
  if [[ -n "${LLAMINAR_SKIP_MODEL_DOWNLOAD:-}" ]]; then
    echo "[fetch_test_models] Skipping Qwen3 download for $fname due to LLAMINAR_SKIP_MODEL_DOWNLOAD"
    continue
  fi
  attempted+=("$fname")
  if ! preflight_check "$url"; then
    echo "[fetch_test_models][WARN] Qwen3 preflight failed (404?) $fname" >&2
    missing+=("$fname")
    continue
  fi
  echo "[fetch_test_models] Downloading Qwen3 model $fname from $url" >&2
  if curl -L --fail --retry 3 --retry-delay 3 -o "$MODEL_DIR/$fname.part" "$url" 2>&1; then
    mv "$MODEL_DIR/$fname.part" "$MODEL_DIR/$fname"
    size=$(du -h "$MODEL_DIR/$fname" | cut -f1)
    echo "[fetch_test_models] Downloaded Qwen3 model $fname ($size)" >&2
    downloaded+=("$fname:$size")
    have_any=1
  else
    echo "[fetch_test_models][ERROR] Failed to download Qwen3 model $fname" >&2
    rm -f "$MODEL_DIR/$fname.part"
    missing+=("$fname")
  fi
done

# Optional large FP32 model(s) (full precision) – gated to avoid CI timeouts / large bandwidth by default.
if [[ -n "${LLAMINAR_FETCH_FP32:-}" ]]; then
  echo "[fetch_test_models] LLAMINAR_FETCH_FP32 set – attempting full precision model fetch" >&2
  FP32_URLS=(
    "https://huggingface.co/Geraldine/Gemini-Distill-Qwen2.5-0.5B-ead-GGUF/resolve/main/Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf"
  )
  for url in "${FP32_URLS[@]}"; do
    fname=$(basename "$url")
    if [[ -s "$MODEL_DIR/$fname" ]]; then
      echo "[fetch_test_models] Found existing fp32 $fname (skip)"
      skipped_existing+=("$fname")
      continue
    fi
    if [[ -n "${LLAMINAR_SKIP_MODEL_DOWNLOAD:-}" ]]; then
      echo "[fetch_test_models] Skipping fp32 download for $fname due to LLAMINAR_SKIP_MODEL_DOWNLOAD"
      continue
    fi
    attempted+=("$fname")
    if ! preflight_check "$url"; then
      echo "[fetch_test_models][WARN] FP32 preflight failed (404?) $fname" >&2
      missing+=("$fname")
      continue
    fi
    echo "[fetch_test_models] Downloading FP32 model $fname from $url" >&2
    if curl -L --fail --retry 3 --retry-delay 3 -o "$MODEL_DIR/$fname.part" "$url" 2>&1; then
      mv "$MODEL_DIR/$fname.part" "$MODEL_DIR/$fname"
      size=$(du -h "$MODEL_DIR/$fname" | cut -f1)
      echo "[fetch_test_models] Downloaded FP32 $fname ($size)" >&2
      downloaded+=("$fname:$size")
      have_any=1
    else
      echo "[fetch_test_models][ERROR] Failed to download FP32 $fname" >&2
      rm -f "$MODEL_DIR/$fname.part"
      missing+=("$fname")
    fi
  done
fi

# Optional large 7B IQ models for comprehensive IQ format testing.
# Gated to avoid CI timeouts / large bandwidth by default (1-3GB each).
if [[ -n "${LLAMINAR_FETCH_7B_IQ_MODELS:-}" ]]; then
  echo "[fetch_test_models] LLAMINAR_FETCH_7B_IQ_MODELS set – attempting 7B IQ model fetch" >&2
  for url in "${LARGE_7B_IQ_MODELS[@]}"; do
    fname=$(basename "$url")
    if [[ -s "$MODEL_DIR/$fname" ]]; then
      echo "[fetch_test_models] Found existing 7B IQ model $fname (skip)"
      skipped_existing+=("$fname")
      continue
    fi
    if [[ -n "${LLAMINAR_SKIP_MODEL_DOWNLOAD:-}" ]]; then
      echo "[fetch_test_models] Skipping 7B IQ download for $fname due to LLAMINAR_SKIP_MODEL_DOWNLOAD"
      continue
    fi
    attempted+=("$fname")
    if ! preflight_check "$url"; then
      echo "[fetch_test_models][WARN] 7B IQ preflight failed (404?) $fname" >&2
      missing+=("$fname")
      continue
    fi
    echo "[fetch_test_models] Downloading 7B IQ model $fname from $url (this may take several minutes)" >&2
    if curl -L --fail --retry 3 --retry-delay 3 -o "$MODEL_DIR/$fname.part" "$url" 2>&1; then
      mv "$MODEL_DIR/$fname.part" "$MODEL_DIR/$fname"
      size=$(du -h "$MODEL_DIR/$fname" | cut -f1)
      echo "[fetch_test_models] Downloaded 7B IQ model $fname ($size)" >&2
      downloaded+=("$fname:$size")
      have_any=1
    else
      echo "[fetch_test_models][ERROR] Failed to download 7B IQ model $fname" >&2
      rm -f "$MODEL_DIR/$fname.part"
      missing+=("$fname")
    fi
  done
fi

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