#!/bin/bash
# compare_ranks.sh - Compare inference output between different MPI rank configurations
#
# Usage: ./scripts/compare_ranks.sh -m MODEL -p PROMPT [-n TOKENS] [--ranks1 N1] [--ranks2 N2]
#
# This script automatically runs inference with two different MPI rank configurations
# and compares the generated tokens to verify tensor parallelism correctness.
#
# Prerequisites:
#   - Release build exists at build_v2_release/src/v2/llaminar2
#   - Model file exists at specified path
#   - OpenMPI is available
#
# Examples:
#   # Basic comparison (1 rank vs 2 ranks)
#   ./scripts/compare_ranks.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello world" -n 20
#
#   # Custom rank configurations
#   ./scripts/compare_ranks.sh -m model.gguf -p "test" --ranks1 1 --ranks2 4
#
#   # Verbose output
#   ./scripts/compare_ranks.sh -m model.gguf -p "test" -v

set -e

# Defaults
MODEL=""
PROMPT=""
TOKENS=20
RANKS1=1
RANKS2=2
VERBOSE=0
BUILD_DIR="build_v2_release"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -n|--tokens)
            TOKENS="$2"
            shift 2
            ;;
        --ranks1)
            RANKS1="$2"
            shift 2
            ;;
        --ranks2)
            RANKS2="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 -m MODEL -p PROMPT [-n TOKENS] [--ranks1 N1] [--ranks2 N2]"
            echo ""
            echo "Options:"
            echo "  -m, --model      Path to GGUF model file (required)"
            echo "  -p, --prompt     Inference prompt (required)"
            echo "  -n, --tokens     Number of tokens to generate (default: 20)"
            echo "  --ranks1         First rank configuration (default: 1)"
            echo "  --ranks2         Second rank configuration (default: 2)"
            echo "  --build-dir      Build directory (default: build_v2_release)"
            echo "  -v, --verbose    Show detailed output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p 'Hello world' -n 20"
            echo "  $0 -m model.gguf -p 'test' --ranks1 1 --ranks2 4 -v"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$PROMPT" ]]; then
    echo -e "${RED}Error: -m MODEL and -p PROMPT are required${NC}"
    echo "Use -h for help"
    exit 1
fi

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    echo -e "${RED}Error: Model file not found: $MODEL${NC}"
    exit 1
fi

# Check build exists
LLAMINAR_BIN="$BUILD_DIR/src/v2/llaminar2"
if [[ ! -f "$LLAMINAR_BIN" ]]; then
    echo -e "${RED}Error: Binary not found: $LLAMINAR_BIN${NC}"
    echo "Build the release version first:"
    echo "  cmake -B $BUILD_DIR -S src/v2 -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build $BUILD_DIR --parallel"
    exit 1
fi

# Temp files for output
TMPDIR=$(mktemp -d)
OUTPUT1="$TMPDIR/output_ranks${RANKS1}.txt"
OUTPUT2="$TMPDIR/output_ranks${RANKS2}.txt"
TOKENS1="$TMPDIR/tokens_ranks${RANKS1}.txt"
TOKENS2="$TMPDIR/tokens_ranks${RANKS2}.txt"

cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo "=========================================="
echo -e "${YELLOW}Comparison Test: $RANKS1 rank(s) vs $RANKS2 rank(s)${NC}"
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens: $TOKENS"
echo "=========================================="

# Set up optimal environment for MPI/OpenMP
export OMP_NUM_THREADS=28
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
export OMP_NESTED=false
export OMP_DYNAMIC=false
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export OPENBLAS_NUM_THREADS=28
export GOTO_NUM_THREADS=28
export MKL_NUM_THREADS=28
export MKL_DYNAMIC=false
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_openib_allow_ib=1

# Function to run inference and extract tokens
run_inference() {
    local ranks=$1
    local output_file=$2
    local tokens_file=$3
    
    echo ""
    echo "Running with $ranks rank(s)..."
    
    if [[ $ranks -eq 1 ]]; then
        # Single rank - no MPI
        LLAMINAR_LOG_LEVEL=DEBUG "$LLAMINAR_BIN" \
            -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
            2>&1 | tee "$output_file"
    else
        # Multi-rank with MPI
        LLAMINAR_LOG_LEVEL=DEBUG mpirun --allow-run-as-root -np "$ranks" \
            --bind-to socket --map-by socket \
            "$LLAMINAR_BIN" \
            -m "$MODEL" -p "$PROMPT" -n "$TOKENS" -t 0 \
            2>&1 | tee "$output_file"
    fi
    
    # Extract sampled tokens from output
    # Look for lines like "[DEBUG] [Rank 0] Sampled token: 12345"
    grep -oP "Sampled token: \K[0-9]+" "$output_file" > "$tokens_file" || true
    
    # If no tokens found with that pattern, try alternative patterns
    if [[ ! -s "$tokens_file" ]]; then
        grep -oP "sampled=\K[0-9]+" "$output_file" > "$tokens_file" || true
    fi
    
    # Fallback: extract from "Generated:" line if present
    if [[ ! -s "$tokens_file" ]]; then
        grep -oP "Generated: .*" "$output_file" > "$tokens_file" || true
    fi
}

# Run with first rank configuration
run_inference "$RANKS1" "$OUTPUT1" "$TOKENS1"

# Run with second rank configuration
run_inference "$RANKS2" "$OUTPUT2" "$TOKENS2"

# Compare tokens
echo ""
echo "=========================================="
echo -e "${YELLOW}COMPARISON RESULTS${NC}"
echo "=========================================="

TOKEN_COUNT1=$(wc -l < "$TOKENS1" 2>/dev/null || echo "0")
TOKEN_COUNT2=$(wc -l < "$TOKENS2" 2>/dev/null || echo "0")

if [[ $VERBOSE -eq 1 ]] || [[ $TOKEN_COUNT1 -le 20 ]]; then
    TOKENS1_LIST=$(cat "$TOKENS1" 2>/dev/null | tr '\n' ' ' || echo "(none)")
    TOKENS2_LIST=$(cat "$TOKENS2" 2>/dev/null | tr '\n' ' ' || echo "(none)")
    echo "Tokens ($RANKS1 rank):   $TOKENS1_LIST"
    echo "Tokens ($RANKS2 ranks):  $TOKENS2_LIST"
else
    echo "Tokens ($RANKS1 rank):   $TOKEN_COUNT1 tokens generated"
    echo "Tokens ($RANKS2 ranks):  $TOKEN_COUNT2 tokens generated"
fi
echo ""

# Check if we got any tokens
if [[ ! -s "$TOKENS1" ]] || [[ ! -s "$TOKENS2" ]]; then
    echo -e "${YELLOW}⚠ WARNING: Could not extract tokens from output${NC}"
    echo ""
    echo "This may indicate:"
    echo "  - Log level too low (need DEBUG)"
    echo "  - Output format changed"
    echo "  - Inference failed"
    echo ""
    if [[ $VERBOSE -eq 1 ]]; then
        echo "--- Output from $RANKS1 rank(s) ---"
        tail -50 "$OUTPUT1"
        echo ""
        echo "--- Output from $RANKS2 rank(s) ---"
        tail -50 "$OUTPUT2"
    fi
    exit 2
fi

if diff -q "$TOKENS1" "$TOKENS2" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS: Token sequences are IDENTICAL${NC}"
    echo ""
    echo "Both rank configurations produced the same $TOKEN_COUNT1 tokens."
    exit 0
else
    echo -e "${RED}❌ FAIL: Token sequences DIFFER${NC}"
    echo ""
    
    # Find first divergence point
    echo "First divergence:"
    DIVERGE_LINE=$(paste "$TOKENS1" "$TOKENS2" | awk -F'\t' 'BEGIN{i=0} {i++; if($1!=$2){print i": "$1" vs "$2; exit}}')
    echo "  Token $DIVERGE_LINE"
    echo ""
    
    # Show diff if verbose
    if [[ $VERBOSE -eq 1 ]]; then
        echo "Full diff:"
        diff "$TOKENS1" "$TOKENS2" || true
        echo ""
    fi
    
    # Show counts
    echo "Token counts: $RANKS1 rank=$TOKEN_COUNT1, $RANKS2 ranks=$TOKEN_COUNT2"
    
    exit 1
fi
