#!/bin/bash
# Focused parity baseline — 26 tests covering all model families.
#
# Covers: Qwen2, Qwen3, Qwen3.5 (dense), Qwen3.5 MoE (sparse),
#         NodeLocalTP (multi-device), HybridPPTP (pipeline+tensor parallel).
#
# Tests run sequentially (parity tests share GPU resources and use
# RESOURCE_LOCK for serialization within CTest).
#
# Usage:
#   .githooks/run_parity_baseline.sh                    # default build dir
#   .githooks/run_parity_baseline.sh build_v2_custom    # custom build dir

set -e

BUILD_DIR="${1:-build_v2_integration}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS=(
  # Qwen2 SingleDevice (6)
  "Qwen2SingleDeviceParityTest_PrefillParity_CPU_KV_FP16$"
  "Qwen2SingleDeviceParityTest_DecodeParity_CPU_KV_FP16$"
  "Qwen2SingleDeviceParityTest_PrefillParity_CUDA_KV_FP16$"
  "Qwen2SingleDeviceParityTest_DecodeParity_CUDA_KV_FP16$"
  "Qwen2SingleDeviceParityTest_PrefillParity_ROCm_KV_FP16$"
  "Qwen2SingleDeviceParityTest_DecodeParity_ROCm_KV_FP16$"
  # Qwen3 SingleDevice (6)
  "Qwen3SingleDeviceParityTest_PrefillParity_Qwen3_CPU_KV_FP16$"
  "Qwen3SingleDeviceParityTest_DecodeParity_Qwen3_CPU_KV_FP16$"
  "Qwen3SingleDeviceParityTest_PrefillParity_Qwen3_CUDA_KV_FP16$"
  "Qwen3SingleDeviceParityTest_DecodeParity_Qwen3_CUDA_KV_FP16$"
  "Qwen3SingleDeviceParityTest_PrefillParity_Qwen3_ROCm_KV_FP16$"
  "Qwen3SingleDeviceParityTest_DecodeParity_Qwen3_ROCm_KV_FP16$"
  # Qwen3.5 SingleDevice 4B (6)
  "Qwen35SingleDeviceParityTest_PrefillParity_Qwen35_4B_CPU_KV_FP16$"
  "Qwen35SingleDeviceParityTest_DecodeParity_Qwen35_4B_CPU_KV_FP16$"
  "Qwen35SingleDeviceParityTest_PrefillParity_Qwen35_4B_CUDA_KV_FP16$"
  "Qwen35SingleDeviceParityTest_DecodeParity_Qwen35_4B_CUDA_KV_FP16$"
  "Qwen35SingleDeviceParityTest_PrefillParity_Qwen35_4B_ROCm_KV_FP16$"
  "Qwen35SingleDeviceParityTest_DecodeParity_Qwen35_4B_ROCm_KV_FP16$"
  # Qwen3.5 MoE SingleDevice CPU (2)
  "Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_CPU_KV_FP16$"
  "Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_CPU_KV_FP16$"
  # Qwen2 NodeLocalTP CPU (2)
  "Qwen2NodeLocalTPParityTest_PrefillParity_NodeLocalTP_2xMPI_CPU$"
  "Qwen2NodeLocalTPParityTest_DecodeParity_NodeLocalTP_2xMPI_CPU$"
  # Qwen3.5 MoE NodeLocalTP CPU (2)
  "Qwen35MoENodeLocalTPParityTest_PrefillParity_NodeLocalTP_2xMPI_CPU_35B_MoE$"
  "Qwen35MoENodeLocalTPParityTest_DecodeParity_NodeLocalTP_2xMPI_CPU_35B_MoE$"
  # Qwen3.5 MoE HybridPPTP (2)
  "Qwen35MoEHybridPPTPParityTest_PrefillParityWithGpuExpertCache"
  "Qwen35MoEHybridPPTPParityTest_DecodeParityWithGpuExpertCache"
)

TOTAL=${#TESTS[@]}
PASSED=0
FAILED=0
FAILED_NAMES=()

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         PARITY BASELINE: ${TOTAL} tests (sequential)              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Build dir: $BUILD_DIR"
echo "Start: $(date)"
echo ""

for i in "${!TESTS[@]}"; do
  TEST="${TESTS[$i]}"
  NUM=$((i + 1))
  DISPLAY_NAME="${TEST%\$}"  # Strip trailing $ for display
  printf "[%2d/${TOTAL}] %-70s " "$NUM" "$DISPLAY_NAME"

  # MoE 35B decode tests need longer timeout (model is large on CPU)
  TIMEOUT=300
  if [[ "$TEST" == *MoE* ]] || [[ "$TEST" == *HybridPPTP* ]]; then
    TIMEOUT=600
  fi

  if ctest --test-dir "$BUILD_DIR" -R "${TEST}" --no-tests=error --output-on-failure --timeout "$TIMEOUT" > /tmp/parity_baseline_${NUM}.log 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    PASSED=$((PASSED + 1))
  else
    echo -e "${RED}FAIL${NC}"
    FAILED=$((FAILED + 1))
    FAILED_NAMES+=("$DISPLAY_NAME")
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "End: $(date)"
echo -e "PASSED: ${GREEN}$PASSED${NC} / $TOTAL"
echo -e "FAILED: ${RED}$FAILED${NC} / $TOTAL"
if [[ $FAILED -gt 0 ]]; then
  echo ""
  echo -e "${RED}Failed tests:${NC}"
  for name in "${FAILED_NAMES[@]}"; do
    echo "  - $name"
  done
  echo ""
  echo -e "${YELLOW}Re-run a specific failed test with:${NC}"
  echo -e "${YELLOW}  ctest --test-dir $BUILD_DIR -R \"<test_name>\" -V${NC}"
  exit 1
fi
echo "════════════════════════════════════════════════════════════════"
exit 0
