#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTEGRATION_BUILD_DIR="${ROOT_DIR}/build_v2_integration"
RELEASE_BIN="${ROOT_DIR}/build_v2_release/llaminar2"
MODEL_PATH="${ROOT_DIR}/models/qwen2.5-0.5b-instruct-q4_0.gguf"

PHASE15_REGEX='^V2_Unit_DeviceGraphOrchestratorSnapshots$|^V2_Unit_GraphBufferManager$|^V2_Unit_GraphBufferManager_BARAllocation$|^V2_Unit_DeviceGraphOrchestratorBufferManagement$|^V2_Unit_DeviceGraphOrchestrator$|^V2_Unit_DeviceGraphOrchestrator_PhaseAwareWeights$|^V2_Unit_DeviceGraphOrchestrator_WeightStreaming$|^V2_Unit_DeviceGraphOrchestratorDomainWiring$|^V2_Unit_DeviceGraphOrchestratorPPTPInit$|^V2_Unit_GraphResolver$|^V2_Unit_Qwen2Graph_KVCachePP$|^V2_Integration_SegmentedGraphCaptureExecution$'

echo "[phase1.5] Building stable-suite targets..."
cmake --build "${INTEGRATION_BUILD_DIR}" --parallel \
  --target \
  v2_test_graph_buffer_manager \
  v2_test_device_graph_orchestrator \
  v2_test_segmented_graph_capture_execution

echo "[phase1.5] Running focused stable suite..."
ctest --test-dir "${INTEGRATION_BUILD_DIR}" --output-on-failure -R "${PHASE15_REGEX}"

if [[ ! -x "${RELEASE_BIN}" ]]; then
  echo "[phase1.5] ERROR: release binary not found at ${RELEASE_BIN}" >&2
  echo "[phase1.5] Build release first: cmake --build ${ROOT_DIR}/build_v2_release --parallel" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[phase1.5] ERROR: model not found at ${MODEL_PATH}" >&2
  exit 1
fi

echo "[phase1.5] Running benchmark snapshot..."
LLAMINAR_LOG_LEVEL=INFO "${RELEASE_BIN}" --benchmark -m "${MODEL_PATH}" -n 16 | tail -n 20

echo "[phase1.5] Stable suite + benchmark completed successfully."
