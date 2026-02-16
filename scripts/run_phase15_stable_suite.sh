#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_v2_integration"

PHASE15_REGEX='^V2_Unit_DeviceGraphOrchestratorSnapshots$|^V2_Unit_GraphBufferManager$|^V2_Unit_GraphBufferManager_BARAllocation$|^V2_Unit_DeviceGraphOrchestratorBufferManagement$|^V2_Unit_DeviceGraphOrchestrator$|^V2_Unit_DeviceGraphOrchestrator_PhaseAwareWeights$|^V2_Unit_DeviceGraphOrchestrator_WeightStreaming$|^V2_Unit_DeviceGraphOrchestratorDomainWiring$|^V2_Unit_DeviceGraphOrchestratorPPTPInit$|^V2_Unit_GraphResolver$|^V2_Unit_Qwen2Graph_KVCachePP$|^V2_Integration_SegmentedGraphCaptureExecution$'

echo "[phase1.5] Building stable-suite targets..."
cmake --build "${BUILD_DIR}" --parallel \
  --target \
  v2_test_graph_buffer_manager \
  v2_test_device_graph_orchestrator \
  v2_test_segmented_graph_capture_execution

echo "[phase1.5] Running focused stable suite..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "${PHASE15_REGEX}"

echo "[phase1.5] Stable suite completed successfully."
