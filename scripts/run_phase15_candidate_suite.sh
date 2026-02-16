#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_v2_integration"

PHASE15_CANDIDATE_REGEX='^V2_Unit_DeviceGraphOrchestratorSnapshots$|^V2_Unit_GraphBufferManager$|^V2_Unit_GraphBufferManager_BARAllocation$|^V2_Unit_DeviceGraphOrchestratorBufferManagement$|^V2_Unit_DeviceGraphOrchestrator$|^V2_Unit_DeviceGraphOrchestrator_PhaseAwareWeights$|^V2_Unit_DeviceGraphOrchestrator_WeightStreaming$|^V2_Unit_DeviceGraphOrchestratorDomainWiring$|^V2_Unit_DeviceGraphOrchestratorPPTPInit$|^V2_Unit_GraphResolver$|^V2_Unit_Qwen2Graph_KVCachePP$|^V2_Unit_GraphExecutorCollective$|^V2_Unit_ComputeGraphExecution$|^V2_Unit_ComputeGraphMerge$|^V2_Unit_Qwen2GraphConfigBuilder$|^V2_Unit_GraphValidator$|^V2_Unit_Qwen2GraphSchema$|^V2_Unit_Qwen2GraphDomain$|^V2_Unit_Qwen2Graph_PP$|^V2_Integration_SegmentedGraphCaptureExecution$|^V2_Integration_GraphBatchedKVCache$|^V2_Integration_GraphSnapshotCallbackInvocation$'

echo "[phase1.5-candidate] Building candidate-suite targets..."
cmake --build "${BUILD_DIR}" --parallel \
  --target \
  v2_test_graph_buffer_manager \
  v2_test_device_graph_orchestrator \
  v2_test_segmented_graph_capture_execution \
  v2_unit_graph_executor_collective \
  v2_test_compute_graph_execution \
  v2_test_compute_graph_merge \
  v2_test_qwen2_graph_config_builder \
  v2_test_graph_validator \
  v2_test_graph_resolver \
  v2_test_qwen2_graph_schema \
  v2_test_qwen2graph_domain \
  v2_test_qwen2graph_pp \
  v2_test_qwen2graph_kvcache_pp \
  v2_integration_graph_batched_kvcache \
  v2_test_graph_snapshot_callback_invocation

echo "[phase1.5-candidate] Running candidate suite..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "${PHASE15_CANDIDATE_REGEX}"

echo "[phase1.5-candidate] Candidate suite completed successfully."
