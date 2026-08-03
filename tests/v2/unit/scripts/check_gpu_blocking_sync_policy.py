#!/usr/bin/env python3
"""Reject unreviewed host-blocking GPU synchronization in production source.

The runtime's normal ordering primitive is an event recorded by the producer
stream and waited on by the consumer stream. A host-blocking stream or device
synchronization is legal only at an explicit ownership boundary such as final
result materialization, loader shutdown, diagnostics, or heterogeneous
collective coordination.

This sanitizer identifies the enclosing caller for every CUDA, HIP, and
Llaminar blocking synchronization call. Approved callers are listed with an
exact expected call count and an architectural category. Consequently, adding
another synchronization to an already-approved function still fails until the
new ownership boundary is reviewed and recorded deliberately.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import pathlib
import re
import sys
from collections import Counter
from collections.abc import Iterable


SOURCE_SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".hip", ".hpp", ".inc"}

SYNC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "raw_stream",
        re.compile(r"\b(?:cuda|hip)StreamSynchronize\s*\("),
    ),
    (
        "worker_stream",
        re.compile(r"(?:->|\.)synchronizeStream(?:Checked)?\s*\("),
    ),
    (
        "raw_device",
        re.compile(r"\b(?:cuda|hip)DeviceSynchronize\s*\("),
    ),
    (
        "raw_event",
        re.compile(r"\b(?:cuda|hip)EventSynchronize\s*\("),
    ),
    (
        "worker_event",
        re.compile(r"(?:->|\.)synchronizeEvent(?:Checked)?\s*\("),
    ),
    (
        "backend_event",
        re.compile(r"(?:->|\.)waitForEvent\s*\("),
    ),
    (
        "backend_sync_copy",
        re.compile(
            r"(?:->|\.)(?:deviceToHost|hostToDevice|deviceToDevice)\s*\("
        ),
    ),
    (
        "backend_sync_compute",
        re.compile(
            r"(?:->|\.)(?:"
            r"argmaxF32|"
            r"argmaxF32BatchedRows|"
            r"sampleTopKTopPF32|"
            r"topKF32"
            r")\s*\("
        ),
    ),
    (
        "backend_device",
        re.compile(r"(?:->|\.)synchronize\s*\("),
    ),
    (
        "rank_device",
        re.compile(r"(?:->|\.)synchronizeDevices\s*\("),
    ),
)

CONTROL_NAMES = {
    "catch",
    "for",
    "if",
    "requires",
    "switch",
    "while",
}

# Dedicated diagnostics and benchmark translation units do not participate in
# production inference. Keeping these path exemptions narrow avoids hundreds
# of caller entries while still preventing runtime code from acquiring a new
# blocking boundary unnoticed.
NON_PRODUCTION_PATH_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("src/v2/backends/benchmarks/", "benchmark"),
    ("src/v2/utils/CUDAKernelProfiler.cu", "profiler"),
    ("src/v2/utils/ROCmKernelProfiler.hip", "profiler"),
    ("src/v2/backends/cuda/CUDATensorValidation.cu", "validation"),
    ("src/v2/backends/rocm/ROCmTensorValidation.cpp", "validation"),
)


@dataclasses.dataclass(frozen=True, order=True)
class CallerKey:
    """Stable identity and primitive kind for one synchronization caller."""

    path: str
    caller: str
    kind: str


@dataclasses.dataclass(frozen=True)
class Allowance:
    """Reviewed synchronization budget for one production caller."""

    path: str
    caller: str
    kind: str
    count: int
    category: str
    reason: str

    @property
    def key(self) -> CallerKey:
        return CallerKey(self.path, self.caller, self.kind)


CATEGORY_REASONS = {
    "backend_primitive": (
        "Low-level implementation of an explicitly synchronous backend API; "
        "production execution uses its stream/event-aware sibling."
    ),
    "collective_boundary": (
        "Explicit synchronous collective/copy API or communicator lifecycle "
        "boundary; graph-native collectives use stream dependencies."
    ),
    "diagnostic": (
        "Opt-in validation, tracing, profiling, or graph verification which "
        "must observe completed device work."
    ),
    "graph_ownership": (
        "Host ownership transfer for a captured graph resource, fenced by the "
        "exact completion event rather than a device-wide synchronization."
    ),
    "heterogeneous_staging": (
        "Deliberate CPU/GPU or cross-backend staging boundary where host memory "
        "is part of the selected heterogeneous execution policy."
    ),
    "host_archive": (
        "Explicit KV/prefix-cache export, import, or immutable host observation "
        "boundary; this is not live GPU execution state."
    ),
    "host_result": (
        "Public API whose contract is to return a completed value to the host."
    ),
    "lifecycle": (
        "Initialization, reset, loader drain, resource destruction, or final "
        "release boundary outside steady-state graph execution."
    ),
}


def reviewed(
    category: str,
    *entries: tuple[str, str, str, int],
) -> tuple[Allowance, ...]:
    """Attach one architectural reason to exact caller/count entries."""

    reason = CATEGORY_REASONS[category]
    return tuple(
        Allowance(
            path=path,
            caller=caller,
            kind=kind,
            count=count,
            category=category,
            reason=reason,
        )
        for path, caller, kind, count in entries
    )


# Keep every group sorted by path, caller, and kind. These are exact reviewed
# ownership boundaries, never performance fallbacks. A new caller or an
# additional wait in an existing caller fails validation.
ALLOWANCES: tuple[Allowance, ...] = (
    *reviewed(
        "backend_primitive",
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::argmaxF32", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::argmaxF32BatchedRows", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::deviceToDevice", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::deviceToHost", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::hostToDevice", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::sampleTopKTopPF32", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::streamSynchronize", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::synchronize", "raw_device", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::synchronizeStream", "raw_stream", 1),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::topKF32", "raw_stream", 2),
        ("src/v2/backends/cuda/CUDABackend.cu", "CUDABackend::waitForEvent", "raw_event", 1),
        ("src/v2/backends/cuda/NvidiaDeviceContext.cu", "NvidiaDeviceContext::synchronizeEventChecked", "raw_event", 1),
        ("src/v2/backends/cuda/NvidiaDeviceContext.cu", "NvidiaDeviceContext::synchronizeStreamChecked", "raw_stream", 1),
        ("src/v2/backends/cuda/NvidiaDeviceContext.cu", "NvidiaDeviceContext::synchronizeChecked", "raw_device", 1),
        ("src/v2/backends/rocm/AMDDeviceContext.cpp", "AMDDeviceContext::synchronizeEventChecked", "raw_event", 1),
        ("src/v2/backends/rocm/AMDDeviceContext.cpp", "AMDDeviceContext::synchronizeChecked", "raw_device", 1),
        ("src/v2/backends/rocm/AMDDeviceContext.cpp", "AMDDeviceContext::synchronizeStreamChecked", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::argmaxF32", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::argmaxF32BatchedRows", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::deviceToDevice", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::deviceToHost", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::deviceToHostFast", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::hostToDevice", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::sampleTopKTopPF32", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::streamSynchronize", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::synchronize", "raw_device", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::synchronizeStream", "raw_stream", 1),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::topKF32", "raw_stream", 2),
        ("src/v2/backends/rocm/ROCmBackend.cpp", "ROCmBackend::waitForEvent", "raw_event", 1),
        ("src/v2/execution/local_execution/device/DeviceContext.cpp", "CUDADeviceContext::copyToDevice", "backend_sync_copy", 1),
        ("src/v2/execution/local_execution/device/DeviceContext.cpp", "CUDADeviceContext::copyToHost", "backend_sync_copy", 1),
        ("src/v2/execution/local_execution/device/DeviceContext.cpp", "CUDADeviceContext::synchronize", "backend_device", 1),
        ("src/v2/execution/local_execution/device/DeviceContext.cpp", "ROCmDeviceContext::copyToDevice", "backend_sync_copy", 1),
        ("src/v2/execution/local_execution/device/DeviceContext.cpp", "ROCmDeviceContext::copyToHost", "backend_sync_copy", 1),
        ("src/v2/execution/local_execution/device/DeviceContext.cpp", "ROCmDeviceContext::synchronize", "backend_device", 1),
        ("src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu", "cudaFlashAttn_synchronize", "raw_device", 1),
        ("src/v2/kernels/cuda/gdn/CUDAGatedDeltaNetKernels.cu", "cudaGDN_stream_synchronize", "raw_stream", 1),
        ("src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel_CUTLASS.cu", "cudaQuantGemm_streamSync", "raw_device", 1),
        ("src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel_CUTLASS.cu", "cudaQuantGemm_streamSync", "raw_stream", 1),
        ("src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip", "rocmGDN_stream_synchronize", "raw_stream", 1),
    ),
    *reviewed(
        "collective_boundary",
        ("src/v2/collective/LocalPPContext.cpp", "HierarchicalPPContext::synchronize", "backend_device", 3),
        ("src/v2/collective/LocalPPContext.cpp", "LocalPPContext::synchronize", "backend_device", 1),
        ("src/v2/collective/LocalPPContext.cpp", "LocalPPContext::synchronizeStream", "worker_stream", 1),
        ("src/v2/collective/LocalTPContext.cpp", "LocalTPContext::synchronize", "backend_device", 1),
        ("src/v2/collective/backends/NCCLBackend.cpp", "NCCLBackend::synchronize", "backend_device", 1),
        ("src/v2/collective/backends/NCCLBackendCUDA.cu", "cudaSynchronizeStream", "raw_stream", 1),
        ("src/v2/collective/backends/RCCLBackend.cpp", "RCCLBackend::synchronize", "backend_device", 1),
        ("src/v2/collective/backends/RCCLBackendHIP.cpp", "hipSynchronizeStream", "raw_stream", 1),
        ("src/v2/collective/backends/RCCLBackendHIP.cpp", "rcclPrimeAndDestroyComm", "raw_stream", 1),
        ("src/v2/collective/backends/ShmemSpinBackend.cpp", "ShmemSpinBackend::synchronize", "backend_device", 1),
        ("src/v2/collective/coordinators/NCCLCoordinator.cu", "NCCLCoordinator::doCopy", "raw_stream", 2),
        ("src/v2/collective/coordinators/NCCLCoordinator.cu", "NCCLCoordinator::doSynchronizeAll", "raw_stream", 1),
        ("src/v2/collective/coordinators/NCCLCoordinator.cu", "NCCLCoordinator::copy", "raw_stream", 1),
        ("src/v2/collective/coordinators/RCCLCoordinator.cpp", "RCCLCoordinator::cleanupOnThread", "raw_stream", 3),
        ("src/v2/collective/coordinators/RCCLCoordinator.cpp", "RCCLCoordinator::copy", "raw_stream", 1),
        ("src/v2/collective/coordinators/RCCLCoordinator.cpp", "RCCLCoordinator::doCopy", "raw_stream", 2),
        ("src/v2/collective/coordinators/RCCLCoordinator.cpp", "RCCLCoordinator::doSynchronizeAll", "raw_stream", 1),
    ),
    *reviewed(
        "diagnostic",
        ("src/v2/backends/cuda/NvidiaDeviceContext.cu", "NvidiaDeviceContext::debugSynchronize", "raw_device", 1),
        ("src/v2/backends/rocm/AMDDeviceContext.cpp", "AMDDeviceContext::debugSynchronize", "raw_device", 1),
        ("src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp", "tracePrefillAssignmentRuntime", "worker_stream", 1),
        ("src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp", "tracePrefillLLEPStatus", "worker_stream", 1),
        ("src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp", "ForwardExecutionEngine::executePrefillWithGraphCache", "worker_event", 1),
        ("src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp", "DeviceGraphCaptureController::executeCapturedReplaySegmentNormal", "worker_stream", 1),
        ("src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp", "DeviceGraphCaptureController::executeCapturedReplaySegmentVerify", "worker_stream", 3),
        ("src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp", "DeviceGraphCaptureController::executeManualReplaySegment", "worker_stream", 3),
        ("src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp", "DeviceGraphCaptureController::executeStreamOnlyReplay", "worker_stream", 1),
        ("src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp", "DeviceGraphExecutor::runStages", "backend_device", 1),
        ("src/v2/execution/local_execution/graph/StageTimeline.h", "collect", "worker_event", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::exportCompletedDeviceMoERebalanceMaintenanceStats", "worker_stream", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "completeMTPDiagnosticObservation", "backend_event", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "logMTPGraphReuseBoundaryDiagnostics", "backend_event", 1),
        ("src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp", "CUDAQuantisedGemmKernel::multiply_fused_tensor_impl", "raw_stream", 2),
        ("src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp", "CUDAQuantisedGemmKernel::multiply_with_fused_swiglu", "raw_stream", 1),
        ("src/v2/kernels/cuda/gemm/CuBLASGemmKernel.cu", "CuBLASGemmKernel::execute_batched_same_a", "raw_stream", 1),
        ("src/v2/kernels/cuda/ops/CUDAOpsKernels.cpp", "CUDAEmbeddingKernelT::apply_tensor", "raw_stream", 1),
        ("src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip", "rocmGDN_chunk_forward_batched_kernel_route", "raw_stream", 2),
        ("src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip", "rocmGDN_chunk_forward_effective", "raw_stream", 1),
        ("src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip", "rocmGDN_deinterleave_qkv", "raw_stream", 1),
        ("src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip", "rocmGDN_recurrent_step", "raw_stream", 1),
        ("src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip", "rocmGDN_recurrent_step_effective_row", "raw_stream", 1),
        ("src/v2/kernels/rocm/ops/ROCmEmbeddingKernelT.cpp", "ROCmEmbeddingKernelT::apply_tensor", "raw_stream", 4),
    ),
    *reviewed(
        "graph_ownership",
        ("src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp", "DeviceGraphExecutor::GraphSegmentCache::waitForCaptureStreamFence", "worker_event", 1),
    ),
    *reviewed(
        "heterogeneous_staging",
        ("src/v2/collective/LocalTPContext.cpp", "LocalTPContext::allreduceCpuBarrier", "backend_sync_copy", 2),
        ("src/v2/collective/backends/HeterogeneousBackend.cpp", "HeterogeneousBackend::executePartialReduceScatter", "backend_device", 3),
        ("src/v2/collective/backends/HeterogeneousBackend.cpp", "HeterogeneousBackend::executePhase1_IntraDomainReduce", "backend_device", 2),
        ("src/v2/collective/backends/HeterogeneousBackend.cpp", "HeterogeneousBackend::synchronize", "backend_device", 3),
        ("src/v2/collective/backends/HostBackendCUDA.cu", "cudaCopyFromHost", "raw_stream", 1),
        ("src/v2/collective/backends/HostBackendCUDA.cu", "cudaCopyToHost", "raw_stream", 1),
        ("src/v2/collective/backends/HostBackendROCm.cpp", "hipCopyFromHost", "raw_stream", 1),
        ("src/v2/collective/backends/HostBackendROCm.cpp", "hipCopyToHost", "raw_stream", 1),
        ("src/v2/execution/local_execution/coherence/CrossDomainTransfer.cpp", "CrossDomainTransfer::transferCpuToGpuImpl", "backend_sync_copy", 1),
        ("src/v2/execution/local_execution/coherence/CrossDomainTransfer.cpp", "CrossDomainTransfer::transferGpuToCpuImpl", "backend_sync_copy", 1),
        ("src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp", "RankOrchestrator::buildRankStochasticDistributionFromLocalTP", "backend_sync_compute", 1),
        ("src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp", "RankOrchestrator::buildRankStochasticTargetDistributionsFromLocalTPRows", "backend_sync_compute", 1),
        ("src/v2/transfer/TransferEngine.cpp", "TransferEngine::copyActivation", "backend_sync_copy", 3),
        ("src/v2/transfer/TransferEngine.cpp", "TransferEngine::executeDeviceToHost", "backend_sync_copy", 1),
        ("src/v2/transfer/TransferEngine.cpp", "TransferEngine::executeHostStaged", "backend_sync_copy", 2),
        ("src/v2/transfer/TransferEngine.cpp", "TransferEngine::executeHostToDevice", "backend_sync_copy", 1),
        ("src/v2/utils/MPIStager.cpp", "MPIStager::deviceToHost", "backend_sync_copy", 1),
        ("src/v2/utils/MPIStager.cpp", "MPIStager::hostToDevice", "backend_sync_copy", 1),
    ),
    *reviewed(
        "host_archive",
        ("src/v2/execution/prefix_cache/PrefixStorageBackend.cpp", "PrefixPayloadReadiness::waitOnHost", "backend_event", 1),
        ("src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h", "exportHybridPrefixState", "worker_stream", 1),
        ("src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h", "importHybridPrefixState", "worker_stream", 1),
        ("src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu", "CUDARingKVCache<Precision>::exportLogicalBlock", "raw_stream", 1),
        ("src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.cpp", "CUDARingKVCacheBase::observeDeviceSequenceState", "raw_device", 1),
        ("src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu", "CUDARingKVCacheTQ::exportLogicalBlock", "raw_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h", "exportHybridPrefixState", "worker_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h", "importHybridPrefixState", "worker_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp", "ROCmRingKVCache<Precision>::exportLogicalBlock", "raw_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp", "ROCmRingKVCacheBase::observeDeviceSequenceState", "raw_device", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip", "ROCmRingKVCacheTQ::exportLogicalBlock", "raw_stream", 1),
    ),
    *reviewed(
        "host_result",
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::copyDeviceSpeculativeOutcomesToHost", "worker_stream", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::finishDeviceResidentStochasticGeneration", "worker_stream", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::forwardMTPBatchAndSampleGreedy", "backend_sync_compute", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::forwardMTPBatchFromLastDraftAndSampleGreedy", "backend_sync_compute", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "sampleGreedyCandidateFromTensor", "backend_sync_compute", 2),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows", "backend_sync_compute", 2),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::sampleOnDeviceAtLogicalPosition", "backend_sync_compute", 1),
        ("src/v2/execution/local_execution/orchestrators/LogitsGatherer.cpp", "LogitsGatherer::copyLocalSpanToHost", "backend_sync_copy", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceSampler.cpp", "DeviceSampler::sample", "backend_sync_compute", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceSampler.cpp", "DeviceSampler::sampleGreedyFromLocalInfos", "backend_sync_compute", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceSampler.cpp", "DeviceSampler::sampleGreedyRowsFromLocalInfos", "backend_sync_compute", 1),
        ("src/v2/transfer/TransferEngine.cpp", "TransferEngine::waitForEventWithProxy", "backend_event", 1),
    ),
    *reviewed(
        "lifecycle",
        ("src/v2/execution/local_execution/engine/ForwardGraphTypes.h", "reset", "worker_stream", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::retirePublishedDeviceWorkBeforeArenaRelease", "backend_event", 1),
        ("src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp", "DeviceGraphOrchestrator::retirePendingPrefixPayloadUses", "backend_event", 1),
        ("src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp", "RankOrchestrator::synchronizeDevices", "backend_device", 2),
        ("src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp", "RankOrchestrator::synchronizeDevices", "rank_device", 1),
        ("src/v2/execution/moe/DeviceMoETransferSlotDirectory.cpp", "DeviceMoETransferSlotDirectory::create", "worker_stream", 1),
        ("src/v2/execution/moe/MoEExpertWeightService.cpp", "finish", "worker_stream", 1),
        ("src/v2/execution/moe/MoEExpertWeightService.cpp", "~ScopedGpuDirectTransferStream", "worker_stream", 2),
        ("src/v2/execution/moe/MoERuntimeTable.cpp", "synchronizeMirror", "worker_stream", 1),
        ("src/v2/execution/moe/MoERuntimeTable.cpp", "copyHostToMirror", "backend_sync_copy", 1),
        ("src/v2/execution/runner/OrchestrationRunner.cpp", "synchronizeRunnerDevicesBeforeRelease", "backend_device", 1),
        ("src/v2/execution/runner/OrchestrationRunner.cpp", "synchronizeRunnerDevicesBeforeRelease", "rank_device", 1),
        ("src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel_CUTLASS.cu", "cudaQuantGemm_uploadRawBytes", "backend_sync_copy", 1),
        ("src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvShardImpl.cu.inc", "cudaRowMajorWeights_create", "raw_stream", 1),
        ("src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu", "CUDARingKVCache<Precision>::initializeBatchedEntryPointerTables", "backend_sync_copy", 2),
        ("src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu", "CUDARingKVCacheTQ::CUDARingKVCacheTQ", "raw_stream", 2),
        ("src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu", "CUDARingKVCacheTQ::publishBatchedEntryTables", "backend_sync_copy", 2),
        ("src/v2/kernels/cuda/kvcache/CUDATurboQuantKernels.cu", "cuda_tq_upload_codebooks", "raw_stream", 1),
        ("src/v2/kernels/cuda/kvcache/CUDATurboQuantKernels.cu", "cuda_tq_upload_rope_freqs", "raw_stream", 1),
        ("src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp", "ROCmQuantisedGemmKernel::ensureWeightsConverted", "raw_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp", "ROCmRingKVCache<Precision>::allocate_pool", "raw_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp", "ROCmRingKVCache<Precision>::initializeBatchedEntryPointerTables", "backend_sync_copy", 2),
        ("src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip", "ROCmRingKVCacheTQ::ROCmRingKVCacheTQ", "raw_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip", "ROCmRingKVCacheTQ::publishBatchedEntryTables", "backend_sync_copy", 2),
        ("src/v2/kernels/rocm/kvcache/ROCmTurboQuantKernels.hip", "hip_tq_upload_codebooks", "raw_stream", 1),
        ("src/v2/kernels/rocm/kvcache/ROCmTurboQuantKernels.hip", "hip_tq_upload_rope_freqs", "raw_stream", 1),
        ("src/v2/kernels/rocm/gemm/ROCmWeightPacker.cpp", "MoEBatchPackedWeightsROCm::uploadToDevice", "backend_sync_copy", 1),
        ("src/v2/kernels/KernelFactory.cpp", "KernelFactory::prepareEmbeddingHandleLocal", "backend_sync_copy", 1),
        ("src/v2/loaders/gpu_pipeline/DeviceLoadPipeline.cpp", "workerLoop", "backend_event", 1),
        ("src/v2/loaders/gpu_pipeline/DeviceLoadPipeline.cpp", "DeviceLoadPipeline::processJobs", "worker_stream", 3),
        ("src/v2/loaders/gpu_pipeline/WeightTranslator.h", "packGpuWeightsForTransfer", "backend_sync_copy", 4),
        ("src/v2/loaders/gpu_pipeline/WeightTranslator.h", "uploadGpuPackedWeights", "backend_sync_copy", 4),
        ("src/v2/loaders/gpu_pipeline/PinnedRingBuffer.cpp", "PinnedRingBuffer::release", "backend_device", 1),
        ("src/v2/loaders/gpu_pipeline/WeightVRAMPool.cpp", "WeightVRAMPool::releaseStaging", "backend_device", 1),
        ("src/v2/transfer/TransferEngine.cpp", "TransferEngine::downloadFull", "backend_sync_copy", 1),
        ("src/v2/transfer/TransferEngine.cpp", "TransferEngine::uploadFull", "backend_sync_copy", 1),
    ),
)


@dataclasses.dataclass(frozen=True)
class Callsite:
    """One detected blocking synchronization call."""

    key: CallerKey
    line: int


def strip_comments_and_literals(source: str) -> str:
    """Replace comments and literals with spaces while preserving newlines."""

    out = list(source)
    index = 0
    size = len(source)
    while index < size:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end < 0:
                end = size
            for cursor in range(index, end):
                out[cursor] = " "
            index = end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = size if end < 0 else end + 2
            for cursor in range(index, end):
                if out[cursor] != "\n":
                    out[cursor] = " "
            index = end
            continue
        if source[index] in {'"', "'"}:
            quote = source[index]
            out[index] = " "
            index += 1
            while index < size:
                if source[index] == "\\":
                    out[index] = " "
                    if index + 1 < size and source[index + 1] != "\n":
                        out[index + 1] = " "
                    index += 2
                    continue
                if source[index] == quote:
                    out[index] = " "
                    index += 1
                    break
                if source[index] != "\n":
                    out[index] = " "
                index += 1
            continue
        index += 1
    return "".join(out)


def matching_braces(source: str) -> dict[int, int]:
    """Return opening-to-closing brace positions for lexically valid scopes."""

    stack: list[int] = []
    matches: dict[int, int] = {}
    for index, char in enumerate(source):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            matches[stack.pop()] = index
    return matches


FUNCTION_PATTERN = re.compile(
    r"""
    (?P<name>
        (?:[A-Za-z_~][A-Za-z0-9_:~<>]*::)*
        (?:operator\s*[^\s(]+|[A-Za-z_~][A-Za-z0-9_~<>]*)
    )
    \s*\(
        [^;{}]*
    \)
    \s*
    (?:
        const\s*
        |noexcept(?:\s*\([^)]*\))?\s*
        |override\s*
        |final\s*
        |->[^{;]+\s*
        |requires[^{;]+\s*
    )*
    \{
    """,
    re.VERBOSE | re.MULTILINE,
)


def function_intervals(source: str) -> list[tuple[int, int, str]]:
    """Discover C++ function bodies sufficiently for stable caller policies."""

    braces = matching_braces(source)
    intervals: list[tuple[int, int, str]] = []
    for match in FUNCTION_PATTERN.finditer(source):
        name_start = match.start("name")
        preceding = source[name_start - 1] if name_start > 0 else ""
        declaration = source[match.start() : match.end()]
        if preceding in {".", ">"} or "[" in declaration:
            # Calls which accept an inline lambda can otherwise resemble a
            # function definition because the lambda contributes the final
            # `() {`. Attribute synchronization in that lambda to the real
            # enclosing method so each allowance names the production caller.
            continue
        name = re.sub(r"\s+", "", match.group("name"))
        short_name = name.rsplit("::", 1)[-1]
        if short_name in CONTROL_NAMES:
            continue
        opening = source.rfind("{", match.start(), match.end())
        closing = braces.get(opening)
        if closing is not None:
            intervals.append((opening, closing, name))
    return intervals


def enclosing_caller(
    intervals: Iterable[tuple[int, int, str]],
    position: int,
) -> str:
    """Return the innermost discovered function containing a callsite."""

    candidates = [
        interval
        for interval in intervals
        if interval[0] < position < interval[1]
    ]
    if not candidates:
        return "<global>"
    return min(candidates, key=lambda interval: interval[1] - interval[0])[2]


def scan_file(repo_root: pathlib.Path, path: pathlib.Path) -> list[Callsite]:
    """Collect blocking synchronization calls from one source file."""

    raw_source = path.read_text(encoding="utf-8", errors="replace")
    if not any(
        token in raw_source
        for token in (
            "StreamSynchronize",
            "DeviceSynchronize",
            "EventSynchronize",
            "synchronizeStream",
            "synchronizeEvent",
            "waitForEvent",
            "deviceToHost",
            "hostToDevice",
            "deviceToDevice",
            "applyLogitPenaltiesF32",
            "argmaxF32",
            "argmaxF32BatchedRows",
            "sampleTopKTopPF32",
            "topKF32",
            "->synchronize(",
            ".synchronize(",
            "->synchronizeDevices(",
            ".synchronizeDevices(",
        )
    ):
        return []
    source = strip_comments_and_literals(raw_source)
    intervals = function_intervals(source)
    line_starts = [0]
    line_starts.extend(
        index + 1 for index, char in enumerate(source) if char == "\n"
    )
    relative_path = path.relative_to(repo_root).as_posix()
    callsites: list[Callsite] = []
    for kind, pattern in SYNC_PATTERNS:
        for match in pattern.finditer(source):
            caller = enclosing_caller(intervals, match.start())
            line = bisect.bisect_right(line_starts, match.start())
            callsites.append(
                Callsite(
                    key=CallerKey(relative_path, caller, kind),
                    line=line,
                )
            )
    return callsites


def exempt_non_production_path(path: str) -> str | None:
    """Return the dedicated non-production category for a path, if any."""

    for prefix, category in NON_PRODUCTION_PATH_CATEGORIES:
        if path == prefix or path.startswith(prefix):
            return category
    return None


def source_files(repo_root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield runtime source files in deterministic order."""

    source_root = repo_root / "src" / "v2"
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            yield path


def validate(
    repo_root: pathlib.Path,
    allowances: tuple[Allowance, ...] | None = None,
) -> list[str]:
    """Validate actual caller budgets against the reviewed allowlist."""

    reviewed_allowances = ALLOWANCES if allowances is None else allowances
    callsites = [
        callsite
        for path in source_files(repo_root)
        for callsite in scan_file(repo_root, path)
        if exempt_non_production_path(callsite.key.path) is None
    ]
    actual = Counter(callsite.key for callsite in callsites)
    expected = {
        allowance.key: allowance for allowance in reviewed_allowances
    }
    failures: list[str] = []

    duplicate_keys = [
        key
        for key, count in Counter(
            allowance.key for allowance in reviewed_allowances
        ).items()
        if count != 1
    ]
    for key in duplicate_keys:
        failures.append(f"duplicate allowance: {key}")

    for key, count in sorted(actual.items()):
        allowance = expected.get(key)
        lines = sorted(
            callsite.line for callsite in callsites if callsite.key == key
        )
        if allowance is None:
            failures.append(
                f"unapproved {key.kind}: {key.path}:{','.join(map(str, lines))} "
                f"in {key.caller} (count={count})"
            )
        elif count != allowance.count:
            failures.append(
                f"sync budget changed for {key.path}::{key.caller} "
                f"[{key.kind}]: expected {allowance.count}, found {count} "
                f"at lines {lines}"
            )

    for key, allowance in sorted(expected.items()):
        if key not in actual:
            failures.append(
                f"stale allowance for {key.path}::{key.caller} "
                f"[{key.kind}] category={allowance.category}"
            )
    return failures


def parse_args() -> argparse.Namespace:
    """Parse the optional repository root used by CTest and local runs."""

    default_root = pathlib.Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=default_root,
        help="Repository root containing src/v2 (default: inferred)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the sanitizer and print actionable caller-level diagnostics."""

    args = parse_args()
    repo_root = args.repo_root.resolve()
    failures = validate(repo_root)
    if failures:
        print("GPU blocking synchronization policy violations:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("GPU blocking synchronization policy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
