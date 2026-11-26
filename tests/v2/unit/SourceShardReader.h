/**
 * @file SourceShardReader.h
 * @brief Shared test helper that reads a primary source file together with its split TU shards.
 *
 * Large translation units under src/v2/ are sharded into <= 6,000-line sibling
 * files for parallel build throughput while preserving a single logical module.
 * Source-contract unit tests use this helper so both positive invariants and
 * forbidden-pattern scans apply uniformly across the entire split family.
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2::test_support
{
    inline std::vector<std::filesystem::path> sourceShardFamilyPaths(
        const std::filesystem::path &primary_path)
    {
        static const std::unordered_map<std::string, std::vector<std::string>> kOrderedFamilies = {
            {"DeviceGraphOrchestrator.cpp", {
                "DeviceGraphOrchestratorInternal.h",
                "DeviceGraphOrchestrator.cpp",
                "DeviceGraphOrchestratorTopology.cpp",
                "DeviceGraphOrchestratorRebalance.cpp",
                "DeviceGraphOrchestratorForward.cpp",
                "DeviceGraphOrchestratorMTPGraph.cpp",
                "DeviceGraphOrchestratorMTPShifted.cpp",
                "DeviceGraphOrchestratorMTPCommit.cpp",
                "DeviceGraphOrchestratorLiveState.cpp",
                "DeviceGraphOrchestratorPrefixCache.cpp",
                "DeviceGraphOrchestratorPrefixRestore.cpp",
                "DeviceGraphOrchestratorSampling.cpp",
                "DeviceGraphOrchestratorDiagnostics.cpp",
                "DeviceGraphOrchestratorOverlay.cpp",
                "DeviceGraphOrchestratorRuntime.cpp",
            }},
            {"RankOrchestrator.cpp", {
                "RankOrchestratorInternal.h",
                "RankOrchestrator.cpp",
                "RankOrchestratorForward.cpp",
                "RankOrchestratorMTP.cpp",
                "RankOrchestratorOverlay.cpp",
            }},
            {"OrchestrationRunner.cpp", {
                "OrchestrationRunnerInternal.h",
                "OrchestrationRunner.cpp",
                "OrchestrationRunnerPrefillAndBatch.cpp",
                "OrchestrationRunnerMTPHelpers.cpp",
                "OrchestrationRunnerMTP.cpp",
                "OrchestrationRunnerOverlay.cpp",
                "OrchestrationRunnerRuntime.cpp",
            }},
            {"Qwen35MoEGraph.cpp", {
                "Qwen35MoEGraphInternal.h",
                "Qwen35MoEGraph.cpp",
                "Qwen35MoEGraphFFN.cpp",
                "Qwen35MoEGraphOverlay.cpp",
            }},
            {"MoEExpertComputeStage.cpp", {
                "MoEExpertComputeStage.cpp",
                "MoEExpertComputeStagePrefill.cpp",
                "MoEExpertComputeStageOverlay.cpp",
                "MoEExpertComputeStageInternal.h",
            }},
            {"ROCmMoEKernel.cpp", {
                "ROCmMoEKernelInternal.h",
                "ROCmMoEKernel.cpp",
                "ROCmMoEKernelRouting.cpp",
                "ROCmMoEKernelExecution.cpp",
            }},
            {"CUDAMoEKernel.cpp", {
                "CUDAMoEKernelInternal.h",
                "CUDAMoEKernel.cpp",
                "CUDAMoEKernelRouting.cpp",
                "CUDAMoEKernelExecution.cpp",
            }},
            {"WeightManager.cpp", {
                "WeightManagerInternal.h",
                "WeightManager.cpp",
                "WeightManagerPreloadAndShard.cpp",
            }},
            {"ForwardExecutionEngine.cpp", {
                "ForwardExecutionEngineInternal.h",
                "ForwardExecutionEngineDecodeCache.cpp",
                "ForwardExecutionEngine.cpp",
            }},
            {"DeviceGraphCaptureController.cpp", {
                "DeviceGraphCaptureControllerInternal.h",
                "DeviceGraphCaptureController.cpp",
                "DeviceGraphCaptureControllerPhases.cpp",
            }},
            {"MoEOverlayPhysicalResidencyFabric.cpp", {
                "MoEOverlayPhysicalResidencyFabricInternal.h",
                "MoEOverlayPhysicalResidencyFabric.cpp",
                "MoEOverlayPhysicalResidencyFabricOperations.cpp",
            }},
            {"MoEOverlayResidencyAuthority.cpp", {
                "MoEOverlayResidencyAuthorityInternal.h",
                "MoEOverlayResidencyAuthority.cpp",
                "MoEOverlayResidencyAuthorityPlanning.cpp",
            }},
            {"MoEOverlayParticipantGraphRunner.cpp", {
                "MoEOverlayParticipantGraphRunnerInternal.h",
                "MoEOverlayParticipantGraphRunner.cpp",
                "MoEOverlayParticipantGraphRunnerExecution.cpp",
            }},
            {"ROCmQuantisedGemmKernel.cpp", {
                "ROCmQuantisedGemmKernelInternal.h",
                "ROCmQuantisedGemmKernel.cpp",
                "ROCmQuantisedGemmKernelDispatch.cpp",
            }},
            {"RoPEPrimitives.cpp", {
                "RoPEPrimitivesInternal.h",
                "RoPEPrimitives.cpp",
                "RoPEPrimitivesQ16.cpp",
            }},
            {"ROCmBackend.cpp", {
                "ROCmBackendInternal.h",
                "ROCmBackend.cpp",
                "ROCmBackendSampling.cpp",
            }},
            {"CUDABackend.cu", {
                "CUDABackendInternal.cuh",
                "CUDABackend.cu",
                "CUDABackendSampling.cu",
            }},
            {"CUDAMoEKernels.cu", {
                "CUDAMoEKernelsCommon.cuh",
                "CUDAMoEDecodeRouterBody.cuh",
                "CUDAMoERebalanceCommon.cuh",
                "CUDAMoERebalanceProjectionKernels.cu",
                "CUDAMoERebalanceApplyKernels.cu",
                "CUDAMoERouterAndGatherKernels.cu",
                "CUDAMoEKernels.cu",
                "CUDAMoENativeQuantisedDecodeHelpers.cuh",
                "CUDAMoEGroupedPrefillLaunchKernels.cu",
                "CUDAMoEGroupingKernels.cu",
                "CUDAMoERebalancePlannerKernels.cu",
            }},
            {"ROCmMoEKernels.hip", {
                "ROCmMoEKernelsCommon.hip.inc",
                "ROCmMoERouterAndGatherKernels.hip",
                "ROCmMoERebalanceCommon.hip.inc",
                "ROCmMoERebalanceProjectionKernels.hip",
                "ROCmMoERebalanceApplyKernels.hip",
                "ROCmMoEKernels.hip",
                "ROCmMoERebalancePlannerKernels.hip",
            }},
            {"CUDASamplingKernels.cu", {
                "CUDASamplingCommon.cuh",
                "CUDASamplingKernels.cu",
                "CUDASamplingSpeculativeKernels.cu",
                "CUDASamplingGenerationControlKernels.cu",
            }},
            {"ROCmSamplingKernels.hip", {
                "ROCmSamplingCommon.hip.inc",
                "ROCmSamplingKernels.hip",
                "ROCmSamplingSpeculativeKernels.hip",
                "ROCmSamplingGenerationControlKernels.hip",
            }},
            {"ROCmGemvKernel_INT8_VNNI.hip", {
                "ROCmGemvKernel_INT8_VNNI_common.hip.inc",
                "ROCmGemvKernel_INT8_VNNI.hip",
                "ROCmGemvKernel_INT8_VNNI_fused.hip",
            }},
            {"ROCmGemvKernel_native_quantised.hip", {
                "ROCmGemvKernel_native_quantised.hip",
                "ROCmMoEGroupedDecodeKernels.hip",
            }},
        };

        std::vector<std::filesystem::path> paths;
        const std::string filename = primary_path.filename().string();
        const auto it = kOrderedFamilies.find(filename);
        if (it != kOrderedFamilies.end())
        {
            const auto dir = primary_path.parent_path();
            for (const auto &member : it->second)
            {
                const auto candidate = dir / member;
                if (std::filesystem::exists(candidate))
                    paths.push_back(candidate);
            }
        }
        else
        {
            paths.push_back(primary_path);
        }
        return paths;
    }

    inline std::string readSourceFileOrFamily(const std::filesystem::path &primary_path)
    {
        std::string combined;
        for (const auto &p : sourceShardFamilyPaths(primary_path))
        {
            std::ifstream in(p);
            if (!in.is_open())
            {
                if (p == primary_path)
                    return {};
                continue;
            }
            if (!combined.empty())
                combined.push_back('\n');
            combined.append(
                std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>());
        }
        return combined;
    }
} // namespace llaminar2::test_support
