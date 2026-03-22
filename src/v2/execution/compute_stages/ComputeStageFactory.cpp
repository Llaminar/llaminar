/**
 * @file ComputeStageFactory.cpp
 * @brief Implementation of ComputeStageFactory
 */

#include "ComputeStageFactory.h"
#include "stages/AllGatherStage.h"
#include "stages/AllGatherVStage.h"
#include "stages/AllreduceStage.h"
#include "stages/AttentionComputeStage.h"
#include "stages/AttentionWithKVCacheStage.h"
#include "stages/FusedGateUpGEMMStage.h"
#include "stages/FusedQKVGEMMStage.h"
#include "stages/GEMMStage.h"
#include "stages/KVCacheAppendStage.h"
#include "stages/KVCacheGatherStage.h"
#include "stages/LMHeadStage.h"
#include "stages/MoEStages.h"
#include "stages/ReceiveActivationsStage.h"
#include "stages/ResidualAddStage.h"
#include "stages/RMSNormStage.h"\n #include "stages/QKNormStage.h"
#include "stages/RoPEStage.h"
#include "stages/SendActivationsStage.h"

namespace llaminar2
{

    // =============================================================================
    // ComputeStageFactory Implementation
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGEMM(
        const GEMMStage::Params &params)
    {
        // Unified: GEMMStage handles all backends via KernelFactory dispatch at execute-time
        return std::make_unique<GEMMStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedQKVGEMM(
        const FusedQKVGEMMStage::Params &params)
    {
        // Unified: FusedQKVGEMMStage will use KernelFactory at execute-time
        return std::make_unique<FusedQKVGEMMStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedGateUpGEMM(
        const FusedGateUpGEMMStage::Params &params)
    {
        // Unified: FusedGateUpGEMMStage will use KernelFactory at execute-time
        return std::make_unique<FusedGateUpGEMMStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRMSNorm(
        const RMSNormStage::Params &params)
    {
        // Unified: RMSNormStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<RMSNormStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createQKNorm(
        const QKNormStage::Params &params)
    {
        return std::make_unique<QKNormStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createRoPE(
        const RoPEStage::Params &params)
    {
        // Unified: RoPEStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<RoPEStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createResidualAdd(
        const ResidualAddStage::Params &params)
    {
        // Unified: ResidualAddStage uses KernelFactory at execute-time for device dispatch
        return std::make_unique<ResidualAddStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedResidualNorm(
        const FusedResidualNormStage::Params &params)
    {
        return std::make_unique<FusedResidualNormStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoERouter(
        const MoERouterStage::Params &params)
    {
        // Unified: MoERouterStage will use KernelFactory at execute-time
        return std::make_unique<MoERouterStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpert(
        const MoEExpertStage::Params &params)
    {
        // Unified: MoEExpertStage will use KernelFactory at execute-time
        return std::make_unique<MoEExpertStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoECombine(
        const MoECombineStage::Params &params)
    {
        // Unified: MoECombineStage will use KernelFactory at execute-time
        return std::make_unique<MoECombineStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAllreduce(
        const AllreduceStage::Params &params)
    {
        // Allreduce is backend-agnostic (uses MPI directly)
        return std::make_unique<AllreduceStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAllGather(
        const AllGatherStage::Params &params)
    {
        // AllGather is backend-agnostic (uses MPI directly)
        return std::make_unique<AllGatherStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAllGatherV(
        const AllGatherVStage::Params &params)
    {
        // AllGatherV is backend-agnostic (uses MPI_Allgatherv or CollectiveContext)
        return std::make_unique<AllGatherVStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createSendActivations(
        const SendActivationsStage::Params &params)
    {
        // SendActivations is backend-agnostic (uses MPI point-to-point)
        return std::make_unique<SendActivationsStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createReceiveActivations(
        const ReceiveActivationsStage::Params &params)
    {
        // ReceiveActivations is backend-agnostic (uses MPI point-to-point)
        return std::make_unique<ReceiveActivationsStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttentionWithKVCache(
        const AttentionWithKVCacheStage::Params &params)
    {
        // Unified: AttentionWithKVCacheStage uses KernelFactory at execute-time
        return std::make_unique<AttentionWithKVCacheStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createKVCacheAppend(
        const KVCacheAppendStage::Params &params)
    {
        // KV cache append is backend-agnostic (pure memory operations)
        return std::make_unique<KVCacheAppendStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createKVCacheGather(
        const KVCacheGatherStage::Params &params)
    {
        // KV cache gather is backend-agnostic (pure memory operations)
        return std::make_unique<KVCacheGatherStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttentionCompute(
        const AttentionComputeStage::Params &params)
    {
        // Unified: AttentionComputeStage uses KernelFactory at execute-time
        return std::make_unique<AttentionComputeStage>(params);
    }

    // =============================================================================
    // Model-Level Stage Factories
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createEmbedding(
        const EmbeddingStage::Params &params)
    {
        return std::make_unique<EmbeddingStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createLMHead(
        const LMHeadStage::Params &params)
    {
        return std::make_unique<LMHeadStage>(params);
    }

} // namespace llaminar2
