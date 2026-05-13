/**
 * @file ComputeStageFactory.cpp
 * @brief Implementation of ComputeStageFactory
 */

#include "ComputeStageFactory.h"
#include "stages/AllGatherStage.h"
#include "stages/AllGatherVStage.h"
#include "stages/AllreduceStage.h"
#include "stages/AttentionComputeStage.h"
#include "stages/FusedGateUpGEMMStage.h"
#include "stages/FusedQKVGEMMStage.h"
#include "stages/GEMMStage.h"
#include "stages/KVCacheAppendStage.h"
#include "stages/KVCacheGatherStage.h"
#include "stages/LMHeadStage.h"
#include "stages/MoEExpertDispatchStage.h"
#include "stages/MoEExpertOverlayCPUFallbackStage.h"
#include "stages/MoEExpertOverlayLocalTPStage.h"
#include "stages/MoEOverlayDomainRuntimeStage.h"
#include "stages/MoEExpertParallelReduceStage.h"
#include "stages/MoERoutingStage.h"
#include "stages/ReceiveActivationsStage.h"
#include "stages/ResidualAddStage.h"
#include "stages/RMSNormStage.h"
#include "stages/QKNormStage.h"
#include "stages/RoPEStage.h"
#include "stages/SendActivationsStage.h"
#include "stages/GDNProjectionStage.h"
#include "stages/ShortConv1dStage.h"
#include "stages/GDNRecurrenceStage.h"
#include "stages/GatedRMSNormStage.h"
#include "stages/AttentionOutputGateStage.h"
#include "stages/QGateSplitStage.h"

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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedAddAllreduce(
        const FusedAddAllreduceStage::Params &params)
    {
        return std::make_unique<FusedAddAllreduceStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpertCompute(
        const MoEExpertComputeStage::Params &params)
    {
        return std::make_unique<MoEExpertComputeStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEFFN(
        const MoEExpertComputeStage::Params &params)
    {
        return std::make_unique<MoEExpertComputeStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoERouting(
        const MoERoutingStage::Params &params)
    {
        return std::make_unique<MoERoutingStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpertDispatch(
        const MoEExpertDispatchStage::Params &params)
    {
        return std::make_unique<MoEExpertDispatchStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpertOverlayCPUFallback(
        const MoEExpertOverlayCPUFallbackStage::Params &params)
    {
        return std::make_unique<MoEExpertOverlayCPUFallbackStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpertOverlayLocalTP(
        const MoEExpertOverlayLocalTPStage::Params &params)
    {
        return std::make_unique<MoEExpertOverlayLocalTPStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEOverlayDomainRuntime(
        const MoEOverlayDomainRuntimeStage::Params &params)
    {
        return std::make_unique<MoEOverlayDomainRuntimeStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpertParallelReduce(
        const MoEExpertParallelReduceStage::Params &params)
    {
        return std::make_unique<MoEExpertParallelReduceStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createSharedExpertFFN(
        const SharedExpertFFNStage::Params &params)
    {
        return std::make_unique<SharedExpertFFNStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createSharedExpertGate(
        const SharedExpertGateStage::Params &params)
    {
        return std::make_unique<SharedExpertGateStage>(params);
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
    // GDN (Gated Delta Net) Stage Factories
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGDNProjection(
        const GDNProjectionStage::Params &params)
    {
        return std::make_unique<GDNProjectionStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createShortConv1d(
        const ShortConv1dStage::Params &params)
    {
        return std::make_unique<ShortConv1dStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGDNRecurrence(
        const GDNRecurrenceStage::Params &params)
    {
        return std::make_unique<GDNRecurrenceStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGatedRMSNorm(
        const GatedRMSNormStage::Params &params)
    {
        return std::make_unique<GatedRMSNormStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAttentionOutputGate(
        const AttentionOutputGateStage::Params &params)
    {
        return std::make_unique<AttentionOutputGateStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createQGateSplit(
        const QGateSplitStage::Params &params)
    {
        return std::make_unique<QGateSplitStage>(params);
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
