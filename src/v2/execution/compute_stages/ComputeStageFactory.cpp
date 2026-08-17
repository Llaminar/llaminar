/**
 * @file ComputeStageFactory.cpp
 * @brief Implementation of ComputeStageFactory
 */

#include "ComputeStageFactory.h"

#include <stdexcept>
#include <utility>
#include "stages/AllGatherStage.h"
#include "stages/AllGatherVStage.h"
#include "stages/AllreduceStage.h"
#include "stages/TPAllreduceStage.h"
#include "stages/AttentionComputeStage.h"
#include "stages/FusedGateUpGEMMStage.h"
#include "stages/FusedQKVGEMMStage.h"
#include "stages/GEMMStage.h"
#include "stages/KVCacheAppendStage.h"
#include "stages/KVCacheGatherStage.h"
#include "stages/HiddenStateRowSelectStage.h"
#include "stages/HiddenStateRowsSelectStage.h"
#include "stages/LMHeadStage.h"
#include "stages/MoEOverlayTicketPublishStage.h"
#include "stages/MoEOverlayTicketConsumeStage.h"
#include "stages/MoEExpertDispatchStage.h"
#include "stages/MoELocalExpertStage.h"
#include "stages/MoESparseDispatchStage.h"
#include "stages/MoESparseReturnReduceStage.h"
#include "stages/MoEDeviceRebalanceStage.h"
#include "stages/MoEGPUCurrentBatchLLEPStage.h"
#include "stages/MoEDeviceDecodeCommitBoundaryStage.h"
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createFusedKVGEMM(
        FusedQKVGEMMStage::Params params)
    {
        if (params.wq || params.output_q || params.bias_q || params.n_q != 0 ||
            params.prepared_ref_q.has_value() ||
            params.output_q_buffer_id.has_value())
        {
            throw std::invalid_argument(
                "createFusedKVGEMM rejects query state; K/V-only graphs must not bind or publish Q");
        }
        params.projection_set = AttentionProjectionSet::KeyValueOnly;
        return std::make_unique<FusedQKVGEMMStage>(std::move(params));
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createKeyOnlyRoPE(
        RoPEStage::Params params)
    {
        if (!params.K || params.Q || params.Q_out || params.K_out ||
            params.q_buffer_id || params.q_out_buffer_id ||
            params.k_out_buffer_id || params.skip_k || params.n_heads != 0 ||
            params.n_kv_heads <= 0)
        {
            throw std::invalid_argument(
                "createKeyOnlyRoPE requires one in-place K operand, a positive "
                "KV-head count, and no query/output/skip state");
        }
        params.operand_set = RoPEOperandSet::KeyOnly;
        return std::make_unique<RoPEStage>(std::move(params));
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

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayTicketPublish(
        const MoEOverlayTicketPublishStage::Params &params)
    {
        return std::make_unique<MoEOverlayTicketPublishStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayTicketConsume(
        const MoEOverlayTicketConsumeStage::Params &params)
    {
        return std::make_unique<MoEOverlayTicketConsumeStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayActivationDispatchPack(
        const MoEOverlayActivationDispatchPackStage::Params &params)
    {
        return std::make_unique<MoEOverlayActivationDispatchPackStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayActivationDispatchPackBatch(
        const MoEOverlayActivationDispatchPackBatchStage::Params &params)
    {
        return std::make_unique<
            MoEOverlayActivationDispatchPackBatchStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayActivationDispatchConsume(
        const MoEOverlayActivationDispatchConsumeStage::Params &params)
    {
        return std::make_unique<MoEOverlayActivationDispatchConsumeStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayActivationReturnPack(
        const MoEOverlayActivationReturnPackStage::Params &params)
    {
        return std::make_unique<MoEOverlayActivationReturnPackStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayActivationReturnConsume(
        const MoEOverlayActivationReturnConsumeStage::Params &params)
    {
        return std::make_unique<MoEOverlayActivationReturnConsumeStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEOverlayActivationReturnConsumeBatch(
        const MoEOverlayActivationReturnConsumeBatchStage::Params &params)
    {
        return std::make_unique<
            MoEOverlayActivationReturnConsumeBatchStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEExpertDispatch(
        const MoEExpertDispatchStage::Params &params)
    {
        return std::make_unique<MoEExpertDispatchStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoESparseDispatch(
        const MoESparseDispatchStage::Params &params)
    {
        return std::make_unique<MoESparseDispatchStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoERankBatchDispatch(
        const MoERankBatchDispatchStage::Params &params)
    {
        return std::make_unique<MoERankBatchDispatchStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoELocalExpert(
        const MoELocalExpertStage::Params &params)
    {
        return std::make_unique<MoELocalExpertStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoELocalExpertCompletion(
        const MoELocalExpertCompletionStage::Params &params)
    {
        return std::make_unique<MoELocalExpertCompletionStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoESparseReturnReduce(
        const MoESparseReturnReduceStage::Params &params)
    {
        return std::make_unique<MoESparseReturnReduceStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoERankBatchReturnReduce(
        const MoERankBatchReturnReduceStage::Params &params)
    {
        return std::make_unique<MoERankBatchReturnReduceStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoEDeviceRebalance(
        const MoEDeviceRebalanceStage::Params &params)
    {
        return std::make_unique<MoEDeviceRebalanceStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEGPUCurrentBatchLLEP(
        const MoEGPUCurrentBatchLLEPStage::Params &params)
    {
        return std::make_unique<MoEGPUCurrentBatchLLEPStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoEDeviceDecodeCommitBoundary(
        const MoEDeviceDecodeCommitBoundaryStage::Params &params)
    {
        return std::make_unique<MoEDeviceDecodeCommitBoundaryStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoECPUCurrentBatchLLEP(
        const MoECPUCurrentBatchLLEPStage::Params &params)
    {
        return std::make_unique<MoECPUCurrentBatchLLEPStage>(params);
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMoECanonicalRouteReduce(
        const MoECanonicalRouteReduceStage::Params &params)
    {
        return std::make_unique<MoECanonicalRouteReduceStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoECanonicalRouteGather(
        const MoECanonicalRouteGatherStage::Params &params)
    {
        return std::make_unique<MoECanonicalRouteGatherStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoECanonicalOutputBroadcast(
        const MoECanonicalOutputBroadcastStage::Params &params)
    {
        return std::make_unique<MoECanonicalOutputBroadcastStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoESharedExpertRankBankPublish(
        const MoESharedExpertRankBankPublishStage::Params &params)
    {
        return std::make_unique<MoESharedExpertRankBankPublishStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMoECanonicalPublicationFinalize(
        const MoECanonicalPublicationFinalizeStage::Params &params)
    {
        return std::make_unique<MoECanonicalPublicationFinalizeStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createAllreduce(
        const AllreduceStage::Params &params)
    {
        // Allreduce is backend-agnostic (uses MPI directly)
        return std::make_unique<AllreduceStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createTPLocalRootedCollective(
        const TPLocalRootedCollectiveStage::Params &params)
    {
        return std::make_unique<TPLocalRootedCollectiveStage>(params);
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createTPKVCacheStateAllGather(
        const TPKVCacheStateAllGatherStage::Params &params)
    {
        return std::make_unique<TPKVCacheStateAllGatherStage>(params);
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGDNLiveStateLocalize(
        const GDNLiveStateLocalizeStage::Params &params)
    {
        return std::make_unique<GDNLiveStateLocalizeStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createGDNLiveStateAllGather(
        const GDNLiveStateAllGatherStage::Params &params)
    {
        return std::make_unique<GDNLiveStateAllGatherStage>(params);
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

    std::unique_ptr<IComputeStage> ComputeStageFactory::createMTPConcat(
        const MTPConcatStage::Params &params)
    {
        return std::make_unique<MTPConcatStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMTPVerifierOutcome(
        const MTPVerifierOutcomeStage::Params &params)
    {
        return std::make_unique<MTPVerifierOutcomeStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createMTPVerifierPreparation(
        const MTPVerifierPreparationStage::Params &params)
    {
        return std::make_unique<MTPVerifierPreparationStage>(params);
    }

    std::unique_ptr<IComputeStage>
    ComputeStageFactory::createPrefillChunkMaterialization(
        const PrefillChunkMaterializationStage::Params &params)
    {
        return std::make_unique<PrefillChunkMaterializationStage>(params);
    }

    // =============================================================================
    // Model-Level Stage Factories
    // =============================================================================

    std::unique_ptr<IComputeStage> ComputeStageFactory::createEmbedding(
        const EmbeddingStage::Params &params)
    {
        return std::make_unique<EmbeddingStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createHiddenStateRowSelect(
        const HiddenStateRowSelectStage::Params &params)
    {
        return std::make_unique<HiddenStateRowSelectStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createHiddenStateRowsSelect(
        const HiddenStateRowsSelectStage::Params &params)
    {
        return std::make_unique<HiddenStateRowsSelectStage>(params);
    }

    std::unique_ptr<IComputeStage> ComputeStageFactory::createLMHead(
        const LMHeadStage::Params &params)
    {
        return std::make_unique<LMHeadStage>(params);
    }

} // namespace llaminar2
