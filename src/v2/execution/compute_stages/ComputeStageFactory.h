/**
 * @file ComputeStageFactory.h
 * @brief Factory for creating compute stage instances
 */

#pragma once

#include "IComputeStage.h"

// Include all stage headers to expose Params structs
#include "stages/GEMMStage.h"
#include "stages/FusedQKVGEMMStage.h"
#include "stages/FusedGateUpGEMMStage.h"
#include "stages/RMSNormStage.h"
#include "stages/RoPEStage.h"
#include "stages/ResidualAddStage.h"
#include "stages/AttentionWithKVCacheStage.h"
#include "stages/KVCacheAppendStage.h"
#include "stages/KVCacheGatherStage.h"
#include "stages/AttentionComputeStage.h"
#include "stages/EmbeddingStage.h"
#include "stages/LMHeadStage.h"
#include "stages/AllreduceStage.h"
#include "stages/AllGatherStage.h"
#include "stages/AllGatherVStage.h"
#include "stages/SendActivationsStage.h"
#include "stages/ReceiveActivationsStage.h"
#include "stages/MoEStages.h"
#include "stages/MoEFFNStage.h"
#include "stages/QKNormStage.h"
#include "stages/FusedResidualNormStage.h"
#include "stages/GDNProjectionStage.h"
#include "stages/ShortConv1dStage.h"
#include "stages/GDNRecurrenceStage.h"
#include "stages/GatedRMSNormStage.h"
#include "stages/AttentionOutputGateStage.h"
#include "stages/QGateSplitStage.h"
#include "stages/FusedAddAllreduceStage.h"

namespace llaminar2
{

    /**
     * @brief Factory for creating compute stage instances
     *
     * Provides a centralized interface for creating all compute stages.
     * The factory methods take typed Params structs for type safety.
     */
    class ComputeStageFactory
    {
    public:
        // =====================================================================
        // GEMM Stages
        // =====================================================================

        /**
         * @brief Create a GEMM stage
         *
         * The stage uses KernelFactory at execute-time to select the appropriate
         * kernel based on IDeviceContext::deviceType(). No backend selection needed
         * at construction time.
         */
        static std::unique_ptr<IComputeStage> createGEMM(
            const GEMMStage::Params &params);

        /**
         * @brief Create a fused QKV GEMM stage
         *
         * This stage quantizes input once and runs Q, K, V projections using a
         * shared Q8_1 buffer. More efficient than separate Quantize + 3x GEMM stages.
         */
        static std::unique_ptr<IComputeStage> createFusedQKVGEMM(
            const FusedQKVGEMMStage::Params &params);

        /**
         * @brief Create a fused Gate/Up GEMM stage for FFN
         *
         * This stage quantizes input once and runs gate + up projections using a
         * shared Q8_1 buffer. More efficient than separate Quantize + 2x GEMM stages.
         */
        static std::unique_ptr<IComputeStage> createFusedGateUpGEMM(
            const FusedGateUpGEMMStage::Params &params);

        // =====================================================================
        // Normalization and Position Encoding
        // =====================================================================

        /**
         * @brief Create an RMSNorm stage
         */
        static std::unique_ptr<IComputeStage> createRMSNorm(
            const RMSNormStage::Params &params);

        /**
         * @brief Create a per-head QK RMSNorm stage (Qwen3)
         */
        static std::unique_ptr<IComputeStage> createQKNorm(
            const QKNormStage::Params &params);

        /**
         * @brief Create a RoPE stage
         */
        static std::unique_ptr<IComputeStage> createRoPE(
            const RoPEStage::Params &params);

        // =====================================================================
        // FFN and Residual
        // =====================================================================

        /**
         * @brief Create a residual add stage
         */
        static std::unique_ptr<IComputeStage> createResidualAdd(
            const ResidualAddStage::Params &params);

        /**
         * @brief Create a fused residual add + RMSNorm stage (GPU optimization)
         */
        static std::unique_ptr<IComputeStage> createFusedResidualNorm(
            const FusedResidualNormStage::Params &params);

        /**
         * @brief Create a fused residual-add + TP allreduce stage
         */
        static std::unique_ptr<IComputeStage> createFusedAddAllreduce(
            const FusedAddAllreduceStage::Params &params);

        // =====================================================================
        // Attention Stages
        // =====================================================================

        /**
         * @brief Create a production attention stage with KV cache and MPI support
         *
         * This is the recommended factory for attention in production pipelines.
         * For simple testing, use createAttention() instead.
         *
         * @param params Attention parameters including KV cache and MPI config
         * @return AttentionWithKVCacheStage instance
         */
        static std::unique_ptr<IComputeStage> createAttentionWithKVCache(
            const AttentionWithKVCacheStage::Params &params);

        /**
         * @brief Create a KV cache append stage for explicit cache management
         *
         * For advanced use cases where cache operations need to be pipelined
         * separately from attention computation.
         */
        static std::unique_ptr<IComputeStage> createKVCacheAppend(
            const KVCacheAppendStage::Params &params);

        /**
         * @brief Create a KV cache gather stage for batched decode
         *
         * Gathers K/V from multiple cache slots into batched output tensors.
         * Use after KVCacheAppendStage and before AttentionComputeStage for
         * batched decode with KV cache history.
         *
         * @param params Gather parameters (cache, batch_size, output tensors)
         * @return KVCacheGatherStage instance
         */
        static std::unique_ptr<IComputeStage> createKVCacheGather(
            const KVCacheGatherStage::Params &params);

        /**
         * @brief Create a pure attention compute stage using KernelFactory
         *
         * This is the new architecture for attention that:
         * - Uses TensorBase* for type-safe tensor handling
         * - Delegates to KernelFactory::createAttention() for kernel dispatch
         * - Supports CPU and GPU backends transparently
         *
         * Use this with KVCacheAppendStage for composable DAG execution.
         * For legacy integrated cache+attention, use createAttentionWithKVCache().
         *
         * @param params Attention compute parameters (Q, K, V, output, dimensions)
         * @return AttentionComputeStage instance
         */
        static std::unique_ptr<IComputeStage> createAttentionCompute(
            const AttentionComputeStage::Params &params);

        // =====================================================================
        // MoE (Mixture of Experts) Stages
        // =====================================================================

        /**
         * @brief Create a MoE router stage for expert selection
         */
        static std::unique_ptr<IComputeStage> createMoERouter(
            const MoERouterStage::Params &params);

        /**
         * @brief Create an expert FFN stage for MoE
         */
        static std::unique_ptr<IComputeStage> createMoEExpert(
            const MoEExpertStage::Params &params);

        /**
         * @brief Create a MoE combine stage for weighted expert output combination
         */
        static std::unique_ptr<IComputeStage> createMoECombine(
            const MoECombineStage::Params &params);

        /**
         * @brief Create a unified MoE FFN stage (router + expert SwiGLU + combine)
         */
        static std::unique_ptr<IComputeStage> createMoEFFN(
            const MoEFFNStage::Params &params);

        /**
         * @brief Create a shared expert FFN stage (always-active dense SwiGLU)
         */
        static std::unique_ptr<IComputeStage> createSharedExpertFFN(
            const SharedExpertFFNStage::Params &params);

        /**
         * @brief Create a shared expert sigmoid gate stage
         */
        static std::unique_ptr<IComputeStage> createSharedExpertGate(
            const SharedExpertGateStage::Params &params);

        // =====================================================================
        // GDN (Gated Delta Net) Stages
        // =====================================================================

        /**
         * @brief Create a GDN 4-way projection stage (QKV + Z + alpha + beta)
         */
        static std::unique_ptr<IComputeStage> createGDNProjection(
            const GDNProjectionStage::Params &params);

        /**
         * @brief Create a short 1D causal convolution stage for GDN
         */
        static std::unique_ptr<IComputeStage> createShortConv1d(
            const ShortConv1dStage::Params &params);

        /**
         * @brief Create a GDN delta-rule linear attention recurrence stage
         */
        static std::unique_ptr<IComputeStage> createGDNRecurrence(
            const GDNRecurrenceStage::Params &params);

        /**
         * @brief Create a gated RMSNorm stage: RMSNorm(x) * SiLU(gate)
         */
        static std::unique_ptr<IComputeStage> createGatedRMSNorm(
            const GatedRMSNormStage::Params &params);

        /**
         * @brief Create an attention output gate stage: sigmoid(gate) * input
         */
        static std::unique_ptr<IComputeStage> createAttentionOutputGate(
            const AttentionOutputGateStage::Params &params);

        /**
         * @brief Create a Q/gate split stage for FA layers with interleaved Q+gate projection
         */
        static std::unique_ptr<IComputeStage> createQGateSplit(
            const QGateSplitStage::Params &params);

        // =====================================================================
        // MPI Communication Stages
        // =====================================================================

        /**
         * @brief Create an Allreduce stage for MPI collective sum
         *
         * Used after row-parallel GEMM to combine partial results across ranks.
         */
        static std::unique_ptr<IComputeStage> createAllreduce(
            const AllreduceStage::Params &params);

        /**
         * @brief Create an AllGather stage for MPI collective gather
         *
         * Used after column-parallel GEMM (e.g., LM head) to collect distributed
         * output slices into a full tensor on all ranks. Each rank contributes
         * its local slice, and all ranks receive the complete result.
         *
         * Input: local_input [seq_len, dim_local] - this rank's portion
         * Output: full_output [seq_len, dim_full] - complete tensor on ALL ranks
         */
        static std::unique_ptr<IComputeStage> createAllGather(
            const AllGatherStage::Params &params);

        /**
         * @brief Create a variable-sized AllGather stage for heterogeneous tensor parallelism
         *
         * Unlike regular AllGather which assumes equal send counts, AllGatherV supports
         * variable send counts per rank. This is needed when devices have different
         * head counts (e.g., 20 heads on NVIDIA vs 8 heads on AMD).
         *
         * Input: local_input [seq_len, local_dim] - this rank's data
         * Output: full_output [seq_len, sum(recv_counts)] - all gathered data
         *
         * @param params AllGatherV parameters including recv_counts and displacements per rank
         * @return AllGatherVStage instance
         */
        static std::unique_ptr<IComputeStage> createAllGatherV(
            const AllGatherVStage::Params &params);

        /**
         * @brief Create a SendActivations stage for pipeline parallelism
         *
         * Sends activations to the next pipeline stage (MPI rank).
         * Supports both synchronous and asynchronous modes.
         *
         * @param params Send parameters including buffer, destination rank, tag
         * @return SendActivationsStage instance
         */
        static std::unique_ptr<IComputeStage> createSendActivations(
            const SendActivationsStage::Params &params);

        /**
         * @brief Create a ReceiveActivations stage for pipeline parallelism
         *
         * Receives activations from the previous pipeline stage (MPI rank).
         * Supports both synchronous and asynchronous modes.
         *
         * @param params Receive parameters including buffer, source rank, tag
         * @return ReceiveActivationsStage instance
         */
        static std::unique_ptr<IComputeStage> createReceiveActivations(
            const ReceiveActivationsStage::Params &params);

        // =====================================================================
        // Model-Level Stage Factories (for ModelExecutor)
        // =====================================================================

        /**
         * @brief Create an embedding lookup stage
         *
         * Used at the start of forward pass to convert token IDs to embeddings.
         *
         * @param params Embedding parameters including token IDs and output tensor
         * @return EmbeddingStage instance
         */
        static std::unique_ptr<IComputeStage> createEmbedding(
            const EmbeddingStage::Params &params);

        /**
         * @brief Create an LM head projection stage
         *
         * Used at the end of forward pass to project hidden states to logits.
         *
         * @param params LM head parameters including hidden states and output logits
         * @return LMHeadStage instance
         */
        static std::unique_ptr<IComputeStage> createLMHead(
            const LMHeadStage::Params &params);
    };

} // namespace llaminar2
