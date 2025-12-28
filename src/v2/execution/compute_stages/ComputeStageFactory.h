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
#include "stages/SwiGLUStage.h"
#include "stages/ResidualAddStage.h"
#include "stages/AttentionWithKVCacheStage.h"
#include "stages/KVCacheAppendStage.h"
#include "stages/KVCacheGatherStage.h"
#include "stages/AttentionComputeStage.h"
#include "stages/FusedAttentionWoStage.h"
#include "stages/EmbeddingStage.h"
#include "stages/LMHeadStage.h"
#include "stages/QuantizeToQ16_1Stage.h"
#include "stages/AllreduceStage.h"
#include "stages/AllGatherStage.h"
#include "stages/MoEStages.h"

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
         * @brief Create a RoPE stage
         */
        static std::unique_ptr<IComputeStage> createRoPE(
            const RoPEStage::Params &params);

        // =====================================================================
        // FFN and Residual
        // =====================================================================

        /**
         * @brief Create a SwiGLU stage
         */
        static std::unique_ptr<IComputeStage> createSwiGLU(
            const SwiGLUStage::Params &params);

        /**
         * @brief Create a residual add stage
         */
        static std::unique_ptr<IComputeStage> createResidualAdd(
            const ResidualAddStage::Params &params);

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

        /**
         * @brief Create a fused attention + Wo projection stage
         *
         * Combines attention computation with output projection (Wo) for:
         * - Better cache locality (context stays in registers)
         * - Eliminated context quantization round-trip
         * - Reduced memory bandwidth
         *
         * Replaces separate AttentionComputeStage + GEMMStage (wo_proj) in the DAG.
         * Output goes directly to attn_proj buffer.
         *
         * @param params Fused attention parameters including Q, K, V, Wo, output
         * @return FusedAttentionWoStage instance
         */
        static std::unique_ptr<IComputeStage> createFusedAttentionWo(
            const FusedAttentionWoStage::Params &params);

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

        /**
         * @brief Create a quantize stage for FP32 → Q16_1 conversion
         *
         * Used to initialize the Q16_1 residual stream in HybridQ16 mode.
         *
         * @param params Quantize parameters including input and output tensors
         * @return QuantizeToQ16_1Stage instance
         */
        static std::unique_ptr<IComputeStage> createQuantizeToQ16_1(
            const QuantizeToQ16_1Stage::Params &params);
    };

} // namespace llaminar2
