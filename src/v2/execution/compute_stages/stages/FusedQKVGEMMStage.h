/**
 * @file FusedQKVGEMMStage.h
 * @brief Shared-quantization attention projection stage for Q/K/V or K/V.
 *
 * Attention and MTP cache-publication graphs consume the same prepared query,
 * key, and value weights but do not always need the same projections. Normal
 * attention requires Q/K/V. Shifted MTP prefill only publishes K/V cache bytes
 * and must not spend bandwidth, workspace, or launches producing an unused
 * query and query gate. This file makes that distinction a typed stage policy
 * while retaining one backend-neutral shared-input quantization transaction.
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../../loaders/WeightPlan.h"
#include "../../../tensors/TensorKernels.h"

#include <memory>
#include <optional>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Exact set of attention projections produced by a fused stage.
     *
     * This is deliberately not a bit mask. The graph has two supported,
     * semantically meaningful transactions and callers must select one of them
     * explicitly. Arbitrary nullable combinations would make weight, workspace,
     * and buffer ownership contracts ambiguous.
     */
    enum class AttentionProjectionSet
    {
        QueryKeyValue, ///< Produce Q, K, and V for an attention calculation.
        KeyValueOnly,  ///< Produce only K and V for cache publication.
    };

    // Forward declarations for cached kernel pointers
    class FP32Tensor;
    class PreparedWeightStore;

    /**
     * @brief Shared-quantization Q/K/V or K/V projection stage.
     *
     * Efficiently computes multiple linear projections (Q, K, V) from a shared
     * input. Uses individual ITensorGemm kernels with multiply_fused() for shared
     * input quantization, aligned with the FusedGateUpGEMM pattern.
     *
     * Implements IWorkspaceConsumerStage to delegate workspace requirements to the
     * underlying GEMM kernels for GPU execution.
     */
    class FusedQKVGEMMStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            /** Exact projection transaction represented by this stage. */
            AttentionProjectionSet projection_set =
                AttentionProjectionSet::QueryKeyValue;

            // Type-safe tensor pointers (required)
            const ITensor *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                      ///< Batch size * seq_len
            int k = 0;                      ///< Input dimension (d_model)

            // Q projection
            const ITensor *wq = nullptr;
            ITensor *output_q = nullptr;
            int n_q = 0;
            const TensorBase *bias_q = nullptr; ///< Optional bias tensor for tensor-aware GPU path

            // K projection
            const ITensor *wk = nullptr;
            ITensor *output_k = nullptr;
            int n_k = 0;
            const TensorBase *bias_k = nullptr; ///< Optional bias tensor for tensor-aware GPU path

            // V projection
            const ITensor *wv = nullptr;
            ITensor *output_v = nullptr;
            int n_v = 0;
            const TensorBase *bias_v = nullptr; ///< Optional bias tensor for tensor-aware GPU path

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> output_q_buffer_id;
            std::optional<BufferId> output_k_buffer_id;
            std::optional<BufferId> output_v_buffer_id;

            /**
             * @brief Require grouped rows to preserve serial-decode arithmetic.
             *
             * MTP publication compares a grouped transaction against accepting
             * the same rows one at a time. Quantized GEMM kernels may otherwise
             * select an M-dependent reduction order. This flag requires the real
             * grouped implementation for every represented M to produce bytes
             * identical to serial decode; row replay and relaxed numerical gates
             * are not production substitutes.
             */
            bool force_decode_equivalent_verifier_prefill = false;

            // =================================================================
            // Phase 7: PreparedWeightRef for direct kernel resolution
            // =================================================================
            std::optional<PreparedWeightRef> prepared_ref_q;
            std::optional<PreparedWeightRef> prepared_ref_k;
            std::optional<PreparedWeightRef> prepared_ref_v;
            PreparedWeightStore *prepared_store = nullptr;
        };

        explicit FusedQKVGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        bool validatePreparedWeights(std::string *error) const override;
        ComputeStageType type() const override
        {
            return includesQuery()
                       ? ComputeStageType::GEMM_FUSED_QKV
                       : ComputeStageType::GEMM_FUSED_KV;
        }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        // =================================================================
        // IWorkspaceConsumerStage Implementation
        // =================================================================

        /**
         * @brief Get a GEMM kernel as IWorkspaceConsumer for delegation
         *
         * Returns the transaction anchor kernel from PreparedWeightStore. Q is
         * the anchor for Q/K/V; K is the anchor for K/V-only publication.
         *
         * @return Kernel implementing IWorkspaceConsumer, or nullptr if not available
         */
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;

        /**
         * @brief Merge workspace requirements from every active projection.
         *
         * Override the default single-kernel delegation so projection-specific
         * dimensions are represented before workspace requirements are merged.
         */
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;

        /**
         * @brief Bind workspace to every active projection kernel.
         *
         * Override the default single-kernel binding to bind every active
         * projection kernel since each needs workspace for GPU execution.
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /**
         * @brief Unbind workspace from every active projection kernel.
         */
        void unbindWorkspace() override;
        void resetSessionState() override;
        void resetSessionStatePreservingCapturedReplay() override;
        void resetSessionStatePreservingLazyInitialization() override;

        /**
         * @brief Provision fused-projection backend resources before graph capture.
         *
         * Exact-shape prefill can legitimately avoid the concurrent projection
         * route even when decode uses it. This capture-only preparation makes
         * decode independent of whether an earlier warmup happened to create
         * the shared stream/event pool.
         */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return GraphLaunchPreparationPolicy::CaptureOnly;
        }

    private:
        Params params_;

        // === Cached kernel pointers (avoid KernelFactory mutex per execute) ===
        ITensorGemm *cached_gemm_q_ = nullptr;
        ITensorGemm *cached_gemm_k_ = nullptr;
        ITensorGemm *cached_gemm_v_ = nullptr;
        bool cache_resolved_individual_ = false;

        /**
         * Stable descriptors consumed during capture/eager execution. Building
         * this vector while resolving prepared kernels keeps allocation and
         * container growth out of the graph execution path.
         */
        std::vector<ITensorGemm::TensorProjectionDesc> cached_projections_;

        bool includesQuery() const noexcept;
        size_t projectionCount() const noexcept;
        ITensorGemm *anchorKernel() const noexcept;
        bool resolveIndividualKernels(const char *caller);
        bool executeDecodeEquivalentVerifierPrefill(
            TensorBase *input_base);
        void rebuildProjectionDescriptors();
        void clearCachedGemmStreams();

    };

} // namespace llaminar2
