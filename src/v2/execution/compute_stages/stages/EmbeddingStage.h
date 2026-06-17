/**
 * @file EmbeddingStage.h
 * @brief Embedding lookup stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../memory/BufferId.h"
#include "../../../loaders/WeightPlan.h"

#include <optional>

namespace llaminar2
{
    class PreparedWeightStore;

    /**
     * @brief Embedding lookup stage
     *
     * Performs token → embedding lookup for transformer input processing.
     * Supports output to various precision formats (FP32, BF16, Q8_1).
     *
     * The embedding table is typically FP32 (stored in model weights).
     * Output format is determined by the output tensor's native_type().
     *
     * Implements IWorkspaceConsumerStage to delegate workspace requirements
     * to the underlying embedding kernel (required for ROCm kernels).
     */
    class EmbeddingStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input/output tensors
            const ITensor *embed_table = nullptr;
            const int *token_ids = nullptr;
            const void *token_ids_device = nullptr; ///< Optional GPU INT32 token IDs [num_tokens]
            ITensor *output = nullptr;

            // Dimensions
            int num_tokens = 0;
            int d_model = 0;
            int vocab_size = 0;
            int vocab_offset = 0;
            int local_vocab_size = 0;

            // Batched input (alternative to token_ids)
            const std::vector<std::vector<int>> *token_batches = nullptr;
            int padded_seq_len = 0;

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> output_buffer_id;

            // Phase 7/8: Prepared embedding data owned by PreparedWeightStore.
            std::optional<PreparedWeightRef> prepared_ref;
            PreparedWeightStore *prepared_store = nullptr;
        };

        explicit EmbeddingStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::EMBEDDING; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool validatePreparedWeights(std::string *error) const override;
        bool isGraphCapturable() const override { return true; }
        bool hasDynamicParams() const override { return true; }
        bool supportsDeviceResidentDynamicPositionReplay() const override
        {
            return true;
        }
        /// In vocab-parallel TP, each device only embeds tokens in its shard;
        /// tokens outside the shard produce all-zero output (summed via AllReduce).
        /// This applies to both MPI-based TP (world_size > 1) and LOCAL TP
        /// (single rank, multi-device) where the embedding table is column-sharded.
        bool allowsZeroOutput() const override
        {
            // Sharded embedding: table rows < full vocab_size
            if (params_.embed_table && params_.vocab_size > 0 &&
                static_cast<int>(params_.embed_table->rows()) < params_.vocab_size)
                return true;
            // Legacy MPI TP check
            return params_.mpi_ctx && params_.mpi_ctx->world_size() > 1;
        }
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            (void)pos_offset;
            (void)seq_len;
            if (cached_kernel_ && params_.num_tokens > 0)
            {
                cached_kernel_->setGPUStream(gpuStream());
                if (params_.token_ids_device)
                {
                    cached_kernel_->setDynamicDeviceTokenIds(
                        params_.token_ids_device,
                        params_.num_tokens);
                }
                else if (params_.token_ids)
                {
                    cached_kernel_->setDynamicTokenIds(params_.token_ids, params_.num_tokens);
                }
            }
        }

        void resetSessionState() override
        {
            IComputeStage::resetSessionState();
            if (cached_kernel_)
            {
                cached_kernel_->resetDynamicState();
                cached_kernel_->setGPUStream(nullptr);
            }
        }

        /// Check if this stage is ready for prefill graph capture.
        /// Requires: GPU device, workspace bound, kernel created.
        bool isPrefillGraphCaptureReady() const
        {
            if (!params_.device_id.is_gpu())
                return false;
            if (!cached_kernel_)
                return false;
            auto *ws = dynamic_cast<const IWorkspaceConsumer *>(cached_kernel_);
            return ws && ws->hasWorkspace();
        }

        /// Update the token_ids pointer to point at stable graph cache storage.
        /// Called when the graph cache takes ownership of the token buffer.
        void setStableTokenPointer(const int *stable_token_ids)
        {
            params_.token_ids = stable_token_ids;
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        // IWorkspaceConsumerStage Implementation
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;

    private:
        Params params_;
        ITensorEmbedding *cached_kernel_ = nullptr;
        int cached_kernel_tensor_type_ = -1;

        bool executeQ16_1Output();
        static void zero_output_row(ITensor *output, int row_idx, int d_model);

        // Get or create the embedding kernel (caches for workspace binding)
        ITensorEmbedding *getOrCreateKernel();
    };

} // namespace llaminar2
