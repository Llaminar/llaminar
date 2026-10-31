/**
 * @file ROCmEmbeddingKernelT.h
 * @brief ROCm implementation of ITensorEmbedding interface
 *
 * Provides device-owned embedding lookup on AMD GPUs. Quantized tables use a
 * model-owned PreparedEmbeddingWeights allocation; FP32 tables must already be
 * resident on the target device.
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "ROCmEmbeddingExecutionBinding.h"
#include <stdexcept>

namespace llaminar2
{

    /**
     * @class ROCmEmbeddingKernelT
     * @brief ROCm/HIP embedding lookup kernel implementation
     *
     * Implements ITensorEmbedding interface for AMD GPUs using HIP.
     * Supports FP32, BF16, FP16, and Q8_1 output formats.
     *
     * ## Workspace Support (REQUIRED)
     *
     * Implements IWorkspaceConsumer for allocation-free hot-path execution.
     * **Workspace MUST be bound via bindWorkspace() before calling apply_tensor().**
     * The workspace provides pre-allocated buffers from DeviceWorkspaceManager.
     *
     * Without a bound workspace, apply_tensor() will return false with an error.
     */
    class ROCmEmbeddingKernelT : public ITensorEmbedding, public IWorkspaceConsumer
    {
    public:
        /**
         * @brief Construct for one explicit immutable device owner
         * @param device_idx Exact nonnegative HIP device ordinal
         */
        explicit ROCmEmbeddingKernelT(int device_idx) : binding_(device_idx) {}

        /**
         * @brief Construct with device context (Phase 4 pattern)
         * @param ctx Device context for shared handles/streams
         */
        explicit ROCmEmbeddingKernelT(IWorkerGPUContext *ctx)
            : ROCmEmbeddingKernelT(requireContextOwner(ctx))
        {
            device_ctx_ = ctx;
        }

        ~ROCmEmbeddingKernelT() override;

        // ===== Device Context Support (Phase 4) =====
        /** @brief Bind only an initialized context for this immutable participant.
         * @param ctx Borrowed initialized context whose device matches this instance.
         * @throws std::invalid_argument Before replacing a valid context on mismatch. */
        void setDeviceContext(IWorkerGPUContext *ctx)
        {
            (void)binding_.resolveOrdinal(requireContextOwner(ctx));
            device_ctx_ = ctx;
        }
        IWorkerGPUContext *deviceContext() const { return device_ctx_; }
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }
        void *getStream() const;

        // ITensorKernel interface
        bool supports_device(int device_idx) const override
        {
            return device_idx == binding_.ordinal();
        }

        /** @brief Authenticate the stream's native owner before replacing its borrowed binding.
         * @param stream Explicit non-null native producer stream owned by this participant.
         * @throws std::runtime_error On failed native ownership observation.
         * @throws std::invalid_argument For a stream belonging to another participant. */
        void bindGPUStream(ExplicitGPUStream stream) override;
        /** @brief Withdraw only the borrowed stream; retain the immutable participant. */
        void clearGPUStreamBinding() override;
        void setPreparedEmbeddingHandle(const PreparedEmbeddingHandle *handle) override
        {
            prepared_embedding_handle_ = handle;
        }

        void setVocabRange(int vocab_offset, int local_vocab_size) override
        {
            explicit_vocab_range_ = true;
            vocab_offset_ = vocab_offset;
            local_vocab_size_ = local_vocab_size;
        }

        void setAllowOutOfRangeTokenIds(bool allow) override
        {
            allow_out_of_range_token_ids_ = allow;
        }

        // ITensorEmbedding interface - FP32 output
        bool apply(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            float *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        // ITensorEmbedding interface - BF16 output
        bool apply_bf16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        // ITensorEmbedding interface - FP16 output
        bool apply_fp16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        // ITensorEmbedding interface - Q8_1 output
        bool apply_q8_1(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            void *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        // ITensorEmbedding interface - tensor-based dispatch
        bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        void setDynamicTokenIds(const int *token_ids, int num_tokens) override;
        void setDynamicDeviceTokenIds(const void *token_ids_device, int num_tokens) override;

        // =====================================================================
        // Session Lifecycle (ITensorKernel overrides)
        // =====================================================================

        /**
         * @brief Reset input-dependent cached state
         *
         * Clears the dynamic_params_active_ flag and token count so that the
         * next apply_tensor() call re-uploads token IDs from scratch.
         * Called by KernelFactory::resetAllDynamicState() on session boundary.
         */
        void resetDynamicState() override;

        /**
         * @brief Check if dynamic token state is cached
         * @return true if a previous setDynamicTokenIds() preload is active
         */
        bool hasDynamicStateActive() const override { return dynamic_params_active_; }

        // =========================================================================
        // IWorkspaceConsumer Interface
        // =========================================================================

        /**
         * @brief Get workspace requirements for embedding lookup
         *
         * Returns the stable INT32 token-ID buffer used by prefill and by graph
         * replay when a device sampler has not supplied token IDs directly.
         * Embedding weights are persistent model state and never workspace.
         *
         * @param m Maximum sequence length (num_tokens)
         * @param n Not used (pass 0)
         * @param k Unused (embedding weights are prepared separately)
         * @return WorkspaceRequirements describing all needed buffers
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const override;

        /**
         * @brief Bind workspace manager (REQUIRED for apply_tensor)
         *
         * **MUST be called before apply_tensor().** After binding, the kernel uses
         * pre-allocated buffers from the workspace manager for allocation-free execution.
         *
         * @param workspace Pointer to workspace manager (NOT owned, must outlive kernel)
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /**
         * @brief Check if a workspace is currently bound
         */
        bool hasWorkspace() const override;

        /**
         * @brief Get the currently bound workspace manager
         */
        DeviceWorkspaceManager *getWorkspace() const override;

    private:
        /** @brief Authenticate the context before deriving the immutable participant ordinal. */
        static int requireContextOwner(IWorkerGPUContext *ctx)
        {
            if (!ctx || !ctx->isInitialized() || ctx->deviceOrdinal() < 0)
                throw std::invalid_argument("ROCm embedding requires an initialized explicit device context");
            return ctx->deviceOrdinal();
        }
        ROCmEmbeddingExecutionBinding<DeviceWorkspaceManager> binding_;
        IWorkerGPUContext *device_ctx_ = nullptr;
        const PreparedEmbeddingHandle *prepared_embedding_handle_ = nullptr;

        int *h_token_ids_ = nullptr;
        int max_token_ids_ = 0;
        int dynamic_token_count_ = 0;
        bool dynamic_params_active_ = false;
        const int *device_token_ids_ = nullptr;
        int device_token_count_ = 0;
        bool device_token_ids_active_ = false;
        void *preload_stream_ = nullptr; ///< Stream used for the last setDynamicTokenIds H2D copy

        int vocab_offset_ = 0;
        int local_vocab_size_ = 0;
        bool explicit_vocab_range_ = false;
        bool allow_out_of_range_token_ids_ = false;
    };

} // namespace llaminar2
