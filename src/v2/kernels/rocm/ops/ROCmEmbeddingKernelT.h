/**
 * @file ROCmEmbeddingKernelT.h
 * @brief ROCm implementation of ITensorEmbedding interface
 *
 * Provides FP32, BF16, FP16, and Q8_1 embedding lookup on AMD GPUs.
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include <unordered_map>
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
         * @brief Construct with optional default device index
         * @param device_idx Default HIP device index (-1 for current device)
         */
        explicit ROCmEmbeddingKernelT(int device_idx = 0) : device_idx_(device_idx) {}

        /**
         * @brief Construct with device context (Phase 4 pattern)
         * @param ctx Device context for shared handles/streams
         */
        explicit ROCmEmbeddingKernelT(IWorkerGPUContext* ctx)
        {
            if (!ctx) throw std::runtime_error("ROCmEmbeddingKernelT: Device context is null");
            if (!ctx->isInitialized()) throw std::runtime_error("ROCmEmbeddingKernelT: Device context not initialized");
            device_ctx_ = ctx;
            device_idx_ = ctx->deviceOrdinal();
        }

        ~ROCmEmbeddingKernelT() override = default;

        // ===== Device Context Support (Phase 4) =====
        void setDeviceContext(IWorkerGPUContext* ctx) { device_ctx_ = ctx; }
        IWorkerGPUContext* deviceContext() const { return device_ctx_; }
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }
        void* getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

        // ITensorKernel interface
        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0; // ROCm supports any valid device index
        }

        // ITensorEmbedding interface - FP32 output
        bool apply(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            float *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - BF16 output
        bool apply_bf16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - FP16 output
        bool apply_fp16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - Q8_1 output
        bool apply_q8_1(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            void *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - tensor-based dispatch
        bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // =========================================================================
        // IWorkspaceConsumer Interface
        // =========================================================================

        /**
         * @brief Get workspace requirements for embedding lookup
         *
         * Returns buffers needed for embedding:
         * - embed_token_ids [max_seq_len]: INT32 token IDs on GPU
         * - embed_table_temp [vocab_size × d_model]: FP32 temp buffer for non-GPU embed tables
         *
         * @param m Maximum sequence length (num_tokens)
         * @param n Not used (pass 0)
         * @param k d_model dimension (embedding dimension)
         * @return WorkspaceRequirements describing all needed buffers
         *
         * @note For embed_table_temp, vocab_size is estimated as 151936 (Qwen2 vocab).
         *       Actual vocab size may be smaller; the buffer will be sufficient.
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

        /**
         * @brief Clear the cached embedding table pointer for this workspace
         *
         * Call this if the model changes or the workspace is reset.
         * The next apply_tensor() call will re-upload the embedding table.
         */
        void clearEmbeddingCache()
        {
            if (workspace_)
            {
                s_workspace_embed_cache_.erase(workspace_);
            }
        }

        /**
         * @brief Static method to clear ALL embedding caches (for model unload)
         */
        static void clearGlobalEmbeddingCache() { s_workspace_embed_cache_.clear(); }

    private:
        int device_idx_;
        IWorkerGPUContext* device_ctx_ = nullptr;

        // IWorkspaceConsumer state
        DeviceWorkspaceManager *workspace_ = nullptr; ///< Bound workspace manager (not owned)

        // Embedding table caching state (STATIC MAP - per-workspace cache)
        // This is critical for performance: kernel instances are recreated every forward pass
        // due to graph rebuild, but the embedding table in GPU workspace is persistent.
        // Using static ensures we don't re-upload 500+ MB every decode step.
        // KEY FIX: Use per-workspace cache to support LOCAL TP with multiple devices.
        // Each device has its own workspace, so we cache separately per workspace.
        static inline std::unordered_map<DeviceWorkspaceManager *, const TensorBase *> s_workspace_embed_cache_;
    };

} // namespace llaminar2
