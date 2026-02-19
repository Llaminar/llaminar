/**
 * @file ROCmRoPEKernelT.h
 * @brief ROCm RoPE kernel with typed precision support and workspace caching
 *
 * HIP/ROCm implementation of Rotary Positional Embeddings (RoPE) following
 * the same template pattern as CUDARoPEKernelT. Specialized for FP32,
 * BF16, and FP16 precisions.
 *
 * OPTIMIZATIONS (matching CUDA v3 strategy):
 * - Pre-computed inverse frequency table cached in device memory
 * - Workspace-based position_ids buffer to avoid per-call hipMalloc/hipFree
 * - Shared memory sin/cos tables in HIP kernels
 * - Fused Q+K kernel to reduce launch overhead
 *
 * @author Llaminar Team
 * @date 2025-01-17
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../kernels/rope/RoPEDeviceParams.h"
#include <cstdint>
#include <stdexcept>

// Forward declaration
namespace llaminar2
{
    class DeviceWorkspaceManager;
}

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // Primary Template (Unsupported Precisions)
        // =========================================================================

        template <ActivationPrecision Precision>
        class ROCmRoPEKernelT
        {
            static_assert(Precision == ActivationPrecision::FP32 ||
                              Precision == ActivationPrecision::BF16 ||
                              Precision == ActivationPrecision::FP16,
                          "ROCmRoPEKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // =========================================================================
        // FP32 Specialization
        // =========================================================================

        template <>
        class ROCmRoPEKernelT<ActivationPrecision::FP32> : public ITensorRoPE, public IWorkspaceConsumer
        {
        public:
            using StorageType = float;

            /// Maximum head_dim/2 for worst-case workspace allocation (covers head_dim up to 256)
            static constexpr int MAX_HALF_DIM = 128;

            explicit ROCmRoPEKernelT(int device_idx = -1, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             * @param rope_theta RoPE theta parameter
             */
            explicit ROCmRoPEKernelT(IWorkerGPUContext *ctx, float rope_theta = 10000.0f)
                : rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmRoPEKernelT<FP32>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmRoPEKernelT<FP32>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmRoPEKernelT() override;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== GPU Stream Support (Graph Capture) =====
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            /// Update the pos_offset in pinned host memory for graph replay
            void setDynamicPosOffset(int pos_offset) override;

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Tensor API (for RoPEStage) =====
            bool apply_tensor(
                TensorBase *Q,
                TensorBase *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int pos_offset = 0) override;

            // ===== Typed API =====
            bool apply_typed(
                float *Q,
                float *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
            static const char *precision_name() { return "FP32"; }

            // ===== IWorkspaceConsumer interface =====

            /**
             * @brief Get workspace requirements for RoPE
             *
             * @param m Maximum sequence length
             * @param n Not used (pass 0)
             * @param k head_dim (for inverse frequency table size)
             * @return WorkspaceRequirements describing needed buffers
             */
            WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;

            /**
             * @brief Bind workspace manager (recommended for hot-path performance)
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
            int device_idx_;
            float rope_theta_;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;

            // Workspace state
            DeviceWorkspaceManager *workspace_ = nullptr;

            // Workspace-based inverse frequency state (v3: tracked per instance)
            mutable bool inv_freq_initialized_ = false;
            mutable int inv_freq_head_dim_ = 0;
            mutable float inv_freq_theta_ = 0.0f;

            /// Pinned host memory for graph-captured H2D copy of device params
            rope::RoPEDeviceParams *h_device_params_ = nullptr;
        };

        // =========================================================================
        // BF16 Specialization
        // =========================================================================

        template <>
        class ROCmRoPEKernelT<ActivationPrecision::BF16> : public ITensorRoPE, public IWorkspaceConsumer
        {
        public:
            using StorageType = uint16_t;

            /// Maximum head_dim/2 for worst-case workspace allocation (covers head_dim up to 256)
            static constexpr int MAX_HALF_DIM = 128;

            explicit ROCmRoPEKernelT(int device_idx = -1, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             * @param rope_theta RoPE theta parameter
             */
            explicit ROCmRoPEKernelT(IWorkerGPUContext *ctx, float rope_theta = 10000.0f)
                : rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmRoPEKernelT<BF16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmRoPEKernelT<BF16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmRoPEKernelT() override;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== GPU Stream Support (Graph Capture) =====
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }
            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            /// Update the pos_offset in pinned host memory for graph replay
            void setDynamicPosOffset(int pos_offset) override;

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)interleaved;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override;

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Tensor API (for RoPEStage) =====
            bool apply_tensor(
                TensorBase *Q,
                TensorBase *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int pos_offset = 0) override;

            // ===== Typed API =====
            bool apply_typed(
                uint16_t *Q,
                uint16_t *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
            static const char *precision_name() { return "BF16"; }

            // ===== IWorkspaceConsumer interface =====
            WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
            void bindWorkspace(DeviceWorkspaceManager *workspace) override;
            bool hasWorkspace() const override;
            DeviceWorkspaceManager *getWorkspace() const override;

        private:
            int device_idx_;
            float rope_theta_;
            DeviceWorkspaceManager *workspace_ = nullptr;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;

            // Workspace-based inverse frequency state (v3: tracked per instance)
            mutable bool inv_freq_initialized_ = false;
            mutable int inv_freq_head_dim_ = 0;
            mutable float inv_freq_theta_ = 0.0f;

            /// Pinned host memory for graph-captured H2D copy of device params
            rope::RoPEDeviceParams *h_device_params_ = nullptr;
        };

        // =========================================================================
        // FP16 Specialization
        // =========================================================================

        template <>
        class ROCmRoPEKernelT<ActivationPrecision::FP16> : public ITensorRoPE, public IWorkspaceConsumer
        {
        public:
            using StorageType = uint16_t;

            /// Maximum head_dim/2 for worst-case workspace allocation (covers head_dim up to 256)
            static constexpr int MAX_HALF_DIM = 128;

            explicit ROCmRoPEKernelT(int device_idx = -1, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             * @param rope_theta RoPE theta parameter
             */
            explicit ROCmRoPEKernelT(IWorkerGPUContext *ctx, float rope_theta = 10000.0f)
                : rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f)
            {
                if (!ctx)
                    throw std::runtime_error("ROCmRoPEKernelT<FP16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("ROCmRoPEKernelT<FP16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~ROCmRoPEKernelT() override;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== GPU Stream Support (Graph Capture) =====
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }
            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            /// Update the pos_offset in pinned host memory for graph replay
            void setDynamicPosOffset(int pos_offset) override;

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)interleaved;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override;

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Tensor API (for RoPEStage) =====
            bool apply_tensor(
                TensorBase *Q,
                TensorBase *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int pos_offset = 0) override;

            // ===== Typed API =====
            bool apply_typed(
                uint16_t *Q,
                uint16_t *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1);

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
            static const char *precision_name() { return "FP16"; }

            // ===== IWorkspaceConsumer interface =====
            WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
            void bindWorkspace(DeviceWorkspaceManager *workspace) override;
            bool hasWorkspace() const override;
            DeviceWorkspaceManager *getWorkspace() const override;

        private:
            int device_idx_;
            float rope_theta_;
            DeviceWorkspaceManager *workspace_ = nullptr;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;

            // Workspace-based inverse frequency state (v3: tracked per instance)
            mutable bool inv_freq_initialized_ = false;
            mutable int inv_freq_head_dim_ = 0;
            mutable float inv_freq_theta_ = 0.0f;

            /// Pinned host memory for graph-captured H2D copy of device params
            rope::RoPEDeviceParams *h_device_params_ = nullptr;
        };

    } // namespace rocm
} // namespace llaminar2
