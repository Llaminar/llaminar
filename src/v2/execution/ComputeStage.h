/**
 * @file ComputeStage.h
 * @brief Compute stage abstraction for device-agnostic kernel dispatch
 * @author David Sanftenberg
 * @date December 2025
 *
 * ComputeStage represents a single parallelizable operation that can execute
 * on any device (CPU, GPU). Stages are the unit of work for layer-level
 * parallelism and enable clean separation of serial setup from parallel compute.
 *
 * Key benefits:
 * 1. Device-agnostic interface - same API for CPU and GPU kernels
 * 2. Composable - build compute graphs from atomic operations
 * 3. Introspectable - stages know their FLOP counts, memory needs
 * 4. MoE-ready - expert FFN stages for parallel expert execution
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "DeviceContext.h"

namespace llaminar2
{

    // Forward declarations
    class TensorBase;
    class IKVCache;

    /**
     * @brief Types of compute operations
     */
    enum class ComputeStageType
    {
        // Matrix operations
        GEMM,           ///< General matrix multiplication
        GEMM_BIAS,      ///< GEMM with bias addition
        GEMM_FUSED_QKV, ///< Fused Q/K/V projection

        // Normalization
        RMS_NORM,   ///< RMS normalization
        LAYER_NORM, ///< Layer normalization (future)

        // Activations
        SWIGLU, ///< SwiGLU activation (FFN)
        GELU,   ///< GELU activation (future)
        SILU,   ///< SiLU activation

        // Attention
        ROPE,              ///< Rotary position encoding
        ATTENTION,         ///< Full attention (Q*K^T, softmax, *V)
        ATTENTION_QK,      ///< Q*K^T only
        ATTENTION_SOFTMAX, ///< Softmax only
        ATTENTION_V,       ///< Softmax @ V only

        // Element-wise
        ADD_RESIDUAL, ///< Element-wise addition (residual connection)
        SCALE,        ///< Element-wise scaling

        // MoE specific
        MOE_ROUTER,     ///< Expert routing (softmax + top-k)
        MOE_EXPERT_FFN, ///< Single expert FFN execution
        MOE_COMBINE,    ///< Combine expert outputs with weights

        // Collective
        ALLREDUCE, ///< MPI allreduce (sum)
        ALLGATHER, ///< MPI allgather

        // Utility
        COPY,       ///< Memory copy
        QUANTIZE,   ///< Quantization (FP32 → Q8_1, etc.)
        DEQUANTIZE, ///< Dequantization
    };

    /**
     * @brief Convert stage type to string for logging
     */
    const char *computeStageTypeName(ComputeStageType type);

    /**
     * @brief Base class for all compute stages
     *
     * Derived classes implement device-specific kernels while maintaining
     * a common interface for orchestration.
     */
    class IComputeStage
    {
    public:
        virtual ~IComputeStage() = default;

        // =========================================================================
        // Execution
        // =========================================================================

        /**
         * @brief Execute this stage on the given device context
         *
         * The stage must be compatible with the device type (CPU stages on CPU, etc.)
         * GPU stages may enqueue work asynchronously - call ctx->synchronize() if
         * you need completion.
         *
         * @param ctx Device context to execute on
         * @return true on success, false on error
         */
        virtual bool execute(IDeviceContext *ctx) = 0;

        // =========================================================================
        // Introspection
        // =========================================================================

        /**
         * @brief Get the operation type
         */
        virtual ComputeStageType type() const = 0;

        /**
         * @brief Human-readable name (for profiling/logging)
         */
        virtual std::string name() const
        {
            return computeStageTypeName(type());
        }

        /**
         * @brief Estimated floating-point operations
         *
         * Used for load balancing and performance modeling.
         * Returns 0 if not applicable (e.g., memory ops).
         */
        virtual size_t estimatedFlops() const { return 0; }

        /**
         * @brief Estimated memory traffic in bytes
         *
         * Includes reads and writes. Used for bandwidth estimation.
         */
        virtual size_t estimatedMemoryBytes() const { return 0; }

        /**
         * @brief Does this stage require MPI synchronization after?
         *
         * True for stages like row-parallel GEMM that need allreduce.
         */
        virtual bool requiresAllreduce() const { return false; }

        /**
         * @brief Can this stage execute on the given backend?
         */
        virtual bool supportsBackend(ComputeBackendType backend) const = 0;
    };

    // =============================================================================
    // Concrete Stage Implementations (CPU)
    // =============================================================================

    /**
     * @brief GEMM stage: C = alpha * A * B + beta * C
     */
    class GEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *A;            ///< Activation matrix (m × k)
            const TensorBase *B;      ///< Weight tensor (k × n, may be quantized)
            void *C;                  ///< Output matrix (m × n)
            int m, n, k;              ///< Matrix dimensions
            float alpha = 1.0f;       ///< Scale factor for A*B
            float beta = 0.0f;        ///< Scale factor for existing C
            bool transpose_B = false; ///< Whether B is transposed (n × k)
        };

        explicit GEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief RMS normalization stage
     */
    class RMSNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            void *input;        ///< Input/output tensor (in-place)
            const float *gamma; ///< Scale weights
            int seq_len;        ///< Sequence length
            int hidden_dim;     ///< Hidden dimension
            float eps;          ///< Epsilon for numerical stability
        };

        explicit RMSNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief Rotary position encoding stage
     */
    class RoPEStage : public IComputeStage
    {
    public:
        struct Params
        {
            void *tensor;     ///< Q or K tensor to apply RoPE
            int seq_len;      ///< Sequence length
            int n_heads;      ///< Number of heads
            int head_dim;     ///< Dimension per head
            int pos_offset;   ///< Position offset (for KV cache)
            float theta_base; ///< RoPE theta base (default 10000.0)
        };

        explicit RoPEStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief Full attention stage (Q*K^T, softmax, *V)
     */
    class AttentionStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *Q;  ///< Query tensor [seq_len, n_heads, head_dim]
            const void *K;  ///< Key tensor [kv_len, n_kv_heads, head_dim]
            const void *V;  ///< Value tensor [kv_len, n_kv_heads, head_dim]
            void *output;   ///< Output tensor [seq_len, n_heads, head_dim]
            int seq_len;    ///< Query sequence length
            int kv_len;     ///< Key/value sequence length
            int n_heads;    ///< Number of query heads
            int n_kv_heads; ///< Number of KV heads (GQA)
            int head_dim;   ///< Dimension per head
            bool causal;    ///< Apply causal mask
            float scale;    ///< Attention scale (1/sqrt(head_dim))
        };

        explicit AttentionStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief SwiGLU activation stage
     */
    class SwiGLUStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *gate; ///< Gate tensor [seq_len, intermediate_dim]
            const void *up;   ///< Up tensor [seq_len, intermediate_dim]
            void *output;     ///< Output tensor [seq_len, intermediate_dim]
            int seq_len;
            int intermediate_dim;
        };

        explicit SwiGLUStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief Residual addition stage: output = input + residual
     */
    class ResidualAddStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *input;    ///< Input tensor
            const void *residual; ///< Residual tensor
            void *output;         ///< Output tensor (can be same as input)
            size_t num_elements;  ///< Total elements
        };

        explicit ResidualAddStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief MPI Allreduce stage
     */
    class AllreduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            void *buffer;   ///< Buffer to allreduce (in-place)
            size_t count;   ///< Number of elements
            void *mpi_comm; ///< MPI communicator (cast to MPI_Comm)
        };

        explicit AllreduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    // =============================================================================
    // MoE Stages
    // =============================================================================

    /**
     * @brief MoE router stage: compute expert selection
     */
    class MoERouterStage : public IComputeStage
    {
    public:
        struct Params
        {
            const void *hidden;       ///< Hidden states [seq_len, d_model]
            const void *gate_weights; ///< Router weights [d_model, num_experts]
            float *router_logits;     ///< Output: router logits [seq_len, num_experts]
            int seq_len;
            int d_model;
            int num_experts;
        };

        explicit MoERouterStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief Single expert FFN execution
     *
     * This stage handles tokens routed to one specific expert.
     * Multiple MoEExpertStages can execute in parallel on different devices.
     */
    class MoEExpertStage : public IComputeStage
    {
    public:
        struct Params
        {
            int expert_id;                         ///< Which expert this is
            const void *input;                     ///< Input tokens for this expert
            void *output;                          ///< Output buffer
            const TensorBase *gate_w;              ///< Expert gate weights
            const TensorBase *up_w;                ///< Expert up weights
            const TensorBase *down_w;              ///< Expert down weights
            const std::vector<int> *token_indices; ///< Which tokens to process
            int d_model;
            int intermediate_dim;
        };

        explicit MoEExpertStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override;
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    /**
     * @brief Combine expert outputs with router weights
     */
    class MoECombineStage : public IComputeStage
    {
    public:
        struct Params
        {
            const std::vector<const void *> *expert_outputs; ///< Outputs from each expert
            const std::vector<float> *expert_weights;        ///< Router weights per token-expert
            const std::vector<int> *token_expert_map;        ///< Which experts each token used
            void *output;                                    ///< Combined output [seq_len, d_model]
            int seq_len;
            int d_model;
            int top_k; ///< Experts per token
        };

        explicit MoECombineStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_COMBINE; }
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        Params params_;
    };

    // =============================================================================
    // Stage Factory
    // =============================================================================

    /**
     * @brief Factory for creating device-appropriate compute stages
     */
    class ComputeStageFactory
    {
    public:
        /**
         * @brief Create a GEMM stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createGEMM(
            const GEMMStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create an RMSNorm stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createRMSNorm(
            const RMSNormStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a RoPE stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createRoPE(
            const RoPEStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create an attention stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createAttention(
            const AttentionStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a SwiGLU stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createSwiGLU(
            const SwiGLUStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a residual add stage for the target backend
         */
        static std::unique_ptr<IComputeStage> createResidualAdd(
            const ResidualAddStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a MoE router stage for expert selection
         */
        static std::unique_ptr<IComputeStage> createMoERouter(
            const MoERouterStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create an expert FFN stage for MoE
         */
        static std::unique_ptr<IComputeStage> createMoEExpert(
            const MoEExpertStage::Params &params,
            ComputeBackendType target_backend);

        /**
         * @brief Create a MoE combine stage for weighted expert output combination
         */
        static std::unique_ptr<IComputeStage> createMoECombine(
            const MoECombineStage::Params &params,
            ComputeBackendType target_backend);
    };

    // =============================================================================
    // GPU Compute Stages (CUDA + ROCm)
    // =============================================================================

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

    /**
     * @brief GPU GEMM stage using IBackend
     *
     * Delegates to CUDABackend/ROCmBackend for actual computation.
     * Supports quantized formats via backend's gemmIQ4NL or falls back to cuBLAS/rocBLAS.
     */
    class GPUGEMMStage : public IComputeStage
    {
    public:
        explicit GPUGEMMStage(GEMMStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        GEMMStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU RMSNorm stage
     *
     * Custom CUDA/HIP kernel with warp-level reduction.
     */
    class GPURMSNormStage : public IComputeStage
    {
    public:
        explicit GPURMSNormStage(RMSNormStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        RMSNormStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU SwiGLU stage
     *
     * Fused silu(gate) * up kernel.
     */
    class GPUSwiGLUStage : public IComputeStage
    {
    public:
        explicit GPUSwiGLUStage(SwiGLUStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        SwiGLUStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU Residual Add stage
     *
     * Simple element-wise addition kernel.
     */
    class GPUResidualAddStage : public IComputeStage
    {
    public:
        explicit GPUResidualAddStage(ResidualAddStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        ResidualAddStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU RoPE stage
     *
     * Rotary position embedding kernel.
     */
    class GPURoPEStage : public IComputeStage
    {
    public:
        explicit GPURoPEStage(RoPEStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        RoPEStage::Params params_;
        ComputeBackendType backend_;
    };

    /**
     * @brief GPU Attention stage
     *
     * Flash attention or standard attention implementation.
     */
    class GPUAttentionStage : public IComputeStage
    {
    public:
        explicit GPUAttentionStage(AttentionStage::Params params, ComputeBackendType backend);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;

    private:
        AttentionStage::Params params_;
        ComputeBackendType backend_;
    };

#endif // HAVE_CUDA || HAVE_ROCM

} // namespace llaminar2
