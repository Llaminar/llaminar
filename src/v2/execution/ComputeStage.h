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
#include "BufferRole.h"                  // For buffer requirements
#include "../pipelines/PipelineConfig.h" // For ActivationPrecision
#include "../tensors/BlockStructures.h"  // For Q8_1Block in StageDumpInfo
#include "../tensors/TensorKernels.h"    // For AttentionMode enum

namespace llaminar2
{

    // Forward declarations (needed for StageDumpInfo before full TensorBase definition)
    class TensorBase;

    /**
     * @brief Detailed dump info for stage debugging
     *
     * Contains all input, output, and parameter buffers needed to fully
     * reproduce and debug a stage execution. Used by StageDumper for
     * first-class debugging support.
     */
    struct StageDumpInfo
    {
        // Input buffers (read during execute)
        struct InputBuffer
        {
            const char *name = nullptr; ///< Buffer name (e.g., "A", "input", "gate")
            const void *data = nullptr; ///< Pointer to data
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";          ///< "FP32", "Q8_1", "IQ4_NL", etc.
            size_t element_size = sizeof(float); ///< Bytes per element (or block)
        };

        // Output buffers (written during execute)
        struct OutputBuffer
        {
            const char *name = nullptr; ///< Buffer name (e.g., "C", "output")
            const void *data = nullptr; ///< Pointer to data
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = "FP32";
            size_t element_size = sizeof(float);
        };

        // Weight/parameter buffers (read-only during execute)
        struct WeightBuffer
        {
            const char *name = nullptr;         ///< Weight name (e.g., "B", "gamma", "wq")
            const TensorBase *tensor = nullptr; ///< Original quantized tensor
            const void *raw_data = nullptr;     ///< Raw quantized data
            size_t raw_size = 0;                ///< Raw data size in bytes
            size_t rows = 0;
            size_t cols = 0;
            const char *dtype = nullptr; ///< Quantization type
        };

        // Scalar parameters
        struct ScalarParam
        {
            const char *name = nullptr;
            double value = 0;
            const char *dtype = "float"; ///< "int", "float", "bool"
        };

        std::vector<InputBuffer> inputs;
        std::vector<OutputBuffer> outputs;
        std::vector<WeightBuffer> weights;
        std::vector<ScalarParam> scalars;

        // Convenience methods for building dump info
        StageDumpInfo &addInput(const char *name, const float *data, size_t rows, size_t cols)
        {
            inputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        /**
         * @brief Add input tensor (implementation in ComputeStage.cpp to avoid header deps)
         */
        StageDumpInfo &addInput(const char *name, const TensorBase *tensor, size_t rows, size_t cols);

        StageDumpInfo &addInputQ8_1(const char *name, const void *data, size_t rows, size_t cols)
        {
            // For Q8_1, cols should be in blocks (cols/32 blocks per row)
            inputs.push_back({name, data, rows, cols, "Q8_1", sizeof(Q8_1Block)});
            return *this;
        }

        StageDumpInfo &addOutput(const char *name, const float *data, size_t rows, size_t cols)
        {
            outputs.push_back({name, data, rows, cols, "FP32", sizeof(float)});
            return *this;
        }

        /**
         * @brief Add output tensor (implementation in ComputeStage.cpp to avoid header deps)
         */
        StageDumpInfo &addOutput(const char *name, const TensorBase *tensor, size_t rows, size_t cols);

        /**
         * @brief Add weight tensor with explicit dimensions (avoids header dependency)
         */
        StageDumpInfo &addWeight(const char *name, const TensorBase *tensor,
                                 size_t rows, size_t cols, const char *dtype)
        {
            weights.push_back({name, tensor, nullptr, 0, rows, cols, dtype});
            return *this;
        }

        /**
         * @brief Add weight tensor (implementation in ComputeStage.cpp to avoid header deps)
         */
        StageDumpInfo &addWeight(const char *name, const TensorBase *tensor);

        StageDumpInfo &addScalar(const char *name, double value, const char *dtype = "float")
        {
            scalars.push_back({name, value, dtype});
            return *this;
        }

        StageDumpInfo &addScalarInt(const char *name, int value)
        {
            scalars.push_back({name, static_cast<double>(value), "int"});
            return *this;
        }

        StageDumpInfo &addScalarBool(const char *name, bool value)
        {
            scalars.push_back({name, value ? 1.0 : 0.0, "bool"});
            return *this;
        }
    };

    // Forward declarations (TensorBase already declared above)
    class IKVCache;
    class IUnifiedKVCache;
    class MPIContext;
    struct MpiAttentionConfig;

    /**
     * @brief Types of compute operations
     */
    enum class ComputeStageType
    {
        // Matrix operations
        GEMM,               ///< General matrix multiplication
        GEMM_BIAS,          ///< GEMM with bias addition
        GEMM_FUSED_QKV,     ///< Fused Q/K/V projection
        GEMM_FUSED_GATE_UP, ///< Fused Gate/Up projection (FFN)

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

        // Model-level operations (ModelExecutor)
        EMBEDDING,  ///< Token embedding lookup
        LM_HEAD,    ///< Language model head projection
        FINAL_NORM, ///< Final layer normalization
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

        /**
         * @brief Get detailed dump info for debugging/snapshotting
         *
         * Returns comprehensive information about all inputs, outputs, weights,
         * and parameters needed to reproduce this stage execution.
         * Used by StageDumper for first-class debugging support.
         *
         * @return Dump info with all buffers and parameters
         */
        virtual StageDumpInfo getDumpInfo() const
        {
            return StageDumpInfo{}; // Default: empty info
        }

        // =========================================================================
        // Buffer Management (Phase 1: GraphExecutor buffer management)
        // =========================================================================

        /**
         * @brief Get buffer requirements for this stage
         *
         * Returns descriptors for all buffers this stage needs:
         * - INPUT: Read-only data consumed by the stage
         * - OUTPUT: Data produced by the stage
         * - INOUT: Read-modify-write buffers (e.g., residual accumulation)
         * - SCRATCH: Temporary workspace (eligible for aliasing)
         * - WEIGHT: Read-only model parameters
         *
         * GraphBufferManager uses this for:
         * - Pre-allocating all buffers before graph execution
         * - Liveness analysis for SCRATCH buffer reuse
         * - Memory budget estimation
         *
         * CRITICAL: Stages MUST NOT allocate buffers internally in execute().
         * All buffers should come from GraphExecutor's managed pool.
         *
         * Default implementation returns empty requirements (for backward compat).
         * Stages should override to declare their actual requirements.
         *
         * @return Buffer requirements for this stage
         * @see BufferRole for role semantics
         * @see GraphBufferManager for buffer allocation
         */
        virtual StageBufferRequirements getBufferRequirements() const
        {
            return StageBufferRequirements{}; // Default: no requirements declared
        }
    };

    // =============================================================================
    // Concrete Stage Implementations (CPU)
    // =============================================================================

    /**
     * @brief GEMM stage: C = alpha * A * B + beta * C
     *
     * Takes FP32 activations and quantized/FP32 weights, handles quantization internally.
     * For multi-projection patterns (Q/K/V or gate/up), use FusedQKVGEMMStage or
     * FusedGateUpGEMMStage which efficiently share quantization across projections.
     */
    class GEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *A = nullptr; ///< Input activation tensor [m, k]
            const TensorBase *B = nullptr; ///< Weight tensor [k, n] (may be quantized)
            TensorBase *C = nullptr;       ///< Output tensor [m, n]
            int m = 0, n = 0, k = 0;       ///< Matrix dimensions
            float alpha = 1.0f;            ///< Scale factor for A*B
            float beta = 0.0f;             ///< Scale factor for existing C
            bool transpose_B = false;      ///< Whether B is transposed (n × k)

            // Extended fields (use designated initializers)
            const float *bias = nullptr;            ///< Fused bias [n] (nullptr if none)
            const TensorBase *gate_input = nullptr; ///< SwiGLU gate input tensor [m, n] (nullptr if not SwiGLU)
            bool do_swiglu = false;                 ///< Whether to apply SwiGLU fusion

            // MPI context for tensor-parallel execution (optional)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit GEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Fused Q/K/V projection stage
     *
     * Efficiently computes multiple linear projections (Q, K, V) from a shared
     * input by quantizing the input once and reusing the Q8_1 buffer for all
     * projections. This avoids redundant quantization and improves cache locality.
     *
     * This stage delegates to QuantisedGemmKernel::multiply_fused_multi(), which
     * handles the quantization and multi-projection execution internally.
     */
    class FusedQKVGEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                         ///< Batch size * seq_len
            int k = 0;                         ///< Input dimension (d_model)

            // Q projection
            const TensorBase *wq = nullptr;
            TensorBase *output_q = nullptr;
            int n_q = 0;
            const float *bias_q = nullptr; ///< Optional bias for Q [n_q]

            // K projection
            const TensorBase *wk = nullptr;
            TensorBase *output_k = nullptr;
            int n_k = 0;
            const float *bias_k = nullptr; ///< Optional bias for K [n_k]

            // V projection
            const TensorBase *wv = nullptr;
            TensorBase *output_v = nullptr;
            int n_v = 0;
            const float *bias_v = nullptr; ///< Optional bias for V [n_v]
        };

        explicit FusedQKVGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_QKV; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Fused Gate/Up projection stage for FFN
     *
     * Efficiently computes both gate and up projections from a shared input
     * by quantizing the input once and reusing the Q8_1 buffer for both
     * projections. This avoids redundant quantization and improves cache locality.
     *
     * Pattern: input → [gate_proj, up_proj] → SwiGLU → down_proj
     *
     * This stage delegates to QuantisedGemmKernel::multiply_fused_multi(), which
     * handles the quantization and multi-projection execution internally.
     */
    class FusedGateUpGEMMStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *input = nullptr; ///< Input activation tensor [m, k]
            int m = 0;                         ///< Batch size * seq_len
            int k = 0;                         ///< Input dimension (d_model)

            // Gate projection
            const TensorBase *w_gate = nullptr;
            TensorBase *output_gate = nullptr;
            int n_gate = 0;                   ///< Intermediate dimension (d_ff)
            const float *bias_gate = nullptr; ///< Optional bias for gate [n_gate]

            // Up projection
            const TensorBase *w_up = nullptr;
            TensorBase *output_up = nullptr;
            int n_up = 0;                   ///< Intermediate dimension (d_ff, same as n_gate)
            const float *bias_up = nullptr; ///< Optional bias for up [n_up]

            // Optional MPI context for distributed execution (needed for Q8_1 path)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit FusedGateUpGEMMStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GEMM_FUSED_GATE_UP; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief RMS normalization stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRMSNorm() for polymorphic device dispatch.
     *
     * Device-agnostic: IActivationTensor handles CPU/GPU dispatch internally.
     */
    class RMSNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            TensorBase *input = nullptr;       ///< Input activation tensor (IActivationTensor*)
            TensorBase *output = nullptr;      ///< Output tensor (can be same as input for in-place)
            const TensorBase *gamma = nullptr; ///< Gamma weights tensor

            float eps = 1e-6f; ///< Epsilon for numerical stability
            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;

            // Explicit seq_len override for pre-allocated buffers
            // If 0, derives from input tensor dimensions
            // CRITICAL: Must be set during decode when using pre-allocated buffers
            int seq_len = 0;
        };

        explicit RMSNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RMS_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Rotary position encoding stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses IActivationTensor::applyRoPE() for polymorphic device dispatch.
     */
    class RoPEStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            TensorBase *Q = nullptr; ///< Query tensor (IActivationTensor*, modified in-place)
            TensorBase *K = nullptr; ///< Key tensor (IActivationTensor*, modified in-place, optional)

            // Configuration
            int n_heads = 0;             ///< Number of query heads
            int n_kv_heads = 0;          ///< Number of KV heads (for GQA)
            int head_dim = 0;            ///< Dimension per head
            int pos_offset = 0;          ///< Position offset (for KV cache)
            float theta_base = 10000.0f; ///< RoPE theta base
            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit RoPEStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ROPE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    // Forward declarations for attention infrastructure
    class IUnifiedKVCache;
    class MPIContext;
    class MpiAttentionConfig;

    /**
     * @brief Production attention stage with KV cache and MPI support
     *
     * This is the full-featured attention implementation that integrates with:
     * - **KV Cache**: Automatic append during prefill, retrieval during decode
     * - **MPI Parallelism**: Tensor-parallel attention via MpiAttentionOrchestrator
     * - **Batched Execution**: Multiple sequences with padding masks
     * - **Asymmetric Lengths**: Different Q and KV lengths (prefill vs decode)
     * - **GQA/MHA/MQA**: All attention variants supported
     *
     * Execution Modes:
     * 1. **Prefill**: Process all tokens, append to KV cache, full attention
     * 2. **Decode**: Single token query, use cached K/V, efficient O(n) per token
     * 3. **Batched**: Multiple sequences, padding-aware masking
     *
     * Usage in LayerExecutor:
     * @code
     * AttentionWithKVCacheStage::Params params{
     *     .Q = buffers.Q.get(),
     *     .K = buffers.K.get(),
     *     .V = buffers.V.get(),
     *     .output = buffers.attn_output.get(),
     *     .kv_cache = kv_cache_.get(),
     *     .layer_idx = layer_idx,
     *     // ... configuration
     * };
     * graph.addNode("attention", ComputeStageFactory::createAttentionWithKVCache(params), device);
     * @endcode
     *
     * MoE Compatibility:
     * For MoE models, attention is typically shared across experts (not expert-specific).
     * The KV cache is shared, and this stage can be used directly in MoE pipelines.
     * Expert parallelism affects FFN routing, not attention computation.
     *
     * @see MpiAttentionOrchestrator for the underlying attention computation
     * @see KVCacheAppendStage for explicit cache operations (advanced pipelining)
     */
    class AttentionWithKVCacheStage : public IComputeStage
    {
    public:
        /**
         * @brief Legacy mode enum for backward compatibility
         * @deprecated Use AttentionMode from TensorKernels.h for new code
         */
        enum class Mode
        {
            AUTO,    ///< Detect from seq_len and KV cache state
            PREFILL, ///< Process multiple tokens, populate cache
            DECODE,  ///< Single token, use cached K/V
            BATCHED  ///< Multiple sequences with different lengths
        };

        struct Params
        {
            // Input/output tensors (already projected)
            TensorBase *Q = nullptr;      ///< Query [batch_size * seq_len, n_heads * head_dim]
            TensorBase *K = nullptr;      ///< Key [batch_size * seq_len, n_kv_heads * head_dim]
            TensorBase *V = nullptr;      ///< Value [batch_size * seq_len, n_kv_heads * head_dim]
            TensorBase *output = nullptr; ///< Output [batch_size * seq_len, n_heads * head_dim]

            // KV cache integration
            IUnifiedKVCache *kv_cache = nullptr; ///< KV cache (nullptr = no caching)
            int layer_idx = 0;                   ///< Layer index for cache access

            // Execution mode (legacy - prefer attention_mode)
            Mode mode = Mode::AUTO;

            // Attention configuration
            int batch_size = 1;
            int seq_len = 0;      ///< Query sequence length
            int n_heads = 0;      ///< Number of query heads
            int n_kv_heads = 0;   ///< Number of KV heads (GQA: n_kv_heads < n_heads)
            int head_dim = 0;     ///< Dimension per head
            bool causal = true;   ///< Apply causal mask
            int window_size = -1; ///< Sliding window attention (-1 = full)

            // MPI configuration (for tensor-parallel execution)
            std::shared_ptr<MPIContext> mpi_ctx;
            int mpi_strategy = 0; ///< MPIStrategy cast to int (avoid header dep)

            // Workspace buffers (pre-allocated by pipeline for zero hot-path allocation)
            // If nullptr, stage will allocate internally (slower)
            TensorBase *workspace_scores = nullptr;  ///< [n_heads * max_seq, max_seq]
            TensorBase *workspace_context = nullptr; ///< [max_seq, head_dim]
            TensorBase *workspace_mask = nullptr;    ///< [max_seq, max_seq]

            // Per-sequence lengths for batched attention (nullptr = all same length)
            const std::vector<int> *sequence_lengths = nullptr;

            // Position offset for decode mode (current position in sequence)
            int position_offset = 0;

            // Device placement (-1 = CPU, >=0 = GPU device index)
            int device_idx = -1;
        };

        explicit AttentionWithKVCacheStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;

        /**
         * @brief Get the actual execution mode that will be used
         *
         * If mode is AUTO, this detects based on seq_len and cache state.
         */
        Mode effectiveMode() const;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        /// Prefill: Process all tokens, populate KV cache
        bool executePrefill(IDeviceContext *ctx);

        /// Decode: Single token against cached K/V
        bool executeDecode(IDeviceContext *ctx);

        /// Batched: Multiple sequences with padding
        bool executeBatched(IDeviceContext *ctx);

        /// Build MpiAttentionConfig from params
        MpiAttentionConfig buildAttentionConfig() const;
    };

    /**
     * @brief Explicit KV cache append stage
     *
     * Separates cache operations from attention computation, enabling:
     * - Pipelined execution: Append on one device while attending on another
     * - Explicit control: Manual cache management for advanced use cases
     * - Cross-device caches: Cache on GPU while computing on CPU
     *
     * For most use cases, prefer AttentionWithKVCacheStage which handles
     * cache operations internally. Use this stage when you need fine-grained
     * control over cache timing.
     *
     * @see AttentionWithKVCacheStage for integrated attention + cache
     */
    class KVCacheAppendStage : public IComputeStage
    {
    public:
        struct Params
        {
            const TensorBase *K = nullptr; ///< Key to append [seq_len, n_kv_heads * head_dim]
            const TensorBase *V = nullptr; ///< Value to append [seq_len, n_kv_heads * head_dim]
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int seq_idx = 0;    ///< Sequence index (for batched caches)
            int num_tokens = 0; ///< Number of tokens to append (0 = full tensor)
        };

        explicit KVCacheAppendStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::COPY; } // Cache ops are data movement
        bool supportsBackend(ComputeBackendType backend) const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Pure attention compute stage (no KV cache management)
     *
     * Delegates to KernelFactory::createAttention() for device-appropriate kernel
     * selection. This stage handles ONLY the attention computation:
     *   output = softmax(Q @ K^T / sqrt(head_dim) + mask) @ V
     *
     * For KV cache management, use KVCacheAppendStage separately in the DAG.
     * For integrated cache+attention, use AttentionWithKVCacheStage.
     *
     * Benefits of separation:
     * - **Composable**: Cache append and attention can run on different devices
     * - **Testable**: Pure attention kernel can be tested without cache
     * - **GPU-ready**: KernelFactory handles CPU/GPU dispatch transparently
     *
     * DAG usage pattern:
     * @code
     * // Stage 1: Append K/V to cache
     * graph.addNode("kv_append", createKVCacheAppend(...), device);
     *
     * // Stage 2: Get cached K/V (done in graph builder, not a stage)
     * TensorBase* K_cached = kv_cache->get_k_base(layer_idx, 0);
     * TensorBase* V_cached = kv_cache->get_v_base(layer_idx, 0);
     *
     * // Stage 3: Pure attention compute
     * graph.addNode("attention", createAttentionCompute({
     *     .Q = Q, .K = K_cached, .V = V_cached, .output = output,
     *     .batch_size = 1, .seq_len = 1, .kv_len = cached_tokens,
     *     .n_heads = n_heads, .n_kv_heads = n_kv_heads, .head_dim = head_dim,
     *     .causal = true
     * }), device);
     * graph.addDependency("kv_append", "attention");
     * @endcode
     *
     * @see KVCacheAppendStage for cache operations
     * @see AttentionWithKVCacheStage for integrated cache+attention
     * @see KernelFactory::createAttention() for kernel dispatch
     */
    class AttentionComputeStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            TensorBase *Q = nullptr;      ///< Query [batch_size * seq_len, n_heads * head_dim]
            TensorBase *K = nullptr;      ///< Key [batch_size * kv_len, n_kv_heads * head_dim]
            TensorBase *V = nullptr;      ///< Value [batch_size * kv_len, n_kv_heads * head_dim]
            TensorBase *output = nullptr; ///< Output [batch_size * seq_len, n_heads * head_dim]

            // Dimensions
            int batch_size = 1; ///< Number of sequences (1 for single sequence)
            int seq_len = 0;    ///< Query sequence length
            int kv_len = 0;     ///< Key/Value length (may differ from seq_len in decode)
            int n_heads = 0;    ///< Number of query heads
            int n_kv_heads = 0; ///< Number of KV heads (GQA: n_heads % n_kv_heads == 0)
            int head_dim = 0;   ///< Dimension per head

            // Attention configuration
            bool causal = true;   ///< Apply causal (lower-triangular) mask
            int window_size = -1; ///< Sliding window size (-1 = full attention)

            /// Execution mode (PREFILL, DECODE, BATCHED_DECODE, CHUNKED_PREFILL)
            /// Default: AUTO detection from batch_size, seq_len, kv_len
            AttentionMode attention_mode = AttentionMode::PREFILL;
            bool auto_detect_mode = true; ///< If true, detect mode from dimensions

            // Workspace buffers (pre-allocated for zero hot-path allocation)
            TensorBase *workspace_scores = nullptr;  ///< [n_heads * seq_len, kv_len]
            TensorBase *workspace_context = nullptr; ///< [seq_len, head_dim]
            TensorBase *workspace_mask = nullptr;    ///< [seq_len, kv_len] or nullptr

            // KV cache for dynamic length query at execution time
            // When set, kv_len is queried from cache instead of using the static value
            // This enables declarative graph construction where the actual kv_len
            // depends on prior KVCacheAppendStage execution
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = -1; ///< Layer index for KV cache query

            // Position offset for decode mode causal masking
            // In decode mode (seq_len < kv_len), this indicates the query's global position
            // For proper causal masking, query at position_offset can attend to [0, position_offset]
            // If 0 (default), will be auto-computed as (kv_len - seq_len) for decode mode
            int position_offset = 0;

            // Optional MPI context (for distributed logging, not tensor-parallel here)
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1; ///< Target device (-1 = CPU, ≥0 = GPU)
        };

        explicit AttentionComputeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief SwiGLU activation stage
     *
     * Type-safe implementation using TensorBase* instead of void*.
     * The tensor's native_type() determines precision dispatch.
     * Uses typed kernel dispatch based on tensor precision.
     */
    class SwiGLUStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *gate = nullptr; ///< Gate tensor [seq_len, intermediate_dim]
            const TensorBase *up = nullptr;   ///< Up tensor [seq_len, intermediate_dim]
            TensorBase *output = nullptr;     ///< Output tensor [seq_len, intermediate_dim]

            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit SwiGLUStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SWIGLU; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

    /**
     * @brief Residual addition stage: output = input + residual
     *
     * Precision-aware implementation:
     * - FP32: Simple float addition
     * - BF16/FP16: Convert to FP32, add, convert back
     * - Q8_1: Native block addition via simd::q8_1_add_q8_1()
     *
     * The Q8_1 path performs fused dequant-add-requant in registers,
     * avoiding memory traffic for intermediate FP32 values.
     */
    class ResidualAddStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Type-safe tensor pointers (required)
            const TensorBase *input = nullptr;    ///< Input tensor (projection output)
            const TensorBase *residual = nullptr; ///< Residual tensor (previous hidden state)
            TensorBase *output = nullptr;         ///< Output tensor (can be same as residual for in-place)

            // Number of elements to process (0 = use input->numel())
            // IMPORTANT: For decode mode with pre-allocated buffers, this must be set to
            // seq_len * hidden_dim to avoid processing garbage data beyond the actual sequence.
            size_t num_elements = 0;

            // Optional MPI context for distributed execution
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit ResidualAddStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ADD_RESIDUAL; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        // Type-specific implementations (dispatch based on tensor native_type)
        // num_elements parameter avoids out-of-bounds access with pre-allocated buffers
        bool executeFP32(IDeviceContext *ctx, size_t num_elements);
        bool executeBF16(IDeviceContext *ctx, size_t num_elements);
        bool executeFP16(IDeviceContext *ctx, size_t num_elements);
        bool executeQ8_1(IDeviceContext *ctx, size_t num_elements);
    };

    // =============================================================================
    // Model-Level Stages (for ModelExecutor)
    // =============================================================================

    /**
     * @brief Embedding lookup stage
     *
     * Performs token → embedding lookup for transformer input processing.
     * Supports output to various precision formats (FP32, BF16, Q8_1).
     *
     * The embedding table is typically FP32 (stored in model weights).
     * Output format is determined by the output tensor's native_type().
     */
    class EmbeddingStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            const TensorBase *embed_table = nullptr; ///< Embedding table [vocab_size, d_model]
            const int *token_ids = nullptr;          ///< Token IDs to look up
            TensorBase *output = nullptr;            ///< Output [num_tokens, d_model]

            // Dimensions
            int num_tokens = 0; ///< Number of tokens to look up
            int d_model = 0;    ///< Embedding dimension
            int vocab_size = 0; ///< Vocabulary size (for bounds checking)

            // Batched input (alternative to token_ids)
            const std::vector<std::vector<int>> *token_batches = nullptr;
            int padded_seq_len = 0; ///< Padded sequence length for batched

            // Device placement
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit EmbeddingStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::EMBEDDING; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        /// Zero out a row of the output tensor (for padding)
        static void zero_output_row(TensorBase *output, int row_idx, int d_model);
    };

    /**
     * @brief Language model head projection stage
     *
     * Projects hidden states to vocabulary logits for token prediction.
     * Typically the final stage in a forward pass before sampling.
     *
     * This stage wraps a GEMM operation but provides semantic clarity
     * in compute graphs and enables LM head-specific optimizations.
     */
    class LMHeadStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            const TensorBase *hidden_states = nullptr;  ///< Hidden states [seq_len, d_model]
            const TensorBase *lm_head_weight = nullptr; ///< LM head weights [vocab_size, d_model]
            TensorBase *logits = nullptr;               ///< Output logits [seq_len, vocab_size]

            // Dimensions
            int seq_len = 0;    ///< Sequence length
            int d_model = 0;    ///< Model dimension
            int vocab_size = 0; ///< Vocabulary size

            // Optional bias (some models have LM head bias)
            const float *bias = nullptr; ///< LM head bias [vocab_size] (nullable)

            // Device placement
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit LMHeadStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::LM_HEAD; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

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
            TensorBase *buffer = nullptr; ///< Buffer to allreduce (in-place)
            void *mpi_comm = nullptr;     ///< MPI communicator (cast to MPI_Comm)
        };

        explicit AllreduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

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
            const TensorBase *hidden = nullptr;       ///< Hidden states [seq_len, d_model]
            const TensorBase *gate_weights = nullptr; ///< Router weights [d_model, num_experts]
            TensorBase *router_logits = nullptr;      ///< Output: router logits [seq_len, num_experts]
            int seq_len = 0;
            int d_model = 0;
            int num_experts = 0;
        };

        explicit MoERouterStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

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
            int expert_id = 0;                               ///< Which expert this is
            const TensorBase *input = nullptr;               ///< Input tokens for this expert
            TensorBase *output = nullptr;                    ///< Output buffer
            const TensorBase *gate_w = nullptr;              ///< Expert gate weights
            const TensorBase *up_w = nullptr;                ///< Expert up weights
            const TensorBase *down_w = nullptr;              ///< Expert down weights
            const std::vector<int> *token_indices = nullptr; ///< Which tokens to process
            int d_model = 0;
            int intermediate_dim = 0;
        };

        explicit MoEExpertStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override;
        size_t estimatedFlops() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

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
            const std::vector<const TensorBase *> *expert_outputs = nullptr; ///< Outputs from each expert
            const std::vector<float> *expert_weights = nullptr;              ///< Router weights per token-expert
            const std::vector<int> *token_expert_map = nullptr;              ///< Which experts each token used
            TensorBase *output = nullptr;                                    ///< Combined output [seq_len, d_model]
            int seq_len = 0;
            int d_model = 0;
            int top_k = 0; ///< Experts per token
        };

        explicit MoECombineStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_COMBINE; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;

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
         * @brief Create an Allreduce stage for MPI collective sum
         *
         * Used after row-parallel GEMM to combine partial results across ranks.
         */
        static std::unique_ptr<IComputeStage> createAllreduce(
            const AllreduceStage::Params &params);

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
